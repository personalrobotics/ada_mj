# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Feeding behaviors — plain functions returning Outcome.

Each behavior is a single manipulation step in the feeding pipeline.
They call mj_manipulator primitives (servo_to_pose, ft_guarded_move,
arm.plan_to_configuration, ctx.execute) and return structured Outcomes.

Architectural rules:
- Import only: ada_mj.feeding.domain, mj_manipulator.outcome,
  mj_manipulator.servo, mj_manipulator.force_control
- No: import mujoco, import viser, import rclpy
- Every function returns Outcome
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from mj_manipulator.outcome import FailureKind, Outcome, failure, success
from mj_manipulator.servo import ft_guarded_move, servo_to_pose

from ada_mj.feeding.domain import (
    MOUTH_APPROACH_SPEED,
    MOUTH_FT_THRESHOLD,
    MOUTH_POSITION_TOL,
    MOUTH_RETRACT_SPEED,
    AcquisitionSchema,
    FoodItem,
)

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import ExecutionContext

logger = logging.getLogger(__name__)


def tare_ft(arm: Arm) -> Outcome:
    """Zero the F/T sensor baseline.

    Call before acquisition so force readings reflect food contact only.
    Skipped silently if no F/T sensor or in kinematic mode.
    """
    if not arm.has_ft_sensor or not arm.ft_valid:
        return success()
    try:
        arm.tare_ft()
        return success()
    except Exception as e:
        return failure(
            FailureKind.PRECONDITION_FAILED,
            "tare_ft:sensor_error",
            error=str(e),
        )


def move_above(
    food: FoodItem,
    schema: AcquisitionSchema,
    *,
    arm: Arm,
    ctx: ExecutionContext,
) -> Outcome:
    """Plan and execute motion to the approach pose above a food item.

    The approach pose is the food position offset by the schema's
    approach_offset (typically a few cm above).
    """
    target_pos = food.position + schema.approach_offset
    q_goal = _ik_to_position(arm, target_pos)
    if q_goal is None:
        return failure(
            FailureKind.PLANNING_FAILED,
            "move_above:no_ik_solution",
            food=food.name,
            target_pos=target_pos.tolist(),
        )

    path = arm.plan_to_configuration(q_goal)
    if path is None:
        return failure(
            FailureKind.PLANNING_FAILED,
            "move_above:no_path",
            food=food.name,
        )

    traj = arm.retime(path)
    if not ctx.execute(traj):
        return failure(
            FailureKind.EXECUTION_FAILED,
            "move_above:execution_failed",
            food=food.name,
        )

    return success(food=food.name)


def tilt_fork(angle: float, *, ctx: ExecutionContext) -> Outcome:
    """Set the articutool tilt angle.

    Controls the fork pitch for skewering (tilted down) or
    leveling (horizontal for transport).
    """
    controller = ctx._controller
    if controller is None:
        return failure(FailureKind.PRECONDITION_FAILED, "tilt_fork:no_controller")

    entity_state = controller._entities.get("articutool")
    if entity_state is None:
        return failure(FailureKind.PRECONDITION_FAILED, "tilt_fork:no_articutool")

    # Set tilt (joint 0), preserve roll (joint 1)
    current = entity_state.target_position.copy()
    current[0] = angle
    entity_state.target_position = current
    controller.step()

    return success(tilt_angle=angle)


def acquire_food(
    food: FoodItem,
    schema: AcquisitionSchema,
    *,
    arm: Arm,
    ctx: ExecutionContext,
) -> Outcome:
    """Stab into food with F/T monitoring.

    Applies the schema's insertion_twist for insertion_duration while
    monitoring F/T against grasp_thresholds. Contact detection
    (threshold exceeded) is expected — it means the fork hit food.
    """
    result = ft_guarded_move(
        schema.insertion_twist,
        arm,
        ctx,
        ft_threshold=schema.grasp_thresholds,
        duration=schema.insertion_duration,
    )
    if not result:
        return result

    contact = result.details.get("contact", False)
    logger.info(
        "Acquisition %s: contact=%s, force=%.1fN",
        food.name,
        contact,
        result.details.get("force_n", 0.0),
    )

    return success(food=food.name, contact=contact)


def extract_food(
    schema: AcquisitionSchema,
    *,
    arm: Arm,
    ctx: ExecutionContext,
) -> Outcome:
    """Pull the fork up after stabbing.

    Applies the schema's extraction_twist with extraction_thresholds.
    High thresholds (50N) allow pulling through resistance.
    """
    return ft_guarded_move(
        schema.extraction_twist,
        arm,
        ctx,
        ft_threshold=schema.extraction_thresholds,
        duration=schema.extraction_duration,
    )


def level_fork(*, arm: Arm, ctx: ExecutionContext) -> Outcome:
    """Level the articutool to keep food horizontal during transport.

    Computes the articutool tilt angle that keeps the fork level
    given the current arm configuration. This prevents food from
    sliding off during the move-to-mouth phase.
    """
    # TODO: compute gravity-compensating tilt from arm FK.
    # Full implementation: extract EE pitch from arm.get_ee_pose(),
    # compute articutool angle that keeps fork horizontal.
    # For now, set tilt to 0 (horizontal).
    return tilt_fork(0.0, ctx=ctx)


def detect_mouth(robot) -> np.ndarray:
    """Read the mouth site pose from the head model.

    Returns:
        4x4 world-frame transform of the mouth.
    """
    return robot.head.get_mouth_pose()


def transfer_to_mouth(
    mouth_pose: np.ndarray,
    *,
    arm: Arm,
    ctx: ExecutionContext,
    abort_fn=None,
) -> Outcome:
    """Move the loaded fork to the user's mouth.

    Two phases:
    1. Plan and execute trajectory to a staging pose near the mouth
    2. Servo the final approach with deceleration and F/T monitoring

    The speed profile decelerates from 0.15 m/s to 0.06 m/s as the
    fork approaches the mouth (over the last 30cm). F/T threshold is
    1N — very sensitive to detect lip/face contact.
    """
    # Compute approach target: offset from mouth by fork length
    # The fork approaches along the mouth's +x axis (forward out of mouth)
    approach_distance = 0.02  # stop 2cm from mouth
    target = mouth_pose.copy()
    target[:3, 3] += mouth_pose[:3, 0] * approach_distance

    return servo_to_pose(
        target,
        arm,
        ctx,
        speed_profile=MOUTH_APPROACH_SPEED,
        ft_threshold=MOUTH_FT_THRESHOLD,
        position_tol=MOUTH_POSITION_TOL,
        timeout=15.0,
    )


def wait_for_bite(
    *,
    arm: Arm,
    ctx: ExecutionContext,
    timeout: float = 10.0,
) -> Outcome:
    """Wait for the user to take a bite (F/T spike from pulling food off).

    Holds position and monitors F/T. Returns success when force exceeds
    the bite detection threshold or timeout elapses.
    """
    from mj_manipulator.force_control import ForceThresholds

    bite_threshold = ForceThresholds(force_n=2.0, torque_nm=1.0)

    # Zero twist — hold position, just monitor F/T
    return ft_guarded_move(
        np.zeros(6),
        arm,
        ctx,
        ft_threshold=bite_threshold,
        duration=timeout,
        timeout=timeout + 1.0,
    )


def retract_from_mouth(
    *,
    arm: Arm,
    ctx: ExecutionContext,
) -> Outcome:
    """Retract the fork from the mouth area.

    Servos backward from the current position with the retraction
    speed profile (slightly faster than approach).
    """
    # Retract 15cm straight back along -x of current EE frame
    ee_pose = arm.get_ee_pose()
    target = ee_pose.copy()
    target[:3, 3] -= ee_pose[:3, 0] * 0.15  # 15cm back along approach axis

    return servo_to_pose(
        target,
        arm,
        ctx,
        speed_profile=MOUTH_RETRACT_SPEED,
        timeout=10.0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ik_to_position(arm: Arm, position: np.ndarray) -> np.ndarray | None:
    """Solve IK for a target position, keeping current orientation.

    Returns the closest joint configuration or None if unreachable.
    """
    target_pose = arm.get_ee_pose().copy()
    target_pose[:3, 3] = position

    if arm.ik_solver is None:
        return None

    solutions = arm.ik_solver.solve(target_pose, q_init=arm.get_joint_positions())
    if not solutions:
        return None

    # Pick closest to current
    q_current = arm.get_joint_positions()
    best = min(solutions, key=lambda q: float(np.linalg.norm(np.array(q) - q_current)))
    return np.array(best)
