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
from mj_manipulator.force_control import ForceThresholds
from mj_manipulator.outcome import FailureKind, Outcome, failure, success
from mj_manipulator.servo import ft_guarded_move, servo_to_pose
from mj_manipulator.teleop import SafetyMode

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
    approach_offset (typically a few cm above). Uses IK + planner if
    an IK solver is available, otherwise plans to the target pose
    directly.
    """
    target_pos = food.position + schema.approach_offset

    # Build target pose: food approach position, current EE orientation
    target_pose = arm.get_ee_pose().copy()
    target_pose[:3, 3] = target_pos

    # Try IK → plan_to_configuration (precise, works for EAIK arms)
    q_goal = _ik_to_position(arm, target_pos)
    if q_goal is not None:
        path = arm.plan_to_configuration(q_goal)
        if path is not None:
            traj = arm.retime(path)
            if ctx.execute(traj):
                return success(food=food.name)
            return failure(
                FailureKind.EXECUTION_FAILED,
                "move_above:execution_failed",
                food=food.name,
            )

    # Fallback: plan_to_pose (for arms without IK, e.g., JACO2 with mink)
    if hasattr(arm, "plan_to_pose"):
        path = arm.plan_to_pose(target_pose)
        if path is not None:
            traj = arm.retime(path)
            if ctx.execute(traj):
                return success(food=food.name)

    return failure(
        FailureKind.PLANNING_FAILED,
        "move_above:no_path",
        food=food.name,
        target_pos=target_pos.tolist(),
    )


def tilt_fork(angle: float, *, ctx: ExecutionContext) -> Outcome:
    """Set the articutool tilt angle.

    Controls the fork pitch for skewering (tilted down) or
    leveling (horizontal for transport).
    """
    # NOTE: accesses Controller._entities directly because there's no
    # public API for entity target control yet. When one is added, this
    # should use it instead.
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
        safety_mode=SafetyMode.ALLOW,  # contact with food is expected
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
    from mj_manipulator.teleop import SafetyMode

    return ft_guarded_move(
        schema.extraction_twist,
        arm,
        ctx,
        ft_threshold=schema.extraction_thresholds,
        duration=schema.extraction_duration,
        safety_mode=SafetyMode.ALLOW,  # pulling through food resistance is expected
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


def detect_mouth(robot) -> np.ndarray | None:
    """Read the mouth site pose from the head model.

    In simulation, reads the MuJoCo site directly. On hardware,
    this would dispatch to face detection (MediaPipe/YOLO + depth).

    Returns:
        4x4 world-frame transform of the mouth, or None if
        unavailable (e.g., head not in model, face not detected).
    """
    if not hasattr(robot, "head") or robot.head is None:
        return None
    try:
        return robot.head.get_mouth_pose()
    except Exception:
        return None


def transfer_to_mouth(
    mouth_pose: np.ndarray,
    *,
    arm: Arm,
    ctx: ExecutionContext,
) -> Outcome:
    """Move the loaded fork to the user's mouth.

    Servos from the current pose (staging) toward the mouth position,
    maintaining the current fork orientation. The speed profile
    decelerates from 0.15 m/s to 0.06 m/s near the mouth.
    F/T threshold is 1N — sensitive to detect lip/face contact.

    The fork orientation is set by the staging pose (which the planner
    achieved). The servo preserves it — we translate to the mouth,
    we don't rotate to match the mouth frame.
    """
    # Target: mouth position, current fork orientation.
    # The mouth frame orientation is the face's forward direction —
    # not related to how the fork should be oriented.
    approach_distance = 0.02  # stop 2cm from mouth
    target = arm.get_ee_pose().copy()  # keep current orientation
    target[:3, 3] = mouth_pose[:3, 3] + mouth_pose[:3, 0] * approach_distance

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
    mouth_pose: np.ndarray,
    *,
    arm: Arm,
    ctx: ExecutionContext,
    retract_distance: float = 0.15,
) -> Outcome:
    """Retract the fork away from the mouth.

    Moves the EE along the mouth's approach axis (+x of mouth frame),
    away from the face. Maintains current fork orientation.

    Args:
        mouth_pose: 4x4 mouth pose (for approach axis direction).
        arm: Arm instance.
        ctx: Execution context.
        retract_distance: How far to retract (meters).
    """
    mouth_approach_axis = mouth_pose[:3, 0]  # +x = forward out of mouth
    target = arm.get_ee_pose().copy()
    target[:3, 3] += mouth_approach_axis * retract_distance

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
