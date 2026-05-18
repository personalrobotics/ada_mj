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
    fork_tsr,
    xy_slop: float = 0.008,
) -> Outcome:
    """Plan and execute motion to a fork-tip-above-food pose.

    Builds a placement TSR over the food using ``ForkTSR.above_plate``
    (fork tip pointing down, x/y free within ``xy_slop`` of the food,
    yaw free, z at the schema's hover height above the food), then
    calls ``arm.plan_to_tsrs`` so the planner picks any collision-free
    configuration in that region.

    Args:
        food: Target food item (world-frame position).
        schema: Acquisition schema; ``schema.approach_offset[2]`` sets the
            hover height above the food (in meters).
        arm: The arm to plan with.
        ctx: Execution context (sim or hardware).
        fork_tsr: Fork TSR generator (``robot.fork_tsr``).
        xy_slop: Half-width of the xy region the planner may pick the
            fork tip within, around the food's xy (meters). Tight by
            default so the subsequent acquisition stab actually hits the
            food.
    """
    hover_height = abs(float(schema.approach_offset[2]))

    # Reference frame: food position, identity rotation (z-up world).
    T_food_world = np.eye(4)
    T_food_world[:3, 3] = food.position

    # k=1 → single height template at exactly hover_height; the planner
    # gets x/y slop and yaw freedom but pins the tip at the requested z.
    templates = fork_tsr.above_plate(
        plate_radius=xy_slop, hover_height=hover_height, k=1,
    )
    goal_tsrs = [t.instantiate(T_food_world) for t in templates]

    path = arm.plan_to_tsrs(goal_tsrs)
    if path is None:
        return failure(
            FailureKind.PLANNING_FAILED,
            "move_above:no_path",
            food=food.name,
            food_pos=food.position.tolist(),
        )

    traj = arm.retime(path)
    if ctx.execute(traj):
        return success(food=food.name)
    return failure(
        FailureKind.EXECUTION_FAILED,
        "move_above:execution_failed",
        food=food.name,
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
    fork_tsr,
    plan_approach_distance: float = 0.05,
    servo_approach_distance: float = 0.02,
) -> Outcome:
    """Move the loaded fork to the user's mouth in two phases.

    Phase 1 (long-range, collision-aware): plan an arm path to a fork-tip
    approach TSR — fork tip ``plan_approach_distance`` in front of the
    mouth along its +x axis, pointing toward the mouth, with small
    lateral and orientation freedom for the planner to find a feasible
    config. Uses ``arm.plan_to_tsrs`` so the planner samples within the
    TSR and picks any collision-free configuration.

    Phase 2 (close-range, force-aware): from the planned pose, servo the
    last few cm to ``servo_approach_distance`` from the mouth using
    ``servo_to_pose`` with the mouth approach speed profile and a tight
    F/T threshold (1 N) so lip / face contact aborts safely.

    The articutool joints are frozen at planning time (snapshotted into
    the TSR's ``Tw_e``); only arm joints move during phase 1, and the
    servo in phase 2 preserves the EE orientation.

    Args:
        mouth_pose: 4×4 world-frame pose of the mouth (+x out of face).
        arm: The arm to plan / servo with.
        ctx: Execution context (sim or hardware).
        fork_tsr: Fork TSR generator (``robot.fork_tsr``).
        plan_approach_distance: How far in front of the mouth the planner
            should target the fork tip (meters). Default 5 cm.
        servo_approach_distance: Final standoff distance after the servo
            (meters). Default 2 cm.
    """
    # Phase 1: plan into the approach TSR
    templates = fork_tsr.approach_mouth(
        approach_distance=plan_approach_distance, k=3,
    )
    goal_tsrs = [t.instantiate(mouth_pose) for t in templates]

    path = arm.plan_to_tsrs(goal_tsrs)
    if path is None:
        return failure(
            FailureKind.PLANNING_FAILED,
            "transfer_to_mouth:no_path",
            mouth_pos=mouth_pose[:3, 3].tolist(),
        )
    traj = arm.retime(path)
    if not ctx.execute(traj):
        return failure(
            FailureKind.EXECUTION_FAILED,
            "transfer_to_mouth:plan_execute_failed",
        )

    # Phase 2: servo the last few cm with F/T monitoring.
    # The TSR plan landed the fork tip ~plan_approach_distance from the
    # mouth. We want the fork tip at servo_approach_distance from the
    # mouth, preserving the EE orientation that the plan achieved. Since
    # the articutool joints are frozen during the servo, translating the
    # EE by the desired fork-tip delta translates the fork tip by the
    # same amount.
    tip_now = fork_tsr.tip_world_pos()
    tip_target = mouth_pose[:3, 3] + mouth_pose[:3, 0] * servo_approach_distance
    target = arm.get_ee_pose().copy()
    target[:3, 3] = target[:3, 3] + (tip_target - tip_now)

    return servo_to_pose(
        target,
        arm,
        ctx,
        speed_profile=MOUTH_APPROACH_SPEED,
        ft_threshold=MOUTH_FT_THRESHOLD,
        position_tol=MOUTH_POSITION_TOL,
        timeout=10.0,
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


