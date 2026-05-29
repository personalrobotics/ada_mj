# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Feeding task — the complete bite cycle.

Composes feeding behaviors into a single ``feed_bite`` function.
This is the ADA feeding pipeline in ~40 lines of readable Python.

The same code runs in MuJoCo simulation and on real hardware —
only the ExecutionContext changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mj_manipulator.outcome import FailureKind, Outcome, failure, success

from ada_mj.feeding.behaviors import (
    acquire_food,
    detect_mouth,
    extract_food,
    level_fork,
    move_above,
    retract_from_mouth,
    tare_ft,
    tilt_fork,
    transfer_to_mouth,
    wait_for_bite,
)
from ada_mj.feeding.domain import AcquisitionSchema, FoodItem, straight_skewer

if TYPE_CHECKING:
    from mj_manipulator.protocols import ExecutionContext

logger = logging.getLogger(__name__)


def feed_bite(
    food: FoodItem,
    schema: AcquisitionSchema | None = None,
    *,
    robot,
    ctx: ExecutionContext,
) -> Outcome:
    """Execute one complete feeding cycle: acquire → transfer → retract.

    Args:
        food: The food item to acquire.
        schema: Acquisition strategy. If None, uses straight_skewer().
        robot: ADA robot instance (for mouth detection, arm access).
        ctx: Execution context (SimContext or HardwareContext).

    Returns:
        Outcome with success=True if the full cycle completed.
        On failure, returns the specific FailureKind from the
        failing step. The caller can retry with a different schema
        or escalate.
    """
    if schema is None:
        schema = straight_skewer()

    arm = robot.arm

    def _failed(step: str, result: Outcome) -> Outcome:
        # The underlying ``failure_code`` (e.g. "ft_guarded_move:no_progress")
        # only names the *behavior*; three pipeline steps call the same
        # behavior, so the step name has to come from here.
        logger.warning(
            "feed_bite[%s]: %s failed (%s)",
            food.name, step, result.failure_code or result.failure_kind,
        )
        return result

    logger.info("feed_bite[%s]: tare_ft", food.name)
    result = tare_ft(arm)
    if not result:
        return _failed("tare_ft", result)

    # Tilt before move_above so the articutool pose is fixed when ForkTSR
    # captures T_ee_to_fork_tip and the planner solves arm IK.
    logger.info("feed_bite[%s]: tilt_fork (%.1f deg)", food.name, schema.tilt_angle)
    result = tilt_fork(schema.tilt_angle, ctx=ctx)
    if not result:
        return _failed("tilt_fork", result)

    logger.info("feed_bite[%s]: move_above", food.name)
    result = move_above(food, schema, arm=arm, ctx=ctx, fork_tsr=robot.fork_tsr)
    if not result:
        return _failed("move_above", result)

    logger.info("feed_bite[%s]: acquire_food", food.name)
    result = acquire_food(food, schema, arm=arm, ctx=ctx)
    if result.failure_kind == FailureKind.SAFETY_ABORTED:
        return _failed("acquire_food", result)  # never retry safety aborts
    if not result:
        return _failed("acquire_food", result)

    logger.info("feed_bite[%s]: extract_food", food.name)
    result = extract_food(schema, arm=arm, ctx=ctx)
    if not result:
        return _failed("extract_food", result)

    # level_fork is best-effort: proceed even if leveling fails.
    logger.info("feed_bite[%s]: level_fork (best-effort)", food.name)
    level_result = level_fork(arm=arm, ctx=ctx)
    if not level_result:
        logger.warning(
            "feed_bite[%s]: level_fork failed (%s) — continuing",
            food.name, level_result.failure_code or level_result.failure_kind,
        )

    logger.info("feed_bite[%s]: detect_mouth", food.name)
    mouth_pose = detect_mouth(robot)
    if mouth_pose is None:
        return _failed("detect_mouth", failure(
            FailureKind.PERCEPTION_FAILED,
            "feed_bite:mouth_not_detected",
        ))

    logger.info("feed_bite[%s]: transfer_to_mouth", food.name)
    result = transfer_to_mouth(mouth_pose, arm=arm, ctx=ctx, fork_tsr=robot.fork_tsr)
    if not result:
        return _failed("transfer_to_mouth", result)

    logger.info("feed_bite[%s]: wait_for_bite", food.name)
    wait_for_bite(arm=arm, ctx=ctx, timeout=10.0)

    logger.info("feed_bite[%s]: retract_from_mouth", food.name)
    retract_from_mouth(mouth_pose, arm=arm, ctx=ctx)

    logger.info("feed_bite[%s]: complete", food.name)
    return success(food=food.name)


def feeding_demo(
    food_items: list[FoodItem],
    *,
    robot,
    ctx: ExecutionContext,
) -> Outcome:
    """Run a full feeding session — acquire and deliver each food item.

    Continues through the list, skipping items that fail (e.g., can't
    reach, can't plan). Stops immediately on safety abort.

    Args:
        food_items: List of food items to feed.
        robot: ADA robot instance.
        ctx: Execution context.

    Returns:
        Outcome with details including which items succeeded/failed.
    """
    succeeded = []
    failed = []

    for food in food_items:
        logger.info("Feeding: %s", food.name)
        result = feed_bite(food, robot=robot, ctx=ctx)

        if result.failure_kind == FailureKind.SAFETY_ABORTED:
            logger.warning("Safety abort during %s — stopping", food.name)
            robot.go_to("stow")
            failed.append(food.name)
            return failure(
                FailureKind.SAFETY_ABORTED,
                "feeding_demo:safety_abort",
                succeeded=succeeded,
                failed=failed,
                aborted_on=food.name,
            )

        if result:
            succeeded.append(food.name)
        else:
            logger.warning(
                "Failed to feed %s (%s) — skipping",
                food.name,
                result.failure_kind.value if result.failure_kind else "unknown",
            )
            failed.append(food.name)

    robot.go_to("stow")
    return success(succeeded=succeeded, failed=failed)
