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

    # 1. Tare F/T sensor (baseline for contact detection)
    result = tare_ft(arm)
    if not result:
        return result

    # 2. Move above the food item
    result = move_above(food, schema, arm=arm, ctx=ctx)
    if not result:
        return result

    # 3. Tilt fork for acquisition
    result = tilt_fork(schema.tilt_angle, ctx=ctx)
    if not result:
        return result

    # 4. Stab into food
    result = acquire_food(food, schema, arm=arm, ctx=ctx)
    if result.failure_kind == FailureKind.SAFETY_ABORTED:
        return result  # never retry safety aborts
    if not result:
        return result

    # 5. Extract food from plate
    result = extract_food(schema, arm=arm, ctx=ctx)
    if not result:
        return result

    # 6. Level fork for transport (best-effort — proceed even if leveling fails)
    level_fork(arm=arm, ctx=ctx)

    # 7. Detect mouth
    mouth_pose = detect_mouth(robot)
    if mouth_pose is None:
        return failure(
            FailureKind.PERCEPTION_FAILED,
            "feed_bite:mouth_not_detected",
        )

    # 8. Move to mouth
    result = transfer_to_mouth(mouth_pose, arm=arm, ctx=ctx)
    if not result:
        return result

    # 9. Wait for bite
    wait_for_bite(arm=arm, ctx=ctx, timeout=10.0)

    # 10. Retract from mouth (along mouth approach axis)
    retract_from_mouth(mouth_pose, arm=arm, ctx=ctx)

    logger.info("Feed bite complete: %s", food.name)
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
