# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Feeding demo — acquire food from the plate and deliver it to the mouth.

Helpers available in the console after launching with ``--demo feeding``:

  food_items()              — list of FoodItem instances for the food bodies
                              currently in the scene
  observe(tilt_max=0)       — move to the observe_plate staging config: the
                              camera framing the whole plate. Solves a CameraTSR
                              once and caches the config; returns to it after.
  feed(food=None)           — run one feed_bite cycle for ``food`` (defaults
                              to the first food on the plate)
  feed_all()                — full session: observe → bite → re-observe until
                              the plate is clear (hides each eaten item)
  move_above_food(food=None)— just the pre-acquisition phase (TSR-plan the
                              fork tip above a food item, schema-tilted)
  transfer(plan=0.15, servo=0.10)
                            — just the mouth-transfer phase (TSR-plan to a
                              staging distance, then servo to a standoff)

The demo expects the ``table`` scene (ADAConfig.scene = "table"), which the
loader sets automatically when ``--demo feeding`` is passed on the CLI.
"""

from __future__ import annotations

scene = {
    "name": "table",
    # Start at "above_plate" — the wrist camera looks down at the plate
    # from here, which is the natural starting view for a feeding session.
    # (The default "stow" keyframe parks the fork inside the plate, so we
    # need a different starting pose anyway.)
    "initial_pose": "above_plate",
}


def _on_data_thread(fn):
    """Run ``fn`` on the thread that owns MuJoCo ``data``.

    Reads/writes of live state (detect, hide) must happen on the physics
    owner thread. ``run_on_physics_thread`` runs ``fn`` directly when already
    there (the console/tick-driven case) and marshals it otherwise.
    """
    ctx = robot._active_context
    el = getattr(ctx, "_event_loop", None) if ctx is not None else None
    return el.run_on_physics_thread(fn) if el is not None else fn()


def food_items():
    """Return the :class:`FoodItem` list currently on the plate.

    Reflects live MuJoCo state, so eaten items (hidden by the feeding loop) no
    longer appear.
    """
    from ada_mj.scenes.table import detect_food

    return _on_data_thread(lambda: detect_food(robot.model, robot.data))


def observe(tilt_max=0.0, force_replan=False):
    """Move to the ``observe_plate`` staging config — camera framing the plate.

    Solves a CameraTSR over the plate (camera looking down at the plate center,
    far enough that the whole plate fits the view), executes the plan, and
    caches the achieved config for fast repeatable returns. After this,
    ``robot.camera.render_color()`` shows the entire plate.

    Args:
        tilt_max: Half-angle (degrees) of the look-at cone off vertical. 0 →
            straight-down-overhead. Widen if the overhead pose is unreachable.
        force_replan: Re-solve the TSR even if a config is already cached.
    """
    import numpy as np

    from ada_mj.feeding.behaviors import observe_plate
    from ada_mj.scenes.table import PLATE_RADIUS, plate_pose

    pose = plate_pose(robot.model, robot.data)
    if pose is None:
        print("No plate_center site — is the table scene loaded?")
        return None

    return observe_plate(
        robot,
        robot._active_context,
        plate_pose=pose,
        plate_radius=PLATE_RADIUS,
        tilt_max=np.radians(tilt_max),
        force_replan=force_replan,
    )


def feed(food=None, schema=None):
    """Run one ``feed_bite`` cycle.

    Args:
        food: A :class:`FoodItem`. Defaults to the first item from
            :func:`food_items`.
        schema: Acquisition schema. Defaults to ``straight_skewer()``.
    """
    from ada_mj.feeding.task import feed_bite

    if food is None:
        items = food_items()
        if not items:
            print("No food items found.")
            return None
        food = items[0]
    return feed_bite(food, schema, robot=robot, ctx=robot._active_context)


def feed_all(tilt_max=0.0):
    """Run the full feeding session: observe → bite → re-observe until clear.

    Returns to the plate-framing observe pose after each bite and re-detects,
    hiding each eaten item so the loop terminates when the plate is clear.

    Args:
        tilt_max: Look-at cone half-angle (degrees) for the observe pose.
    """
    import numpy as np

    from ada_mj.feeding.task import feeding_session
    from ada_mj.scenes.table import PLATE_RADIUS, detect_food, hide_food, plate_pose

    pose = _on_data_thread(lambda: plate_pose(robot.model, robot.data))
    if pose is None:
        print("No plate_center site — is the table scene loaded?")
        return None

    return feeding_session(
        robot,
        robot._active_context,
        detect_food=lambda: _on_data_thread(
            lambda: detect_food(robot.model, robot.data)
        ),
        consume_food=lambda food: _on_data_thread(
            lambda: hide_food(robot.model, robot.data, food.name)
        ),
        plate_pose=pose,
        plate_radius=PLATE_RADIUS,
        tilt_max=np.radians(tilt_max),
    )


def move_above_food(food=None):
    """Just the pre-acquisition phase: tilt the fork, then TSR-plan above food.

    Useful for visualizing what ``move_above`` does without running the whole
    bite cycle.
    """
    from ada_mj.feeding.behaviors import move_above, tilt_fork
    from ada_mj.feeding.domain import straight_skewer

    if food is None:
        items = food_items()
        if not items:
            print("No food items found.")
            return None
        food = items[0]
    schema = straight_skewer()
    res = tilt_fork(schema.tilt_angle, ctx=robot._active_context)
    if not res:
        return res
    return move_above(
        food, schema, arm=robot.arm, ctx=robot._active_context, fork_tsr=robot.fork_tsr,
    )


def transfer(plan: float = 0.15, servo: float = 0.10):
    """Just the mouth-transfer phase: TSR-plan to ``plan`` m, servo to ``servo`` m.

    Useful for visualizing what ``transfer_to_mouth`` does without running
    the whole bite cycle. Defaults are conservative (15 cm staging,
    10 cm servo standoff) to stay clear of the head collision envelope.
    """
    from ada_mj.feeding.behaviors import transfer_to_mouth

    mouth_pose = robot.head.get_mouth_pose()
    return transfer_to_mouth(
        mouth_pose, arm=robot.arm, ctx=robot._active_context, fork_tsr=robot.fork_tsr,
        plan_approach_distance=plan, servo_approach_distance=servo,
    )
