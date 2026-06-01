# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the feeding_session loop (observe → detect → bite → consume).

The loop logic is tested in isolation from feed_bite/observe_plate (which are
gated on separate motion blockers) by monkeypatching them: this catches loop
bugs — non-termination, consuming on failure, not skipping a failed item,
mishandling a safety abort — that the full pipeline would hide.
"""

from __future__ import annotations

import numpy as np
import pytest
from mj_manipulator.outcome import FailureKind, failure, success

import ada_mj.feeding.task as task
from ada_mj.feeding.domain import FoodItem


def _foods(n):
    return [
        FoodItem(name=f"food/x_{i}", position=np.zeros(3), food_type="x")
        for i in range(n)
    ]


def _run(monkeypatch, *, feed_bite_result, foods, consume_removes=True, max_bites=None):
    """Drive feeding_session with stubbed observe/feed and a mutable plate.

    Returns (outcome, remaining_names, consumed_names).
    """
    remaining = {f.name: f for f in foods}
    consumed = []

    monkeypatch.setattr(task, "observe_plate", lambda *a, **k: success())
    monkeypatch.setattr(task, "feed_bite", lambda food, **k: feed_bite_result(food))

    def detect():
        return list(remaining.values())

    def consume(f):
        consumed.append(f.name)
        if consume_removes:
            remaining.pop(f.name, None)

    outcome = task.feeding_session(
        robot=object(),
        ctx=object(),
        detect_food=detect,
        consume_food=consume,
        plate_pose=np.eye(4),
        plate_radius=0.1,
        max_bites=max_bites,
    )
    return outcome, set(remaining), consumed


def test_session_eats_and_consumes_all(monkeypatch):
    foods = _foods(3)
    outcome, remaining, consumed = _run(
        monkeypatch, feed_bite_result=lambda f: success(food=f.name), foods=foods
    )
    assert outcome
    assert sorted(outcome.details["succeeded"]) == [f.name for f in foods]
    assert sorted(consumed) == [f.name for f in foods]
    assert remaining == set()  # plate cleared → loop terminated


def test_session_skips_failed_items_and_terminates(monkeypatch):
    """A non-safety failure must skip the item (not consume) and still halt."""
    foods = _foods(2)
    outcome, remaining, consumed = _run(
        monkeypatch,
        feed_bite_result=lambda f: failure(FailureKind.PLANNING_FAILED, "x:no_path"),
        foods=foods,
        max_bites=10,  # guard: test fails loudly if the loop spins instead of skipping
    )
    assert outcome  # session completed (didn't hang)
    assert sorted(outcome.details["failed"]) == [f.name for f in foods]
    assert consumed == []  # failures are never consumed
    assert remaining == {f.name for f in foods}  # food untouched on the plate


def test_session_safety_abort_stops_immediately(monkeypatch):
    foods = _foods(3)
    outcome, remaining, consumed = _run(
        monkeypatch,
        feed_bite_result=lambda f: failure(FailureKind.SAFETY_ABORTED, "x:estop"),
        foods=foods,
    )
    assert not outcome
    assert outcome.failure_kind == FailureKind.SAFETY_ABORTED
    assert outcome.details["aborted_on"] == foods[0].name
    assert consumed == []


def test_session_respects_max_bites(monkeypatch):
    foods = _foods(5)
    outcome, remaining, consumed = _run(
        monkeypatch,
        feed_bite_result=lambda f: success(food=f.name),
        foods=foods,
        max_bites=2,
    )
    assert outcome
    assert len(consumed) == 2  # stopped at the cap, plate not cleared
    assert len(remaining) == 3


def test_session_aborts_if_initial_observe_fails(monkeypatch):
    monkeypatch.setattr(
        task, "observe_plate", lambda *a, **k: failure(FailureKind.PLANNING_FAILED, "o:no_path")
    )
    monkeypatch.setattr(task, "feed_bite", lambda food, **k: success())
    called = []
    outcome = task.feeding_session(
        robot=object(),
        ctx=object(),
        detect_food=lambda: called.append(1) or _foods(1),
        consume_food=lambda f: None,
        plate_pose=np.eye(4),
        plate_radius=0.1,
    )
    assert not outcome
    assert called == []  # never reached detection — bailed on the first observe


@pytest.mark.slow
def test_detect_and_hide_food_in_sim():
    """detect_food / hide_food mechanics against the real table scene."""
    import mujoco

    from ada_mj.config import ADAConfig
    from ada_mj.scenes.table import detect_food, hide_food

    ADA = pytest.importorskip("ada_mj.robot").ADA
    robot = ADA(ADAConfig.default())
    mujoco.mj_forward(robot.model, robot.data)

    foods = detect_food(robot.model, robot.data)
    n0 = len(foods)
    assert n0 >= 1

    assert hide_food(robot.model, robot.data, foods[0].name)
    remaining = detect_food(robot.model, robot.data)
    assert len(remaining) == n0 - 1
    assert foods[0].name not in {f.name for f in remaining}
    assert not hide_food(robot.model, robot.data, "food/nonexistent_99")


@pytest.mark.slow
def test_session_clears_plate_in_sim(monkeypatch):
    """Acceptance: real observe → detect → (stubbed bite) → hide → re-observe
    clears the plate and terminates. Only feed_bite is stubbed — it is gated on
    separate motion blockers — so this exercises the real loop integration:
    observe_plate plans/executes between every bite, detect/hide mutate live
    state, and the loop drives to an empty plate.
    """
    import mujoco

    from ada_mj.config import ADAConfig
    from ada_mj.scenes.table import PLATE_RADIUS, detect_food, hide_food, plate_pose

    ADA = pytest.importorskip("ada_mj.robot").ADA
    robot = ADA(ADAConfig.default())
    monkeypatch.setattr(task, "feed_bite", lambda food, **k: success(food=food.name))

    with robot.sim(physics=False, headless=True) as ctx:
        robot.arm.set_joint_positions(robot.named_poses["above_plate"])
        robot._snap_tool_to_weld()
        mujoco.mj_forward(robot.model, robot.data)
        pose = plate_pose(robot.model, robot.data)
        n0 = len(detect_food(robot.model, robot.data))
        outcome = task.feeding_session(
            robot,
            ctx,
            detect_food=lambda: detect_food(robot.model, robot.data),
            consume_food=lambda f: hide_food(robot.model, robot.data, f.name),
            plate_pose=pose,
            plate_radius=PLATE_RADIUS,
        )
        assert outcome, f"session failed: {outcome.failure_code}"
        assert len(outcome.details["succeeded"]) == n0
        assert detect_food(robot.model, robot.data) == []  # plate cleared
