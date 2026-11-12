"""ADA grasps and lifts its cylinder in open space, and holds it (``ada-lift``).

``ada-lift --seed 0 --record`` grasps the 6 cm cylinder at its reference pose with the
JACO2's enveloping side grasp, lifts it, and holds still for ``HOLD_NS``. Its report,
``lift.json``, is this test's evidence (``artifacts/test_lift_holds_the_cylinder``). It
shows:
- the grasp succeeding, every stage of it;
- ``acquired`` and ``retained`` assumed, unobserved, by the application's recorded
  policy, and nothing else assumed (personalrobotics/ssgrasping#47);
- the cylinder held at the end of the hold, from privileged state: attached, and risen
  at least ``LIFT_M - LIFT_TOLERANCE_M``;
- the same report, byte for byte, from a second run.

Reproduce with ``uv run pytest tests/test_lift.py``, or ``uv run ada-lift --seed 0
--record <dir>`` and read ``<dir>/lift.json``. Its tilt and attachment violation are
recorded, not judged: they measure the stated limitations (README).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ada_mj import lift


def test_lift_holds_the_cylinder(artifacts: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    assert lift.main(["--seed", "0", "--record", str(first)]) == 0
    assert lift.main(["--seed", "0", "--record", str(second)]) == 0
    report = first / "lift.json"
    shutil.copy(report, artifacts / "lift.json")
    assert report.read_bytes() == (second / "lift.json").read_bytes()

    record = json.loads(report.read_text())
    assert record["grip"] == "enveloping"
    grasp = record["grasp"]
    assert grasp["state"] == "succeeded"
    assert [s["name"] for s in grasp["stages"] if not s["succeeded"]] == []
    assert sorted(grasp["assumed"]) == ["acquired", "retained"]
    assert record["policy"] == lift.POLICY
    assert record["privileged"]["held"] is True
    assert record["privileged"]["rise_m"] >= lift.LIFT_M - lift.LIFT_TOLERANCE_M
