"""ADA's bare hand grasps a cylinder standing on a table, and lifts it, in open space.

The scene is the over-bed table's top and a 6 cm cylinder, nothing else: no obstacle is
near any motion. The robot is ``ada_assets.models``, the JACO2 with its
two-finger hand and no tool, in ssrobot's ``MujocoRuntime``. The cylinder's pose is
registered from its privileged pose channel, standing in for perception; every motion is
then planned against the registration (``ObjectPoses.REGISTERED``).

The grasp is ssgrasping's ``grasp``, from sstsr's JACO2 cylinder side regions of one
grip (``GRIPS``): by default the enveloping grasps, the palm a clearance from the
cylinder and the fingers folded around it (personalrobotics/sstsr#204), which the JACO2
holds; ``fingertip``, sstsr's other side grasps, which its curling fingers pull in and
turn, is kept for comparison. The hand is declared as one whose opening does not
establish a grasp: the close can reach its target around the cylinder
(personalrobotics/ssgrasping#47). So ``acquired`` and ``retained`` are unobserved, and
this application proceeds on them by an explicit policy, ``POLICY`` (``assume_unobserved``),
recorded in every run. After the lift the hand holds still for ``HOLD_NS``.

Whether the cylinder is held is read from privileged state, for evaluation only: nothing
here branches on it. It is held when the grasp succeeded and, at the end of the hold, it
has risen at least ``LIFT_M - LIFT_TOLERANCE_M``. (With registered object poses the
scene keeps the attachment however the cylinder moves, so the attachment says nothing
here.) Its largest tilt over the hold, and
when the runtime first found it moved in the hand past the attachment's tolerance, are
recorded beside that: they measure the first limitation below, and are not failures.

**Limitations.**

* Held-object collision geometry is not conservative after the JACO2 seats the object
  in its palm (personalrobotics/ssrobot#242), so ADA demonstrates grasp and lift in open
  space only, not obstacle-constrained carrying or placement.
* The model's fingers barely squeeze a cylinder narrower than about 5 cm: they reach
  their closed target wrapped around it, so it is held lightly, tilts in the hand and
  creeps down (personalrobotics/ada_assets#50). The cylinder is 6 cm for that reason.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sscbirrt import CBiRRTConfig
from ssgrasping import (
    GraspObject,
    GraspRegion,
    Primitive,
    SkillResult,
    grasp,
    grasp_regions,
    run_skill,
    sstsr,
)
from ssrobot import ObjectPoses, ObservationRequest, Pose, RobotContext, load_installed_package
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from tsr import KinovaJaco2TwoFinger

PACKAGE = "ada_assets.models"
REGISTRY_KEY = "kinova_jaco2_two_finger"
"""sstsr's registry name for the JACO2."""
OBJECT = "cylinder"
SUPPORT = "ada_table"
RADIUS_M, HALF_HEIGHT_M = 0.03, 0.06
"""Within the JACO2's enveloping diameters, and wide enough that its fingers squeeze it
(personalrobotics/ada_assets#50)."""
REFERENCE = Pose(position=(0.3082, -0.6416, 0.7950), quat_wxyz=(0.270668, 0.0, 0.0, 0.962673))
"""Where ssrobot's ``references/scenes/ada-cylinder.xml`` stands its 4 cm cylinder, and
where this one stands for seed 0."""
PLACEMENT_M = 0.03
"""How far, in x and y, any other seed's cylinder stands from ``REFERENCE``, at most; its
yaw is drawn uniformly."""
UP = (0.0, 0.0, 1.0)
APPROACH_M = 0.05
LIFT_M = 0.06
CLEARANCE_M = 0.005
"""The clearance the pregrasp keeps through free space, as Geodude's."""
GRIPPER_TIMEOUT_NS = 30_000_000_000
"""The JACO2's close takes about 12 s to seat the cylinder: a failure bound, not a wait."""
HOLD_NS = 10_000_000_000
"""How long the hand holds still after the lift, before the cylinder is scored: long
enough that a cylinder creeping down in the hand is seen to."""
TILT_SAMPLE_NS = 100_000_000
"""How often the cylinder's tilt is read over the hold."""
LIFT_TOLERANCE_M = 0.01
"""How far short of ``LIFT_M`` the cylinder's rise may fall and still count as lifted:
it settles in the hand."""
MAX_TICKS = 100_000
POLICY = {
    "acquired": "assumed",
    "retained": "assumed",
    "basis": (
        "the JACO2's opening does not establish a grasp, and no portable signal yet tells "
        "(personalrobotics/ssgrasping#47)"
    ),
}
"""What this application assumes of the estimates the grasp leaves unobserved."""
GRIPS = ("enveloping", "fingertip")
"""The side grasps a run draws from: sstsr's ``grip="enveloping"`` templates, or the rest."""


class _SideGrip:
    """The JACO2, its cylinder side grasps narrowed to one of ``GRIPS`` by each template's
    provenance: what ``sstsr.SstsrTemplates`` draws a side region's templates from."""

    def __init__(self, hand: Any, grip: str) -> None:
        self.hand, self.grip = hand, grip

    def grasp_cylinder_side(self, *args: Any, **kwargs: Any) -> list[Any]:
        return [
            t
            for t in self.hand.grasp_cylinder_side(*args, **kwargs)
            if (t.provenance is not None and t.provenance.grip == "enveloping")
            == (self.grip == "enveloping")
        ]


def placement(seed: int) -> Pose:
    """The cylinder's pose for ``seed``: ``REFERENCE`` at 0; otherwise moved by up to
    ``PLACEMENT_M`` in x and y, at a uniformly drawn yaw, standing on the table."""
    if seed == 0:
        return REFERENCE
    rng = random.Random(seed)
    x, y, z = REFERENCE.position
    dx, dy = rng.uniform(-PLACEMENT_M, PLACEMENT_M), rng.uniform(-PLACEMENT_M, PLACEMENT_M)
    yaw = rng.uniform(0.0, 2 * math.pi)
    return Pose(
        position=(round(x + dx, 6), round(y + dy, 6), z),
        quat_wxyz=(round(math.cos(yaw / 2), 6), 0.0, 0.0, round(math.sin(yaw / 2), 6)),
    )


def scene(cylinder: Pose) -> str:
    """The open scene: the over-bed table's collision top (ssrobot's reference scene's),
    and the cylinder at ``cylinder``."""
    pos = " ".join(map(str, cylinder.position))
    quat = " ".join(map(str, cylinder.quat_wxyz))
    return f"""<mujoco model="ada-lift">
  <worldbody>
    <body name="{SUPPORT}" pos="0.06 -0.48 0">
      <geom name="{SUPPORT}_top" type="box" pos="0 0 0.714" size="0.926 0.371 0.021"
            friction="0.8 0.005 0.0001" mass="10"/>
    </body>
    <body name="{OBJECT}" pos="{pos}" quat="{quat}">
      <freejoint/>
      <geom type="cylinder" size="{RADIUS_M} {HALF_HEIGHT_M}" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def lift(seed: int, grip: str = GRIPS[0]) -> dict[str, Any]:
    """Grasp and lift the cylinder of ``seed`` by a ``grip`` side grasp, hold, and return
    the run's record."""
    if grip not in GRIPS:
        raise ValueError(f"grip must be one of {GRIPS}, not {grip!r}")
    package = load_installed_package(PACKAGE)
    description = package.description
    (end_effector,) = description.end_effectors
    (manipulator,) = description.manipulators
    assert end_effector.gripper is not None
    planner = CBiRRT.from_package(
        package, manipulator.name, config=CBiRRTConfig(edge_resolution=0.01)
    )
    hand = KinovaJaco2TwoFinger()
    profile = sstsr.profile(
        hand,
        manipulator=manipulator.name,
        end_effector=end_effector.name,
        gripper=end_effector.gripper,
        opening_channel="gripper_opening",
        opening_establishes_grasp=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.xml"
        path.write_text(scene(placement(seed)))
        runtime = MujocoRuntime(package, scene=path)
        with RobotContext(description, runtime, object_poses=ObjectPoses.REGISTERED) as ctx:
            pose_request = ObservationRequest(channels=(f"{OBJECT}.pose",))

            def observed() -> Pose:
                x, y, z, w, i, j, k = ctx.observe(pose_request).readings[0].value  # type: ignore[misc]
                return Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))

            observation = ctx.observe(pose_request)
            x, y, z, w, i, j, k = observation.readings[0].value  # type: ignore[misc]
            registration = Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))
            ctx.register(OBJECT, registration, stamp=observation.stamp, source="pose_channel")
            regions = grasp_regions(
                GraspObject(
                    OBJECT,
                    registration,
                    (
                        GraspRegion(
                            "body",
                            Primitive("cylinder", (RADIUS_M, 2 * HALF_HEIGHT_M)),
                            strategies=("side",),
                        ),
                    ),
                ),
                profile,
                sstsr.SstsrTemplates(_SideGrip(hand, grip), registry_key=REGISTRY_KEY),
                approach_m=APPROACH_M,
            )
            spec = regions.spec(
                support=SUPPORT,
                normal=UP,
                lift_m=LIFT_M,
                clearance_m=CLEARANCE_M,
                gripper_timeout_ns=GRIPPER_TIMEOUT_NS,
                assume_unobserved=True,
            )
            before = registration.position[2]
            result = run_skill(ctx, grasp(ctx, planner, spec), max_ticks=MAX_TICKS)
            hold_ended = None
            max_tilt = 0.0
            if result.succeeded:
                end = ctx.now.time_ns + HOLD_NS
                sample = ctx.now.time_ns
                while ctx.now.time_ns < end:
                    if ctx.now.time_ns >= sample:  # privileged, for evaluation only
                        max_tilt = max(max_tilt, _tilt(observed()))
                        sample += TILT_SAMPLE_NS
                    ctx.step()
                hold_ended = ctx.now.time_ns
            # Privileged, for evaluation only.
            final = observed()
            if hold_ended is not None:
                max_tilt = max(max_tilt, _tilt(final))
            rise = final.position[2] - before
            left = ctx.violation(OBJECT)
            privileged = {
                "rise_m": round(rise, 4),
                "held": hold_ended is not None and rise >= LIFT_M - LIFT_TOLERANCE_M,
                "hold_ended_ns": hold_ended,
                "max_tilt_rad": round(max_tilt, 3),
                "attachment_violation": None if left is None else _violation(left),
            }
    return {
        "seed": seed,
        "grip": grip,
        "cylinder": {
            "position": [round(v, 6) for v in registration.position],
            "quat_wxyz": [round(v, 6) for v in registration.quat_wxyz],
        },
        "grasp": _skill(result),
        "policy": POLICY,
        "privileged": privileged,
    }


def _violation(left: Any) -> dict[str, Any]:
    """The runtime's attachment violation, its errors rounded to 0.1 mm and 1 mrad."""
    record = dataclasses.asdict(left)
    record["position_error"] = round(record["position_error"], 4)
    record["rotation_error"] = round(record["rotation_error"], 3)
    return record


def _tilt(pose: Pose) -> float:
    """The angle between the cylinder's axis and the vertical, in radians."""
    _, x, y, _ = pose.quat_wxyz
    up = 1.0 - 2.0 * (x * x + y * y)  # the z component of the body's z axis
    return math.acos(max(-1.0, min(1.0, abs(up))))


def _skill(result: SkillResult) -> dict[str, Any]:
    return {
        "state": result.state.value,
        "stages": [
            {
                "name": s.name,
                "succeeded": s.succeeded,
                "reason": None if s.reason is None else s.reason.value,
                "cause": s.cause,
                "detail": s.detail,
            }
            for s in result.stages
        ],
        "assumed": [t.value for t in result.assumed],
        "transitions": [
            {
                "transition": t.transition.value,
                "stage": t.stage,
                "outcome": t.outcome.value,
                "reason": t.reason.value,
            }
            for t in result.transitions
        ],
        "evaluations": [
            {
                "transition": e.estimate.transition.value,
                "estimate": e.estimate.outcome.value,
                "reference": e.reference.value,
                "source": e.source,
            }
            for e in result.evaluations
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ada-lift", description="ADA grasps and lifts a cylinder in open space."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the cylinder's placement (0: reference)"
    )
    parser.add_argument(
        "--grip", choices=GRIPS, default=GRIPS[0], help="the side grasps drawn from"
    )
    parser.add_argument("--record", type=Path, help="a directory to write lift.json to")
    args = parser.parse_args(argv)
    record = lift(args.seed, args.grip)
    if args.record is not None:
        args.record.mkdir(parents=True, exist_ok=True)
        out = args.record / "lift.json"
        partial = out.with_suffix(".json.partial")
        partial.write_text(json.dumps(record, indent=2) + "\n")
        partial.replace(out)
    privileged = record["privileged"]
    print(
        f"seed {args.seed}, {args.grip}: grasp {record['grasp']['state']}; "
        f"rose {privileged['rise_m'] * 1000:.1f} mm; "
        f"held after the hold: {privileged['held']}"
    )
    return 0 if record["grasp"]["state"] == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
