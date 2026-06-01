#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Validate the observe_plate CameraTSR: is the whole plate inside the frustum?

Two independent checks, both exit non-zero on failure (CI-ready):

1. Closed-form framing math — the analytic ``min_standoff`` must place the
   worst-case plate-rim point exactly on the FOV cone boundary (verified by
   brute force over the rim), proving the formula is exact, not a heuristic.

2. In-context TSR construction — build the real ``CameraTSR.observe`` templates,
   instantiate them at the plate pose, sample poses across the look-at cone, and
   confirm every sampled camera pose (a) keeps the optical axis through the plate
   center, (b) tilts no more than ``tilt_max`` off vertical, and (c) projects the
   entire *true* plate rim inside the camera's vertical FOV cone.

Run::

    uv run python scripts/validate_observe.py
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from ada_mj.config import ADAConfig
from ada_mj.robot import ADA
from ada_mj.scenes.table import PLATE_RADIUS, plate_pose

TILTS_DEG = [0.0, 15.0, 30.0, 45.0]
AXIS_EPS = 2e-3  # m: optical axis must pass within 2 mm of the plate center
TILT_EPS = np.radians(0.5)  # rad: tilt magnitude slack


def worst_rim_half_angle(R: float, d: float, tilt: float, n: int = 2000) -> float:
    """Max angle between the optical axis and a plate-rim ray.

    Camera at distance ``d`` from the plate center, looking at it, tilted
    ``tilt`` off vertical. Plate is a disc of radius ``R`` in the z=0 plane.
    """
    u = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    C = d * u
    a = -u
    al = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    P = np.stack([R * np.cos(al), R * np.sin(al), np.zeros_like(al)], axis=1)
    v = P - C
    cosang = (v @ a) / np.linalg.norm(v, axis=1)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)).max())


def check_closed_form(cam_tsr, beta: float, frame_margin: float = 0.15) -> bool:
    """The analytic d_min must drive the worst rim angle exactly to beta."""
    r_eff = PLATE_RADIUS * (1.0 + frame_margin)
    print("\n[1] Closed-form framing math (worst rim angle == fovy/2):")
    ok = True
    for tdeg in [0.0, 5.0, np.degrees(beta), 30.0, 45.0]:
        t = np.radians(tdeg)
        # min_standoff includes the frame margin; strip standoff_margin (0 here).
        d = cam_tsr.min_standoff(PLATE_RADIUS, tilt=t, frame_margin=frame_margin)
        wr = worst_rim_half_angle(r_eff, d, t)
        err = np.degrees(wr - beta)
        flag = "" if abs(err) < 0.05 else "  <-- MISMATCH"
        ok = ok and abs(err) < 0.05
        print(f"    tilt={tdeg:5.1f}°  d_min={d:.4f}m  worst_rim={np.degrees(wr):7.3f}°  err={err:+.4f}°{flag}")
    print(f"    => {'PASS' if ok else 'FAIL'}")
    return ok


def check_tsr_algebra(robot, cam_tsr, plate_T: np.ndarray, beta: float) -> bool:
    """Sample the TSR cone and verify the rotation ALGEBRA frames the plate.

    NOTE: this reconstructs the camera as ``T_ee @ T_ee_cam``, where the same
    ``T_ee_cam`` is baked (inverted) into ``Tw_e`` — so it cancels exactly and
    this check sees the hand-authored canonical pose, NOT the real link_6→camera
    mounting. It proves the Euler-box construction (axis stays on center, tilt
    bounded by tilt_max) across the whole cone. The mounting itself is verified
    independently in `check_executed_direct`, which reads MuJoCo FK.
    """
    from tsr.tsr import TSR

    T_ee_cam = cam_tsr._get_T_ee_to_cam()
    center = plate_T[:3, 3]
    down = np.array([0.0, 0.0, -1.0])

    print("\n[2] TSR rotation algebra (sampled cone keeps axis on center, tilt bounded):")
    all_ok = True
    for tdeg in TILTS_DEG:
        tilt_max = np.radians(tdeg)
        templates = cam_tsr.observe(PLATE_RADIUS, tilt_max=tilt_max)

        worst_axis_err = 0.0
        worst_tilt = 0.0
        worst_rim = 0.0
        n = 0
        # Deterministic grid over the cone: every standoff template, pitch from
        # 0..tilt_max, full azimuth ring. Hits the worst case (max tilt) exactly.
        pitches = np.linspace(0.0, tilt_max, 4)
        yaws = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        for tmpl in templates:
            tsr = TSR(T0_w=plate_T @ tmpl.T_ref_tsr, Tw_e=tmpl.Tw_e, Bw=tmpl.Bw)
            for p in pitches:
                for y in yaws:
                    T_ee = tsr.to_transform([0.0, 0.0, 0.0, 0.0, p, y])
                    T_cam = T_ee @ T_ee_cam
                    C = T_cam[:3, 3]
                    axis = -T_cam[:3, 2]  # MuJoCo camera looks down -z
                    axis = axis / np.linalg.norm(axis)

                    # (a) optical axis passes through the plate center
                    w = center - C
                    perp = np.linalg.norm(w - (w @ axis) * axis)
                    worst_axis_err = max(worst_axis_err, perp)

                    # (b) tilt of optical axis off vertical
                    tilt = np.arccos(np.clip(axis @ down, -1.0, 1.0))
                    worst_tilt = max(worst_tilt, tilt)

                    # (c) true plate rim inside the half-FOV cone
                    al = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
                    rim = center + PLATE_RADIUS * np.stack(
                        [np.cos(al), np.sin(al), np.zeros_like(al)], axis=1
                    )
                    v = rim - C
                    cosang = (v @ axis) / np.linalg.norm(v, axis=1)
                    rim_ang = np.arccos(np.clip(cosang, -1.0, 1.0)).max()
                    worst_rim = max(worst_rim, rim_ang)
                    n += 1

        ok = (
            worst_axis_err <= AXIS_EPS
            and worst_tilt <= tilt_max + TILT_EPS
            and worst_rim <= beta + 1e-6
        )
        all_ok = all_ok and ok
        print(
            f"    tilt_max={tdeg:5.1f}°  n={n:4d}  "
            f"axis_off_center<={worst_axis_err * 1e3:5.2f}mm  "
            f"max_tilt={np.degrees(worst_tilt):5.2f}°  "
            f"rim_angle={np.degrees(worst_rim):6.3f}° (half_fov={np.degrees(beta):.2f}°)  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print(f"    => {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def check_executed_direct(robot, beta: float, frame_margin: float = 0.15) -> bool:
    """Solve observe_plate, then read the camera straight from MuJoCo FK.

    The independent oracle: the camera pose comes from MuJoCo's own forward
    kinematics at the achieved config, so a bug in the EE→camera mounting
    transform (which cancels in `check_tsr_algebra`) is caught here. Also
    asserts the framing margin is actually present (rim well inside the cone),
    not merely rim ≤ beta — closing the ~5° slack a bare containment check
    would leave.
    """
    from ada_mj.feeding.behaviors import observe_plate

    print("\n[3] Executed pose (camera read from MuJoCo FK — independent of T_ee_cam):")
    pose = plate_pose(robot.model, robot.data)
    center = pose[:3, 3]
    cid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera/d415_color")
    expected = beta / (1.0 + frame_margin)
    all_ok = True
    with robot.sim(physics=False, headless=True) as ctx:
        for tdeg in [0.0, 30.0]:
            robot._observe_config = None  # force a fresh solve each tilt
            robot.arm.set_joint_positions(robot.named_poses["above_plate"])
            robot._snap_tool_to_weld()
            mujoco.mj_forward(robot.model, robot.data)
            res = observe_plate(
                robot, ctx, plate_pose=pose, plate_radius=PLATE_RADIUS,
                tilt_max=np.radians(tdeg),
            )
            if not res:
                print(f"    tilt_max={tdeg:5.1f}°  PLAN FAILED ({res.failure_code})")
                all_ok = False
                continue
            axis = -robot.data.cam_xmat[cid].reshape(3, 3)[:, 2]
            axis = axis / np.linalg.norm(axis)
            C = robot.data.cam_xpos[cid].copy()
            w = center - C
            perp = np.linalg.norm(w - (w @ axis) * axis)
            al = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
            rim = center + PLATE_RADIUS * np.stack(
                [np.cos(al), np.sin(al), np.zeros_like(al)], axis=1
            )
            v = rim - C
            rim_ang = np.arccos(np.clip((v @ axis) / np.linalg.norm(v, axis=1), -1, 1)).max()
            ok = perp <= 3e-3 and rim_ang <= beta and rim_ang <= expected + np.radians(1.0)
            all_ok = all_ok and ok
            print(
                f"    tilt_max={tdeg:5.1f}°  axis_off_center={perp * 1e3:5.2f}mm  "
                f"rim_angle={np.degrees(rim_ang):6.3f}° "
                f"(≤half_fov {np.degrees(beta):.2f}°, expect ≤{np.degrees(expected):.2f}°)  "
                f"{'PASS' if ok else 'FAIL'}"
            )
    print(f"    => {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    robot = ADA(ADAConfig.default())
    mujoco.mj_forward(robot.model, robot.data)
    cam_tsr = robot.camera_tsr
    beta = cam_tsr.half_fov

    pose = plate_pose(robot.model, robot.data)
    if pose is None:
        print("plate_center site not found — is the table scene loaded?")
        return 2

    print(
        f"fovy = {np.degrees(cam_tsr.fovy):.1f}°   "
        f"half_fov = {np.degrees(beta):.2f}°   plate_radius = {PLATE_RADIUS} m"
    )
    ok1 = check_closed_form(cam_tsr, beta)
    ok2 = check_tsr_algebra(robot, cam_tsr, pose, beta)
    ok3 = check_executed_direct(robot, beta)

    ok = ok1 and ok2 and ok3
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
