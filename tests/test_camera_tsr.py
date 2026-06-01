# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the observe_plate framing geometry (CameraTSR).

Guards the invariant the whole feature rests on: the closed-form minimum
standoff must place the worst-case disc-rim point exactly on the FOV cone
boundary — i.e. the formula is exact, not a heuristic. Pure geometry, no model.
"""

from __future__ import annotations

import numpy as np
import pytest

from ada_mj.feeding.camera_tsr import min_standoff_for_disc


def _worst_rim_half_angle(R: float, d: float, tilt: float, n: int = 4000) -> float:
    """Brute-force max angle between optical axis and a disc-rim ray.

    Camera at distance ``d`` from the disc center, looking at it, tilted
    ``tilt`` off the disc normal. Disc of radius ``R`` in the z=0 plane.
    """
    u = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    C = d * u
    a = -u
    al = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    P = np.stack([R * np.cos(al), R * np.sin(al), np.zeros_like(al)], axis=1)
    v = P - C
    cosang = (v @ a) / np.linalg.norm(v, axis=1)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)).max())


@pytest.mark.parametrize("beta_deg", [21.5, 30.0, 45.0])
@pytest.mark.parametrize("tilt_deg", [0.0, 5.0, 15.0, 30.0, 45.0, 60.0])
def test_min_standoff_puts_worst_rim_on_fov_boundary(beta_deg, tilt_deg):
    """d_min must drive the worst rim angle exactly to the half-FOV beta."""
    R = 0.133
    beta = np.radians(beta_deg)
    tilt = np.radians(tilt_deg)

    d = min_standoff_for_disc(R, beta, tilt=tilt)
    worst = _worst_rim_half_angle(R, d, tilt)

    # The binding rim point sits exactly on the FOV cone (never outside).
    assert worst == pytest.approx(beta, abs=np.radians(0.05))


def test_straight_down_matches_simple_formula():
    """At zero tilt the standoff reduces to R / tan(beta)."""
    R, beta = 0.133, np.radians(21.5)
    assert min_standoff_for_disc(R, beta, tilt=0.0) == pytest.approx(R / np.tan(beta))


def test_standoff_saturates_beyond_beta():
    """For tilt > beta the standoff is constant at R / sin(beta)."""
    R, beta = 0.133, np.radians(21.5)
    saturated = R / np.sin(beta)
    for tilt_deg in [22.0, 30.0, 45.0, 70.0]:
        assert min_standoff_for_disc(R, beta, tilt=np.radians(tilt_deg)) == pytest.approx(
            saturated
        )


def test_standoff_monotone_nondecreasing_in_tilt():
    """More tilt never requires less standoff."""
    R, beta = 0.133, np.radians(21.5)
    tilts = np.radians(np.linspace(0.0, 80.0, 50))
    ds = [min_standoff_for_disc(R, beta, tilt=t) for t in tilts]
    assert all(b >= a - 1e-12 for a, b in zip(ds, ds[1:]))


def test_nonpositive_half_fov_raises():
    """A degenerate (zero/negative) FOV must raise, not silently return inf."""
    with pytest.raises(ValueError):
        min_standoff_for_disc(0.133, 0.0)


@pytest.mark.slow
def test_observe_plate_frames_plate_via_direct_camera_read():
    """End-to-end, independent of _get_T_ee_to_cam: solve observe_plate, then
    read the camera pose straight from MuJoCo FK and confirm the whole real
    plate projects inside the FOV cone with the intended framing margin.

    This is the oracle the validator's algebraic check cannot be: the camera
    pose comes from MuJoCo's own forward kinematics, so a bug in the EE→camera
    transform (which cancels in T_ee @ T_ee_cam) would be caught here.
    """
    import mujoco

    from ada_mj.config import ADAConfig
    from ada_mj.feeding.behaviors import observe_plate
    from ada_mj.scenes.table import PLATE_RADIUS, plate_pose

    ADA = pytest.importorskip("ada_mj.robot").ADA
    robot = ADA(ADAConfig.default())
    pose = plate_pose(robot.model, robot.data)
    assert pose is not None, "table scene must provide a plate_center site"
    center = pose[:3, 3]
    beta = robot.camera_tsr.half_fov
    frame_margin = 0.15  # observe()'s default → expected framing headroom

    with robot.sim(physics=False, headless=True) as ctx:
        robot.arm.set_joint_positions(robot.named_poses["above_plate"])
        robot._snap_tool_to_weld()
        mujoco.mj_forward(robot.model, robot.data)
        res = observe_plate(robot, ctx, plate_pose=pose, plate_radius=PLATE_RADIUS)
        assert res, f"observe_plate failed: {res.failure_code}"

        # Independent read: camera pose from MuJoCo FK at the achieved config.
        cid = mujoco.mj_name2id(
            robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera/d415_color"
        )
        axis = -robot.data.cam_xmat[cid].reshape(3, 3)[:, 2]
        axis = axis / np.linalg.norm(axis)
        C = robot.data.cam_xpos[cid].copy()

        # Optical axis passes through the plate center.
        w = center - C
        perp = np.linalg.norm(w - (w @ axis) * axis)
        assert perp <= 3e-3, f"optical axis {perp * 1e3:.1f} mm off plate center"

        # The whole TRUE plate rim is inside the FOV cone — and tighter than
        # beta by the framing margin (guards against silent under-framing).
        al = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        rim = center + PLATE_RADIUS * np.stack(
            [np.cos(al), np.sin(al), np.zeros_like(al)], axis=1
        )
        v = rim - C
        rim_ang = np.arccos(np.clip((v @ axis) / np.linalg.norm(v, axis=1), -1, 1)).max()
        assert rim_ang <= beta, f"plate rim {np.degrees(rim_ang):.2f}° exceeds FOV {np.degrees(beta):.2f}°"
        assert rim_ang <= beta / (1.0 + frame_margin) + np.radians(1.0), (
            f"framing margin missing: rim {np.degrees(rim_ang):.2f}° not within "
            f"expected {np.degrees(beta / (1.0 + frame_margin)):.2f}°"
        )
