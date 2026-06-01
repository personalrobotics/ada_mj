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


@pytest.mark.parametrize("fovy_deg", [43.0, 60.0, 90.0])
@pytest.mark.parametrize("tilt_deg", [0.0, 5.0, 15.0, 30.0, 45.0, 60.0])
def test_min_standoff_puts_worst_rim_on_fov_boundary(fovy_deg, tilt_deg):
    """d_min must drive the worst rim angle exactly to beta = fovy/2."""
    R = 0.133
    fovy = np.radians(fovy_deg)
    beta = fovy / 2.0
    tilt = np.radians(tilt_deg)

    d = min_standoff_for_disc(R, fovy, tilt=tilt)
    worst = _worst_rim_half_angle(R, d, tilt)

    # The binding rim point sits exactly on the FOV cone (never outside).
    assert worst == pytest.approx(beta, abs=np.radians(0.05))


def test_straight_down_matches_simple_formula():
    """At zero tilt the standoff reduces to R / tan(fovy/2)."""
    R, fovy = 0.133, np.radians(43.0)
    assert min_standoff_for_disc(R, fovy, tilt=0.0) == pytest.approx(
        R / np.tan(fovy / 2.0)
    )


def test_standoff_saturates_beyond_beta():
    """For tilt > beta the standoff is constant at R / sin(beta)."""
    R, fovy = 0.133, np.radians(43.0)
    beta = fovy / 2.0
    saturated = R / np.sin(beta)
    for tilt_deg in [22.0, 30.0, 45.0, 70.0]:
        assert min_standoff_for_disc(R, fovy, tilt=np.radians(tilt_deg)) == pytest.approx(
            saturated
        )


def test_standoff_monotone_nondecreasing_in_tilt():
    """More tilt never requires less standoff."""
    R, fovy = 0.133, np.radians(43.0)
    tilts = np.radians(np.linspace(0.0, 80.0, 50))
    ds = [min_standoff_for_disc(R, fovy, tilt=t) for t in tilts]
    assert all(b >= a - 1e-12 for a, b in zip(ds, ds[1:]))
