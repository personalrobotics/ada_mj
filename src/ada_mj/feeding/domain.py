# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Feeding domain types — pure Python, no infrastructure imports.

These are the concepts the feeding logic reasons about. They are
importable on any machine without MuJoCo, ROS, or viser installed.

Parameters are drawn from the original ada_feeding system's
acquisition_library.yaml and action server configs, validated with
real patients over multiple feeding studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from mj_manipulator.force_control import ForceThresholds, SpeedProfile

# ---------------------------------------------------------------------------
# Food representation
# ---------------------------------------------------------------------------


@dataclass
class FoodItem:
    """A food item on the plate.

    In simulation, position comes from MuJoCo body pose. On hardware,
    it comes from the perception pipeline (camera + segmentation).

    Args:
        name: MuJoCo body name or perception instance ID.
        position: (3,) world-frame position of the food centroid.
        food_type: Category for acquisition strategy selection
            (e.g., "strawberry", "cracker", "noodle").
        properties: Extensible metadata (hardness, size, etc.).
    """

    name: str
    position: np.ndarray  # (3,) world frame
    food_type: str | None = None
    properties: dict = field(default_factory=dict)


class ForkState(Enum):
    """Whether the fork is carrying food."""

    EMPTY = "empty"
    LOADED = "loaded"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Acquisition strategy
# ---------------------------------------------------------------------------


@dataclass
class AcquisitionSchema:
    """Parameters for one food acquisition attempt.

    Describes the complete motion sequence: approach pose, insertion
    direction/speed, extraction direction/speed, articutool tilt angle,
    and per-phase F/T thresholds.

    The original ada_feeding system stores these in acquisition_library.yaml
    with 6 primitive types (straight skewer, tilted scoop, twirl, etc.).
    Each primitive maps to a different schema with different thresholds
    and twist profiles.

    Args:
        approach_offset: (3,) offset from food position to approach pose
            in the food frame (typically [0, 0, -height_above]).
        tilt_angle: Articutool tilt angle for this acquisition (rad).
        insertion_twist: 6D twist [vx,vy,vz,wx,wy,wz] for stabbing/scooping.
        insertion_duration: How long to apply the insertion twist (s).
        extraction_twist: 6D twist for pulling food out.
        extraction_duration: Duration of extraction motion (s).
        approach_thresholds: F/T limits during approach (abort if exceeded).
        grasp_thresholds: F/T limits during insertion/grasp phase.
        extraction_thresholds: F/T limits during extraction.
    """

    approach_offset: np.ndarray  # (3,) offset from food to approach pose
    tilt_angle: float  # articutool tilt (rad)
    insertion_twist: np.ndarray  # 6D twist for stabbing
    insertion_duration: float  # seconds
    extraction_twist: np.ndarray  # 6D twist for pulling out
    extraction_duration: float  # seconds
    approach_thresholds: ForceThresholds
    grasp_thresholds: ForceThresholds
    extraction_thresholds: ForceThresholds


# ---------------------------------------------------------------------------
# Mouth approach
# ---------------------------------------------------------------------------

# Default speed profile for mouth approach — from original ada_feeding
# move_to_mouth_tree.py parameters, validated with real users.
MOUTH_APPROACH_SPEED = SpeedProfile(
    max_linear=0.15,  # m/s at full distance
    min_linear=0.06,  # m/s near mouth
    max_angular=0.15,  # rad/s
    min_angular=0.075,  # rad/s
    ramp_distance=0.3,  # m — start decelerating at 30cm
)

# F/T thresholds during mouth approach — very sensitive to detect
# contact with the user's face/lips.
MOUTH_FT_THRESHOLD = ForceThresholds(force_n=1.0, torque_nm=1.0)

# Position tolerance for "arrived at mouth" (meters)
MOUTH_POSITION_TOL = 0.0075  # 7.5mm — tight tolerance

# Retraction speed profile — slightly faster than approach
MOUTH_RETRACT_SPEED = SpeedProfile(
    max_linear=0.15,
    min_linear=0.08,
    max_angular=0.3,
    min_angular=0.15,
    ramp_distance=0.3,
)


# ---------------------------------------------------------------------------
# Default acquisition schemas
# ---------------------------------------------------------------------------


def straight_skewer() -> AcquisitionSchema:
    """Standard vertical skewer — the most common acquisition action.

    Parameters from ada_feeding acquisition_library.yaml action index 0.
    """
    return AcquisitionSchema(
        approach_offset=np.array([0.0, 0.0, -0.025]),  # 2.5cm above food
        tilt_angle=-np.pi / 4,  # 45° forward tilt
        insertion_twist=np.array([0.0, 0.0, -0.03, 0.0, 0.0, 0.0]),  # 3cm/s downward
        insertion_duration=1.0,
        extraction_twist=np.array([0.0, 0.0, 0.08, 0.0, 0.0, 0.0]),  # 8cm/s upward
        extraction_duration=1.0,
        approach_thresholds=ForceThresholds(force_n=20.0, torque_nm=4.0),
        grasp_thresholds=ForceThresholds(force_n=15.0, torque_nm=4.0),
        extraction_thresholds=ForceThresholds(force_n=50.0, torque_nm=4.0),
    )
