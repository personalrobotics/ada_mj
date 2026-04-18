# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Path resolution for ADA model assets.

Locates ada_ros2 and ada_feeding repos in the workspace for mesh files.
Both are expected as siblings of this package's repo root (the robot-code
workspace convention), or can be overridden via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]  # ada_mj/src/ada_mj → robot-code


def find_ada_description() -> Path:
    """Return the path to ada_ros2/ada_description."""
    if "ADA_DESCRIPTION_PATH" in os.environ:
        p = Path(os.environ["ADA_DESCRIPTION_PATH"])
        if p.is_dir():
            return p

    p = _WORKSPACE_ROOT / "ada_ros2" / "ada_description"
    if p.is_dir():
        return p

    raise FileNotFoundError(
        "ada_description not found. Clone it:\n"
        "  git clone https://github.com/personalrobotics/ada_ros2\n"
        "Or set: export ADA_DESCRIPTION_PATH=/path/to/ada_description"
    )


def find_ada_planning_scene() -> Path:
    """Return the path to ada_feeding/ada_planning_scene."""
    if "ADA_PLANNING_SCENE_PATH" in os.environ:
        p = Path(os.environ["ADA_PLANNING_SCENE_PATH"])
        if p.is_dir():
            return p

    p = _WORKSPACE_ROOT / "ada_feeding" / "ada_planning_scene"
    if p.is_dir():
        return p

    raise FileNotFoundError(
        "ada_planning_scene not found. Clone it:\n"
        "  git clone https://github.com/personalrobotics/ada_feeding\n"
        "Or set: export ADA_PLANNING_SCENE_PATH=/path/to/ada_planning_scene"
    )


def arm_mesh(name: str) -> Path:
    """Path to a JACO2 arm mesh (DAE format)."""
    return find_ada_description() / "meshes" / name


def forque_mesh(name: str) -> Path:
    """Path to a forque tool mesh (STL format, scale 0.001)."""
    return find_ada_description() / "meshes" / "forque" / name


def scene_mesh(name: str) -> Path:
    """Path to a planning scene mesh (STL format)."""
    return find_ada_planning_scene() / "assets" / name
