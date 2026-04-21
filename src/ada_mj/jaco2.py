# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Kinova JACO2 j2n6s200 arm constants and factory.

The j2n6s200 is a 6-DOF arm with a **non-spherical** wrist (the "n"
in the model name). Joints 4-6 have 37-73 mm offsets between their
axes, which means EAIK has no analytical decomposition. The factory
uses ``resolve_ik_solver(with_ik="auto")`` which falls back to mink
numerical IK automatically.

All joint specs from ``ada_ros2/ada_description/urdf/j2n6s200.xacro``.
Named configurations from ``ada_feeding/config/ada_feeding_action_servers_default.yaml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.arm import Arm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JACO2_JOINT_NAMES = [f"j2n6s200_joint_{i}" for i in range(1, 7)]

# Joint origins relative to parent (from j2n6s200.xacro).
# The RPY rotations (π, -π/2, π/3 etc.) are handled by MuJoCo's
# joint axis definitions at build time — these XYZ offsets are what
# positions each link relative to its parent.
JACO2_JOINT_ORIGINS = [
    [0.0, 0.0, 0.15675],  # J1: base → shoulder
    [0.0, 0.0016, -0.11875],  # J2: shoulder → arm
    [0.0, -0.410, 0.0],  # J3: arm → forearm
    [0.0, 0.2073, -0.0114],  # J4: forearm → wrist_1
    [0.0, -0.03703, -0.06414],  # J5: wrist_1 → wrist_2
    [0.0, -0.03703, -0.06414],  # J6: wrist_2 → hand
]

JACO2_EE_ORIGIN = [0.0, 0.0, -0.1600]  # hand → end_effector

# Joint RPY rotations from the xacro (applied as frame rotations).
JACO2_JOINT_RPYS = [
    [0.0, np.pi, 0.0],  # J1
    [-np.pi / 2, 0.0, np.pi],  # J2
    [0.0, np.pi, 0.0],  # J3
    [-np.pi / 2, 0.0, np.pi],  # J4
    [np.pi / 3, 0.0, np.pi],  # J5
    [np.pi / 3, 0.0, np.pi],  # J6
]

# Velocity limits (rad/s) from URDF.
# J1-3: 36°/s = 0.628 rad/s, J4-6: 48°/s = 0.838 rad/s
JACO2_VELOCITY_LIMITS = np.array([0.628, 0.628, 0.628, 0.838, 0.838, 0.838])

# Acceleration limits — not published; derived from v_max / 0.1s
JACO2_ACCELERATION_LIMITS = np.array([6.28, 6.28, 6.28, 8.38, 8.38, 8.38])

# Position limits. Continuous joints (J1, J4, J5, J6) get full rotation.
# J2 and J3 are revolute with limits from the URDF.
JACO2_LOWER = np.array([-2 * np.pi, 0.820, 0.332, -2 * np.pi, -2 * np.pi, -2 * np.pi])
JACO2_UPPER = np.array([2 * np.pi, 5.464, 5.948, 2 * np.pi, 2 * np.pi, 2 * np.pi])

# Effort limits (N·m) from URDF.
JACO2_EFFORT_LIMITS = np.array([40.0, 80.0, 40.0, 20.0, 20.0, 20.0])

# ---------------------------------------------------------------------------
# Named configurations (radians, from ada_feeding config YAML)
# ---------------------------------------------------------------------------

JACO2_ABOVE_PLATE = np.array([-2.579, 3.010, 1.770, -2.076, -1.791, 2.858])
JACO2_RESTING = np.array([-1.860, 2.181, 0.364, -5.187, -0.470, -0.814])
JACO2_STAGING = np.array([-2.122, 4.496, 4.022, -4.710, -2.493, -1.926])
JACO2_STOW = np.array([-1.521, 2.601, 0.348, -4.000, 0.228, 3.879])

# Fingers locked holding the fork (from ada_moveit initial_positions.yaml)
JACO2_FINGER_CLOSED = 1.33  # radians, both fingers

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_jaco2_arm(
    env: "Environment",
    *,
    ee_site: str = "fork_tip",
    with_ik="auto",
    extra_arm_body_names: list[str] | None = None,
    grasp_manager=None,
) -> "Arm":
    """Create a JACO2 arm with mink IK (EAIK has no decomposition for this arm).

    Args:
        env: MuJoCo environment containing the ADA model.
        ee_site: Name of the end-effector site.
        with_ik: IK solver mode (default "auto" → mink fallback).
        extra_arm_body_names: Bodies to treat as part of the arm for
            collision checking (e.g., welded tool root body).

    Returns:
        Arm ready for planning and execution.
    """
    from mj_manipulator.arm import Arm
    from mj_manipulator.arms._ik_factory import resolve_ik_solver
    from mj_manipulator.config import ArmConfig, KinematicLimits, PlanningDefaults

    config = ArmConfig(
        name="jaco2",
        entity_type="arm",
        joint_names=list(JACO2_JOINT_NAMES),
        kinematic_limits=KinematicLimits(
            velocity=JACO2_VELOCITY_LIMITS.copy(),
            acceleration=JACO2_ACCELERATION_LIMITS.copy(),
        ),
        ee_site=ee_site,
        extra_arm_body_names=extra_arm_body_names,
        planning_defaults=PlanningDefaults(smoothing_iterations=25),
    )

    arm = Arm(env, config)
    ik_solver = resolve_ik_solver(arm, with_ik=with_ik)
    return Arm(env, config, ik_solver=ik_solver, grasp_manager=grasp_manager)
