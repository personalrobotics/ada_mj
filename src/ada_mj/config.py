# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Configuration for the ADA feeding robot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ArticutoolConfig:
    """Joint, actuator, and sensor names for the articutool (prefixed by assembly)."""

    tilt_joint: str = "articutool/atool_joint1"
    roll_joint: str = "articutool/atool_joint2"
    tilt_actuator: str = "articutool/act_atool_tilt"
    roll_actuator: str = "articutool/act_atool_roll"
    weld_name: str = "articutool_grasp_weld"
    ft_force_sensor: str = "articutool/ft_force"
    ft_torque_sensor: str = "articutool/ft_torque"
    imu_accel_sensor: str = "articutool/imu_accel"
    imu_gyro_sensor: str = "articutool/imu_gyro"


@dataclass
class ForqueConfig:
    """Sensor names for the forque (prefixed by assembly)."""

    weld_name: str = "forque_grasp_weld"
    ft_force_sensor: str = "forque/ft_force"
    ft_torque_sensor: str = "forque/ft_torque"


@dataclass
class HeadConfig:
    """Mocap head body and mouth site names."""

    body_name: str = "human/head"
    mouth_site: str = "human/mouth"


@dataclass
class CameraConfig:
    """D415 wrist camera configuration."""

    color_camera: str = "camera/d415_color"
    depth_camera: str = "camera/d415_depth"
    resolution: tuple[int, int] = (1280, 720)
    depth_range: tuple[float, float] = (0.16, 10.0)


@dataclass
class DebugConfig:
    """Debug logging flags."""

    verbose: bool = False
    planning: bool = False
    primitives: bool = False

    @classmethod
    def from_env(cls) -> DebugConfig:
        """Create from ADA_DEBUG environment variable (comma-separated flags)."""
        env = os.environ.get("ADA_DEBUG", "")
        flags = {f.strip().lower() for f in env.split(",") if f.strip()}
        return cls(
            verbose="verbose" in flags or "all" in flags,
            planning="planning" in flags or "all" in flags,
            primitives="primitives" in flags or "all" in flags,
        )


@dataclass
class ADAConfig:
    """Full ADA robot configuration.

    Controls which components are assembled and how they're addressed.
    The model is built dynamically by ada_assets.assembly.assemble_ada().

    Example::

        config = ADAConfig.default()          # articutool + fork + camera + human
        config = ADAConfig.with_forque()      # forque (rigid fork, no motors)
        config = ADAConfig(tool_tip="spoon")  # articutool with spoon
    """

    # Assembly options (passed to assemble_ada)
    tool: str | None = "articutool"
    tool_tip: str = "fork"
    with_human: bool = True
    with_camera: bool = True

    # End-effector site (changes based on tool)
    ee_site: str = "articutool/fork_tip"

    # Sub-configs
    articutool: ArticutoolConfig = field(default_factory=ArticutoolConfig)
    forque: ForqueConfig = field(default_factory=ForqueConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    debug: DebugConfig = field(default_factory=DebugConfig.from_env)

    @classmethod
    def default(cls) -> ADAConfig:
        """Default config: articutool + fork + camera + human."""
        return cls()

    @classmethod
    def with_forque(cls) -> ADAConfig:
        """Config with forque (rigid fork, no 2-DOF motors)."""
        return cls(tool="forque", ee_site="forque/fork_tip")

    @classmethod
    def bare(cls) -> ADAConfig:
        """Config with no tool, no camera, no human (just arm on wheelchair)."""
        return cls(tool=None, with_camera=False, with_human=False, ee_site="ee_site")
