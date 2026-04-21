# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ADA robot-assisted feeding system.

Single JACO2 arm on a wheelchair with articutool (2-DOF fork/spoon)
or forque (rigid fork), wrist camera, and seated human.

Example::

    from ada_mj import ADA

    robot = ADA()
    with robot.sim() as ctx:
        robot.go_to("above_plate")
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from ada_mj.config import ADAConfig
from ada_mj.jaco2 import (
    JACO2_ABOVE_PLATE,
    JACO2_RESTING,
    JACO2_STAGING,
    JACO2_STOW,
    create_jaco2_arm,
)

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.sim_context import SimContext

logger = logging.getLogger(__name__)


class _ADASimContext:
    """Wrapper that sets robot._active_context on enter/exit."""

    def __init__(self, inner: SimContext, robot: ADA):
        self._inner = inner
        self._robot = robot

    def __enter__(self):
        ctx = self._inner.__enter__()
        self._robot._active_context = ctx
        return ctx

    def __exit__(self, *args):
        self._robot._active_context = None
        return self._inner.__exit__(*args)


class ADA:
    """ADA robot-assisted feeding system.

    Composes ada_assets for the MuJoCo model and mj_manipulator for
    arm control, planning, and execution.

    Args:
        config: Robot configuration. Defaults to articutool + fork + camera + human.

    Example::

        robot = ADA()
        with robot.sim() as ctx:
            robot.go_to("above_plate")
            mouth = robot.head.get_mouth_pose()
            robot.arm.plan_to_pose(mouth)
    """

    def __init__(self, config: ADAConfig | None = None):
        self.config = config or ADAConfig.default()

        # Build the MuJoCo model from ada_assets
        self.model, self.data = self._build_model()

        # Wrap in Environment (for mj_manipulator Arm compatibility)
        from mj_environment import Environment

        self._env = Environment.from_model(self.model, self.data)

        # Grasp manager — tracks the welded tool as a grasped object
        # so the collision checker moves it with the arm during planning.
        from mj_manipulator.grasp_manager import GraspManager

        self._grasp_manager = GraspManager(self.model, self.data)

        if self.config.tool is not None:
            tool_body = f"{self.config.tool}/fork_base"
            self._grasp_manager.mark_grasped(tool_body, "jaco2")
            self._grasp_manager.attach_object(tool_body, "j2n6s200_link_6")

        # Create the JACO2 arm — include tool bodies in collision set,
        # pass grasp_manager so the planner moves the tool with the arm.
        extra_bodies = None
        if self.config.tool == "articutool":
            extra_bodies = ["articutool/fork_base"]
        elif self.config.tool == "forque":
            extra_bodies = ["forque/fork_base"]

        self._arm = create_jaco2_arm(
            self._env,
            ee_site=self.config.ee_site,
            with_ik="auto",
            extra_arm_body_names=extra_bodies,
            grasp_manager=self._grasp_manager,
        )

        # Articutool entity (if using articutool)
        self._articutool = None
        if self.config.tool == "articutool":
            from ada_mj.articutool import Articutool

            self._articutool = Articutool(self.model, self.data, self.config.articutool)

        # Head controller (if human present)
        self._head = None
        if self.config.with_human:
            from ada_mj.head import HeadController

            self._head = HeadController(self.model, self.data, self.config.head)

        # Wrist camera (if camera present)
        self._camera = None
        if self.config.with_camera:
            from ada_mj.sensors import WristCamera

            self._camera = WristCamera(self.model, self.data, self.config.camera)

        # Initialize ctrl to hold current pose (prevents arm collapse on physics start)
        self._init_ctrl_from_qpos()

        # Named poses (flat — single arm)
        self._named_poses: dict[str, np.ndarray] = {
            "above_plate": JACO2_ABOVE_PLATE.copy(),
            "resting": JACO2_RESTING.copy(),
            "staging": JACO2_STAGING.copy(),
            "stow": JACO2_STOW.copy(),
        }

        # Execution state
        self._active_context: SimContext | None = None
        self._abort_event = threading.Event()

        logger.info(
            "ADA initialized: tool=%s, tip=%s, nbody=%d, njnt=%d, nu=%d",
            self.config.tool,
            self.config.tool_tip,
            self.model.nbody,
            self.model.njnt,
            self.model.nu,
        )

    # -- Internal helpers -----------------------------------------------------

    def _snap_tool_to_weld(self) -> None:
        """Reset tool freejoint qpos to the weld attachment site.

        During physics, the freejoint drifts from the weld target.
        Call before planning to ensure the planner's forked MjData
        has the tool at the correct position.
        """
        if self.config.tool is None:
            return
        from ada_assets.assembly import TOOLS, _init_tool_pose

        _init_tool_pose(self.model, self.data, self.config.tool, self.config.tool)

    def _init_ctrl_from_qpos(self) -> None:
        """Set ctrl to current joint positions so position actuators hold pose."""
        from ada_mj.jaco2 import JACO2_FINGER_CLOSED

        for i in range(self.model.nu):
            trntype = self.model.actuator_trntype[i]
            if trntype == 0:  # joint transmission
                jnt_id = self.model.actuator_trnid[i, 0]
                qadr = self.model.jnt_qposadr[jnt_id]
                self.data.ctrl[i] = self.data.qpos[qadr]
            elif trntype == 3:  # tendon transmission (fingers)
                self.data.ctrl[i] = JACO2_FINGER_CLOSED

    # -- Model building -------------------------------------------------------

    def _build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Build the MuJoCo model from ada_assets."""
        from ada_assets.assembly import assemble_ada

        return assemble_ada(
            tool=self.config.tool,
            tool_tip=self.config.tool_tip,
            with_human=self.config.with_human,
            with_camera=self.config.with_camera,
        )

    # -- Core access ----------------------------------------------------------

    @property
    def arm(self) -> Arm:
        """The JACO2 arm."""
        return self._arm

    @property
    def arms(self) -> dict[str, Arm]:
        """For mj_manipulator protocol compatibility."""
        return {"jaco2": self._arm}

    @property
    def articutool(self):
        """Articutool 2-DOF entity (None if using forque or no tool)."""
        return self._articutool

    @property
    def head(self):
        """Head mocap controller (None if no human)."""
        return self._head

    @property
    def camera(self):
        """Wrist camera (None if no camera)."""
        return self._camera

    @property
    def grasp_manager(self) -> GraspManager:
        """Shared grasp manager."""
        return self._grasp_manager

    @property
    def named_poses(self) -> dict[str, np.ndarray]:
        """Named arm configurations."""
        return self._named_poses

    # -- Simulation context ---------------------------------------------------

    def sim(
        self,
        physics: bool = True,
        headless: bool = False,
        viewer=None,
        viewer_fps: float = 30.0,
        event_loop=None,
    ) -> _ADASimContext:
        """Create a simulation execution context.

        Example::

            with robot.sim(physics=True) as ctx:
                path = robot.arm.plan_to_pose(target)
                traj = robot.arm.retime(path)
                ctx.execute(traj)
        """
        from mj_manipulator.sim_context import SimContext

        entities = {}
        if self._articutool is not None:
            entities[self._articutool.name] = self._articutool

        inner = SimContext(
            self.model,
            self.data,
            {"jaco2": self._arm},
            physics=physics,
            headless=headless,
            viewer=viewer,
            viewer_fps=viewer_fps,
            entities=entities,
            abort_fn=self.is_abort_requested,
            event_loop=event_loop,
        )
        return _ADASimContext(inner, self)

    # -- Named poses ----------------------------------------------------------

    def go_to(self, pose_name: str) -> bool:
        """Move the arm to a named configuration.

        Args:
            pose_name: One of "above_plate", "resting", "staging", "stow".

        Returns:
            True if the arm reached the target.
        """
        if pose_name not in self._named_poses:
            raise ValueError(f"Unknown pose '{pose_name}'. Available: {sorted(self._named_poses)}")

        q_goal = self._named_poses[pose_name]
        ctx = self._active_context
        if ctx is None:
            # No sim context — just set qpos directly
            self._arm.set_joint_positions(q_goal)
            self.forward()
            return True

        path = self._arm.plan_to_configuration(q_goal)
        if path is None:
            logger.warning("go_to(%s): planning failed", pose_name)
            return False

        traj = self._arm.retime(path)
        ctx.execute(traj)
        return True

    # -- Sensor convenience ---------------------------------------------------

    def get_ft_wrench(self) -> np.ndarray:
        """F/T wrench [fx,fy,fz,tx,ty,tz] from the active tool's sensor."""
        if self._articutool is not None:
            return self._articutool.get_ft_wrench()
        # Forque: read from sensor directly
        cfg = self.config.forque
        force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, cfg.ft_force_sensor)
        torque_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, cfg.ft_torque_sensor)
        force = self.data.sensordata[self.model.sensor_adr[force_id]:self.model.sensor_adr[force_id] + 3].copy()
        torque = self.data.sensordata[self.model.sensor_adr[torque_id]:self.model.sensor_adr[torque_id] + 3].copy()
        return np.concatenate([force, torque])

    def get_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """IMU (accel, gyro) from the articutool. Raises if no articutool."""
        if self._articutool is None:
            raise RuntimeError("IMU only available with articutool")
        return self._articutool.get_imu()

    # -- Tool management (ada_mj#4) -------------------------------------------

    def release_tool(self) -> None:
        """Disable the tool weld constraint (tool drops)."""
        weld_name = self._get_weld_name()
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
        if eq_id >= 0:
            self.data.eq_active[eq_id] = 0

    def acquire_tool(self) -> None:
        """Re-enable the tool weld constraint.

        Teleports the tool to the attachment site before enabling the weld
        to avoid large impulses.
        """
        from ada_assets.assembly import _init_tool_pose

        weld_name = self._get_weld_name()
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
        if eq_id >= 0:
            _init_tool_pose(self.model, self.data, self.config.tool, self.config.tool)
            self.forward()
            self.data.eq_active[eq_id] = 1

    def _get_weld_name(self) -> str:
        if self.config.tool == "articutool":
            return self.config.articutool.weld_name
        elif self.config.tool == "forque":
            return self.config.forque.weld_name
        raise RuntimeError("No tool configured")

    # -- State ----------------------------------------------------------------

    def forward(self) -> None:
        """Run mj_forward to update derived quantities."""
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset to initial state (stow keyframe)."""
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stow")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self._init_ctrl_from_qpos()
        self.forward()

    def request_abort(self) -> None:
        """Request abort of the current motion."""
        self._abort_event.set()

    def clear_abort(self) -> None:
        """Clear the abort flag."""
        self._abort_event.clear()

    def is_abort_requested(self) -> bool:
        """Check if abort was requested."""
        return self._abort_event.is_set()
