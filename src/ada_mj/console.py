# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""IPython console for interactive ADA control.

Delegates to mj_manipulator.console.start_console with
ADA-specific panels and namespace entries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ada_mj.robot import ADA

logger = logging.getLogger(__name__)


def start_console(
    robot: ADA,
    *,
    physics: bool = False,
    viewer: bool = False,
    viser: bool = False,
) -> None:
    """Launch the ADA IPython console.

    Args:
        robot: ADA robot instance.
        physics: Enable physics simulation.
        viewer: Launch native MuJoCo viewer (requires mjpython).
        viser: Launch browser viewer at http://localhost:8080.
    """
    from mj_manipulator.console import start_console as _start_console

    def commands() -> None:
        """Print a quick reference of available commands."""
        print(
            """
ADA Quick Reference
===================

Arm:
  robot.arm.get_ee_pose()             — 4x4 end-effector pose
  robot.arm.get_joint_positions()     — current joint angles
  robot.arm.plan_to_pose(T)           — plan to 4x4 target
  robot.arm.plan_to_configuration(q)  — plan to joint config
  robot.go_to("above_plate")          — move to named pose
  robot.go_to("stow")                 — stow the arm
  robot.named_poses                   — list all named configs

Articutool:
  robot.articutool.get_tilt()         — current tilt angle
  robot.articutool.get_roll()         — current roll angle
  robot.articutool.set_tilt(0.5)      — set tilt (radians)
  robot.articutool.set_roll(1.0)      — set roll (radians)
  robot.articutool.is_attached()      — weld state

Sensors:
  robot.get_ft_wrench()               — [fx,fy,fz,tx,ty,tz]
  robot.get_imu()                     — (accel[3], gyro[3])

Head:
  robot.head.get_mouth_pose()         — 4x4 mouth target
  robot.head.set_position([x,y,z])    — move head
  robot.head.get_position()           — head position

Camera:
  robot.camera.render_color()         — (720, 1280, 3) RGB
  robot.camera.render_depth()         — (720, 1280) depth

Tool:
  robot.release_tool()                — drop tool
  robot.acquire_tool()                — re-attach tool

State:
  robot.forward()                     — run mj_forward
  robot.reset()                       — reset to stow
  commands()                          — this help

IPython:
  robot.<tab>                         — tab completion
  ?robot.articutool                   — docstring
"""
        )

    extra_ns: dict = {
        "commands": commands,
        "go_to": lambda pose_name: robot.go_to(pose_name),
    }

    # -- Panel setup callback (same pattern as geodude) -------------------------
    def _setup_ada_panels(gui, viewer, robot, event_loop, tabs):
        """Add ADA-specific panels to the viser GUI."""
        import mujoco
        import viser as _viser

        all_panels = []

        # Fix teleop ghost — the generic console creates it with empty prefix
        # (shows full robot). Replace with tool prefix so only the fork shows.
        tool_prefix = ""
        if robot.config.tool == "articutool":
            tool_prefix = "articutool/"
        elif robot.config.tool == "forque":
            tool_prefix = "forque/"
        if tool_prefix:
            for panel in viewer._panels:
                if hasattr(panel, "_ghost") and panel._ghost is not None:
                    panel._ghost.remove()
                    from mj_viser.teleop_panel import GhostHand

                    panel._ghost = GhostHand(
                        viewer._server,
                        robot.model,
                        robot.data,
                        gripper_body_prefix=tool_prefix,
                        ee_site_id=robot.arm.ee_site_id,
                    )

        with tabs.add_tab("ADA"):
            # Keyframe dropdown
            keyframe_names = []
            for i in range(robot.model.nkey):
                name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_KEY, i)
                if name:
                    keyframe_names.append(name)

            if keyframe_names:
                _moving = [False]

                # One button per keyframe — no stale selection state
                _keyframe_btns = {}
                for kf_name in keyframe_names:
                    if kf_name not in robot.named_poses:
                        continue
                    btn = gui.add_button(kf_name.replace("_", " ").title())
                    _keyframe_btns[kf_name] = btn

                    def _make_handler(name):
                        def _on_click(_: _viser.GuiEvent) -> None:
                            if _moving[0]:
                                return

                            q_goal = robot.named_poses[name].copy()
                            ctx = robot._active_context
                            if ctx is None:
                                return

                            _moving[0] = True
                            for b in _keyframe_btns.values():
                                b.disabled = True

                            try:

                                def _plan():
                                    return robot.arm.plan_to_configuration(q_goal)

                                path = event_loop.run_on_physics_thread(_plan)
                                if path is None:
                                    logger.warning("Planning to %s failed", name)
                                    return

                                traj = robot.arm.retime(path)
                                ctx.execute(traj)
                            except Exception as e:
                                logger.warning("go_to(%s): %s", name, e)
                            finally:
                                _moving[0] = False
                                for b in _keyframe_btns.values():
                                    b.disabled = False

                        return _on_click

                    btn.on_click(_make_handler(kf_name))

            # Articutool sliders — update PhysicsController entity target
            # so the controller drives the joints (same pattern as teleop
            # updating arm targets via step_cartesian).
            if robot.articutool is not None:
                import numpy as np

                atool = robot.articutool

                tilt_slider = gui.add_slider(
                    "Tilt (joint1)",
                    min=-1.5708,
                    max=1.5708,
                    step=0.01,
                    initial_value=0.0,
                )
                roll_slider = gui.add_slider(
                    "Roll (joint2)",
                    min=-3.14159,
                    max=3.14159,
                    step=0.01,
                    initial_value=0.0,
                )

                def _set_tool_target(tilt: float, roll: float) -> None:
                    """Set articutool target through the PhysicsController."""
                    ctx = robot._active_context
                    if ctx is None or ctx._controller is None:
                        return
                    entity_state = ctx._controller._entities.get("articutool")
                    if entity_state is not None:
                        entity_state.target_position = np.array([tilt, roll])

                @tilt_slider.on_update
                def _on_tilt(_: _viser.GuiEvent) -> None:
                    _set_tool_target(tilt_slider.value, roll_slider.value)

                @roll_slider.on_update
                def _on_roll(_: _viser.GuiEvent) -> None:
                    _set_tool_target(tilt_slider.value, roll_slider.value)

        viewer._panels.extend(all_panels)

    # -- Delegate to generic console (exact geodude pattern) --------------------
    _start_console(
        robot,
        physics=physics,
        viser=viser,
        headless=not viewer,
        robot_name="ADA",
        extra_ns=extra_ns,
        panel_setup=_setup_ada_panels if viser else None,
    )
