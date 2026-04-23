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
                import numpy as np

                initial = "stow" if "stow" in keyframe_names else keyframe_names[0]
                dropdown = gui.add_dropdown("Keyframe", keyframe_names, initial_value=initial)

                # Tolerance for matching joint config to a named pose (rad)
                _KEYFRAME_TOL = 0.05
                _moving = [False]

                def _current_keyframe() -> str | None:
                    """Return the named pose matching the arm's actual joints, or None."""
                    q = robot.arm.get_joint_positions()
                    for name, q_target in robot.named_poses.items():
                        if np.max(np.abs(q - np.array(q_target))) < _KEYFRAME_TOL:
                            return name
                    return None

                @dropdown.on_update
                def _on_keyframe(_: _viser.GuiEvent) -> None:
                    if _moving[0]:
                        return  # ignore clicks while moving
                    pose_name = dropdown.value
                    if pose_name not in robot.named_poses:
                        return

                    q_goal = robot.named_poses[pose_name].copy()
                    ctx = robot._active_context
                    if ctx is None:
                        return

                    _moving[0] = True
                    dropdown.disabled = True

                    def _plan_and_execute():
                        # Check on physics thread (safe MuJoCo access)
                        if _current_keyframe() == pose_name:
                            return  # already there
                        path = robot.arm.plan_to_configuration(q_goal)
                        if path is not None:
                            traj = robot.arm.retime(path)
                            ctx.execute(traj)
                        else:
                            logger.warning("Planning to %s failed", pose_name)

                    try:
                        event_loop.run_on_physics_thread(_plan_and_execute)
                    except Exception as e:
                        logger.warning("go_to(%s): %s", pose_name, e)
                    finally:
                        _moving[0] = False
                        dropdown.disabled = False

                # Sync dropdown to actual arm position each frame
                class _KeyframeSyncPanel:
                    def setup(self, gui, viewer):
                        pass

                    def on_sync(self, viewer):
                        if _moving[0]:
                            return
                        actual = _current_keyframe()
                        if actual is not None and dropdown.value != actual:
                            dropdown.value = actual

                _kf_sync = _KeyframeSyncPanel()
                viewer._panels.append(_kf_sync)

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
