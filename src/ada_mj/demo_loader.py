# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Demo discovery and loading for ADA.

Demos are Python files with a ``scene`` dict (currently just ``{"name": ...}``,
picking which ADA scene to build) and a handful of public helper functions.
They live in :mod:`ada_mj.demos` or any user-provided file.

Example demo file::

    \"\"\"My demo — describe in one line.\"\"\"
    scene = {"name": "table"}

    def hello():
        print(robot)

The console injects ``robot`` (and, while inside ``robot.sim()``, the active
context is reachable via ``robot._active_context``) into the loaded demo
module and re-exports the demo's public functions into the IPython namespace.

This mirrors the same pattern used in :mod:`geodude.demo_loader`. Scene
resolution differs: ada_mj scenes are deterministic (built by
:mod:`ada_mj.scenes`), so there is no object-spawning step.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ada_mj.robot import ADA

logger = logging.getLogger(__name__)

DEMOS_DIR = Path(__file__).parent / "demos"


def discover_demos() -> dict[str, Path]:
    """Find all demos in the demos/ directory. Returns ``{name: path}``."""
    demos: dict[str, Path] = {}
    if not DEMOS_DIR.is_dir():
        return demos
    for p in sorted(DEMOS_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue
        demos[p.stem] = p
    return demos


def _get_demo_description(path: Path) -> str:
    """Extract the first line of a demo file's docstring without importing."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                quote = line[:3]
                content = line[3:]
                if content.endswith(quote):
                    return content[:-3].strip()
                return content.strip()
            if line and not line.startswith("#"):
                break
    return path.stem


def list_demos() -> None:
    """Print available demos with their one-line descriptions."""
    found = discover_demos()
    if not found:
        print("No demos found.")
        return
    print("\nAvailable demos:\n")
    for name, path in found.items():
        print(f"  {name:20s} — {_get_demo_description(path)}")
    print()


def load_demo(name_or_path: str) -> ModuleType:
    """Load a demo by name (from ``demos/``) or file path."""
    path = Path(name_or_path)
    if path.is_file():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load demo from '{path}'")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    demos = discover_demos()
    if name_or_path in demos:
        return load_demo(str(demos[name_or_path]))

    raise ValueError(
        f"Demo '{name_or_path}' not found. "
        f"Available: {', '.join(demos.keys()) or 'none'}"
    )


def inject_robot(demo_module: ModuleType, robot: ADA) -> None:
    """Set ``robot`` as a module-level global on ``demo_module``."""
    demo_module.robot = robot


def apply_initial_pose(demo_module: ModuleType, robot: ADA) -> None:
    """If the demo declares ``scene["initial_pose"]``, move the arm there.

    Demos use this when the default ``stow`` keyframe collides with
    something the demo's scene adds (e.g. the feeding demo's plate
    overlaps the fork at stow). Set ``initial_pose`` in the demo's
    ``scene`` dict to one of ``robot.named_poses``.
    """
    pose_name = getattr(demo_module, "scene", {}).get("initial_pose")
    if pose_name is None:
        return
    if pose_name not in robot.named_poses:
        raise ValueError(
            f"Demo declares initial_pose='{pose_name}', "
            f"but robot has no such named pose. Available: "
            f"{sorted(robot.named_poses)}"
        )
    robot.arm.set_joint_positions(robot.named_poses[pose_name])
    # Snap the welded tool freejoint to the new link_6 pose and re-run FK so
    # the visible state matches the new joint config before the console
    # starts.
    robot._snap_tool_to_weld()
    import mujoco

    mujoco.mj_forward(robot.model, robot.data)
    # Reset ctrl so position actuators don't fight the new qpos when physics
    # starts later.
    robot._init_ctrl_from_qpos()


def get_demo_functions(demo_module: ModuleType) -> dict[str, Callable]:
    """Return the public functions defined in ``demo_module``.

    Skips any function whose name starts with ``_`` or that was imported from
    elsewhere.
    """
    return {
        name: obj
        for name, obj in inspect.getmembers(demo_module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == demo_module.__name__
    }
