# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ADA CLI entry point.

Usage::

    uv run python -m ada_mj --viser                       # bare table scene
    uv run python -m ada_mj --demo feeding --viser        # feeding demo
    uv run python -m ada_mj --list-demos                  # list demos and exit
    uv run python -m ada_mj --viser --physics             # with physics
    uv run python -m ada_mj --tool forque --viser         # forque tool
    uv run python -m ada_mj --tool-tip spoon --viser      # spoon tip
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ada",
        description="ADA interactive console",
    )
    parser.add_argument("--demo", type=str, default=None, help="Demo name (see --list-demos) or path to a demo file.")
    parser.add_argument("--list-demos", action="store_true", help="List available demos and exit.")
    parser.add_argument("--tool", choices=["articutool", "forque"], default="articutool",
                        help="Tool to attach (default: articutool).")
    parser.add_argument("--tool-tip", choices=["fork", "spoon"], default="fork",
                        help="Articutool tip (default: fork).")
    parser.add_argument("--no-tool", action="store_true", help="No tool.")
    parser.add_argument("--no-camera", action="store_true", help="No wrist camera.")
    parser.add_argument("--no-human", action="store_true", help="No seated human.")
    parser.add_argument("--physics", action="store_true", help="Physics simulation.")
    parser.add_argument("--viewer", action="store_true",
                        help="Launch native MuJoCo viewer (requires mjpython).")
    parser.add_argument("--viser", action="store_true",
                        help="Launch browser viewer at http://localhost:8080.")
    args = parser.parse_args()

    if args.list_demos:
        from ada_mj.demo_loader import list_demos

        list_demos()
        sys.exit(0)

    from ada_mj.config import ADAConfig

    demo_module = None
    if args.demo is not None:
        from ada_mj.demo_loader import load_demo

        demo_module = load_demo(args.demo)

    tool = None if args.no_tool else args.tool
    scene = demo_module.scene["name"] if demo_module is not None else ADAConfig().scene
    config = ADAConfig(
        tool=tool,
        tool_tip=args.tool_tip,
        with_human=not args.no_human,
        with_camera=not args.no_camera,
        scene=scene,
    )

    from ada_mj.robot import ADA

    print(f"\nLoading ADA (tool={tool}, tip={args.tool_tip}, scene={scene})...", flush=True)
    robot = ADA(config)

    if demo_module is not None:
        from ada_mj.demo_loader import apply_initial_pose

        apply_initial_pose(demo_module, robot)

    from ada_mj.console import start_console

    start_console(
        robot,
        physics=args.physics,
        viewer=args.viewer,
        viser=args.viser,
        demo_module=demo_module,
    )
