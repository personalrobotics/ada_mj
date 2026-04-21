# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ADA CLI entry point.

Usage::

    uv run python -m ada_mj --viser                  # browser viewer (default)
    uv run python -m ada_mj --viser --physics         # with physics
    uv run python -m ada_mj --tool forque --viser     # forque instead of articutool
    uv run python -m ada_mj --tool-tip spoon --viser  # spoon tip
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ada",
        description="ADA interactive console",
    )
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

    from ada_mj.config import ADAConfig

    tool = None if args.no_tool else args.tool
    config = ADAConfig(
        tool=tool,
        tool_tip=args.tool_tip,
        with_human=not args.no_human,
        with_camera=not args.no_camera,
    )

    from ada_mj.robot import ADA

    print(f"\nLoading ADA (tool={tool}, tip={args.tool_tip})...", flush=True)
    robot = ADA(config)

    from ada_mj.console import start_console

    start_console(
        robot,
        physics=args.physics,
        viewer=args.viewer,
        viser=args.viser,
    )
