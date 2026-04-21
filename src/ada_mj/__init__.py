# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ADA robot-assisted feeding system — MuJoCo simulation.

Quick start::

    from ada_mj import ADA

    robot = ADA()
    with robot.sim() as ctx:
        robot.go_to("above_plate")
        robot.move_to_mouth()
"""

from ada_mj.config import ADAConfig

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy imports for heavy classes."""
    if name == "ADA":
        from ada_mj.robot import ADA

        return ADA
    if name == "Articutool":
        from ada_mj.articutool import Articutool

        return Articutool
    raise AttributeError(f"module 'ada_mj' has no attribute {name!r}")


__all__ = ["ADA", "ADAConfig", "Articutool"]
