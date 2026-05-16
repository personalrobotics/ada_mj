# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Demo scenes for ADA.

Each scene composes the base ADA robot (from ada_assets) with task-specific
furniture and objects (table, plate, food; bedside tray; etc.).
"""

from ada_mj.scenes.table import assemble_table_demo

__all__ = ["assemble_table_demo"]
