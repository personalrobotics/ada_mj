# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ADA feeding pipeline — domain types, behaviors, and task.

Pure Python feeding logic with no viser, ROS, or MuJoCo imports.
The same code runs on SimContext (MuJoCo) and HardwareContext (real robot).

Usage::

    from ada_mj.feeding import feed_bite, FoodItem, AcquisitionSchema

    result = feed_bite(food_item, schema, robot=robot, ctx=ctx)
    if not result:
        print(f"Failed: {result.failure_kind.value}")
"""
