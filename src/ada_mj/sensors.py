# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Wrist camera rendering (Intel RealSense D415)."""

from __future__ import annotations

import mujoco
import numpy as np

from ada_mj.config import CameraConfig


class WristCamera:
    """Renders color and depth images from the D415 camera.

    The camera is at the calibrated color_optical_frame on the wrist.
    MuJoCo fovy=43° approximates the D415 nominal fy=911. For pixel-
    accurate rendering with the full intrinsic matrix, see ada_mj#5.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: CameraConfig):
        self._model = model
        self._data = data
        self._config = config

        self._color_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.color_camera)
        self._depth_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.depth_camera)

        if self._color_cam_id < 0:
            raise ValueError(f"Color camera '{config.color_camera}' not found")
        if self._depth_cam_id < 0:
            raise ValueError(f"Depth camera '{config.depth_camera}' not found")

        w, h = config.resolution
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, h)

        self._renderer = mujoco.Renderer(model, height=h, width=w)

    def render_color(self) -> np.ndarray:
        """RGB image at [H, W, 3] uint8."""
        self._renderer.update_scene(self._data, camera=self._config.color_camera)
        return self._renderer.render()

    def render_depth(self) -> np.ndarray:
        """Depth image at [H, W] float32, clipped to D415 range [0.16, 10.0] m."""
        self._renderer.enable_depth_rendering(True)
        self._renderer.update_scene(self._data, camera=self._config.depth_camera)
        depth = self._renderer.render()
        self._renderer.enable_depth_rendering(False)

        lo, hi = self._config.depth_range
        depth = np.clip(depth, lo, hi)
        return depth.astype(np.float32)

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """(color, depth) pair."""
        return self.render_color(), self.render_depth()
