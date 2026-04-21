# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Head mocap controller for the seated human."""

from __future__ import annotations

import mujoco
import numpy as np

from ada_mj.config import HeadConfig


class HeadController:
    """Control the user's head position via mocap body.

    Simulates face detection output: the head (and mouth site) can be
    repositioned to test different user postures. On the real robot,
    the head pose comes from face detection (MediaPipe/YOLO + depth).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: HeadConfig):
        self._model = model
        self._data = data

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.body_name)
        if body_id < 0:
            raise ValueError(f"Head body '{config.body_name}' not found")
        self._mocap_id = model.body_mocapid[body_id]
        if self._mocap_id < 0:
            raise ValueError(f"Body '{config.body_name}' is not a mocap body")

        self._mouth_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.mouth_site)
        if self._mouth_site_id < 0:
            raise ValueError(f"Mouth site '{config.mouth_site}' not found")

    def get_position(self) -> np.ndarray:
        """Head position [3] in world frame."""
        return self._data.mocap_pos[self._mocap_id].copy()

    def get_orientation(self) -> np.ndarray:
        """Head orientation as quaternion [w,x,y,z]."""
        return self._data.mocap_quat[self._mocap_id].copy()

    def set_position(self, pos: np.ndarray) -> None:
        """Set head position [3]."""
        self._data.mocap_pos[self._mocap_id] = pos

    def set_orientation(self, quat: np.ndarray) -> None:
        """Set head orientation [w,x,y,z]."""
        self._data.mocap_quat[self._mocap_id] = quat

    def set_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Set head position and orientation."""
        self.set_position(pos)
        self.set_orientation(quat)

    def get_mouth_pose(self) -> np.ndarray:
        """4x4 world-frame transform of the mouth site (feeding target)."""
        pos = self._data.site_xpos[self._mouth_site_id]
        mat = self._data.site_xmat[self._mouth_site_id].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = mat
        T[:3, 3] = pos
        return T
