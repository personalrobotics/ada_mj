# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Fork tool TSR generator for ADA feeding.

Generates TSRTemplates for fork-related tasks:
- ``above_plate``: fork tip hovering above the plate surface
- ``approach_mouth``: fork tip approaching the mouth from the front

The TSRs constrain the fork TIP position/orientation relative to
the plate or mouth. ``Tw_e`` bakes in the inverse of ``T_ee_to_tip``
so the planner gets EE targets (what it needs for IK).

This is ADA-specific (lives in ada_mj, not the tsr repo).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tsr.template import TSRTemplate

if TYPE_CHECKING:
    import mujoco


class ForkTSR:
    """Generate TSR templates for fork-based manipulation.

    Reads the EE-to-fork-tip transform live from the MuJoCo model
    (not cached) because the articutool joints change the relationship.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ee_site_name: Name of the EE site (on the arm kinematic chain).
        tip_site_name: Name of the fork tip site.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "ee_site",
        tip_site_name: str = "articutool/fork_tip",
    ):
        import mujoco as mj

        self._model = model
        self._data = data
        self._ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, ee_site_name)
        self._tip_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, tip_site_name)

        if self._ee_site_id < 0:
            raise ValueError(f"EE site '{ee_site_name}' not found")
        if self._tip_site_id < 0:
            raise ValueError(f"Fork tip site '{tip_site_name}' not found")

    def _get_T_ee_to_tip(self) -> np.ndarray:
        """Compute the current EE-to-fork-tip transform from the model.

        Must be called after mj_forward so site poses are up to date.
        The transform changes with articutool joint angles.
        """
        T_world_ee = self._site_pose(self._ee_site_id)
        T_world_tip = self._site_pose(self._tip_site_id)
        return np.linalg.inv(T_world_ee) @ T_world_tip

    def _site_pose(self, site_id: int) -> np.ndarray:
        """Read a 4x4 pose from a MuJoCo site."""
        T = np.eye(4)
        T[:3, :3] = self._data.site_xmat[site_id].reshape(3, 3)
        T[:3, 3] = self._data.site_xpos[site_id]
        return T

    def above_plate(
        self,
        plate_radius: float,
        hover_height: float = 0.03,
        *,
        k: int = 3,
    ) -> list[TSRTemplate]:
        """TSR templates for fork tip hovering above a plate.

        The plate frame has z pointing up and origin at the plate center
        surface. The fork tip should point downward (into the food).

        Freedoms:
        - x, y: free within plate radius (any position on plate)
        - z: fixed at hover_height above plate surface
        - yaw: free (any approach angle)
        - roll, pitch: constrained (fork tip must point down)

        The planner samples from these templates to find a collision-free
        configuration with the fork above the plate.

        Args:
            plate_radius: Radius of the plate (meters).
            hover_height: Height above plate surface (meters).
            k: Number of depth levels to generate.

        Returns:
            List of TSRTemplates. Instantiate with the plate's world pose.
        """
        import mujoco

        mujoco.mj_forward(self._model, self._data)
        T_ee_tip = self._get_T_ee_to_tip()

        # TSR frame: at plate surface, z-up.
        # Fork tip should be at z=hover_height, pointing down (-z).
        # Tw_e maps from TSR frame (plate) to EE: tip_pose → EE_pose.
        #
        # At Bw=0 the fork tip is at [0, 0, hover_height] in plate frame,
        # pointing down. Tw_e = T_tip_to_ee at this canonical pose.

        # Fork tip pointing down means tip z-axis = [0, 0, -1] in plate frame.
        # Construct T_plate_to_tip_canonical:
        T_plate_tip = np.eye(4)
        T_plate_tip[2, 3] = hover_height
        # Rotate tip frame so its z-axis points down (180° about x)
        T_plate_tip[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        )

        # T_tip_to_ee: inverse of T_ee_to_tip, used to compose Tw_e per height
        T_tip_ee = np.linalg.inv(T_ee_tip)

        # Bounds: x/y free on plate, z fixed, yaw free
        Bw = np.array(
            [
                [-plate_radius, plate_radius],  # x: slide on plate
                [-plate_radius, plate_radius],  # y: slide on plate
                [0.0, 0.0],  # z: fixed at hover_height
                [0.0, 0.0],  # roll: fixed
                [0.0, 0.0],  # pitch: fixed
                [-np.pi, np.pi],  # yaw: any approach angle
            ]
        )

        templates = []
        heights = np.linspace(hover_height, hover_height + 0.05, max(k, 1))
        for i, h in enumerate(heights):
            T_plate_tip_h = T_plate_tip.copy()
            T_plate_tip_h[2, 3] = h
            Tw_e_h = T_plate_tip_h @ T_tip_ee

            templates.append(
                TSRTemplate(
                    T_ref_tsr=np.eye(4),
                    Tw_e=Tw_e_h,
                    Bw=Bw,
                    task="above",
                    subject="fork_tip",
                    reference="plate",
                    name=f"Fork above plate — {h * 100:.0f}cm",
                    description=f"Fork tip {h * 100:.1f}cm above plate, "
                    f"x/y free within {plate_radius * 100:.0f}cm radius, "
                    f"yaw free",
                )
            )

        return templates

    def approach_mouth(
        self,
        approach_distance: float = 0.05,
        *,
        k: int = 3,
    ) -> list[TSRTemplate]:
        """TSR templates for fork tip approaching the mouth.

        The mouth frame has +x pointing forward (out of the mouth).
        The fork should approach along -x (toward the mouth), with
        the tip approximately horizontal.

        Freedoms:
        - x: at approach_distance from mouth (offset along +x)
        - y, z: small freedom (±2cm) for head position uncertainty
        - yaw: small freedom (±30°) for approach angle variation
        - roll, pitch: small freedom (±15°) for fork tilt

        Args:
            approach_distance: How far from mouth to position the fork tip.
            k: Number of offset levels.

        Returns:
            List of TSRTemplates. Instantiate with the mouth's world pose.
        """
        import mujoco

        mujoco.mj_forward(self._model, self._data)
        T_ee_tip = self._get_T_ee_to_tip()
        T_tip_ee = np.linalg.inv(T_ee_tip)

        # Fork tip at [approach_distance, 0, 0] in mouth frame,
        # approaching along -x (mouth's +x = forward out of face).
        # Fork tip z-axis should point along -x of mouth frame (toward mouth).
        T_mouth_tip = np.eye(4)
        T_mouth_tip[0, 3] = approach_distance  # offset along mouth +x
        # Rotate tip frame so its z-axis = mouth -x (approaching mouth)
        T_mouth_tip[:3, :3] = np.array(
            [
                [0, 0, 1],  # tip x = mouth z (up)
                [0, 1, 0],  # tip y = mouth y (left)
                [-1, 0, 0],  # tip z = mouth -x (toward mouth)
            ]
        )

        # Bounds: position tight, orientation has some freedom
        Bw = np.array(
            [
                [0.0, 0.0],  # x: fixed at approach distance
                [-0.02, 0.02],  # y: ±2cm lateral
                [-0.02, 0.02],  # z: ±2cm vertical
                [-np.radians(15), np.radians(15)],  # roll: ±15°
                [-np.radians(15), np.radians(15)],  # pitch: ±15°
                [-np.radians(30), np.radians(30)],  # yaw: ±30°
            ]
        )

        templates = []
        distances = np.linspace(approach_distance, approach_distance + 0.05, max(k, 1))
        for i, d in enumerate(distances):
            T_mouth_tip_d = T_mouth_tip.copy()
            T_mouth_tip_d[0, 3] = d
            Tw_e_d = T_mouth_tip_d @ T_tip_ee

            templates.append(
                TSRTemplate(
                    T_ref_tsr=np.eye(4),
                    Tw_e=Tw_e_d,
                    Bw=Bw,
                    task="approach",
                    subject="fork_tip",
                    reference="mouth",
                    name=f"Fork toward mouth — {d * 100:.0f}cm",
                    description=f"Fork tip {d * 100:.1f}cm from mouth along approach axis, "
                    f"y/z ±2cm, yaw ±30°, roll/pitch ±15°",
                )
            )

        return templates
