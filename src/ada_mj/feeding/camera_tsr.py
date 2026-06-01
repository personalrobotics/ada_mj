# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Camera TSR generator for ADA plate observation.

Generates TSRTemplates that place the wrist camera so the whole plate
falls inside its view frustum — the ``observe_plate`` staging region the
feeding loop returns to after each bite.

The TSR constrains the *camera* pose (not the fork tip): a general "look-at"
region in which the optical axis passes through the plate center from any
direction within a cone of half-angle ``tilt_max`` off vertical, at a standoff
far enough that the whole plate fits the view. ``Tw_e`` bakes in the inverse of
``T_ee_to_cam`` so the planner gets EE targets. ``tilt_max=0`` degenerates to a
single straight-down-overhead view.

The camera is rigidly mounted on ``link_6`` (the same body as ``ee_site``,
no intervening joints), so ``T_ee_to_cam`` is a constant — this is what lets us
author the TSR in *camera space* (the natural space for a look-at constraint)
and push it through a constant ``Tw_e`` to the EE-space region the planner
solves. With a wrist-moving camera the offset would depend on configuration and
this would not be possible. (Contrast ``ForkTSR``, whose EE→tip transform does
change with the articutool joints — there the tilt is fixed before planning.)

Geometry — the look-at cone (reference frame at the plate center):
  Because the sampled displacement ``Δ`` (translations zeroed) is a pure
  rotation about the reference origin, and the canonical optical axis already
  points at that origin, EVERY sampled pose keeps the optical axis through the
  plate center. The TSR's Euler box (``T0_e = T0_w·Trans·Rz(yaw)·Ry(pitch)·
  Rx(roll)·Tw_e``) is used as: roll=0, pitch∈[0,tilt_max] (tilt magnitude,
  since tilt = arccos(cos·roll·cos·pitch) = pitch when roll=0), yaw∈[-π,π] free
  (azimuth orbit when tilted; image rotation when not — rotation about vertical
  preserves tilt magnitude). No lateral translation: offsetting the optical
  axis from the plate center would break the framing guarantee.

Framing guarantee — minimum standoff for a disc of radius R at tilt θ, with
``β = fovy/2`` the limiting half field-of-view:
      d_min(θ) = R·(sinθ + cosθ/tanβ)   for θ ≤ β   (near rim governs)
               = R/sinβ                  for θ > β   (rim tangent governs)
  Derived from requiring every plate-rim point to fall inside the FOV cone of
  half-angle β. β is the *inscribed* circle of the FOV rectangle (= fovy/2,
  since the 1280×720 sensor makes the horizontal FOV wider): with yaw/image-roll
  free, the rectangle rotates arbitrarily relative to the plate, so the largest
  cone guaranteed inside it for all rolls is exactly that inscribed circle.

This is ADA-specific (lives in ada_mj, not the tsr repo).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tsr.template import TSRTemplate

if TYPE_CHECKING:
    import mujoco


def min_standoff_for_disc(disc_radius: float, fovy: float, *, tilt: float = 0.0) -> float:
    """Minimum standoff for a disc to fit a camera's FOV cone (closed form).

    A disc of radius ``disc_radius`` is viewed with the optical axis through its
    center, tilted ``tilt`` off the disc normal. Every rim point stays inside
    the FOV cone of half-angle ``β = fovy/2`` when::

        d_min(θ) = disc_radius·(sinθ + cosθ/tanβ)   for θ ≤ β   (near rim governs)
                 = disc_radius/sinβ                  for θ > β   (rim tangent governs)

    Continuous at ``θ = β`` and saturating beyond it. Derived by maximizing the
    rim-point angle over the disc; see the module docstring. Pure geometry — no
    model, no margins (pass an already-inflated radius for a framing margin).

    Args:
        disc_radius: Disc radius (m).
        fovy: Full vertical field of view (rad); ``β = fovy/2`` is the limiting
            half-angle (the FOV's inscribed circle).
        tilt: Optical-axis tilt off the disc normal (rad).

    Returns:
        Minimum standoff distance (m).
    """
    beta = fovy / 2.0
    if tilt <= beta:
        return disc_radius * (np.sin(tilt) + np.cos(tilt) / np.tan(beta))
    return disc_radius / np.sin(beta)


class CameraTSR:
    """Generate TSR templates that frame the plate in the wrist camera.

    Reads the EE-to-camera transform live from the MuJoCo model. Unlike the
    fork tip, the camera is rigid relative to the EE, so this transform is
    constant — but reading it live keeps the generator robust to model
    changes (e.g. a recalibrated camera mount).

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ee_site_name: Name of the EE site (on the arm kinematic chain).
        camera_name: Name of the MuJoCo camera (color optical frame).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "ee_site",
        camera_name: str = "camera/d415_color",
    ):
        import mujoco as mj

        self._model = model
        self._data = data
        self._ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, ee_site_name)
        self._cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

        if self._ee_site_id < 0:
            raise ValueError(f"EE site '{ee_site_name}' not found")
        if self._cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found")

    @property
    def fovy(self) -> float:
        """Camera vertical field of view (radians)."""
        return float(np.radians(self._model.cam_fovy[self._cam_id]))

    def min_standoff(
        self,
        plate_radius: float,
        *,
        tilt: float = 0.0,
        frame_margin: float = 0.15,
    ) -> float:
        """Minimum camera standoff (m) for the whole plate to fit the frustum.

        Closed form (see module docstring for the derivation). The plate is a
        disc of radius ``R_eff = plate_radius·(1+frame_margin)``; with the
        optical axis through the plate center, every rim point stays inside the
        FOV cone of half-angle ``β = fovy/2`` when::

            d_min(θ) = R_eff·(sinθ + cosθ/tanβ)   for θ ≤ β
                     = R_eff/sinβ                  for θ > β

        For ``θ > β`` the binding ray is tangent to the rim and the standoff
        saturates at ``R_eff/sinβ`` — tilting further needs no extra distance.

        Args:
            plate_radius: Plate radius (m).
            tilt: Camera tilt off vertical (rad). The guarantee must hold for
                the *largest* tilt the TSR permits, so pass ``tilt_max`` here.
            frame_margin: Fractional border to leave around the plate
                (0.15 → plate occupies at most ~87% of the limiting dimension).

        Returns:
            Minimum standoff distance ``d`` (m).
        """
        r_eff = plate_radius * (1.0 + frame_margin)
        return min_standoff_for_disc(r_eff, self.fovy, tilt=tilt)

    def _site_pose(self, site_id: int) -> np.ndarray:
        """Read a 4x4 pose from a MuJoCo site."""
        T = np.eye(4)
        T[:3, :3] = self._data.site_xmat[site_id].reshape(3, 3)
        T[:3, 3] = self._data.site_xpos[site_id]
        return T

    def _camera_pose(self) -> np.ndarray:
        """Read the camera's 4x4 world pose (MuJoCo -z = view direction)."""
        T = np.eye(4)
        T[:3, :3] = self._data.cam_xmat[self._cam_id].reshape(3, 3)
        T[:3, 3] = self._data.cam_xpos[self._cam_id]
        return T

    def _get_T_ee_to_cam(self) -> np.ndarray:
        """Compute the (constant) EE-to-camera transform from the model.

        Runs forward kinematics so site and camera poses are consistent,
        then returns ``inv(T_world_ee) @ T_world_cam``. The result is
        independent of arm configuration (camera and EE share ``link_6``).
        """
        import mujoco as mj

        mj.mj_forward(self._model, self._data)
        T_world_ee = self._site_pose(self._ee_site_id)
        T_world_cam = self._camera_pose()
        return np.linalg.inv(T_world_ee) @ T_world_cam

    def observe(
        self,
        plate_radius: float,
        *,
        tilt_max: float = 0.0,
        frame_margin: float = 0.15,
        standoff_margin: float = 0.05,
        standoff_range: float = 0.10,
        k: int = 3,
    ) -> list[TSRTemplate]:
        """TSR templates for the camera framing the whole plate (look-at cone).

        The reference frame is the plate center (z up). At ``Bw = 0`` the camera
        sits at ``[0, 0, d]`` looking straight down (MuJoCo camera -z = view
        direction = world -z), optical axis through the plate center. The Euler
        box opens this into a look-at cone (see module docstring for why each
        sampled pose keeps the axis on the plate center):

        - x, y, z: 0 (z standoff baked per template; no lateral offset, which
          would move the optical axis off center and break the guarantee)
        - roll: 0 (keeps tilt magnitude exactly equal to pitch)
        - pitch: ``[0, tilt_max]`` — tilt of the optical axis off vertical
        - yaw: ``[-π, π]`` — azimuth orbit (when tilted) / image rotation

        Standoff is sized for the *worst-case* tilt (``tilt_max``), so every
        sampled view in the cone frames the whole plate. ``tilt_max=0`` yields a
        single straight-down-overhead view (best for perception); a larger cone
        trades view quality for reachability.

        Args:
            plate_radius: Plate radius (m).
            tilt_max: Half-angle (rad) of the look-at cone off vertical. 0 →
                straight down only.
            frame_margin: Fractional border to leave around the plate.
            standoff_margin: Extra standoff (m) beyond the geometric minimum.
            standoff_range: Span (m) of standoff levels above ``d_min``.
            k: Number of standoff levels to generate.

        Returns:
            List of TSRTemplates. Instantiate each with the plate's world pose.
        """
        d_min = (
            self.min_standoff(plate_radius, tilt=tilt_max, frame_margin=frame_margin)
            + standoff_margin
        )

        # T_ee_to_cam is constant; T_cam_to_ee composes the canonical camera
        # pose (in plate frame) into the EE target the planner solves for.
        T_cam_ee = np.linalg.inv(self._get_T_ee_to_cam())

        # Canonical camera pose in plate frame: identity rotation places
        # z_cam = +z_plate (= world up), so -z_cam (view direction) points
        # down at the plate. Position is set per standoff level below.
        Bw = np.array(
            [
                [0.0, 0.0],  # x: no lateral offset
                [0.0, 0.0],  # y: no lateral offset
                [0.0, 0.0],  # z: fixed at the template's standoff
                [0.0, 0.0],  # roll: fixed → tilt magnitude == pitch
                [0.0, tilt_max],  # pitch: tilt off vertical (the look-at cone)
                [-np.pi, np.pi],  # yaw: azimuth orbit / image rotation
            ]
        )

        templates = []
        standoffs = np.linspace(d_min, d_min + standoff_range, max(k, 1))
        for d in standoffs:
            T_plate_cam = np.eye(4)
            T_plate_cam[2, 3] = d
            Tw_e = T_plate_cam @ T_cam_ee

            templates.append(
                TSRTemplate(
                    T_ref_tsr=np.eye(4),
                    Tw_e=Tw_e,
                    Bw=Bw,
                    task="observe",
                    subject="camera",
                    reference="plate",
                    name=f"Camera over plate — {d * 100:.0f}cm",
                    description=f"Camera {d * 100:.1f}cm from plate center, "
                    f"look-at, tilt ≤{np.degrees(tilt_max):.0f}°, yaw free",
                )
            )

        return templates
