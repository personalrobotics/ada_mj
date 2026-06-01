# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Table demo: ADA + hospital over-bed table + plate + food items.

Composition:
- Base robot (wheelchair + JACO2 + articutool + human) from ada_assets
- ``ada_table`` from prl_assets, positioned in front of the user using the
  ada_feeding planning-scene pose
- ``plastic_plate`` from prl_assets, placed on the table via a StablePlacer
  TSR (canonical upright placement at the table center)
- N small bite-sized food cylinders as freejoint objects on the plate

The plate is a static body (no joint) — it stays put for the demo. Food
items are freejoint so the fork can stab and lift them.
"""

from __future__ import annotations

import numpy as np
import mujoco

from ada_assets.assembly import build_spec, compile_and_init
from prl_assets import OBJECTS_DIR
from tsr.placement.stable_placer import StablePlacer


# Table position in the wheelchair/world frame. x and y are derived from
# the ada_feeding planning scene (config: ada_planning_scene.yaml, namespace
# "seated"): table at [0.08, -0.5, -0.56] in the `root` (arm-base) frame,
# with arm-base at world [-0.02, 0.02, 0.5112]. z is forced to 0 so the
# cart rests on the visible world floor (our wheelchair model's arm-base
# height differs by ~5 cm from ada_feeding's calibration; following their
# z literally would sink the cart below our floor).
TABLE_POS = np.array([0.06, -0.48, 0.0])
TABLE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # identity (wxyz)
TABLE_SURFACE_HEIGHT = 0.735  # from prl_assets/ada_table/meta.yaml

# Plastic plate dimensions (prl_assets/plastic_plate/meta.yaml)
PLATE_RADIUS = 0.133
PLATE_HEIGHT = 0.027

# Name of the plate-center site (top surface, z-up) added in _add_table_and_plate.
PLATE_CENTER_SITE = "plate_center"


def plate_pose(model, data) -> np.ndarray | None:
    """World pose (4x4) of the plate-center site, or ``None`` if absent.

    Returns the full site frame — orientation included, not just translation —
    so callers honor a tilted plate's actual disc normal (the framing standoff
    is sized relative to that normal). Runs forward kinematics so the read
    reflects the current state. Shared by the feeding loop, the ``observe``
    demo helper, and ``scripts/validate_observe.py``.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PLATE_CENTER_SITE)
    if sid < 0:
        return None
    mujoco.mj_forward(model, data)
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = data.site_xpos[sid]
    return T

# Usable half-extents of the table surface for placement sampling.
# The table mesh is 1.85 × 0.74 m but the C-shape leaves a smaller usable
# rectangle near the user. Conservative half-extents keep the plate centered.
TABLE_PLACEMENT_X = 0.20
TABLE_PLACEMENT_Y = 0.15

# Default food primitives (small bite-sized cylinders)
DEFAULT_FOOD = [
    # (label, radius, height, rgba)
    ("strawberry", 0.013, 0.022, (0.85, 0.10, 0.15, 1.0)),
    ("carrot",     0.011, 0.030, (0.95, 0.50, 0.10, 1.0)),
    ("broccoli",   0.014, 0.020, (0.20, 0.55, 0.20, 1.0)),
]


def assemble_table_demo(
    *,
    with_human: bool = True,
    with_camera: bool = True,
    tool: str = "articutool",
    tool_tip: str = "fork",
    with_floor: bool = True,
    food: list[tuple[str, float, float, tuple[float, float, float, float]]] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Assemble the table feeding demo and return (model, data).

    Args:
        with_human, with_camera, tool, tool_tip, with_floor: forwarded to
            ``ada_assets.assembly.build_spec``.
        food: List of ``(label, radius, height, rgba)`` for food cylinders.
            Defaults to one strawberry, one carrot, one broccoli.

    Returns:
        Compiled model and data, with the JACO2 at stow keyframe and food
        items resting on the plate.
    """
    if food is None:
        food = DEFAULT_FOOD

    spec = build_spec(
        with_human=with_human,
        with_camera=with_camera,
        tool=tool,
        tool_tip=tool_tip,
        with_floor=with_floor,
    )

    plate_world_pos = _add_table_and_plate(spec)
    food_positions = _add_food_items(spec, food, plate_world_pos)
    model, data = compile_and_init(spec, tool=tool)
    _init_food_qpos(model, data, food, food_positions)
    mujoco.mj_forward(model, data)
    return model, data


def _add_table_and_plate(spec: mujoco.MjSpec) -> np.ndarray:
    """Attach the ada_table and plastic_plate to ``spec``.

    Uses ``StablePlacer`` to compute the plate's canonical upright pose
    on the table surface (Bw=0 sample → centered, yaw=0).

    Returns:
        ``(3,)`` world-frame position of the plate body origin (= plate center).
    """
    # --- Table -----------------------------------------------------------
    table_dir = OBJECTS_DIR / "ada_table"
    table_spec = mujoco.MjSpec.from_file(str(table_dir / "ada_table.xml"))
    table_spec.meshdir = str(table_dir)

    table_frame = spec.worldbody.add_frame(
        pos=TABLE_POS.tolist(),
        quat=TABLE_QUAT.tolist(),
    )
    spec.attach(table_spec, prefix="table/", frame=table_frame)

    # --- Plate placement via StablePlacer --------------------------------
    placer = StablePlacer(
        table_x=TABLE_PLACEMENT_X,
        table_y=TABLE_PLACEMENT_Y,
        reference="table",
    )
    # place_cylinder returns 2 templates: -z face down (upright), +z face down (inverted).
    # Take the upright one.
    template = placer.place_cylinder(
        cylinder_radius=PLATE_RADIUS,
        cylinder_height=PLATE_HEIGHT,
        subject="plastic_plate",
    )[0]

    # Table-surface pose in world: TABLE_POS shifted up by surface height.
    # (Identity orientation — the table mesh's z-up axis matches world z.)
    T_surface_world = np.eye(4)
    T_surface_world[:3, 3] = TABLE_POS + np.array([0.0, 0.0, TABLE_SURFACE_HEIGHT])

    # Canonical pose at Bw=0 (table center, yaw=0): T_surface_world @ Tw_e
    T_plate_world = T_surface_world @ template.Tw_e
    plate_pos = T_plate_world[:3, 3]
    plate_quat = _mat_to_quat(T_plate_world[:3, :3])

    # --- Plate body (static — no freejoint) -------------------------------
    plate_dir = OBJECTS_DIR / "plastic_plate"
    plate_spec = mujoco.MjSpec.from_file(str(plate_dir / "plastic_plate.xml"))
    plate_spec.meshdir = str(plate_dir)
    # Drop the freejoint so the plate is welded to world.
    plate_body = plate_spec.body("plastic_plate")
    for j in list(plate_body.joints):
        plate_spec.delete(j)

    plate_frame = spec.worldbody.add_frame(
        pos=plate_pos.tolist(),
        quat=plate_quat.tolist(),
    )
    spec.attach(plate_spec, prefix="plate/", frame=plate_frame)

    # Plate-center site (top surface, z-up) — used by ForkTSR.above_plate.
    # The plastic_plate body origin is at the cylinder center, so the top
    # surface is at +PLATE_HEIGHT/2 in plate frame.
    plate_center_site = spec.worldbody.add_site()
    plate_center_site.name = "plate_center"
    plate_center_site.pos = (plate_pos + np.array([0.0, 0.0, PLATE_HEIGHT / 2.0])).tolist()
    plate_center_site.size = [0.005, 0, 0]
    plate_center_site.rgba = [0.0, 1.0, 0.0, 0.4]

    return plate_pos


def _add_food_items(
    spec: mujoco.MjSpec,
    food: list[tuple[str, float, float, tuple[float, float, float, float]]],
    plate_pos: np.ndarray,
) -> list[np.ndarray]:
    """Add one freejoint cylinder per food item, geometry only.

    The bodies start at the world origin; ``_init_food_qpos`` sets their
    qpos to rest on the plate after compile.

    Returns the list of intended (x, y, z) world positions for each food.
    """
    # Spread food items in a small ring on the plate surface.
    n = len(food)
    plate_top_z = float(plate_pos[2] + PLATE_HEIGHT / 2.0)
    ring_radius = max(PLATE_RADIUS * 0.4, 0.02)

    positions = []
    for i, (label, r, h, _rgba) in enumerate(food):
        theta = 2.0 * np.pi * i / max(n, 1)
        # Center the food cylinder vertically with its base on the plate top.
        z = plate_top_z + h / 2.0 + 0.001  # tiny gap to avoid initial contact
        positions.append(
            np.array([
                float(plate_pos[0] + ring_radius * np.cos(theta)),
                float(plate_pos[1] + ring_radius * np.sin(theta)),
                z,
            ])
        )

    for i, (label, r, h, rgba) in enumerate(food):
        body = spec.worldbody.add_body()
        body.name = f"food/{label}_{i}"
        fj = body.add_freejoint()
        fj.name = f"food/{label}_{i}/freejoint"
        geom = body.add_geom()
        geom.name = f"food/{label}_{i}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [r, h / 2.0, 0.0]  # cylinder: [radius, half-height, unused]
        geom.rgba = list(rgba)
        geom.density = 600.0  # ~bread/fruit density
        geom.friction = [1.0, 0.01, 0.001]  # high friction so it sits on the plate
        geom.contype = 1
        geom.conaffinity = 1

    return positions


def _init_food_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    food: list[tuple[str, float, float, tuple[float, float, float, float]]],
    positions: list[np.ndarray],
) -> None:
    """Set each food item's freejoint qpos to its target position on the plate."""
    for i, (label, _r, _h, _rgba) in enumerate(food):
        jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"food/{label}_{i}/freejoint",
        )
        if jnt_id < 0:
            continue
        adr = model.jnt_qposadr[jnt_id]
        data.qpos[adr:adr + 3] = positions[i]
        data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]  # identity quat (wxyz)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → MuJoCo (wxyz) quaternion."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=float).flatten())
    return quat
