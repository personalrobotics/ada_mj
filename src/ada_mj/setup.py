# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Build the ADA MuJoCo scene.

Loads the ADA robot from a pre-generated URDF (committed to version
control, generated once via ``scripts/convert_from_xacro.py``), then
adds workspace objects (wheelchair, human, table, head) via MjSpec.

The URDF includes the full assembly: JACO2 arm + wheelchair tilt +
forque F/T sensor + fork + camera mount — all with correct transforms
from the original xacro.

No xacro, no ROS, no runtime generation.

Usage::

    from ada_mj.setup import build_ada_model
    model, data = build_ada_model()
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from ada_mj.jaco2 import (
    JACO2_ABOVE_PLATE,
    JACO2_FINGER_CLOSED,
    JACO2_JOINT_NAMES,
)
from ada_mj.paths import find_ada_description, find_ada_planning_scene

_MODELS_DIR = Path(__file__).parent / "models"

# Kinova dark grey — the real JACO2 is dark carbon fiber
_ARM_RGBA = [0.15, 0.15, 0.18, 1.0]
_RING_RGBA = [0.25, 0.25, 0.28, 1.0]
_FORQUE_RGBA = [0.4, 0.4, 0.4, 1.0]


def build_ada_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the complete ADA scene and return (model, data)."""
    desc = find_ada_description()
    urdf_path = _MODELS_DIR / "ada.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"ADA URDF not found at {urdf_path}. "
            "Run: uv run python scripts/convert_from_xacro.py"
        )

    # Create a flat directory with symlinks to all mesh sources.
    # MuJoCo strips directory components from mesh filenames during
    # URDF parsing, so all meshes must be reachable by basename alone.
    flat_dir = _create_flat_meshdir(desc)

    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.meshdir = flat_dir
    spec.option.gravity = [0, 0, -9.81]

    # Gravcomp on all arm bodies (real Kinova runs internal gravcomp)
    for name in [
        "j2n6s200_link_base", "j2n6s200_link_1", "j2n6s200_link_2",
        "j2n6s200_link_3", "j2n6s200_link_4", "j2n6s200_link_5",
        "j2n6s200_link_6", "FTArmMount", "FTMount", "FT",
        "forkHandle", "forkHandleCover", "forque", "forkTine",
    ]:
        body = spec.body(name)
        if body is not None:
            body.gravcomp = 1

    # Position actuators for arm joints
    for jname in JACO2_JOINT_NAMES:
        act = spec.add_actuator()
        act.name = f"act_{jname}"
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = 50.0
        act.biasprm[1] = -50.0
        act.biasprm[2] = -5.0
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    # Wheelchair tilt actuator
    tilt_act = spec.add_actuator()
    tilt_act.name = "act_robot_tilt"
    tilt_act.target = "robot_tilt"
    tilt_act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    tilt_act.gainprm[0] = 100.0
    tilt_act.biasprm[1] = -100.0
    tilt_act.biasprm[2] = -20.0
    tilt_act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    # EE site on the end effector body
    ee_body = spec.body("j2n6s200_end_effector")
    if ee_body is not None:
        site = ee_body.add_site()
        site.name = "ee_site"
        site.pos = [0, 0, 0]
        site.size = [0.005, 0, 0]
        site.rgba = [0, 1, 0, 1]

    # Fork tip site
    fork_body = spec.body("forkTip")
    if fork_body is not None:
        site = fork_body.add_site()
        site.name = "fork_tip"
        site.pos = [0, 0, 0]
        site.size = [0.003, 0, 0]
        site.rgba = [1, 0, 0, 1]

    # Floor + lighting
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3, 3, 0.1]
    floor.rgba = [0.9, 0.9, 0.9, 1]

    light = spec.worldbody.add_light()
    light.pos = [0, 0, 3]
    light.dir = [0, 0, -1]

    # Workspace (wheelchair, human, table)
    _add_workspace(spec)

    # Head (mocap body with mouth site)
    _add_head(spec)

    # Compile
    model = spec.compile()
    data = mujoco.MjData(model)

    # Color the arm geoms (URDF materials come through as light grey)
    _color_arm_geoms(model)

    # Set arm to above-plate config
    for i, jname in enumerate(JACO2_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = JACO2_ABOVE_PLATE[i]

    # Lock fingers
    for fname in ["j2n6s200_joint_finger_1", "j2n6s200_joint_finger_2"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = JACO2_FINGER_CLOSED

    mujoco.mj_forward(model, data)
    return model, data


def _create_flat_meshdir(desc: Path) -> str:
    """Create a temp directory with symlinks to all mesh sources."""
    flat_dir = tempfile.mkdtemp(prefix="ada_meshes_")

    # Arm meshes (STL)
    for f in (desc / "meshes").glob("*.STL"):
        target = os.path.join(flat_dir, f.name)
        if not os.path.exists(target):
            os.symlink(f, target)

    # Forque meshes (stl, different case)
    for f in (desc / "meshes" / "forque").glob("*.stl"):
        # Normalize to .STL since the URDF was converted to use .STL
        stl_name = f.stem + ".STL"
        target = os.path.join(flat_dir, stl_name)
        if not os.path.exists(target):
            os.symlink(f, target)

    # Camera meshes
    for f in (desc / "meshes" / "camera").glob("*.stl"):
        stl_name = f.stem + ".STL"
        target = os.path.join(flat_dir, stl_name)
        if not os.path.exists(target):
            os.symlink(f, target)

    # Workspace meshes
    try:
        scene_dir = find_ada_planning_scene() / "assets"
        for f in scene_dir.glob("*.stl"):
            stl_name = f.stem + ".STL"
            target = os.path.join(flat_dir, stl_name)
            if not os.path.exists(target):
                os.symlink(f, target)
    except FileNotFoundError:
        pass

    return flat_dir


def _color_arm_geoms(model: mujoco.MjModel) -> None:
    """Set dark colors on arm geoms (URDF materials come through as light)."""
    arm_body_names = {
        "j2n6s200_link_base", "j2n6s200_link_1", "j2n6s200_link_2",
        "j2n6s200_link_3", "j2n6s200_link_4", "j2n6s200_link_5",
        "j2n6s200_link_6", "j2n6s200_link_finger_1", "j2n6s200_link_finger_2",
        "j2n6s200_link_finger_tip_1", "j2n6s200_link_finger_tip_2",
    }
    forque_body_names = {
        "FTArmMount", "FTMount", "FT", "forkHandle", "forkHandleCover",
        "forque", "forkTine",
    }

    arm_body_ids = set()
    for name in arm_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            arm_body_ids.add(bid)

    forque_body_ids = set()
    for name in forque_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            forque_body_ids.add(bid)

    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid in arm_body_ids:
            model.geom_rgba[gid] = _ARM_RGBA
        elif bid in forque_body_ids:
            model.geom_rgba[gid] = _FORQUE_RGBA


def _add_workspace(spec: mujoco.MjSpec) -> None:
    """Add wheelchair, human body, and table."""
    for name in ["wheelchair", "body_collision_in_wheelchair", "table", "tom"]:
        m = spec.add_mesh()
        m.name = f"scene_{name}"
        m.file = f"{name}.STL"

    # Wheelchair (static)
    wc = spec.worldbody.add_body()
    wc.name = "wheelchair"
    wc.pos = [0.02, -0.02, -0.05]

    wcg = wc.add_geom()
    wcg.name = "wheelchair_geom"
    wcg.type = mujoco.mjtGeom.mjGEOM_MESH
    wcg.meshname = "scene_wheelchair"
    wcg.contype = 1
    wcg.conaffinity = 1
    wcg.rgba = [0.3, 0.3, 0.35, 1]

    bg = wc.add_geom()
    bg.name = "human_body_geom"
    bg.type = mujoco.mjtGeom.mjGEOM_MESH
    bg.meshname = "scene_body_collision_in_wheelchair"
    bg.contype = 1
    bg.conaffinity = 1
    bg.rgba = [0.7, 0.55, 0.45, 0.4]

    tg = wc.add_geom()
    tg.name = "human_visual"
    tg.type = mujoco.mjtGeom.mjGEOM_MESH
    tg.meshname = "scene_tom"
    tg.contype = 0
    tg.conaffinity = 0
    tg.rgba = [0.7, 0.55, 0.45, 0.8]

    # Table
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [0.08, -0.5, -0.56]

    tbg = table.add_geom()
    tbg.name = "table_geom"
    tbg.type = mujoco.mjtGeom.mjGEOM_MESH
    tbg.meshname = "scene_table"
    tbg.contype = 1
    tbg.conaffinity = 1
    tbg.rgba = [0.6, 0.4, 0.2, 1]

    # Plate
    plate = table.add_body()
    plate.name = "plate"
    plate.pos = [0.0, 0.15, 0.48]

    pg = plate.add_geom()
    pg.name = "plate_geom"
    pg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    pg.size = [0.12, 0.01, 0]
    pg.rgba = [0.95, 0.95, 0.95, 1]
    pg.contype = 0
    pg.conaffinity = 0


def _add_head(spec: mujoco.MjSpec) -> None:
    """Add a mocap body for the user's head with a mouth site."""
    head = spec.worldbody.add_body()
    head.name = "head"
    head.mocap = True
    head.pos = [0.29, 0.35, 0.85]
    head.quat = [0.704, -0.062, -0.062, -0.680]

    hg = head.add_geom()
    hg.name = "head_visual"
    hg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    hg.size = [0.09, 0, 0]
    hg.rgba = [0.7, 0.55, 0.45, 0.8]
    hg.contype = 1
    hg.conaffinity = 1

    mouth = head.add_site()
    mouth.name = "mouth"
    mouth.pos = [0.05, 0.0, -0.02]
    mouth.size = [0.015, 0, 0]
    mouth.rgba = [1, 0, 0, 1]
    mouth.type = mujoco.mjtGeom.mjGEOM_SPHERE
