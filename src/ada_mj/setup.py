# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Build the ADA MuJoCo scene.

Loads the JACO2 arm from a pre-generated URDF (committed to version
control, generated once via ``scripts/convert_from_xacro.py``), then
adds workspace objects (wheelchair, human, table, head) and the
Articutool 2-DOF fork via MjSpec.

No xacro, no ROS, no runtime generation — just loads a file.

Usage::

    from ada_mj.setup import build_ada_model
    model, data = build_ada_model()
"""

from __future__ import annotations

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


def build_ada_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the complete ADA scene and return (model, data).

    Loads the JACO2 arm from a committed URDF, then adds workspace
    objects and the Articutool via MjSpec.
    """
    desc = find_ada_description()
    urdf_path = _MODELS_DIR / "jaco2.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"JACO2 URDF not found at {urdf_path}. "
            "Run: uv run python scripts/convert_from_xacro.py"
        )

    # Load the JACO2 URDF. MuJoCo's URDF parser strips directory
    # components from mesh filenames, so meshdir must point at the
    # meshes directory. Compile the arm URDF first to resolve meshes,
    # export as MJCF XML, then reload WITHOUT meshdir so we can add
    # non-arm meshes from other directories.
    # Load URDF, set meshdir, then use this spec directly. Non-arm
    # meshes need absolute paths — but MuJoCo strips directories.
    # Workaround: create symlinks in a temp dir that flattens all meshes.
    import os
    import tempfile

    flat_dir = tempfile.mkdtemp(prefix="ada_meshes_")
    # Symlink arm meshes
    for f in (desc / "meshes").glob("*.STL"):
        target = os.path.join(flat_dir, f.name)
        if not os.path.exists(target):
            os.symlink(f, target)
    # Symlink forque meshes (with unique timestamped names, no collision)
    for f in (desc / "meshes" / "forque").glob("*.stl"):
        target = os.path.join(flat_dir, f.name)
        if not os.path.exists(target):
            os.symlink(f, target)
    # Symlink workspace meshes
    try:
        scene_dir = find_ada_planning_scene() / "assets"
        for f in scene_dir.glob("*.stl"):
            target = os.path.join(flat_dir, f.name)
            if not os.path.exists(target):
                os.symlink(f, target)
    except FileNotFoundError:
        pass  # workspace meshes optional

    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.meshdir = flat_dir
    spec.option.gravity = [0, 0, -9.81]

    # Gravcomp on all arm bodies
    for name in [
        "j2n6s200_link_base",
        "j2n6s200_link_1",
        "j2n6s200_link_2",
        "j2n6s200_link_3",
        "j2n6s200_link_4",
        "j2n6s200_link_5",
        "j2n6s200_link_6",
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

    # EE site on the end effector body
    ee_body = spec.body("j2n6s200_end_effector")
    if ee_body is not None:
        site = ee_body.add_site()
        site.name = "ee_site"
        site.pos = [0, 0, 0]
        site.size = [0.005, 0, 0]
        site.rgba = [0, 1, 0, 1]

    # Articutool (2-DOF: tilt + roll)
    _add_articutool(spec, desc)

    # Floor + lighting
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3, 3, 0.1]
    floor.rgba = [0.9, 0.9, 0.9, 1]

    light = spec.worldbody.add_light()
    light.pos = [0, 0, 3]
    light.dir = [0, 0, -1]

    # Workspace objects
    _add_workspace(spec)

    # Head (mocap body with mouth site)
    _add_head(spec)

    # Compile
    model = spec.compile()
    data = mujoco.MjData(model)

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


def _add_articutool(spec: mujoco.MjSpec, desc: Path) -> None:
    """Add the 2-DOF Articutool (tilt + roll) to j2n6s200_link_6."""
    hand = spec.body("j2n6s200_link_6")
    if hand is None:
        return

    # Forque meshes — filenames only (flat_dir has symlinks)
    for name, fname in [
        ("ft_arm_mount", "2024_01_18_FTArmMount.stl"),
        ("fork_handle", "2024_06_24_forkHandleHollow.stl"),
        ("fork_tine", "fork_tine.stl"),
    ]:
        m = spec.add_mesh()
        m.name = name
        m.file = fname
        m.scale = [0.001, 0.001, 0.001]

    # Mount
    mount = hand.add_body()
    mount.name = "articutool_mount"
    mount.pos = [0.0065, -0.011, -0.0075]
    mount.quat = _rpy_to_quat([np.pi / 2, 0, np.pi / 2])

    mg = mount.add_geom()
    mg.type = mujoco.mjtGeom.mjGEOM_MESH
    mg.meshname = "ft_arm_mount"
    mg.contype = 0
    mg.conaffinity = 0
    mg.rgba = [0.4, 0.4, 0.4, 1]

    # Tilt joint
    tilt = mount.add_body()
    tilt.name = "articutool_tilt"
    tilt.pos = [0, 0, 0.05]

    tj = tilt.add_joint()
    tj.name = "articutool_tilt_joint"
    tj.type = mujoco.mjtJoint.mjJNT_HINGE
    tj.axis = [1, 0, 0]
    tj.range = [-1.0, 1.0]
    tj.limited = True
    tj.damping = 0.1

    tg = tilt.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_MESH
    tg.meshname = "fork_handle"
    tg.contype = 0
    tg.conaffinity = 0
    tg.rgba = [0.3, 0.3, 0.3, 1]

    ta = spec.add_actuator()
    ta.name = "act_articutool_tilt"
    ta.target = "articutool_tilt_joint"
    ta.trntype = mujoco.mjtTrn.mjTRN_JOINT
    ta.gainprm[0] = 10.0
    ta.biasprm[1] = -10.0
    ta.biasprm[2] = -1.0
    ta.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    # Roll joint
    roll = tilt.add_body()
    roll.name = "articutool_roll"
    roll.pos = [0, 0, 0.05]

    rj = roll.add_joint()
    rj.name = "articutool_roll_joint"
    rj.type = mujoco.mjtJoint.mjJNT_HINGE
    rj.axis = [0, 0, 1]
    rj.range = [-np.pi, np.pi]
    rj.limited = True
    rj.damping = 0.05

    ra = spec.add_actuator()
    ra.name = "act_articutool_roll"
    ra.target = "articutool_roll_joint"
    ra.trntype = mujoco.mjtTrn.mjTRN_JOINT
    ra.gainprm[0] = 5.0
    ra.biasprm[1] = -5.0
    ra.biasprm[2] = -0.5
    ra.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    # Fork tine + tip site
    tine = roll.add_body()
    tine.name = "fork_tine"
    tine.pos = [0, 0, 0.05]

    fg = tine.add_geom()
    fg.type = mujoco.mjtGeom.mjGEOM_MESH
    fg.meshname = "fork_tine"
    fg.contype = 1
    fg.conaffinity = 1
    fg.rgba = [0.8, 0.8, 0.8, 1]

    tip = tine.add_site()
    tip.name = "fork_tip"
    tip.pos = [0, 0, 0.04]
    tip.size = [0.003, 0, 0]
    tip.rgba = [1, 0, 0, 1]


def _add_workspace(spec: mujoco.MjSpec) -> None:
    """Add wheelchair, human body, and table."""
    # Meshes are symlinked into flat_dir; just use filenames
    for name in ["wheelchair", "body_collision_in_wheelchair", "table", "tom"]:
        m = spec.add_mesh()
        m.name = f"scene_{name}"
        m.file = f"{name}.stl"

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


def _rpy_to_quat(rpy: list[float]) -> list[float]:
    """Convert roll-pitch-yaw to wxyz quaternion."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("xyz", rpy)
    q = R.as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
