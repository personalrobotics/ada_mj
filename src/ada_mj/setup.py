# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Build the ADA MuJoCo scene programmatically via MjSpec.

Constructs the full assembly: JACO2 arm, Articutool 2-DOF fork,
wheelchair tilt joint, seated human with mocap head, and workspace
objects (table, plate). All mesh paths are resolved from the local
ada_ros2 and ada_feeding clones.

Usage::

    from ada_mj.setup import build_ada_model
    model, data = build_ada_model()
"""

from __future__ import annotations

import mujoco
import numpy as np

from ada_mj.jaco2 import (
    JACO2_ABOVE_PLATE,
    JACO2_EE_ORIGIN,
    JACO2_EFFORT_LIMITS,
    JACO2_FINGER_CLOSED,
    JACO2_JOINT_NAMES,
    JACO2_JOINT_ORIGINS,
    JACO2_JOINT_RPYS,
    JACO2_LOWER,
    JACO2_UPPER,
)
from ada_mj.paths import arm_mesh, forque_mesh, scene_mesh


def _rpy_to_quat(rpy: list[float]) -> list[float]:
    """Convert roll-pitch-yaw to wxyz quaternion.

    Uses the XYZ extrinsic convention (same as URDF) via scipy.
    """
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("xyz", rpy)
    q = R.as_quat()  # xyzw
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]  # wxyz


def build_ada_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the complete ADA scene and return (model, data).

    The scene contains:
    - JACO2 6-DOF arm with position actuators and gravcomp
    - 2 finger joints (locked at FINGER_CLOSED for fork grip)
    - Articutool: 2 revolute joints (tilt + roll) with fork tine
    - Wheelchair tilt joint (root → root_tilt)
    - Wheelchair collision mesh + seated human visual
    - Head as mocap body with mouth site
    - Table + plate
    """
    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, -9.81]

    # -- Assets: meshes ------------------------------------------------
    _add_arm_meshes(spec)
    _add_forque_meshes(spec)
    _add_scene_meshes(spec)

    # -- Floor + lighting ----------------------------------------------
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3, 3, 0.1]
    floor.rgba = [0.9, 0.9, 0.9, 1]
    floor.contype = 1
    floor.conaffinity = 1

    light = spec.worldbody.add_light()
    light.pos = [0, 0, 3]
    light.dir = [0, 0, -1]
    light.diffuse = [0.8, 0.8, 0.8]

    # -- Wheelchair + human --------------------------------------------
    _add_wheelchair(spec)

    # -- Table ---------------------------------------------------------
    _add_table(spec)

    # -- Head (mocap body) ---------------------------------------------
    _add_head(spec)

    # -- Compile -------------------------------------------------------
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


# ------------------------------------------------------------------
# Arm chain
# ------------------------------------------------------------------

_ARM_LINKS = ["base", "shoulder", "arm", "forearm", "wrist", "wrist", "hand_2finger"]
_ARM_MESH_NAMES = ["base", "shoulder", "arm", "forearm", "wrist", "wrist", "hand_2finger"]
_RING_MESHES = ["ring_big", "ring_big", "ring_big", "ring_small", "ring_small", "ring_small", None]


def _add_arm_meshes(spec: mujoco.MjSpec) -> None:
    """Register all arm meshes as assets."""
    for name in ["base", "shoulder", "arm", "forearm", "wrist", "hand_2finger", "ring_big", "ring_small", "finger_proximal", "finger_distal"]:
        m = spec.add_mesh()
        m.name = name
        m.file = str(arm_mesh(f"{name}.stl"))


def _add_forque_meshes(spec: mujoco.MjSpec) -> None:
    """Register forque/fork meshes."""
    forque_files = {
        "ft_arm_mount": "2024_01_18_FTArmMount.stl",
        "ft_mount": "2024_01_18_FTMount.stl",
        "ft_sensor": "ATI-9105-TW-NANO25-E.stl",
        "fork_handle": "2024_06_24_forkHandleHollow.stl",
        "fork_tine": "fork_tine.stl",
    }
    for name, fname in forque_files.items():
        m = spec.add_mesh()
        m.name = name
        m.file = str(forque_mesh(fname))
        m.scale = [0.001, 0.001, 0.001]


def _add_scene_meshes(spec: mujoco.MjSpec) -> None:
    """Register workspace meshes."""
    for name in ["wheelchair", "body_collision_in_wheelchair", "table", "tom"]:
        m = spec.add_mesh()
        m.name = name
        m.file = str(scene_mesh(f"{name}.stl"))


def _build_arm_chain(spec: mujoco.MjSpec, parent_body) -> None:
    """Build the JACO2 6-DOF arm kinematic chain under parent_body.

    Each link: body with mesh geom + joint.
    """
    current = parent_body

    for i in range(6):
        # Link body
        link = current.add_body()
        link.name = f"j2n6s200_link_{i + 1}" if i < 6 else "j2n6s200_link_hand"
        link.pos = JACO2_JOINT_ORIGINS[i]
        link.quat = _rpy_to_quat(JACO2_JOINT_RPYS[i])
        link.gravcomp = 1

        # Joint
        j = link.add_joint()
        j.name = JACO2_JOINT_NAMES[i]
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = [0, 0, 1]
        if i in (1, 2):  # J2, J3 have limits
            j.range = [float(JACO2_LOWER[i]), float(JACO2_UPPER[i])]
            j.limited = True
        else:
            j.limited = False

        # Visual geom
        mesh_name = _ARM_MESH_NAMES[i]
        g = link.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh_name
        g.contype = 1
        g.conaffinity = 1
        g.rgba = [0.75, 0.75, 0.75, 1]

        # Actuator (position control with gravcomp)
        act = spec.add_actuator()
        act.name = f"act_{JACO2_JOINT_NAMES[i]}"
        act.target = JACO2_JOINT_NAMES[i]
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = 50.0  # Kp — will tune later
        act.biasprm[1] = -50.0  # position feedback
        act.biasprm[2] = -5.0  # velocity damping
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

        current = link

    # -- End effector virtual link (URDF convention) --
    ee = current.add_body()
    ee.name = "j2n6s200_end_effector"
    ee.pos = JACO2_EE_ORIGIN
    ee.quat = _rpy_to_quat([np.pi, 0, np.pi / 2])

    # -- Fingers (locked, holding fork) --
    _add_fingers(current)

    # -- Articutool (2-DOF: tilt + roll) --
    _add_articutool(spec, current)


def _add_fingers(hand_body) -> None:
    """Add two finger joints, locked at FINGER_CLOSED."""
    for side, y_sign in [("1", 1), ("2", -1)]:
        fb = hand_body.add_body()
        fb.name = f"j2n6s200_link_finger_{side}"
        fb.pos = [-0.0025, y_sign * 0.03095, -0.11482]
        fb.quat = _rpy_to_quat([-np.pi / 2, 0.6493, y_sign * np.pi / 2])

        fj = fb.add_joint()
        fj.name = f"j2n6s200_joint_finger_{side}"
        fj.type = mujoco.mjtJoint.mjJNT_HINGE
        fj.axis = [0, 0, 1]
        fj.range = [0, 1.51]
        fj.limited = True

        fg = fb.add_geom()
        fg.type = mujoco.mjtGeom.mjGEOM_MESH
        fg.meshname = "finger_proximal"
        fg.contype = 1
        fg.conaffinity = 1
        fg.rgba = [0.3, 0.3, 0.3, 1]

        # Finger tip (fixed)
        ft = fb.add_body()
        ft.name = f"j2n6s200_link_finger_tip_{side}"
        ft.pos = [0.044, -0.003, 0]

        ftg = ft.add_geom()
        ftg.type = mujoco.mjtGeom.mjGEOM_MESH
        ftg.meshname = "finger_distal"
        ftg.contype = 1
        ftg.conaffinity = 1
        ftg.rgba = [0.2, 0.2, 0.2, 1]


def _add_articutool(spec: mujoco.MjSpec, hand_body) -> None:
    """Add the 2-DOF Articutool (tilt + roll) at the JACO2 flange.

    The Articutool has:
    - A tilt joint (lateral axis, controls approach angle to food)
    - A roll joint (long axis, for TWIRL_CW/CCW primitives)
    - Fork tine geometry with a fork_tip site
    """
    # Mount body (at the forque attachment point)
    mount = hand_body.add_body()
    mount.name = "articutool_mount"
    mount.pos = [0.0065, -0.011, -0.0075]
    mount.quat = _rpy_to_quat([np.pi / 2, 0, np.pi / 2])

    mg = mount.add_geom()
    mg.type = mujoco.mjtGeom.mjGEOM_MESH
    mg.meshname = "ft_arm_mount"
    mg.contype = 0
    mg.conaffinity = 0
    mg.rgba = [0.4, 0.4, 0.4, 1]

    # Tilt body + joint
    tilt = mount.add_body()
    tilt.name = "articutool_tilt"
    tilt.pos = [0, 0, 0.05]  # offset along mount

    tj = tilt.add_joint()
    tj.name = "articutool_tilt_joint"
    tj.type = mujoco.mjtJoint.mjJNT_HINGE
    tj.axis = [1, 0, 0]  # lateral tilt
    tj.range = [-1.0, 1.0]
    tj.limited = True
    tj.damping = 0.1

    tg = tilt.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_MESH
    tg.meshname = "fork_handle"
    tg.contype = 0
    tg.conaffinity = 0
    tg.rgba = [0.3, 0.3, 0.3, 1]

    # Tilt actuator
    ta = spec.add_actuator()
    ta.name = "act_articutool_tilt"
    ta.target = "articutool_tilt_joint"
    ta.trntype = mujoco.mjtTrn.mjTRN_JOINT
    ta.gainprm[0] = 10.0
    ta.biasprm[1] = -10.0
    ta.biasprm[2] = -1.0
    ta.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    # Roll body + joint
    roll = tilt.add_body()
    roll.name = "articutool_roll"
    roll.pos = [0, 0, 0.05]

    rj = roll.add_joint()
    rj.name = "articutool_roll_joint"
    rj.type = mujoco.mjtJoint.mjJNT_HINGE
    rj.axis = [0, 0, 1]  # roll around fork axis
    rj.range = [-np.pi, np.pi]
    rj.limited = True
    rj.damping = 0.05

    # Roll actuator
    ra = spec.add_actuator()
    ra.name = "act_articutool_roll"
    ra.target = "articutool_roll_joint"
    ra.trntype = mujoco.mjtTrn.mjTRN_JOINT
    ra.gainprm[0] = 5.0
    ra.biasprm[1] = -5.0
    ra.biasprm[2] = -0.5
    ra.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    # Fork tine
    tine = roll.add_body()
    tine.name = "fork_tine"
    tine.pos = [0, 0, 0.05]

    fg = tine.add_geom()
    fg.type = mujoco.mjtGeom.mjGEOM_MESH
    fg.meshname = "fork_tine"
    fg.contype = 1
    fg.conaffinity = 1
    fg.rgba = [0.8, 0.8, 0.8, 1]

    # Fork tip site (the actual EE target for feeding)
    tip_site = tine.add_site()
    tip_site.name = "fork_tip"
    tip_site.pos = [0, 0, 0.04]
    tip_site.size = [0.003, 0, 0]
    tip_site.rgba = [1, 0, 0, 1]


# ------------------------------------------------------------------
# Wheelchair + human
# ------------------------------------------------------------------


def _add_wheelchair(spec: mujoco.MjSpec) -> None:
    """Add wheelchair with tilt joint and arm mount."""
    # Wheelchair base (static)
    wc = spec.worldbody.add_body()
    wc.name = "wheelchair"
    wc.pos = [0, 0, 0]

    wcg = wc.add_geom()
    wcg.name = "wheelchair_visual"
    wcg.type = mujoco.mjtGeom.mjGEOM_MESH
    wcg.meshname = "wheelchair"
    wcg.pos = [0.02, -0.02, -0.05]
    wcg.contype = 1
    wcg.conaffinity = 1
    wcg.rgba = [0.3, 0.3, 0.35, 1]

    # Seated human body (collision + visual, static on wheelchair)
    body_geom = wc.add_geom()
    body_geom.name = "human_body"
    body_geom.type = mujoco.mjtGeom.mjGEOM_MESH
    body_geom.meshname = "body_collision_in_wheelchair"
    body_geom.pos = [0.02, -0.02, -0.05]
    body_geom.contype = 1
    body_geom.conaffinity = 1
    body_geom.rgba = [0.7, 0.55, 0.45, 0.5]

    # Tom visual (the human figure)
    tom_geom = wc.add_geom()
    tom_geom.name = "human_visual"
    tom_geom.type = mujoco.mjtGeom.mjGEOM_MESH
    tom_geom.meshname = "tom"
    tom_geom.pos = [0.02, -0.02, -0.05]
    tom_geom.contype = 0
    tom_geom.conaffinity = 0
    tom_geom.rgba = [0.7, 0.55, 0.45, 0.8]

    # Tilt joint (root → root_tilt)
    tilt_body = wc.add_body()
    tilt_body.name = "root_tilt"
    tilt_body.pos = [0, 0, 0]

    tj = tilt_body.add_joint()
    tj.name = "robot_tilt"
    tj.type = mujoco.mjtJoint.mjJNT_HINGE
    tj.axis = [-1, 0, 0]
    tj.range = [-np.pi, np.pi]
    tj.limited = True
    tj.damping = 5.0

    # Tilt actuator
    ta = spec.add_actuator()
    ta.name = "act_robot_tilt"
    ta.target = "robot_tilt"
    ta.trntype = mujoco.mjtTrn.mjTRN_JOINT
    ta.gainprm[0] = 100.0
    ta.biasprm[1] = -100.0
    ta.biasprm[2] = -20.0
    ta.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    # Arm base (child of tilt)
    arm_base = tilt_body.add_body()
    arm_base.name = "j2n6s200_link_base"
    arm_base.pos = [0, 0, 0]
    arm_base.gravcomp = 1

    bg = arm_base.add_geom()
    bg.type = mujoco.mjtGeom.mjGEOM_MESH
    bg.meshname = "base"
    bg.contype = 1
    bg.conaffinity = 1
    bg.rgba = [0.15, 0.15, 0.15, 1]

    # Build the arm chain on the base
    _build_arm_chain(spec, arm_base)


def _add_table(spec: mujoco.MjSpec) -> None:
    """Add a table in front of the wheelchair."""
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [0.08, -0.5, -0.56]

    tg = table.add_geom()
    tg.name = "table_geom"
    tg.type = mujoco.mjtGeom.mjGEOM_MESH
    tg.meshname = "table"
    tg.contype = 1
    tg.conaffinity = 1
    tg.rgba = [0.6, 0.4, 0.2, 1]

    # Plate on table (simple cylinder for Phase 1)
    plate = table.add_body()
    plate.name = "plate"
    plate.pos = [0.0, 0.15, 0.48]  # on top of table

    pg = plate.add_geom()
    pg.name = "plate_geom"
    pg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    pg.size = [0.12, 0.01, 0]
    pg.rgba = [0.95, 0.95, 0.95, 1]
    pg.contype = 0
    pg.conaffinity = 0


def _add_head(spec: mujoco.MjSpec) -> None:
    """Add a mocap body for the user's head with a mouth site.

    The head is a mocap body — movable via data.mocap_pos/quat to
    simulate face detection. The mouth site is the transfer target.
    """
    head = spec.worldbody.add_body()
    head.name = "head"
    head.mocap = True
    head.pos = [0.29, 0.35, 0.85]
    head.quat = [0.704, -0.062, -0.062, -0.680]  # from planning scene

    # Head visual (sphere for now — tom.stl head is the full body)
    hg = head.add_geom()
    hg.name = "head_visual"
    hg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    hg.size = [0.09, 0, 0]
    hg.rgba = [0.7, 0.55, 0.45, 0.8]
    hg.contype = 1
    hg.conaffinity = 1

    # Mouth site — the feeding target
    mouth = head.add_site()
    mouth.name = "mouth"
    mouth.pos = [0.05, 0.0, -0.02]  # front of face, slightly below center
    mouth.size = [0.015, 0, 0]
    mouth.rgba = [1, 0, 0, 1]
    mouth.type = mujoco.mjtGeom.mjGEOM_SPHERE
