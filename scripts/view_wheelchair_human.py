#!/usr/bin/env python3
"""View wheelchair + human together to verify alignment.

Both components are in floor frame (z=0 = floor).
No offset math needed — just load and compose.
"""

import mujoco
import os
import tempfile
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "src" / "ada_mj" / "models"


def main():
    # Flat meshdir with all assets (MuJoCo strips directories from mesh filenames)
    flat = tempfile.mkdtemp()
    for d in [MODELS / "wheelchair" / "assets", MODELS / "human" / "assets"]:
        for f in d.glob("*.stl"):
            t = os.path.join(flat, f.name)
            if not os.path.exists(t):
                os.symlink(f, t)

    # Load wheelchair as base
    spec = mujoco.MjSpec.from_file(str(MODELS / "wheelchair" / "wheelchair.xml"))
    spec.meshdir = flat

    # Load human XML and merge its content
    human_spec = mujoco.MjSpec.from_file(str(MODELS / "human" / "seated.xml"))
    human_spec.meshdir = flat

    # Add human meshes to wheelchair spec
    for name, fname in [("tom", "tom.stl"), ("body_collision", "body_collision_in_wheelchair.stl")]:
        m = spec.add_mesh()
        m.name = name
        m.file = fname

    # Add human materials
    skin = spec.add_material()
    skin.name = "skin"
    skin.rgba = [0.72, 0.58, 0.47, 0.8]
    skin.specular = 0.2
    skin.shininess = 0.1

    # Body collision — static, floor frame
    user = spec.worldbody.add_body()
    user.name = "user_body"
    user.pos = [0.02, -0.02, 0.4112]
    g = user.add_geom()
    g.name = "body_safety"
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = "body_collision"
    g.contype = 1
    g.conaffinity = 1
    g.group = 3
    g.rgba = [0.72, 0.58, 0.47, 0.15]

    # Head — mocap, floor frame
    head = spec.worldbody.add_body()
    head.name = "head"
    head.mocap = True
    head.pos = [0.29, 0.35, 1.3112]
    head.quat = [0.704416, -0.061628, -0.061628, -0.680442]

    hg = head.add_geom()
    hg.name = "head_visual"
    hg.type = mujoco.mjtGeom.mjGEOM_MESH
    hg.meshname = "tom"
    hg.rgba = [0.72, 0.58, 0.47, 0.8]
    hg.contype = 0
    hg.conaffinity = 0
    hg.group = 2

    hc = head.add_geom()
    hc.name = "head_collision"
    hc.type = mujoco.mjtGeom.mjGEOM_SPHERE
    hc.size = [0.10, 0, 0]
    hc.contype = 1
    hc.conaffinity = 1
    hc.group = 3
    hc.rgba = [0.7, 0.5, 0.4, 0.2]

    ms = head.add_site()
    ms.name = "mouth"
    ms.pos = [0.05, 0.0, -0.02]
    ms.size = [0.015, 0, 0]
    ms.rgba = [1, 0, 0, 1]

    # Face wall — child of head
    fw = head.add_geom()
    fw.name = "face_wall_geom"
    fw.type = mujoco.mjtGeom.mjGEOM_BOX
    fw.size = [0.005, 0.225, 0.2]
    fw.pos = [0.15, 0, 0]
    fw.contype = 0
    fw.conaffinity = 0
    fw.group = 3
    fw.rgba = [1, 0.3, 0.3, 0.1]

    # Floor + light
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3, 3, 0.1]
    floor.rgba = [0.9, 0.9, 0.9, 1]
    light = spec.worldbody.add_light()
    light.pos = [0, 0, 3]
    light.dir = [0, 0, -1]

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Wheelchair + Human: nbody={model.nbody} ngeom={model.ngeom}")

    from mj_viser import MujocoViewer

    viewer = MujocoViewer(model, data, label="ADA: wheelchair + human")
    viewer.launch()


if __name__ == "__main__":
    main()
