#!/usr/bin/env python3
"""View any component standalone in mj_viser.

Usage::

    uv run python scripts/view_component.py jaco2
    uv run python scripts/view_component.py wheelchair
    uv run python scripts/view_component.py human
    uv run python scripts/view_component.py fork
"""

import sys
from pathlib import Path

import mujoco

COMPONENTS = {
    "jaco2": "src/ada_mj/models/jaco2/j2n6s200.xml",
    "wheelchair": "src/ada_mj/models/wheelchair/scene.xml",
    "human": "src/ada_mj/models/human/seated.xml",
    "fork": "src/ada_mj/models/articutool/fork.xml",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMPONENTS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(COMPONENTS.keys())}>")
        return 1

    name = sys.argv[1]
    xml_path = COMPONENTS[name]

    print(f"Loading {name} from {xml_path}...", flush=True)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Apply first keyframe if it has one
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print(f"Applied keyframe '{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0)}'")

    mujoco.mj_forward(model, data)
    print(f"Model: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

    from mj_viser import MujocoViewer

    viewer = MujocoViewer(model, data, label=f"ADA: {name}")
    print(f"Viewer at http://localhost:8080", flush=True)
    viewer.launch()


if __name__ == "__main__":
    sys.exit(main() or 0)
