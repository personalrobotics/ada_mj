#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""One-time conversion: xacro → clean URDF for MuJoCo.

Generates ``src/ada_mj/models/jaco2.urdf`` from the ada_ros2 xacro
files. The generated URDF is committed to version control so there's
no runtime dependency on xacro or ROS.

Run this whenever the upstream xacro changes::

    uv pip install xacro
    uv run python scripts/convert_from_xacro.py

Requires:
    - ``xacro`` Python package (``uv pip install xacro``)
    - ``ada_ros2`` cloned at ``../ada_ros2`` (or set ADA_DESCRIPTION_PATH)
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path


def main() -> int:
    # Resolve ada_description
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from ada_mj.paths import find_ada_description

        desc = find_ada_description()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Check xacro is installed
    try:
        import xacro
    except ImportError:
        print("ERROR: xacro not installed. Run: uv pip install xacro", file=sys.stderr)
        return 1

    # Mock ament_index_python so xacro can resolve $(find ada_description)
    mock_ament = types.ModuleType("ament_index_python")
    mock_packages = types.ModuleType("ament_index_python.packages")

    def _get_pkg(name: str) -> str:
        if name == "ada_description":
            return str(desc)
        raise FileNotFoundError(f"Unknown package: {name}")

    mock_packages.get_package_share_directory = _get_pkg
    mock_ament.packages = mock_packages
    sys.modules["ament_index_python"] = mock_ament
    sys.modules["ament_index_python.packages"] = mock_packages

    # Process xacro
    source = desc / "urdf" / "j2n6s200_standalone.xacro"
    if not source.exists():
        print(f"ERROR: xacro source not found: {source}", file=sys.stderr)
        return 1

    print(f"Processing {source}...", flush=True)
    doc = xacro.process_file(str(source), mappings={})
    urdf = doc.toxml()

    # Clean up for MuJoCo:
    # 1. Make mesh paths relative. The xacro resolves $(find ada_description)
    #    to the absolute path, but may also leave package:// URIs. Strip both
    #    so mesh paths become just "meshes/base.STL" etc.
    urdf = urdf.replace(str(desc) + "/", "")
    urdf = urdf.replace("package://ada_description/", "")
    # Leave mesh paths as "meshes/X.STL" — meshdir will be set at load time
    # to ada_description/ so paths resolve as ada_description/meshes/X.STL

    # 2. Strip Gazebo + transmission elements (confuse MuJoCo's URDF parser)
    urdf = re.sub(r"<gazebo[^>]*>.*?</gazebo>", "", urdf, flags=re.DOTALL)
    urdf = re.sub(r"<transmission[^>]*>.*?</transmission>", "", urdf, flags=re.DOTALL)

    # 3. DAE → STL (MuJoCo doesn't load Collada)
    urdf = urdf.replace(".dae", ".STL")

    # 4. Pretty-print (the xacro output is one long line)
    try:
        import xml.dom.minidom

        urdf = xml.dom.minidom.parseString(urdf).toprettyxml(indent="  ")
        # Remove the XML declaration that minidom adds (URDF already has one)
        if urdf.startswith("<?xml"):
            urdf = urdf.split("\n", 1)[1]
    except Exception:
        pass  # Keep ugly but valid XML if pretty-print fails

    # Write to models/
    out_dir = Path(__file__).resolve().parents[1] / "src" / "ada_mj" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "jaco2.urdf"
    out.write_text(urdf)

    print(f"Written: {out}")
    print(f"  Size: {len(urdf)} chars")

    # Verify it loads in MuJoCo
    import mujoco

    spec = mujoco.MjSpec.from_file(str(out))
    spec.meshdir = str(desc / "meshes")
    model = spec.compile()
    print(f"  MuJoCo verification: nq={model.nq}, nv={model.nv}, nbody={model.nbody}, njnt={model.njnt}")
    print("\nDone. Commit src/ada_mj/models/jaco2.urdf to version control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
