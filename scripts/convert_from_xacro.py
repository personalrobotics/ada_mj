#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""One-time conversion: xacro → clean URDF for MuJoCo.

Generates ``src/ada_mj/models/ada.urdf`` from the ada_ros2 xacro files.
The generated URDF includes the full assembly: JACO2 arm + wheelchair
tilt + forque F/T sensor + fork + camera mount. Committed to version
control — no runtime xacro dependency.

Run whenever the upstream xacro changes::

    uv pip install xacro
    uv run python scripts/convert_from_xacro.py
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from ada_mj.paths import find_ada_description

        desc = find_ada_description()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        import xacro
    except ImportError:
        print("ERROR: xacro not installed. Run: uv pip install xacro", file=sys.stderr)
        return 1

    # Mock ament_index_python for xacro's $(find ada_description)
    mock_ament = types.ModuleType("ament_index_python")
    mock_packages = types.ModuleType("ament_index_python.packages")
    mock_packages.get_package_share_directory = lambda name: str(desc) if name == "ada_description" else (_ for _ in ()).throw(FileNotFoundError(name))
    mock_ament.packages = mock_packages
    sys.modules["ament_index_python"] = mock_ament
    sys.modules["ament_index_python.packages"] = mock_packages

    # Process the full ADA assembly (arm + forque + camera + tilt)
    source = desc / "urdf" / "ada_standalone.xacro"
    if not source.exists():
        print(f"ERROR: {source} not found", file=sys.stderr)
        return 1

    print(f"Processing {source}...", flush=True)
    doc = xacro.process_file(str(source), mappings={"use_forque": "true"})
    urdf = doc.toxml()

    # Clean for MuJoCo
    urdf = urdf.replace(str(desc) + "/", "")
    urdf = urdf.replace("package://ada_description/", "")
    urdf = re.sub(r"<gazebo[^>]*>.*?</gazebo>", "", urdf, flags=re.DOTALL)
    urdf = re.sub(r"<transmission[^>]*>.*?</transmission>", "", urdf, flags=re.DOTALL)
    # Normalize all mesh extensions to .STL
    urdf = urdf.replace(".dae", ".STL")
    urdf = urdf.replace(".stl", ".STL")

    # Strip camera assembly — meshes are too high-poly for MuJoCo
    # (d415.stl = 400K faces, jetsonNano = 300K; limit is 200K).
    # Tracked as ada_mj#2 for mesh decimation. For now, remove the
    # entire camera subtree by parsing as XML and dropping bodies.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf)
    camera_link_names: set[str] = set()
    for link in list(root.findall("link")):
        name = link.get("name", "")
        if any(p in name for p in [
            "nanoMount", "enclosure", "nano", "Stabilizer",
            "cameraMount", "screwHead", "uncalibrated_camera",
            "camera_link", "camera_depth", "camera_color", "sensor_d415",
        ]):
            camera_link_names.add(name)
            root.remove(link)
    for joint in list(root.findall("joint")):
        child = joint.find("child")
        parent = joint.find("parent")
        child_name = child.get("link", "") if child is not None else ""
        parent_name = parent.get("link", "") if parent is not None else ""
        if child_name in camera_link_names or parent_name in camera_link_names:
            root.remove(joint)
    urdf = ET.tostring(root, encoding="unicode")

    # Pretty-print
    try:
        import xml.dom.minidom

        urdf = xml.dom.minidom.parseString(urdf).toprettyxml(indent="  ")
        if urdf.startswith("<?xml"):
            urdf = urdf.split("\n", 1)[1]
    except Exception:
        pass

    out_dir = Path(__file__).resolve().parents[1] / "src" / "ada_mj" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ada.urdf"
    out.write_text(urdf)
    print(f"Written: {out} ({len(urdf)} chars)")

    # Verify in MuJoCo
    import mujoco
    import os
    import tempfile

    # Create flat meshdir with all mesh sources
    flat_dir = tempfile.mkdtemp(prefix="ada_verify_")
    for subdir in ["meshes", "meshes/forque", "meshes/camera"]:
        for f in (desc / subdir).glob("*.STL"):
            target = os.path.join(flat_dir, f.name)
            if not os.path.exists(target):
                os.symlink(f, target)
        for f in (desc / subdir).glob("*.stl"):
            target = os.path.join(flat_dir, f.stem + ".STL")
            if not os.path.exists(target):
                os.symlink(f, target)

    spec = mujoco.MjSpec.from_file(str(out))
    spec.meshdir = flat_dir
    model = spec.compile()
    print(f"MuJoCo: nq={model.nq}, nv={model.nv}, nbody={model.nbody}, njnt={model.njnt}")

    import xml.etree.ElementTree as ET

    root = ET.fromstring(out.read_text())
    non_fixed = [j for j in root.findall(".//joint") if j.get("type") not in ("fixed", None)]
    print(f"Non-fixed joints: {len(non_fixed)}")
    for j in non_fixed:
        print(f"  {j.get('name'):<40} {j.get('type')}")

    print("\nDone. Commit src/ada_mj/models/ada.urdf to version control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
