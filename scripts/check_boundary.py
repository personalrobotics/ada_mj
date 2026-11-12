"""Fail unless importing ada_mj loads none of the legacy stack.

ada_mj is built on ssrobot and ssgrasping only. The legacy application's mj_manipulator,
mj_environment, and asset_manager, and the old mj-viser distribution, must never load or
be installed. The new viewer, ssmjviser, is also imported as ``mj_viser``, so the old one
is told apart by its distribution name.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import sys

PROHIBITED = ("mj_manipulator", "mj_environment", "asset_manager")


def main() -> int:
    import ada_mj

    for module in pkgutil.walk_packages(ada_mj.__path__, "ada_mj."):
        importlib.import_module(module.name)
    loaded = sorted(n for n in sys.modules if n.split(".")[0] in PROHIBITED)
    distributions = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    old_viewer = "mj-viser" in distributions or "mj_viser" in distributions
    if loaded or old_viewer:
        print(f"FAIL: ada_mj loads {loaded}; old mj-viser installed: {old_viewer}")
        return 1
    print(f"OK: ada_mj from {ada_mj.__file__} loads none of {', '.join(PROHIBITED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
