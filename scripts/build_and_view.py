#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Build the ADA model and view it in mj_viser.

Usage::

    uv run python scripts/build_and_view.py
"""

from ada_mj.setup import build_ada_model


def main():
    print("Building ADA model...", flush=True)
    model, data = build_ada_model()
    print(f"Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}", flush=True)

    from mj_viser import MujocoViewer

    viewer = MujocoViewer(model, data, label="ADA")
    print("Launching viewer at http://localhost:8080", flush=True)
    viewer.launch()


if __name__ == "__main__":
    main()
