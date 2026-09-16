#!/usr/bin/env python3
"""Download and inspect the selected official Isaac Sim dinnerware assets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


launcher = AppLauncher(
    headless=True,
    experience=str(PROJECT / "configs/ycb.python.headless.kit"),
    kit_args=f"--portable-root {PROJECT}/outputs/runtime/kit",
)
app = launcher.app

from pxr import Usd, UsdGeom, UsdPhysics

from isaac_assets import prepare_official_assets


def main() -> None:
    cfg = json.loads((PROJECT / "configs/dinnerware_scene.json").read_text())
    paths = prepare_official_assets(PROJECT, cfg["objects"])
    report = {}
    for spec in cfg["objects"]:
        name = spec["name"]
        stage = Usd.Stage.Open(str(paths[name]))
        if stage is None:
            raise RuntimeError(f"Cannot open {paths[name]}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        meshes = []
        rigid_bodies = []
        colliders = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                meshes.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(str(prim.GetPath()))
        report[name] = {
            "path": str(paths[name]),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": UsdGeom.GetStageUpAxis(stage),
            "bounds_min": list(bounds.GetMin()),
            "bounds_max": list(bounds.GetMax()),
            "mesh_prims": meshes,
            "rigid_body_prims": rigid_bodies,
            "collision_prims": colliders,
        }
    output = PROJECT / "outputs/dinnerware-asset-inspection.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
