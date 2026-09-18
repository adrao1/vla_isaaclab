#!/usr/bin/env python3
"""Inspect the converted official YCB bowl and plate physics assets."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

from isaaclab.app import AppLauncher


launcher = AppLauncher(
    headless=True,
    experience=str(PROJECT / "configs/ycb.python.headless.kit"),
    kit_args=f"--portable-root {PROJECT}/outputs/runtime/kit",
)
app = launcher.app

from pxr import Usd, UsdGeom, UsdPhysics

def main() -> None:
    cfg = json.loads((PROJECT / "configs/dinnerware_scene.json").read_text())
    report = {}
    for spec in cfg["objects"]:
        name = spec["name"]
        path = PROJECT / spec["asset"]
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError(f"Cannot open {path}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        meshes = []
        rigid_bodies = []
        colliders = []
        masses = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                meshes.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                approximation = None
                if prim.IsA(UsdGeom.Mesh):
                    approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                colliders.append({"path": str(prim.GetPath()), "approximation": approximation})
            if prim.HasAPI(UsdPhysics.MassAPI):
                masses.append(
                    {"path": str(prim.GetPath()), "mass_kg": UsdPhysics.MassAPI(prim).GetMassAttr().Get()}
                )
        report[name] = {
            "path": str(path),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": UsdGeom.GetStageUpAxis(stage),
            "bounds_min": list(bounds.GetMin()),
            "bounds_max": list(bounds.GetMax()),
            "mesh_prims": meshes,
            "rigid_body_prims": rigid_bodies,
            "collision_prims": colliders,
            "mass_prims": masses,
            "published_dimensions_m": spec["published_dimensions_m"],
            "published_mass_kg": spec["mass"],
        }
    output = PROJECT / "outputs/dinnerware-asset-inspection.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
