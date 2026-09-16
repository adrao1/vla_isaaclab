"""Inspect official asset physics using this environment's bundled USD."""
import json
import os
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))
from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True, experience=str(project / "configs/ycb.python.headless.kit"),
                       kit_args=f"--portable-root {project}/outputs/runtime/kit")
app = launcher.app
try:
    from pxr import Usd, UsdGeom, UsdPhysics
    from ycb_assets import prepare_assets

    cfg = json.loads((project / "configs/scene.json").read_text())
    assets = prepare_assets(project, cfg["objects"])
    report = {}
    for name, path in assets.items():
        stage = Usd.Stage.Open(str(path))
        root = stage.GetDefaultPrim()
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        box = bounds.ComputeWorldBound(root).ComputeAlignedRange()
        mass_properties = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                mass_properties.append({
                    "prim": str(prim.GetPath()),
                    "mass_kg": mass_api.GetMassAttr().Get(),
                    "density_kg_m3": mass_api.GetDensityAttr().Get(),
                    "diagonal_inertia": list(mass_api.GetDiagonalInertiaAttr().Get() or []),
                    "center_of_mass": list(mass_api.GetCenterOfMassAttr().Get() or []),
                })
        report[name] = {
            "default_prim": str(root.GetPath()),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "min": list(box.GetMin()), "max": list(box.GetMax()),
            "rigid_bodies": [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)],
            "colliders": [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)],
            "mass_properties": mass_properties,
        }
    (project / "outputs/asset-inspection.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
finally:
    app.close()
