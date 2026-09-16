"""Custom reusable asset spawners."""

from __future__ import annotations

from dataclasses import MISSING

from pxr import Usd, UsdGeom

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


@clone
def spawn_visual_with_cylinder_collider(prim_path, cfg, translation=None, orientation=None) -> Usd.Prim:
    root = prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    prim_utils.create_prim(f"{prim_path}/Visual", "Xform", usd_path=cfg.usd_path, scale=cfg.visual_scale)
    cylinder = UsdGeom.Cylinder.Define(stage_utils.get_current_stage(), f"{prim_path}/Collision")
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateRadiusAttr(cfg.collider_radius)
    cylinder.CreateHeightAttr(cfg.collider_height)
    cylinder.AddTranslateOp().Set((0.0, 0.0, cfg.collider_height / 2.0))
    cylinder.MakeInvisible()
    sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    sim_utils.define_mass_properties(prim_path, cfg.mass_props)
    sim_utils.define_collision_properties(f"{prim_path}/Collision", cfg.collision_props)
    return root


@configclass
class VisualCylinderRigidObjectCfg(RigidObjectSpawnerCfg):
    func = spawn_visual_with_cylinder_collider
    usd_path: str = MISSING
    visual_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    collider_radius: float = MISSING
    collider_height: float = MISSING
