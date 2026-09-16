"""Official Isaac Sim dinnerware object set."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from ..contracts import ObjectSetDefinition
from ..spawners import VisualCylinderRigidObjectCfg


PROJECT = Path(__file__).resolve().parents[3]
ASSET_ROOT = PROJECT / "assets/Isaac/4.5/NVIDIA/Assets/ArchVis/Residential/Kitchen/Kitchenware/Dinnerware"


def dinnerware_cfg(path, radius, height, mass, position):
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached dinnerware asset: {path}")
    return RigidObjectCfg(
        prim_path="",
        spawn=VisualCylinderRigidObjectCfg(
            usd_path=str(path), collider_radius=radius, collider_height=height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.05, angular_damping=0.05,
                solver_position_iteration_count=8,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def configure_dinnerware(scene, world) -> None:
    scene.object = dinnerware_cfg(ASSET_ROOT / "bowl_plate.usd", 0.112, 0.030, 0.35, world.object_position)
    scene.object.prim_path = "{ENV_REGEX_NS}/Object"
    scene.goal = dinnerware_cfg(ASSET_ROOT / "plate_large.usd", 0.160, 0.012, 0.45, world.goal_position)
    scene.goal.prim_path = "{ENV_REGEX_NS}/Goal"


DINNERWARE_OBJECTS = ObjectSetDefinition(
    component_id="Objects-Dinnerware-v0",
    capabilities=frozenset({"manipulation_object", "physical_goal"}),
    entity_roles={"manipulation_object": "object", "goal": "goal"},
    configure_scene=configure_dinnerware,
)
