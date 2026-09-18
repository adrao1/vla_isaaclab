"""Official YCB bowl and plate with converted physics USD assets."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from ..contracts import ObjectSetDefinition


PROJECT = Path(__file__).resolve().parents[3]
ASSET_ROOT = PROJECT / "assets/YCB/dinnerware"


def dinnerware_cfg(path, position):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing converted YCB dinnerware asset: {path}. "
            "Run scripts/prepare_ycb_dinnerware.sh first."
        )
    return RigidObjectCfg(
        prim_path="",
        spawn=sim_utils.UsdFileCfg(usd_path=str(path)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def configure_dinnerware(scene, world) -> None:
    scene.object = dinnerware_cfg(ASSET_ROOT / "024_bowl/024_bowl_physics.usd", world.object_position)
    scene.object.prim_path = "{ENV_REGEX_NS}/Object"
    scene.goal = dinnerware_cfg(ASSET_ROOT / "029_plate/029_plate_physics.usd", world.goal_position)
    scene.goal.prim_path = "{ENV_REGEX_NS}/Goal"


DINNERWARE_OBJECTS = ObjectSetDefinition(
    component_id="Objects-Dinnerware-v0",
    capabilities=frozenset({"manipulation_object", "physical_goal"}),
    entity_roles={"manipulation_object": "object", "goal": "goal"},
    configure_scene=configure_dinnerware,
    metadata={
        "source": "YCB Object and Model Set",
        "license": "CC BY 4.0",
        "manifest": "assets/YCB/dinnerware/manifest.json",
        "collision_approximation": "convexDecomposition",
        "assets": {
            "object": {
                "name": "024_bowl",
                "mass_kg": 0.147,
                "published_dimensions_m": [0.159, 0.159, 0.053],
            },
            "goal": {
                "name": "029_plate",
                "mass_kg": 0.279,
                "published_dimensions_m": [0.258, 0.258, 0.024],
            },
        },
    },
)
