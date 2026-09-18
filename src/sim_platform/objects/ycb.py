"""Selected local official YCB assets."""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from ..contracts import ObjectSetDefinition


PROJECT = Path(__file__).resolve().parents[3]
YCB_ROOT = PROJECT / "assets/YCB/Axis_Aligned_Physics"


def ycb_cfg(asset, position, half_height, rotation=None):
    path = YCB_ROOT / f"{asset}.usd"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached YCB asset: {path}")
    return RigidObjectCfg(
        prim_path="",
        spawn=sim_utils.UsdFileCfg(usd_path=str(path)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(position[0], position[1], position[2] + half_height),
            rot=rotation or (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
        ),
    )


def configure_ycb(scene, world) -> None:
    scene.object = ycb_cfg("003_cracker_box", world.object_position, 0.107)
    scene.object.prim_path = "{ENV_REGEX_NS}/Object"
    scene.goal = ycb_cfg("005_tomato_soup_can", world.goal_position, 0.051)
    scene.goal.prim_path = "{ENV_REGEX_NS}/Goal"
    scene.distractor_sugar = ycb_cfg("004_sugar_box", (-0.02, 0.18, world.support_height + 0.08), 0.088)
    scene.distractor_sugar.prim_path = "{ENV_REGEX_NS}/DistractorSugar"
    # This asset's published local vertical axis is opposite to the boxes and
    # can, so use the inverse X-axis conversion to keep the cap pointing up.
    scene.distractor_mustard = ycb_cfg(
        "006_mustard_bottle",
        (0.02, -0.12, world.support_height + 0.08),
        0.095,
        rotation=(math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
    )
    scene.distractor_mustard.prim_path = "{ENV_REGEX_NS}/DistractorMustard"


YCB_BASIC_OBJECTS = ObjectSetDefinition(
    component_id="Objects-YCB-Basic-v0",
    capabilities=frozenset({"manipulation_object", "physical_goal", "distractors"}),
    entity_roles={"manipulation_object": "object", "goal": "goal"},
    configure_scene=configure_ycb,
)
