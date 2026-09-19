"""Open-source articulated microwave converted from furniture_sim MJCF."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ..contracts import ObjectSetDefinition


PROJECT = Path(__file__).resolve().parents[3]
MICROWAVE_USD = PROJECT / "assets/furniture_sim/generated/microwave.usd"


def configure_microwave(scene, world) -> None:
    if not MICROWAVE_USD.is_file():
        raise FileNotFoundError(
            f"Missing converted microwave asset: {MICROWAVE_USD}. "
            "Run scripts/prepare_microwave_asset.sh first."
        )
    scene.microwave = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(MICROWAVE_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=world.object_position,
            joint_pos={"micro0joint": 0.0},
            joint_vel={"micro0joint": 0.0},
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["micro0joint"],
                effort_limit_sim=30.0,
                velocity_limit_sim=1.2,
                stiffness=30.0,
                damping=4.0,
            )
        },
    )


MICROWAVE_OBJECTS = ObjectSetDefinition(
    component_id="Objects-Microwave-v0",
    capabilities=frozenset({"articulated_appliance", "openable_door", "container"}),
    entity_roles={"appliance": "microwave", "door_joint": "micro0joint"},
    configure_scene=configure_microwave,
    metadata={
        "source": "https://github.com/vikashplus/furniture_sim",
        "source_commit": "c97995afb81c9e2d7325b0069f9abc9a2c74a2f0",
        "license": "Apache-2.0",
        "manifest": "assets/furniture_sim/manifest.json",
        "asset": "assets/furniture_sim/generated/microwave.usd",
        "door_joint": "micro0joint",
        "door_joint_range_rad": [-2.094, 0.0],
    },
)
