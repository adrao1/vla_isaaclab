"""Manager-based G1 dinnerware task configuration."""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG

from . import mdp
from .actions import G1UpperBodyActionCfg, UPPER_BODY_JOINT_NAMES
from .spawners import DinnerwareSpawnerCfg


PROJECT = Path(__file__).resolve().parents[2]
SCENE_SPEC = json.loads((PROJECT / "configs" / "dinnerware_scene.json").read_text())
ASSET_ROOT = PROJECT / "assets" / "Isaac" / "4.5"
G1_USD = ASSET_ROOT / SCENE_SPEC["robot"]["asset"]
BOWL_USD = ASSET_ROOT / SCENE_SPEC["objects"][0]["asset"]
PLATE_USD = ASSET_ROOT / SCENE_SPEC["objects"][1]["asset"]

for _asset_path in (G1_USD, BOWL_USD, PLATE_USD):
    if not _asset_path.is_file():
        raise FileNotFoundError(
            f"Required cached asset is missing: {_asset_path}. Run the existing asset preparation/validation script first."
        )


def static_box(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.92, 0.92, 0.92), roughness=0.75, metallic=0.0
        ),
    )


G1_FIXED_CFG = G1_CFG.copy()
G1_FIXED_CFG.prim_path = "{ENV_REGEX_NS}/Robot"
G1_FIXED_CFG.spawn.usd_path = str(G1_USD)
G1_FIXED_CFG.spawn.articulation_props.fix_root_link = True
G1_FIXED_CFG.init_state.pos = tuple(SCENE_SPEC["robot"]["position"])
G1_FIXED_CFG.init_state.rot = tuple(SCENE_SPEC["robot"]["orientation_wxyz"])


@configclass
class G1DinnerwareSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.25)),
        collision_group=-1,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1100.0, color=(0.85, 0.88, 1.0)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88), angle=0.8),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.3827, 0.0, 0.0)),
    )

    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        spawn=static_box((1.2, 0.8, 0.05)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.725)),
    )
    table_leg_0 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg_0", spawn=static_box((0.055, 0.055, 0.70)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.52, -0.32, 0.35)),
    )
    table_leg_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg_1", spawn=static_box((0.055, 0.055, 0.70)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.52, 0.32, 0.35)),
    )
    table_leg_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg_2", spawn=static_box((0.055, 0.055, 0.70)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.52, -0.32, 0.35)),
    )
    table_leg_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg_3", spawn=static_box((0.055, 0.055, 0.70)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.52, 0.32, 0.35)),
    )

    robot = G1_FIXED_CFG
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        spawn=DinnerwareSpawnerCfg(
            usd_path=str(BOWL_USD), collider_radius=0.112, collider_height=0.030,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.05, angular_damping=0.05,
                solver_position_iteration_count=8,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.18, 0.04, 0.83)),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=DinnerwareSpawnerCfg(
            usd_path=str(PLATE_USD), collider_radius=0.160, collider_height=0.012,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.05, angular_damping=0.05,
                solver_position_iteration_count=8,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.45),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.02, 0.83)),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ObservationCamera",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0, focus_distance=2.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
    )


@configclass
class ActionsCfg:
    upper_body = G1UpperBodyActionCfg(
        asset_name="robot", joint_names=UPPER_BODY_JOINT_NAMES, scale=0.25
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES, preserve_order=True)},
        )
        bowl_pose = ObsTerm(func=mdp.object_pose, params={"asset_cfg": SceneEntityCfg("bowl")})
        plate_pose = ObsTerm(func=mdp.object_pose, params={"asset_cfg": SceneEntityCfg("plate")})
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    object_distance = RewTerm(func=mdp.negative_object_distance, weight=1.0)


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.task_success, params={"distance_threshold": 0.08})
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class G1DinnerwareEnvCfg(ManagerBasedRLEnvCfg):
    env_name: str = "G1DinnerwarePickPlace-v0"
    scene: G1DinnerwareSceneCfg = G1DinnerwareSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    episode_length_s: float = 10.0
    decimation: int = 4

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.rerender_on_reset = True
        self.wait_for_textures = True
        self.viewer.eye = tuple(SCENE_SPEC["camera_eye"])
        self.viewer.lookat = tuple(SCENE_SPEC["camera_target"])
