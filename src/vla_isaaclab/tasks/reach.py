"""Left-palm reach task that can run in any world with a workspace target."""

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..contracts import TaskDefinition


def end_effector_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, asset_cfg.body_ids[0]] - env.scene.env_origins


def target_position(env: ManagerBasedEnv, target: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(target, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)


def reach_distance(env: ManagerBasedRLEnv, body_name: str, target: tuple[float, float, float]) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    body_ids, _ = robot.find_bodies([body_name], preserve_order=True)
    palm = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    target_tensor = torch.tensor(target, device=env.device).unsqueeze(0)
    return torch.linalg.vector_norm(palm - target_tensor, dim=-1)


def negative_reach_distance(env: ManagerBasedRLEnv, body_name: str, target) -> torch.Tensor:
    return -reach_distance(env, body_name, target)


def reach_success(env: ManagerBasedRLEnv, body_name: str, target, threshold: float) -> torch.Tensor:
    return reach_distance(env, body_name, target) < threshold


def configure_scene(scene, world, robot, objects) -> None:
    del robot, objects
    scene.reach_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReachTarget",
        spawn=sim_utils.SphereCfg(
            radius=0.045,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.2), emissive_color=(0.02, 0.25, 0.04)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=world.reach_target),
    )


def build_managers(world, robot, objects):
    palm_cfg = SceneEntityCfg("robot", body_names=[robot.left_end_effector], preserve_order=True)

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(robot.action_joint_names), preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(robot.action_joint_names), preserve_order=True)},
        )
        end_effector = ObsTerm(func=end_effector_position, params={"asset_cfg": palm_cfg})
        target = ObsTerm(func=target_position, params={"target": world.reach_target})
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RewardsCfg:
        reach_distance = RewTerm(
            func=negative_reach_distance,
            weight=1.0,
            params={"body_name": robot.left_end_effector, "target": world.reach_target},
        )

    @configclass
    class TerminationsCfg:
        success = DoneTerm(
            func=reach_success,
            params={"body_name": robot.left_end_effector, "target": world.reach_target, "threshold": 0.07},
        )
        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    return ObservationsCfg(), RewardsCfg(), TerminationsCfg()


REACH_TASK = TaskDefinition(
    component_id="Task-Reach-v0",
    instruction="Move the left hand to the green target.",
    required_world_capabilities=frozenset({"workspace", "target_region", "camera_mount"}),
    required_robot_capabilities=frozenset({"upper_body_joint_control", "left_end_effector"}),
    required_object_capabilities=frozenset(),
    required_sensor_capabilities=frozenset({"rgb"}),
    configure_scene=configure_scene,
    build_managers=build_managers,
)
