"""Scene-independent object-to-goal task."""

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..contracts import TaskDefinition
from .common import object_pose as object_pose_obs


def object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    goal: RigidObject = env.scene["goal"]
    return torch.linalg.vector_norm(obj.data.root_pos_w - goal.data.root_pos_w, dim=-1)


def negative_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -object_distance(env)


def task_success(env: ManagerBasedRLEnv, support_height: float, threshold: float = 0.08) -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    goal: RigidObject = env.scene["goal"]
    horizontal = torch.linalg.vector_norm(obj.data.root_pos_w[:, :2] - goal.data.root_pos_w[:, :2], dim=-1)
    return torch.logical_and(horizontal < threshold, obj.data.root_pos_w[:, 2] > support_height - 0.03)


def configure_scene(scene, world) -> None:
    return None


def build_managers(world, robot, objects):
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
        object_pose = ObsTerm(func=object_pose_obs, params={"asset_cfg": SceneEntityCfg("object")})
        goal_pose = ObsTerm(func=object_pose_obs, params={"asset_cfg": SceneEntityCfg("goal")})
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RewardsCfg:
        object_distance = RewTerm(func=negative_object_distance, weight=1.0)

    @configclass
    class TerminationsCfg:
        success = DoneTerm(
            func=task_success,
            params={"support_height": world.support_height, "threshold": 0.08},
        )
        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    return ObservationsCfg(), RewardsCfg(), TerminationsCfg()


PICK_PLACE_TASK = TaskDefinition(
    component_id="Task-PickPlace-v0",
    required_world_capabilities=frozenset({"support_surface", "object_spawn_region", "target_region"}),
    required_robot_capabilities=frozenset({"upper_body_joint_control"}),
    required_object_capabilities=frozenset({"manipulation_object", "physical_goal"}),
    required_sensor_capabilities=frozenset({"rgb", "depth"}),
    configure_scene=configure_scene,
    build_managers=build_managers,
)
