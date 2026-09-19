"""Neutral task for validating any composed scene without task-specific behavior."""

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..contracts import TaskDefinition


def configure_scene(scene, world, robot, objects) -> None:
    del scene, world, robot, objects


def build_managers(world, robot, objects):
    del world, objects

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
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RewardsCfg:
        alive = RewTerm(func=base_mdp.is_alive, weight=0.0)

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    return ObservationsCfg(), RewardsCfg(), TerminationsCfg()


SCENE_PREVIEW_TASK = TaskDefinition(
    component_id="Task-ScenePreview-v0",
    instruction="Observe the composed scene while the robot remains standing.",
    required_world_capabilities=frozenset({"support_surface", "camera_mount"}),
    required_robot_capabilities=frozenset({"upper_body_joint_control"}),
    required_object_capabilities=frozenset(),
    required_sensor_capabilities=frozenset({"rgb"}),
    configure_scene=configure_scene,
    build_managers=build_managers,
)
