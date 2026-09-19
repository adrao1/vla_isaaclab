"""Deterministic YCB sugar-box pick-and-place task."""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..contracts import TaskDefinition


# Unitree uses +X forward, +Y left.  With the configured +90 degree world-yaw,
# robot-right (-base Y) is world +X.
ROBOT_RIGHT_WORLD = (1.0, 0.0, 0.0)
SUGAR_BOX_HALF_HEIGHT_M = 0.088
# The YCB asset already needs a +90 degree X rotation to stand upright. Apply
# an additional +90 degree world-Z yaw so its narrow/wide tabletop axes swap.
SUGAR_BOX_ROTATED_90_WXYZ = (0.5, 0.5, 0.5, 0.5)
# Close to the left hand, while retaining clearance from the robot-side edge.
INITIAL_XY = (-0.18, -0.25)
TARGET_DISPLACEMENT_M = 0.02


def left_ee_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    return torch.cat(
        (robot.data.body_pos_w[:, body_id] - env.scene.env_origins, robot.data.body_quat_w[:, body_id]), dim=-1
    )


def object_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[asset_cfg.name]
    return torch.cat(
        (
            obj.data.root_pos_w - env.scene.env_origins,
            obj.data.root_quat_w,
            obj.data.root_lin_vel_w,
            obj.data.root_ang_vel_w,
        ),
        dim=-1,
    )


def target_pose(env: ManagerBasedEnv, support_height: float) -> torch.Tensor:
    pose = torch.tensor(
        [
            INITIAL_XY[0] + TARGET_DISPLACEMENT_M,
            INITIAL_XY[1],
            support_height + SUGAR_BOX_HALF_HEIGHT_M,
            *SUGAR_BOX_ROTATED_90_WXYZ,
        ],
        device=env.device,
    )
    return pose.unsqueeze(0).repeat(env.num_envs, 1)


def task_metrics(env: ManagerBasedRLEnv, palm_body_name: str, support_height: float) -> dict[str, torch.Tensor]:
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]
    palm_ids, _ = robot.find_bodies([palm_body_name], preserve_order=True)
    target = target_pose(env, support_height)[:, :3] + env.scene.env_origins
    return {
        "xy_error": torch.linalg.vector_norm(sugar_box.data.root_pos_w[:, :2] - target[:, :2], dim=-1),
        "height_error": torch.abs(sugar_box.data.root_pos_w[:, 2] - target[:, 2]),
        "linear_speed": torch.linalg.vector_norm(sugar_box.data.root_lin_vel_w, dim=-1),
        "angular_speed": torch.linalg.vector_norm(sugar_box.data.root_ang_vel_w, dim=-1),
        "hand_distance": torch.linalg.vector_norm(
            robot.data.body_pos_w[:, palm_ids[0]] - sugar_box.data.root_pos_w, dim=-1
        ),
    }


def placement_reward(env: ManagerBasedRLEnv, palm_body_name: str, support_height: float) -> torch.Tensor:
    metrics = task_metrics(env, palm_body_name, support_height)
    return 1.0 - torch.tanh(25.0 * metrics["xy_error"] + 10.0 * metrics["height_error"])


def task_success(
    env: ManagerBasedRLEnv, palm_body_name: str, support_height: float, hold_steps: int = 15
) -> torch.Tensor:
    metrics = task_metrics(env, palm_body_name, support_height)
    instantaneous = (
        (metrics["xy_error"] < 0.008)
        & (metrics["height_error"] < 0.015)
        & (metrics["linear_speed"] < 0.04)
        & (metrics["angular_speed"] < 0.30)
        & (metrics["hand_distance"] > 0.14)
    )
    counter = getattr(env, "task_success_counter", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.task_success_counter = counter
    counter[:] = torch.where(instantaneous, counter + 1, torch.zeros_like(counter))
    return counter >= hold_steps


def object_fallen(env: ManagerBasedRLEnv, support_height: float) -> torch.Tensor:
    sugar_box: RigidObject = env.scene["object"]
    return sugar_box.data.root_pos_w[:, 2] < support_height - 0.05


def invalid_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]
    return ~(
        torch.isfinite(robot.data.joint_pos).all(dim=-1)
        & torch.isfinite(sugar_box.data.root_state_w).all(dim=-1)
    )


def configure_scene(scene, world, robot, objects) -> None:
    del robot
    if objects.entity_roles.get("sugar_box") != "object":
        raise ValueError("Task-YCBPickPlaceSugarBox-v0 requires entity role sugar_box='object'")
    scene.object.init_state.pos = (
        INITIAL_XY[0],
        INITIAL_XY[1],
        world.support_height + SUGAR_BOX_HALF_HEIGHT_M,
    )
    scene.object.init_state.rot = SUGAR_BOX_ROTATED_90_WXYZ
    scene.target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SugarBoxTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.095, 0.050, 0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.8, 0.2), opacity=0.35
            ),
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(INITIAL_XY[0] + TARGET_DISPLACEMENT_M, INITIAL_XY[1], world.support_height + 0.002),
            rot=(0.70710678, 0.0, 0.0, 0.70710678),
        ),
    )


def build_managers(world, robot, objects):
    del objects
    palm_cfg = SceneEntityCfg("robot", body_names=[robot.left_end_effector], preserve_order=True)
    joints_cfg = SceneEntityCfg(
        "robot", joint_names=list(robot.action_joint_names), preserve_order=True
    )

    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": joints_cfg})
        joint_velocity = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": joints_cfg})
        left_end_effector_pose = ObsTerm(func=left_ee_pose, params={"asset_cfg": palm_cfg})
        sugar_box_state = ObsTerm(func=object_state, params={"asset_cfg": SceneEntityCfg("object")})
        target_pose = ObsTerm(func=target_pose, params={"support_height": world.support_height})
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RewardsCfg:
        placement = RewTerm(
            func=placement_reward,
            weight=2.0,
            params={"palm_body_name": robot.left_end_effector, "support_height": world.support_height},
        )

    @configclass
    class TerminationsCfg:
        success = DoneTerm(
            func=task_success,
            params={"palm_body_name": robot.left_end_effector, "support_height": world.support_height},
        )
        object_fallen = DoneTerm(func=object_fallen, params={"support_height": world.support_height})
        invalid_state = DoneTerm(func=invalid_state)
        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    return ObservationsCfg(), RewardsCfg(), TerminationsCfg()


YCB_PICK_PLACE_SUGAR_BOX_TASK = TaskDefinition(
    component_id="Task-YCBPickPlaceSugarBox-v0",
    instruction=(
        "Approach and grasp the rotated YCB 004 sugar box from the robot-facing side, "
        "move it 2 cm toward robot-right "
        "(world +X), and place it back on the table."
    ),
    required_world_capabilities=frozenset({"support_surface", "object_spawn_region", "target_region"}),
    required_robot_capabilities=frozenset({"fixed_base", "upper_body_joint_control", "left_end_effector"}),
    required_object_capabilities=frozenset({"manipulation_object", "sugar_box"}),
    required_sensor_capabilities=frozenset({"rgb", "depth"}),
    configure_scene=configure_scene,
    build_managers=build_managers,
)
