"""Physical left-hand bowl-to-plate task definition."""

from __future__ import annotations

import math

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from ..contracts import TaskDefinition


INSTRUCTION = "Use only the left hand to pick up the bowl and place it securely in the center of the plate."
PHASE_COUNT = 10


def body_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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


def relative_pose(env: ManagerBasedEnv, source_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return target pose expressed in source coordinates."""
    if source_cfg.name == "robot":
        source: Articulation = env.scene[source_cfg.name]
        source_pos = source.data.body_pos_w[:, source_cfg.body_ids[0]]
        source_quat = source.data.body_quat_w[:, source_cfg.body_ids[0]]
    else:
        source: RigidObject = env.scene[source_cfg.name]
        source_pos, source_quat = source.data.root_pos_w, source.data.root_quat_w
    target: RigidObject = env.scene[target_cfg.name]
    rel_pos, rel_quat = subtract_frame_transforms(source_pos, source_quat, target.data.root_pos_w, target.data.root_quat_w)
    return torch.cat((rel_pos, rel_quat), dim=-1)


def controller_phase(env: ManagerBasedEnv) -> torch.Tensor:
    phase = getattr(env, "bowl_to_plate_phase", None)
    if phase is None:
        phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return torch.nn.functional.one_hot(phase.clamp(0, PHASE_COUNT - 1), PHASE_COUNT).float()


def placement_metrics(env: ManagerBasedRLEnv, palm_body_name: str, support_height: float) -> dict[str, torch.Tensor]:
    robot: Articulation = env.scene["robot"]
    bowl: RigidObject = env.scene["object"]
    plate: RigidObject = env.scene["goal"]
    palm_ids, _ = robot.find_bodies([palm_body_name], preserve_order=True)
    palm_pos = robot.data.body_pos_w[:, palm_ids[0]]

    delta = bowl.data.root_pos_w - plate.data.root_pos_w
    world_up = torch.zeros_like(delta)
    world_up[:, 2] = 1.0
    bowl_up = quat_apply(bowl.data.root_quat_w, world_up)
    return {
        "horizontal_distance": torch.linalg.vector_norm(delta[:, :2], dim=-1),
        "height_above_plate": delta[:, 2],
        "upright_cosine": bowl_up[:, 2],
        "linear_speed": torch.linalg.vector_norm(bowl.data.root_lin_vel_w, dim=-1),
        "angular_speed": torch.linalg.vector_norm(bowl.data.root_ang_vel_w, dim=-1),
        "hand_distance": torch.linalg.vector_norm(palm_pos - bowl.data.root_pos_w, dim=-1),
        "above_table": bowl.data.root_pos_w[:, 2] >= support_height - 0.004,
    }


def approach_reward(env: ManagerBasedRLEnv, palm_body_name: str) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    bowl: RigidObject = env.scene["object"]
    palm_ids, _ = robot.find_bodies([palm_body_name], preserve_order=True)
    distance = torch.linalg.vector_norm(robot.data.body_pos_w[:, palm_ids[0]] - bowl.data.root_pos_w, dim=-1)
    return 1.0 - torch.tanh(5.0 * distance)


def lift_reward(env: ManagerBasedRLEnv, support_height: float) -> torch.Tensor:
    bowl: RigidObject = env.scene["object"]
    return torch.clamp((bowl.data.root_pos_w[:, 2] - support_height) / 0.10, 0.0, 1.0)


def placement_reward(env: ManagerBasedRLEnv, palm_body_name: str, support_height: float) -> torch.Tensor:
    metrics = placement_metrics(env, palm_body_name, support_height)
    centered = 1.0 - torch.tanh(12.0 * metrics["horizontal_distance"])
    upright = torch.clamp(metrics["upright_cosine"], 0.0, 1.0)
    return centered * upright


def task_success(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    support_height: float,
    hold_steps: int = 15,
) -> torch.Tensor:
    metrics = placement_metrics(env, palm_body_name, support_height)
    instantaneous = (
        (metrics["horizontal_distance"] < 0.04)
        & (metrics["height_above_plate"] > 0.008)
        & (metrics["height_above_plate"] < 0.075)
        & (metrics["upright_cosine"] > math.cos(math.radians(15.0)))
        & (metrics["linear_speed"] < 0.035)
        & (metrics["angular_speed"] < 0.25)
        & (metrics["hand_distance"] > 0.13)
        & metrics["above_table"]
    )
    counter = getattr(env, "bowl_to_plate_success_counter", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.bowl_to_plate_success_counter = counter
    counter[:] = torch.where(instantaneous, counter + 1, torch.zeros_like(counter))
    return counter >= hold_steps


def configure_scene(scene, world, robot, objects) -> None:
    del world, objects
    for index, body_name in enumerate(robot.left_fingertip_body_names):
        setattr(
            scene,
            f"left_fingertip_contact_{index}",
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                update_period=0.0,
                history_length=3,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            ),
        )


def build_managers(world, robot, objects):
    del objects
    palm_cfg = SceneEntityCfg("robot", body_names=[robot.left_end_effector], preserve_order=True)
    bowl_cfg = SceneEntityCfg("object")
    plate_cfg = SceneEntityCfg("goal")
    left_fingers_cfg = SceneEntityCfg(
        "robot", joint_names=list(robot.left_hand_joint_names), preserve_order=True
    )

    @configclass
    class PolicyCfg(ObsGroup):
        left_hand_pose = ObsTerm(func=body_pose, params={"asset_cfg": palm_cfg})
        left_finger_position = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": left_fingers_cfg})
        bowl_state = ObsTerm(func=object_state, params={"asset_cfg": bowl_cfg})
        plate_pose = ObsTerm(
            func=lambda env, asset_cfg: object_state(env, asset_cfg)[:, :7], params={"asset_cfg": plate_cfg}
        )
        bowl_relative_to_hand = ObsTerm(
            func=relative_pose, params={"source_cfg": palm_cfg, "target_cfg": bowl_cfg}
        )
        bowl_relative_to_plate = ObsTerm(
            func=relative_pose, params={"source_cfg": plate_cfg, "target_cfg": bowl_cfg}
        )
        phase = ObsTerm(func=controller_phase)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RewardsCfg:
        approach = RewTerm(func=approach_reward, weight=0.2, params={"palm_body_name": robot.left_end_effector})
        lift = RewTerm(func=lift_reward, weight=1.0, params={"support_height": world.support_height})
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
        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    return ObservationsCfg(), RewardsCfg(), TerminationsCfg()


BOWL_TO_PLATE_TASK = TaskDefinition(
    component_id="Task-BowlToPlate-v0",
    instruction=INSTRUCTION,
    required_world_capabilities=frozenset({"support_surface", "object_spawn_region", "target_region"}),
    required_robot_capabilities=frozenset({"fixed_base", "upper_body_joint_control", "left_end_effector"}),
    required_object_capabilities=frozenset({"manipulation_object", "physical_goal"}),
    required_sensor_capabilities=frozenset({"rgb", "depth"}),
    configure_scene=configure_scene,
    build_managers=build_managers,
)
