"""Sugar-box success and failure termination terms."""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from ...common import (
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_OPEN_JOINT_POSITIONS,
)


def task_metrics(env: ManagerBasedRLEnv, palm_body_name: str, command_name: str) -> dict[str, torch.Tensor]:
    """Metrics for the original pick-and-place task."""
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]

    palm_ids, _ = robot.find_bodies([palm_body_name], preserve_order=True)

    target = env.command_manager.get_command(command_name)[:, :3] + env.scene.env_origins

    return {
        "xy_error": torch.linalg.vector_norm(
            sugar_box.data.root_pos_w[:, :2] - target[:, :2],
            dim=-1,
        ),
        "height_error": torch.abs(
            sugar_box.data.root_pos_w[:, 2] - target[:, 2]
        ),
        "linear_speed": torch.linalg.vector_norm(
            sugar_box.data.root_lin_vel_w,
            dim=-1,
        ),
        "angular_speed": torch.linalg.vector_norm(
            sugar_box.data.root_ang_vel_w,
            dim=-1,
        ),
        "hand_distance": torch.linalg.vector_norm(
            robot.data.body_pos_w[:, palm_ids[0]]
            - sugar_box.data.root_pos_w,
            dim=-1,
        ),
    }


def grasp_metrics(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
) -> dict[str, torch.Tensor]:
    """Metrics used by the grasp-and-lift RL task."""
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]

    # Cache IDs because these reward/termination functions run every step.
    palm_id = getattr(env, "_grasp_reward_palm_id", None)
    if palm_id is None:
        palm_ids, found = robot.find_bodies(
            [palm_body_name],
            preserve_order=True,
        )
        if list(found) != [palm_body_name]:
            raise RuntimeError(
                f"Could not resolve palm body {palm_body_name!r}: {found}"
            )
        palm_id = palm_ids[0]
        env._grasp_reward_palm_id = palm_id

    hand_joint_ids = getattr(env, "_grasp_reward_hand_joint_ids", None)
    if hand_joint_ids is None:
        hand_joint_ids, found = robot.find_joints(
            list(LEFT_HAND_JOINT_NAMES),
            preserve_order=True,
        )
        if list(found) != list(LEFT_HAND_JOINT_NAMES):
            raise RuntimeError(
                "Left-hand joint mismatch: "
                f"expected {list(LEFT_HAND_JOINT_NAMES)}, found {found}"
            )
        env._grasp_reward_hand_joint_ids = hand_joint_ids

    palm_pos = robot.data.body_pos_w[:, palm_id]
    box_pos = sugar_box.data.root_pos_w

    hand_distance = torch.linalg.vector_norm(
        palm_pos - box_pos,
        dim=-1,
    )

    hand_pos = robot.data.joint_pos[:, hand_joint_ids]

    open_pos = torch.tensor(
        LEFT_HAND_OPEN_JOINT_POSITIONS,
        device=env.device,
        dtype=hand_pos.dtype,
    ).unsqueeze(0)

    closed_pos = torch.tensor(
        LEFT_HAND_CLOSED_JOINT_POSITIONS,
        device=env.device,
        dtype=hand_pos.dtype,
    ).unsqueeze(0)

    closure_per_joint = (
        (hand_pos - open_pos)
        / (closed_pos - open_pos).clamp_min(1.0e-6)
    )

    closure_fraction = torch.clamp(
        closure_per_joint,
        0.0,
        1.0,
    ).mean(dim=-1)

    lift_height = sugar_box.data.root_pos_w[:, 2] - initial_box_height

    return {
        "hand_distance": hand_distance,
        "closure_fraction": closure_fraction,
        "lift_height": lift_height,
        "linear_speed": torch.linalg.vector_norm(
            sugar_box.data.root_lin_vel_w,
            dim=-1,
        ),
    }


def task_success(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    command_name: str,
    hold_steps: int = 15,
) -> torch.Tensor:
    """Original pick-and-place success condition."""
    metrics = task_metrics(env, palm_body_name, command_name)

    instantaneous = (
        (metrics["xy_error"] < 0.015)
        & (metrics["height_error"] < 0.015)
        & (metrics["linear_speed"] < 0.04)
        & (metrics["angular_speed"] < 0.30)
        & (metrics["hand_distance"] > 0.20)
    )

    counter = getattr(env, "task_success_counter", None)

    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        env.task_success_counter = counter

    counter[:] = torch.where(
        instantaneous,
        counter + 1,
        torch.zeros_like(counter),
    )

    return counter >= hold_steps


def grasp_success(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
    lift_threshold: float = 0.03,
    closure_threshold: float = 0.60,
    max_hand_distance: float = 0.22,
    hold_steps: int = 10,
) -> torch.Tensor:
    """Success when the Dex3 hand has grasped and lifted the sugar box."""
    metrics = grasp_metrics(
        env,
        palm_body_name,
        initial_box_height,
    )

    instantaneous = (
        (metrics["hand_distance"] < max_hand_distance)
        & (metrics["closure_fraction"] > closure_threshold)
        & (metrics["lift_height"] > lift_threshold)
    )

    counter = getattr(env, "grasp_success_counter", None)

    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        env.grasp_success_counter = counter

    counter[:] = torch.where(
        instantaneous,
        counter + 1,
        torch.zeros_like(counter),
    )

    return counter >= hold_steps


def object_fallen(
    env: ManagerBasedRLEnv,
    support_height: float,
) -> torch.Tensor:
    sugar_box: RigidObject = env.scene["object"]
    return sugar_box.data.root_pos_w[:, 2] < support_height - 0.05


def invalid_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]

    return ~(
        torch.isfinite(robot.data.joint_pos).all(dim=-1)
        & torch.isfinite(sugar_box.data.root_state_w).all(dim=-1)
    )
