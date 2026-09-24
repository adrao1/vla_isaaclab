"""Sugar-box reward terms."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .terminations import grasp_metrics, task_metrics


def placement_reward(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    command_name: str,
) -> torch.Tensor:
    """Original pick-and-place reward."""
    metrics = task_metrics(env, palm_body_name, command_name)

    return 1.0 - torch.tanh(
        25.0 * metrics["xy_error"]
        + 10.0 * metrics["height_error"]
    )


def grasp_reaching_reward(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
) -> torch.Tensor:
    """X-Sim-style dense TCP/palm-to-object reaching reward."""
    metrics = grasp_metrics(
        env,
        palm_body_name,
        initial_box_height,
    )

    distance = metrics["hand_distance"]

    # Mirrors the two-scale reaching structure used by X-Sim:
    # one broad attraction term and one sharper near-object term.
    reward = 1.0 - torch.tanh(3.0 * distance)
    reward += 1.0 - torch.tanh(30.0 * distance)

    return reward / 2.0


def grasp_closure_reward(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
) -> torch.Tensor:
    """Reward finger closure, but primarily when the hand is near the box."""
    metrics = grasp_metrics(
        env,
        palm_body_name,
        initial_box_height,
    )

    distance = metrics["hand_distance"]
    closure = metrics["closure_fraction"]

    proximity = 1.0 - torch.tanh(5.0 * distance)

    return proximity * closure


def grasp_lift_reward(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
    lift_target: float = 0.03,
) -> torch.Tensor:
    """Reward actual upward movement of the sugar box."""
    metrics = grasp_metrics(
        env,
        palm_body_name,
        initial_box_height,
    )

    lift_progress = torch.clamp(
        metrics["lift_height"] / lift_target,
        min=0.0,
        max=1.0,
    )

    # Requiring some closure makes accidental object motion less valuable.
    return lift_progress * metrics["closure_fraction"]


def grasp_success_reward(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
    lift_threshold: float = 0.03,
    closure_threshold: float = 0.60,
    max_hand_distance: float = 0.22,
) -> torch.Tensor:
    """One-step dense bonus when the physical grasp condition is satisfied."""
    metrics = grasp_metrics(
        env,
        palm_body_name,
        initial_box_height,
    )

    return (
        (metrics["hand_distance"] < max_hand_distance)
        & (metrics["closure_fraction"] > closure_threshold)
        & (metrics["lift_height"] > lift_threshold)
    ).float()
