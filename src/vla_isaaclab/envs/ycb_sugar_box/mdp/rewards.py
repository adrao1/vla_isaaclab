"""Sugar-box reward terms."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .terminations import (
    dex3_grasp_contacts,
    grasp_metrics,
    task_metrics,
)


def placement_reward(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    command_name: str,
) -> torch.Tensor:
    """Original pick-and-place reward."""
    metrics = task_metrics(
        env,
        palm_body_name,
        command_name,
    )

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
    reward = (
        1.0
        - torch.tanh(
            3.0 * distance
        )
    )
    reward += (
        1.0
        - torch.tanh(
            30.0 * distance
        )
    )

    return reward / 2.0


def grasp_contact_reward(
    env: ManagerBasedRLEnv,
    min_force: float = 0.5,
) -> torch.Tensor:
    """Reward simultaneous sugar-box contact by all three Dex3 fingers."""
    contacts = dex3_grasp_contacts(
        env,
        min_force=min_force,
    )

    return contacts[
        "is_grasping"
    ].float()


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
        metrics["lift_height"]
        / lift_target,
        min=0.0,
        max=1.0,
    )

    # Keep the existing closure-based shaping here for this first
    # contact-reward experiment. Actual grasp validity is handled by
    # grasp_contact_reward() and grasp_success().
    return (
        lift_progress
        * metrics["closure_fraction"]
    )


def grasp_success_reward(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
    lift_threshold: float = 0.03,
    min_contact_force: float = 0.5,
    max_hand_distance: float = 0.22,
) -> torch.Tensor:
    """Dense bonus for a three-finger contact grasp above the lift threshold."""
    metrics = grasp_metrics(
        env,
        palm_body_name,
        initial_box_height,
        min_contact_force=min_contact_force,
    )

    return (
        (
            metrics["hand_distance"]
            < max_hand_distance
        )
        & metrics["is_grasping"]
        & (
            metrics["lift_height"]
            > lift_threshold
        )
    ).float()
