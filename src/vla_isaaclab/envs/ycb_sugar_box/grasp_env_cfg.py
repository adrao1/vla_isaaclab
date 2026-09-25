"""Grasp-only RL variant of the YCB sugar-box environment."""

import torch

import isaaclab.envs.mdp as base_mdp

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..common import LEFT_END_EFFECTOR, SUPPORT_HEIGHT
from . import mdp
from .env_cfg import (
    INITIAL_XY,
    SUGAR_BOX_HALF_HEIGHT_M,
    YCBSugarBoxEnvCfg,
)


INITIAL_BOX_HEIGHT = SUPPORT_HEIGHT + SUGAR_BOX_HALF_HEIGHT_M


def object_xy_displacement_penalty(
    env: ManagerBasedRLEnv,
    initial_xy: tuple[float, float],
    scale: float = 10.0,
) -> torch.Tensor:
    """Penalize moving the sugar box horizontally away from its reset position.

    Only XY displacement is considered. Vertical motion is intentionally ignored
    so this term does not oppose lifting if lifting is added back later.
    """
    sugar_box: RigidObject = env.scene["object"]

    initial_xy_tensor = torch.tensor(
        initial_xy,
        device=env.device,
        dtype=sugar_box.data.root_pos_w.dtype,
    ).unsqueeze(0)

    # INITIAL_XY is expressed in each environment's local frame, while
    # root_pos_w is expressed in world coordinates.
    initial_xy_world = (
        env.scene.env_origins[:, :2]
        + initial_xy_tensor
    )

    xy_displacement = torch.linalg.vector_norm(
        sugar_box.data.root_pos_w[:, :2]
        - initial_xy_world,
        dim=-1,
    )

    # 0 when the box is exactly at its reset position.
    # Smoothly approaches 1 as the box is pushed farther away.
    return torch.tanh(
        scale * xy_displacement
    )


@configclass
class GraspRewardsCfg:
    # X-Sim-style broad + near-object reaching signal.
    reach = RewTerm(
        func=mdp.grasp_reaching_reward,
        weight=1.0,
        params={
            "palm_body_name": LEFT_END_EFFECTOR,
            "initial_box_height": INITIAL_BOX_HEIGHT,
        },
    )

    # Progressively reward valid Dex3 contact from each required finger.
    grasp = RewTerm(
        func=mdp.grasp_contact_reward,
        weight=1.0,
        params={
            "min_force": 0.5,
        },
    )

    # Discourage solving the task by shoving or dragging the box around first.
    #
    # The function itself returns a positive displacement measure in [0, 1].
    # The negative manager weight turns it into a penalty.
    object_motion = RewTerm(
        func=object_xy_displacement_penalty,
        weight=-0.25,
        params={
            "initial_xy": INITIAL_XY,
            "scale": 10.0,
        },
    )


@configclass
class GraspTerminationsCfg:
    # Evaluation / termination criterion only.
    #
    # Success means the thumb, index, and middle fingers all maintain
    # object contact above the minimum force for 30 consecutive steps.
    success = DoneTerm(
        func=mdp.grasp_success,
        params={
            "min_contact_force": 0.5,
            "hold_steps": 30,
        },
    )

    object_fallen = DoneTerm(
        func=mdp.object_fallen,
        params={
            "support_height": SUPPORT_HEIGHT,
        },
    )

    invalid_state = DoneTerm(
        func=mdp.invalid_state,
    )

    time_out = DoneTerm(
        func=base_mdp.time_out,
        time_out=True,
    )


@configclass
class YCBSugarBoxGraspEnvCfg(YCBSugarBoxEnvCfg):
    """Sugar-box reach-and-grasp task used for PPO training."""

    rewards: GraspRewardsCfg = GraspRewardsCfg()
    terminations: GraspTerminationsCfg = GraspTerminationsCfg()

    episode_length_s: float = 30.0

    task_instruction: str = (
        "Reach for the YCB 004 sugar box with the left Dex3 hand "
        "and grasp it securely with the thumb, index, and middle fingers "
        "without significantly moving it from its initial table position."
    )
