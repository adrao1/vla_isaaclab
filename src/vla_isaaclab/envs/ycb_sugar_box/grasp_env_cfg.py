"""Grasp-only RL variant of the YCB sugar-box environment."""

import isaaclab.envs.mdp as base_mdp

from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from ..common import (
    LEFT_END_EFFECTOR,
    SUPPORT_HEIGHT,
)
from . import mdp
from .env_cfg import (
    SUGAR_BOX_HALF_HEIGHT_M,
    YCBSugarBoxEnvCfg,
)


INITIAL_BOX_HEIGHT = (
    SUPPORT_HEIGHT
    + SUGAR_BOX_HALF_HEIGHT_M
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

    # X-Sim-style physical grasp signal.
    #
    # For Dex3, all three logical fingers must simultaneously contact the
    # sugar box with at least the configured minimum contact force.
    grasp = RewTerm(
        func=mdp.grasp_contact_reward,
        weight=1.0,
        params={
            "min_force": 0.5,
        },
    )


@configclass
class GraspTerminationsCfg:
    # For this simplified experiment, success means maintaining the physical
    # three-finger grasp. Lifting is deliberately not required.
    success = DoneTerm(
        func=mdp.grasp_success,
        params={
            "min_contact_force": 0.5,
            "hold_steps": 90,
        },
    )

    object_fallen = DoneTerm(
        func=mdp.object_fallen,
        params={
            "support_height": SUPPORT_HEIGHT
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
class YCBSugarBoxGraspEnvCfg(
    YCBSugarBoxEnvCfg
):
    """Sugar-box grasp-only task used for PPO training."""

    rewards: GraspRewardsCfg = (
        GraspRewardsCfg()
    )

    terminations: GraspTerminationsCfg = (
        GraspTerminationsCfg()
    )

    # Give the policy enough time to reach the object, establish all three
    # finger contacts, and maintain the grasp.
    episode_length_s: float = 30.0

    task_instruction: str = (
        "Reach for the YCB 004 sugar box with the left Dex3 hand "
        "and grasp it securely with the thumb, index, and middle fingers."
    )
