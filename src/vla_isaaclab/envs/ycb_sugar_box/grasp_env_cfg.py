"""Grasp-and-lift RL variant of the YCB sugar-box environment."""

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
        weight=2.0,
        params={
            "palm_body_name": LEFT_END_EFFECTOR,
            "initial_box_height": INITIAL_BOX_HEIGHT,
        },
    )

    # X-Sim-style grasp signal: reward actual physical grasp rather than
    # merely rewarding finger closure.
    #
    # For Dex3, all three logical fingers must contact the sugar box.
    grasp = RewTerm(
        func=mdp.grasp_contact_reward,
        weight=2.0,
        params={
            "min_force": 0.5,
        },
    )

    # Strong dense signal: actually make the box rise.
    lift = RewTerm(
        func=mdp.grasp_lift_reward,
        weight=4.0,
        params={
            "palm_body_name": LEFT_END_EFFECTOR,
            "initial_box_height": INITIAL_BOX_HEIGHT,
            "lift_target": 0.03,
        },
    )

    # Bonus once a genuine contact grasp-and-lift state is reached.
    success_bonus = RewTerm(
        func=mdp.grasp_success_reward,
        weight=10.0,
        params={
            "palm_body_name": LEFT_END_EFFECTOR,
            "initial_box_height": INITIAL_BOX_HEIGHT,
            "lift_threshold": 0.03,
            "min_contact_force": 0.5,
            "max_hand_distance": 0.22,
        },
    )


@configclass
class GraspTerminationsCfg:
    success = DoneTerm(
        func=mdp.grasp_success,
        params={
            "palm_body_name": LEFT_END_EFFECTOR,
            "initial_box_height": INITIAL_BOX_HEIGHT,
            "lift_threshold": 0.03,
            "min_contact_force": 0.5,
            "max_hand_distance": 0.22,
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
    """Sugar-box grasp-and-lift task used for PPO training."""

    rewards: GraspRewardsCfg = (
        GraspRewardsCfg()
    )
    terminations: GraspTerminationsCfg = (
        GraspTerminationsCfg()
    )

    # Grasping should not require the full 60-second placement horizon.
    episode_length_s: float = 30.0

    task_instruction: str = (
        "Reach for the YCB 004 sugar box with the left Dex3 hand, "
        "grasp it securely, and lift it at least 3 cm above its initial height."
    )
