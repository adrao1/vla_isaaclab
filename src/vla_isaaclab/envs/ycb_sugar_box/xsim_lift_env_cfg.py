"""X-Sim-style reach/grasp/lift task for G1 + left Dex3 (no calibrated grasp pose).

See mdp/xsim.py for which parts follow X-Sim source and which are G1 adaptations.
Task-specific choices made here (ours, not X-Sim):
  * two hand-set lift waypoints (+2.5 cm, +5 cm), not demo-derived
  * goal_thresh = 1 cm, success_radius = 1.5 cm (X-Sim's 5 cm radius would
    count the unlifted box as success, since the final waypoint is 5 cm away)
  * waypoint orientation reward off
  * left-hand fingers start just inside the soft joint limits
  * fixed object reset (X-Sim randomizes +-2.5 cm / +-22.5 deg; added later)
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..common import LEFT_HAND_JOINT_NAMES, SUPPORT_HEIGHT
from . import mdp
from .env_cfg import YCBSugarBoxEnvCfg
from .mdp import xsim as xsim_mdp

# Just inside the soft limits (0.9 factor), so zero finger action does not
# clamp-move the fingers on the first step.
XSIM_HAND_OPEN_JOINT_POS = {
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_thumb_1_joint": 0.0,
    "left_hand_thumb_2_joint": 0.09,
    "left_hand_middle_0_joint": -0.08,
    "left_hand_middle_1_joint": -0.09,
    "left_hand_index_0_joint": -0.08,
    "left_hand_index_1_joint": -0.09,
}


@configclass
class XSimPolicyCfg(ObsGroup):
    tcp_pose = ObsTerm(func=xsim_mdp.obs_tcp_pose)                       # 7
    finger_joint_pos = ObsTerm(                                          # 7 (gripper-width analog)
        func=base_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(LEFT_HAND_JOINT_NAMES), preserve_order=True)},
    )
    achieved_goal = ObsTerm(func=xsim_mdp.obs_object_pose)               # 7
    desired_goal = ObsTerm(func=xsim_mdp.obs_desired_goal)               # 7
    goal_position_diff = ObsTerm(func=xsim_mdp.obs_goal_position_diff)   # 3
    goal_rotation_diff = ObsTerm(func=xsim_mdp.obs_goal_rotation_diff)   # 1
    is_grasped = ObsTerm(func=xsim_mdp.obs_is_grasped)                   # 1

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class XSimObservationsCfg:
    policy: XSimPolicyCfg = XSimPolicyCfg()


@configclass
class XSimRewardsCfg:
    # Weight is set to 1/step_dt in __post_init__ so the per-step reward equals
    # X-Sim's formula (Isaac Lab's RewardManager multiplies by step_dt).
    xsim = RewTerm(
        func=xsim_mdp.xsim_dense_reward,
        weight=1.0,
        params={
            "goal_thresh": 0.01,
            "success_radius": 0.015,
            "success_angle": 0.3,
            "static_thresh": 0.1,
            "min_force": 0.5,
            "max_angle_deg": 85.0,
            "require_grasp": True,
        },
    )


@configclass
class XSimTerminationsCfg:
    # No success termination: X-Sim trains with partial_reset=False.
    object_fallen = DoneTerm(func=mdp.object_fallen, params={"support_height": SUPPORT_HEIGHT})
    invalid_state = DoneTerm(func=mdp.invalid_state)
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class XSimLiftEnvCfg(YCBSugarBoxEnvCfg):
    observations: XSimObservationsCfg = XSimObservationsCfg()
    rewards: XSimRewardsCfg = XSimRewardsCfg()
    terminations: XSimTerminationsCfg = XSimTerminationsCfg()

    episode_length_s: float = 10.0

    task_instruction: str = (
        "Grasp the YCB 004 sugar box with the left Dex3 hand and lift it 5 cm."
    )

    def __post_init__(self):
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        self.scene.robot.init_state.joint_pos.update(XSIM_HAND_OPEN_JOINT_POS)
        self.events.reset_xsim_state = EventTerm(func=xsim_mdp.reset_xsim_state, mode="reset")
        self.rewards.xsim.weight = 1.0 / (self.sim.dt * self.decimation)


@configclass
class XSimLiftRotEnvCfg(XSimLiftEnvCfg):
    """XSimLift with X-Sim's waypoint orientation reward turned on.

    rotation_reward=True is used by all five X-Sim tasks (metadata.npz).
    angle_goal_thresh = 0.3 rad matches X-Sim's Mustard-Place metadata.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.xsim.params["rotation_reward"] = True
        self.rewards.xsim.params["angle_goal_thresh"] = 0.3


# G1 adaptation (not X-Sim): X-Sim's Panda gripper is force-limited (100 N).
# Without a cap, absolute Dex3 grip targets squeeze with hundreds of newtons
# and throw the box (rl/probe_policy_close.py). Cap hand joint torque and
# lower damping so the fingers still close quickly under the cap.
HAND_EFFORT_CAP_NM = 2.0
HAND_DAMPING = 0.5


@configclass
class XSimLiftGripEnvCfg(XSimLiftRotEnvCfg):
    """XSimLiftRot with a force-capped Dex3 hand, for the 7-D grip controller."""

    def __post_init__(self):
        super().__post_init__()
        hands = self.scene.robot.actuators["hands"]
        hands.effort_limit = HAND_EFFORT_CAP_NM
        if hasattr(hands, "effort_limit_sim"):
            hands.effort_limit_sim = HAND_EFFORT_CAP_NM
        hands.damping = HAND_DAMPING
