"""Cracker-box reach then grasp on the original YCB table scene.

G1, table, cameras, and object XY/yaw match the sugar-box environment.
The object USD is YCB 003. Reach and grasp share observations so a reach
checkpoint can warm-start grasp.

The named reach target is one physics-validated palm pose from
assets/grasps/cracker_box.json (object frame). That file is produced by
rl/validate_grasp_candidates.py then rl/build_grasp_library.py; keep a
single candidate so the target cannot jump.

Grip lock/gating is applied in rl/train_ppo.py and rl/eval_ppo.py on the
7-D grip action (the env only sees 43-D joint targets).
"""

from pathlib import Path

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..common import LEFT_HAND_JOINT_NAMES, SUPPORT_HEIGHT
from ..common.managers import EventsCfg
from .env_cfg import INITIAL_XY, SUGAR_BOX_ORIENTATION_WXYZ, YCBSugarBoxEnvCfg
from .grasp_env_cfg import object_xy_displacement_penalty
from .guided_grasp_env_cfg import completion_reward, contact_reward
from .mdp import grasp_guidance, object_fallen, invalid_state, grasp_success
from .xsim_lift_env_cfg import HAND_DAMPING, HAND_EFFORT_CAP_NM, XSIM_HAND_OPEN_JOINT_POS

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CRACKER_BOX_USD = PROJECT_ROOT / "assets/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
CRACKER_GRASP_LIBRARY = PROJECT_ROOT / "assets/grasps/cracker_box.json"

# Same upright rotation as the sugar box (x+90, then table yaw).
CRACKER_ORIENTATION_WXYZ = SUGAR_BOX_ORIENTATION_WXYZ
# YCB 003 upright half-height used by the tabletop preview. Z-yaw does not
# change min-z, so rest height is support plus this half-extent.
CRACKER_HALF_HEIGHT_M = 0.107
CRACKER_REST_Z_M = SUPPORT_HEIGHT + CRACKER_HALF_HEIGHT_M

REACH_MAX_DISTANCE_M = 0.01
REACH_MAX_ANGLE_RAD = 0.17453292519943295  # 10 deg
REACH_HOLD_STEPS = 15


def _library_params():
    return {"library_path": str(CRACKER_GRASP_LIBRARY)}


@configclass
class CrackerPolicyCfg(ObsGroup):
    palm_pose = ObsTerm(func=grasp_guidance.obs_palm_pose)
    object_pose = ObsTerm(func=grasp_guidance.obs_object_pose)
    grasp_target_pose = ObsTerm(
        func=grasp_guidance.obs_grasp_target_pose, params=_library_params()
    )
    grasp_position_error = ObsTerm(
        func=grasp_guidance.obs_grasp_position_error, params=_library_params()
    )
    grasp_angle_error = ObsTerm(
        func=grasp_guidance.obs_grasp_angle_error, params=_library_params()
    )
    finger_joint_pos = ObsTerm(
        func=base_mdp.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(LEFT_HAND_JOINT_NAMES),
                preserve_order=True,
            )
        },
    )
    is_grasping = ObsTerm(func=grasp_guidance.obs_is_grasping)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CrackerObservationsCfg:
    policy: CrackerPolicyCfg = CrackerPolicyCfg()


@configclass
class CrackerReachRewardsCfg:
    reach = RewTerm(
        func=grasp_guidance.guidance_reach, weight=1.0, params=_library_params()
    )
    orientation = RewTerm(
        func=grasp_guidance.guidance_orientation, weight=0.5, params=_library_params()
    )
    object_motion = RewTerm(
        func=object_xy_displacement_penalty,
        weight=-0.25,
        params={"initial_xy": INITIAL_XY, "scale": 10.0},
    )
    completion = RewTerm(func=completion_reward, weight=5.0)


@configclass
class CrackerGraspRewardsCfg(CrackerReachRewardsCfg):
    grasp = RewTerm(func=contact_reward, weight=1.0)


@configclass
class CrackerReachTerminationsCfg:
    success = DoneTerm(
        func=grasp_guidance.palm_at_grasp,
        params={
            **_library_params(),
            "max_distance": REACH_MAX_DISTANCE_M,
            "max_angle_rad": REACH_MAX_ANGLE_RAD,
            "hold_steps": REACH_HOLD_STEPS,
        },
    )
    object_fallen = DoneTerm(func=object_fallen, params={"support_height": SUPPORT_HEIGHT})
    invalid_state = DoneTerm(func=invalid_state)
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class CrackerGraspTerminationsCfg(CrackerReachTerminationsCfg):
    success = DoneTerm(
        func=grasp_success,
        params={"min_contact_force": 0.5, "hold_steps": 30},
    )


@configclass
class CrackerEventsCfg(EventsCfg):
    clear_guidance_counters = EventTerm(
        func=grasp_guidance.reset_guidance_counters, mode="reset"
    )


@configclass
class CrackerBoxEnvCfg(YCBSugarBoxEnvCfg):
    """Original table scene with the YCB cracker box."""

    grasp_library_path: str = str(CRACKER_GRASP_LIBRARY)
    events: CrackerEventsCfg = CrackerEventsCfg()
    episode_length_s: float = 20.0

    def __post_init__(self):
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        if not CRACKER_BOX_USD.is_file():
            raise FileNotFoundError(f"Missing cached YCB asset: {CRACKER_BOX_USD}")
        self.scene.object.spawn.usd_path = str(CRACKER_BOX_USD)
        self.scene.object.init_state.pos = (INITIAL_XY[0], INITIAL_XY[1], CRACKER_REST_Z_M)
        self.scene.object.init_state.rot = CRACKER_ORIENTATION_WXYZ
        self.scene.robot.init_state.joint_pos.update(XSIM_HAND_OPEN_JOINT_POS)
        hands = self.scene.robot.actuators["hands"]
        hands.effort_limit = HAND_EFFORT_CAP_NM
        if hasattr(hands, "effort_limit_sim"):
            hands.effort_limit_sim = HAND_EFFORT_CAP_NM
        hands.damping = HAND_DAMPING


@configclass
class CrackerReachEnvCfg(CrackerBoxEnvCfg):
    """Open-hand reach to the saved cracker-box grasp pose."""

    observations: CrackerObservationsCfg = CrackerObservationsCfg()
    rewards: CrackerReachRewardsCfg = CrackerReachRewardsCfg()
    terminations: CrackerReachTerminationsCfg = CrackerReachTerminationsCfg()
    task_instruction: str = (
        "Reach the saved left Dex3 palm pose on the YCB 003 cracker box with the hand open."
    )

    def __post_init__(self):
        super().__post_init__()
        if not CRACKER_GRASP_LIBRARY.is_file():
            raise FileNotFoundError(
                f"Missing grasp library: {CRACKER_GRASP_LIBRARY} "
                "(validate with rl/validate_grasp_candidates.py --object cracker_box, "
                "then rl/build_grasp_library.py)"
            )


@configclass
class CrackerGraspEnvCfg(CrackerReachEnvCfg):
    """Close a three-finger grasp after the palm is at the saved pose."""

    rewards: CrackerGraspRewardsCfg = CrackerGraspRewardsCfg()
    terminations: CrackerGraspTerminationsCfg = CrackerGraspTerminationsCfg()
    task_instruction: str = (
        "From the saved cracker-box palm pose, close the left Dex3 hand and hold a "
        "three-finger grasp without knocking the box aside."
    )
