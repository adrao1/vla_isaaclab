"""Minimal X-Sim Mustard Place scene port for G1 + Dex3."""

import math
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..common import (
    ACTION_JOINT_NAMES,
    JointLimitActionsCfg,
    VLAEnvCfg,
    camera_cfg,
    light_cfgs,
    make_g1_cfg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

KITCHEN_USD = (
    PROJECT_ROOT
    / "assets/xsim/kitchen_env/usd/Kitchen.usd"
)

MUSTARD_USD = (
    PROJECT_ROOT
    / "assets/xsim/kitchen_env/mustard/usd/mustard.usd"
)


# ---------------------------------------------------------------------------
# X-Sim Mustard-Place scene metadata
# ---------------------------------------------------------------------------

# metadata:
#   kitchen_to_robot_transform = [0.4547, 0.1653, 0.1775]
#
# MustardPlaceEnv then applies:
#   kitchen_xyz[:, 0] -= 0.02
#
# Therefore the actual kitchen position used in X-Sim is:
KITCHEN_POSITION = (
    0.4347,
    0.1653,
    0.1775,
)

# X-Sim rotates the kitchen +90 degrees about world X.
KITCHEN_ORIENTATION_WXYZ = (
    math.cos(math.pi / 4.0),
    math.sin(math.pi / 4.0),
    0.0,
    0.0,
)

GROUND_ALTITUDE = -0.6


# First pose from mustard_rl.npy.
MUSTARD_INITIAL_POSITION = (
    0.51332025,
    -0.27634321,
    0.24780577,
)

MUSTARD_INITIAL_ORIENTATION_WXYZ = (
    0.68878423,
    0.71095663,
    0.09902595,
    0.10154222,
)


# X-Sim camera metadata.
CAMERA_EYE = (-0.97, -0.96, 1.26)

CAMERA_TARGET = (0.55, 0.00, 0.30)


def _kitchen_cfg() -> AssetBaseCfg:
    if not KITCHEN_USD.is_file():
        raise FileNotFoundError(
            f"Missing converted X-Sim kitchen asset: {KITCHEN_USD}"
        )

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_USD),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=KITCHEN_POSITION,
            rot=KITCHEN_ORIENTATION_WXYZ,
        ),
    )


def _mustard_cfg() -> RigidObjectCfg:
    if not MUSTARD_USD.is_file():
        raise FileNotFoundError(
            f"Missing converted X-Sim mustard asset: {MUSTARD_USD}"
        )

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(MUSTARD_USD),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=MUSTARD_INITIAL_POSITION,
            rot=MUSTARD_INITIAL_ORIENTATION_WXYZ,
        ),
    )


_DOME_LIGHT, _KEY_LIGHT = light_cfgs()


@configclass
class MustardPlaceSceneCfg(InteractiveSceneCfg):
    # X-Sim ground altitude is -0.6 m.
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, GROUND_ALTITUDE),
        ),
    )

    dome_light = _DOME_LIGHT
    key_light = _KEY_LIGHT

    kitchen = _kitchen_cfg()

    # Keep the robot at the X-Sim robot-frame origin for this first
    # geometry test. The G1 USD itself expects its fixed root at z=0.74.
    #
    # make_g1_cfg's default yaw/orientation is retained initially.
    robot = make_g1_cfg((0.10, 0.0, 0.1923), orientation_wxyz=(1.0, 0.0, 0.0, 0.0))

    object = _mustard_cfg()

    # Reproduce the X-Sim task camera for geometry inspection.
    cam_side = camera_cfg(
        CAMERA_EYE,
        CAMERA_TARGET,
    )


# ---------------------------------------------------------------------------
# Minimal managers
#
# These are intentionally not the final Mustard Place observations/rewards.
# Their only purpose is to make this a valid ManagerBasedRLEnv while we
# validate scene geometry first.
# ---------------------------------------------------------------------------


@configclass
class MustardPlacePolicyCfg(ObsGroup):
    joint_position = ObsTerm(
        func=base_mdp.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(ACTION_JOINT_NAMES),
            )
        },
    )

    last_action = ObsTerm(
        func=base_mdp.last_action,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: MustardPlacePolicyCfg = MustardPlacePolicyCfg()


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=base_mdp.time_out,
        time_out=True,
    )


@configclass
class MustardPlaceEnvCfg(VLAEnvCfg):
    scene: MustardPlaceSceneCfg = MustardPlaceSceneCfg(
        num_envs=1,
        env_spacing=5.0,
        replicate_physics=True,
    )

    actions: JointLimitActionsCfg = JointLimitActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # X-Sim: 200 control steps at 5 Hz = 40 seconds.
    #
    # We reproduce the physical episode duration here, but we are NOT yet
    # changing Isaac Lab's control frequency to 5 Hz. That comes after the
    # scene geometry is validated.
    episode_length_s: float = 40.0

    task_instruction: str = (
        "Pick up the mustard bottle and place it at the target pose."
    )

    camera_eye: tuple[float, float, float] = CAMERA_EYE
    camera_target: tuple[float, float, float] = CAMERA_TARGET
