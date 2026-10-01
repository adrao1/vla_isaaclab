"""Registered YCB sugar-box manipulation environments."""

import gymnasium as gym


ENV_ID = "VLA-YCBSugarBox-G1-JointPos-v0"
GRASP_ENV_ID = "VLA-YCBSugarBox-G1-Grasp-v0"


if ENV_ID not in gym.registry:
    gym.register(
        id=ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:YCBSugarBoxEnvCfg"
        },
        disable_env_checker=True,
    )


if GRASP_ENV_ID not in gym.registry:
    gym.register(
        id=GRASP_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.grasp_env_cfg:YCBSugarBoxGraspEnvCfg"
            )
        },
        disable_env_checker=True,
    )


__all__ = [
    "ENV_ID",
    "GRASP_ENV_ID",
]


GUIDED_GRASP_ENV_ID = "VLA-YCBSugarBox-G1-GuidedGrasp-v0"
if GUIDED_GRASP_ENV_ID not in gym.registry:
    gym.register(
        id=GUIDED_GRASP_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.guided_grasp_env_cfg:GuidedGraspEnvCfg"
        },
        disable_env_checker=True,
    )

WAYPOINT_ENV_ID = 'VLA-YCBSugarBox-G1-Waypoint-v0'
if WAYPOINT_ENV_ID not in gym.registry:
    gym.register(id=WAYPOINT_ENV_ID, entry_point='isaaclab.envs:ManagerBasedRLEnv', kwargs={'env_cfg_entry_point': f'{__name__}.waypoint_env_cfg:WaypointEnvCfg'}, disable_env_checker=True)

XSIM_LIFT_ENV_ID = "VLA-YCBSugarBox-G1-XSimLift-v0"
if XSIM_LIFT_ENV_ID not in gym.registry:
    gym.register(
        id=XSIM_LIFT_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.xsim_lift_env_cfg:XSimLiftEnvCfg"},
        disable_env_checker=True,
    )

XSIM_LIFT_ROT_ENV_ID = "VLA-YCBSugarBox-G1-XSimLiftRot-v0"
if XSIM_LIFT_ROT_ENV_ID not in gym.registry:
    gym.register(
        id=XSIM_LIFT_ROT_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.xsim_lift_env_cfg:XSimLiftRotEnvCfg"},
        disable_env_checker=True,
    )

XSIM_LIFT_GRIP_ENV_ID = "VLA-YCBSugarBox-G1-XSimLiftGrip-v0"
if XSIM_LIFT_GRIP_ENV_ID not in gym.registry:
    gym.register(
        id=XSIM_LIFT_GRIP_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.xsim_lift_env_cfg:XSimLiftGripEnvCfg"},
        disable_env_checker=True,
    )

MUSTARD_XSIM_LIFT_GRIP_ENV_ID = "VLA-YCBMustard-G1-XSimLiftGrip-v0"
if MUSTARD_XSIM_LIFT_GRIP_ENV_ID not in gym.registry:
    gym.register(
        id=MUSTARD_XSIM_LIFT_GRIP_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.mustard_env_cfg:MustardXSimLiftGripEnvCfg"},
        disable_env_checker=True,
    )

MUSTARD_GUIDED_ENV_ID = "VLA-YCBMustard-G1-GuidedComputed-v0"
if MUSTARD_GUIDED_ENV_ID not in gym.registry:
    gym.register(
        id=MUSTARD_GUIDED_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.mustard_env_cfg:MustardGuidedXSimLiftGripEnvCfg"},
        disable_env_checker=True,
    )

CRACKER_REACH_ENV_ID = "VLA-YCBCrackerBox-G1-Reach-v0"
if CRACKER_REACH_ENV_ID not in gym.registry:
    gym.register(
        id=CRACKER_REACH_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.cracker_env_cfg:CrackerReachEnvCfg"},
        disable_env_checker=True,
    )

CRACKER_GRASP_ENV_ID = "VLA-YCBCrackerBox-G1-Grasp-v0"
if CRACKER_GRASP_ENV_ID not in gym.registry:
    gym.register(
        id=CRACKER_GRASP_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.cracker_env_cfg:CrackerGraspEnvCfg"},
        disable_env_checker=True,
    )
