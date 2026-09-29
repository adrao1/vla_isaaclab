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
