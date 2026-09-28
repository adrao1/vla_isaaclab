"""Guided G1 grasp baseline; uses a calibrated robot-specific grasp pose."""

import torch

from isaaclab.managers import EventTermCfg, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude

from ..common import LEFT_END_EFFECTOR
from ..common.managers import EventsCfg
from .grasp_env_cfg import GraspRewardsCfg, YCBSugarBoxGraspEnvCfg
from .mdp.terminations import dex3_grasp_contacts


# Same calibrated pose as the successful scripted grasp.
GRASP_OFFSET = (-0.10779687797449515, -0.11014619853853247, 0.07505996064897658)
GRASP_QUAT = (0.838739529885833, -0.0562443579616566,
              0.0882924479819968, 0.5343753520080425)


def pose_errors(env):
    robot = env.scene["robot"]
    palm_id = getattr(env, "_guided_palm_id", None)
    if palm_id is None:
        ids, _ = robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
        palm_id = ids[0]
        env._guided_palm_id = palm_id

    obj_pos = env.scene["object"].data.root_pos_w
    desired_pos = obj_pos + obj_pos.new_tensor(GRASP_OFFSET)
    desired_quat = obj_pos.new_tensor(GRASP_QUAT).expand(env.num_envs, -1)

    distance = torch.linalg.vector_norm(
        robot.data.body_pos_w[:, palm_id] - desired_pos, dim=-1
    )
    angle = quat_error_magnitude(
        robot.data.body_quat_w[:, palm_id], desired_quat
    )
    return distance, angle


def approach_reward(env):
    distance, _ = pose_errors(env)
    return 0.5 * (
        1.0 - torch.tanh(3.0 * distance)
        + 1.0 - torch.tanh(20.0 * distance)
    )


def orientation_reward(env):
    distance, angle = pose_errors(env)
    proximity = 1.0 - torch.tanh(5.0 * distance)
    return proximity * (1.0 - torch.tanh(2.0 * angle))


def contact_reward(env):
    contacts = dex3_grasp_contacts(env, min_force=0.5)
    partial = (
        contacts["thumb_contact"].float()
        + contacts["index_contact"].float()
        + contacts["middle_contact"].float()
    ) / 3.0
    full = contacts["is_grasping"].float()
    return 0.15 * partial + 2.0 * full


def completion_reward(env):
    # RewardManager multiplies terms by step_dt.
    # Dividing here makes weight=5 a one-time +5 transition reward.
    return (
        env.termination_manager.get_term("success").float() / env.step_dt
    )


def reset_grasp_counter(env, env_ids):
    counter = getattr(env, "grasp_success_counter", None)
    if counter is not None:
        counter[slice(None) if env_ids is None else env_ids] = 0


@configclass
class GuidedRewardsCfg(GraspRewardsCfg):
    reach = RewardTermCfg(func=approach_reward, weight=1.0)
    orientation = RewardTermCfg(func=orientation_reward, weight=0.5)
    grasp = RewardTermCfg(func=contact_reward, weight=1.0)
    completion = RewardTermCfg(func=completion_reward, weight=5.0)
    # Inherits the original object-motion penalty.


@configclass
class GuidedEventsCfg(EventsCfg):
    clear_grasp_counter = EventTermCfg(
        func=reset_grasp_counter, mode="reset"
    )


@configclass
class GuidedGraspEnvCfg(YCBSugarBoxGraspEnvCfg):
    rewards: GuidedRewardsCfg = GuidedRewardsCfg()
    events: GuidedEventsCfg = GuidedEventsCfg()
