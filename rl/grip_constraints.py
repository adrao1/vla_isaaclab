"""Lock or gate the 7-D Dex3 grip channel before the EE controller runs.

The environment action is 43-D joint targets, so reach-open / close-at-pose
constraints have to be applied to the RL action here.
"""
import math

import torch


def apply_grip_constraints(
    action,
    env,
    lock_open=False,
    gate_until_reach=False,
    max_distance=0.015,
    max_angle_rad=math.radians(15.0),
):
    """Return an action tensor with grip dim 6 possibly overwritten.

    lock_open: force fully open (-1), used for reach training.
    gate_until_reach: keep the hand open until the palm is near the saved pose.
    """
    if action.shape[-1] < 7:
        raise RuntimeError(
            f"Grip constraints require a 7-D or 13-D EE action, got {tuple(action.shape)}"
        )
    if not lock_open and not gate_until_reach:
        return action
    out = action.clone()
    if lock_open:
        out[:, 6] = -1.0
        return out
    path = getattr(env.cfg, "grasp_library_path", None)
    if not path:
        raise RuntimeError("gate_until_reach requires env.cfg.grasp_library_path")
    from vla_isaaclab.envs.ycb_sugar_box.mdp.grasp_guidance import nearest_grasp_errors

    distance, angle, _ = nearest_grasp_errors(env, path)
    far = (distance > max_distance) | (angle > max_angle_rad)
    out[far, 6] = -1.0
    return out
