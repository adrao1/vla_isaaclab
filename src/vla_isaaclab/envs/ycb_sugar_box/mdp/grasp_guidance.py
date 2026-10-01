"""Pose guidance toward saved, physics-validated grasps.

Our addition (NOT X-Sim). Replaces the hardcoded world-frame pose in
guided_grasp_env_cfg.py. Each saved grasp is a palm pose in the OBJECT frame
(produced by rl/grasp_geometry.py and validated in rl/validate_grasp_candidates.py),
so the target follows the object's current pose every step.

Reach/grasp tasks should store one validated pose in the library so the
selected target cannot jump.
"""

import json
from pathlib import Path

import torch

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_matrix,
    quat_mul,
)

from ...common import LEFT_END_EFFECTOR
from .terminations import dex3_grasp_contacts


def _library(env, path):
    cache = getattr(env, "_grasp_library", None)
    if cache is None or cache["path"] != str(path):
        data = json.loads(Path(path).read_text())
        grasps = data["grasps"]
        if not grasps:
            raise RuntimeError(f"Grasp library is empty: {path}")
        device = env.device
        pos = torch.tensor([g["palm_position_asset_m"] for g in grasps], dtype=torch.float32, device=device)
        rot = torch.tensor([g["palm_rotation_asset_from_palm"] for g in grasps], dtype=torch.float32, device=device)
        cache = {
            "path": str(path),
            "pos": pos,  # (G, 3) palm position in object frame
            "quat": quat_from_matrix(rot),  # (G, 4) palm orientation in object frame, wxyz
        }
        env._grasp_library = cache
    return cache


def _palm_id(env):
    palm_id = getattr(env, "_guidance_palm_id", None)
    if palm_id is None:
        ids, _ = env.scene["robot"].find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
        palm_id = ids[0]
        env._guidance_palm_id = palm_id
    return palm_id


def nearest_grasp(env, library_path, angle_weight=0.3):
    """Return distance, angle, index, world target position, and world target quat."""
    robot = env.scene["robot"]
    palm_id = _palm_id(env)
    lib = _library(env, library_path)
    obj = env.scene["object"]
    obj_pos, obj_quat = obj.data.root_pos_w, obj.data.root_quat_w
    obj_rot = matrix_from_quat(obj_quat)

    target_pos = obj_pos[:, None, :] + torch.einsum("nij,gj->ngi", obj_rot, lib["pos"])
    target_quat = quat_mul(
        obj_quat[:, None, :].expand(-1, lib["quat"].shape[0], -1).reshape(-1, 4),
        lib["quat"][None].expand(env.num_envs, -1, -1).reshape(-1, 4),
    ).reshape(env.num_envs, -1, 4)

    palm_pos = robot.data.body_pos_w[:, palm_id]
    palm_quat = robot.data.body_quat_w[:, palm_id]
    distance = torch.linalg.vector_norm(target_pos - palm_pos[:, None, :], dim=-1)
    angle = quat_error_magnitude(
        palm_quat[:, None, :].expand(-1, target_quat.shape[1], -1).reshape(-1, 4),
        target_quat.reshape(-1, 4),
    ).reshape(env.num_envs, -1)
    best = torch.argmin(distance + angle_weight * angle, dim=1)
    rows = torch.arange(env.num_envs, device=env.device)
    return (
        distance[rows, best],
        angle[rows, best],
        best,
        target_pos[rows, best],
        target_quat[rows, best],
    )


def nearest_grasp_errors(env, library_path, angle_weight=0.3):
    """Return (distance, angle, index) of the palm to the nearest saved grasp."""
    distance, angle, index, _, _ = nearest_grasp(env, library_path, angle_weight)
    return distance, angle, index


def guidance_reach(env, library_path):
    distance, _, _ = nearest_grasp_errors(env, library_path)
    return 0.5 * (1.0 - torch.tanh(3.0 * distance) + 1.0 - torch.tanh(20.0 * distance))


def guidance_orientation(env, library_path):
    distance, angle, _ = nearest_grasp_errors(env, library_path)
    proximity = 1.0 - torch.tanh(5.0 * distance)
    return proximity * (1.0 - torch.tanh(2.0 * angle))


def obs_palm_pose(env):
    robot = env.scene["robot"]
    palm_id = _palm_id(env)
    return torch.cat(
        (
            robot.data.body_pos_w[:, palm_id] - env.scene.env_origins,
            robot.data.body_quat_w[:, palm_id],
        ),
        dim=-1,
    )


def obs_object_pose(env):
    obj = env.scene["object"]
    return torch.cat(
        (obj.data.root_pos_w - env.scene.env_origins, obj.data.root_quat_w),
        dim=-1,
    )


def obs_grasp_target_pose(env, library_path):
    _, _, _, target_pos, target_quat = nearest_grasp(env, library_path)
    return torch.cat((target_pos - env.scene.env_origins, target_quat), dim=-1)


def obs_grasp_position_error(env, library_path):
    _, _, _, target_pos, _ = nearest_grasp(env, library_path)
    palm_pos = env.scene["robot"].data.body_pos_w[:, _palm_id(env)]
    return palm_pos - target_pos


def obs_grasp_angle_error(env, library_path):
    _, angle, _ = nearest_grasp_errors(env, library_path)
    return angle.unsqueeze(-1)


def obs_is_grasping(env, min_force=0.5):
    return dex3_grasp_contacts(env, min_force=min_force)["is_grasping"].float().unsqueeze(-1)


def palm_at_grasp(
    env,
    library_path,
    max_distance=0.01,
    max_angle_rad=0.17453292519943295,
    hold_steps=15,
):
    """Named reach success: open-hand palm held at the saved grasp pose."""
    distance, angle, _ = nearest_grasp_errors(env, library_path)
    instantaneous = (distance < max_distance) & (angle < max_angle_rad)
    counter = getattr(env, "reach_success_counter", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.reach_success_counter = counter
    counter[:] = torch.where(instantaneous, counter + 1, torch.zeros_like(counter))
    return counter >= hold_steps


def reset_guidance_counters(env, env_ids):
    ids = slice(None) if env_ids is None else env_ids
    for name in ("reach_success_counter", "grasp_success_counter"):
        counter = getattr(env, name, None)
        if counter is not None:
            counter[ids] = 0
