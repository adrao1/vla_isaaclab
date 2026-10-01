"""Pose guidance toward the nearest of several saved, physics-validated grasps.

Our addition (NOT X-Sim). Replaces the hardcoded world-frame pose in
guided_grasp_env_cfg.py. Each saved grasp is a palm pose in the OBJECT frame
(produced by rl/grasp_geometry.py and validated in rl/validate_grasp_candidates.py),
so the target follows the object's current pose every step.
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


def _library(env, path):
    cache = getattr(env, "_grasp_library", None)
    if cache is None or cache["path"] != str(path):
        data = json.loads(Path(path).read_text())
        grasps = data["grasps"]
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


def nearest_grasp_errors(env, library_path, angle_weight=0.3):
    """Return (distance, angle, index) of the palm to the nearest saved grasp.

    Nearest means smallest distance + angle_weight * angle (metres + radians).
    """
    robot = env.scene["robot"]
    palm_id = getattr(env, "_guidance_palm_id", None)
    if palm_id is None:
        ids, _ = robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
        palm_id = ids[0]
        env._guidance_palm_id = palm_id

    lib = _library(env, library_path)
    obj = env.scene["object"]
    obj_pos, obj_quat = obj.data.root_pos_w, obj.data.root_quat_w  # (N,3), (N,4)
    obj_rot = matrix_from_quat(obj_quat)  # (N,3,3)

    target_pos = obj_pos[:, None, :] + torch.einsum("nij,gj->ngi", obj_rot, lib["pos"])  # (N,G,3)
    target_quat = quat_mul(
        obj_quat[:, None, :].expand(-1, lib["quat"].shape[0], -1).reshape(-1, 4),
        lib["quat"][None].expand(env.num_envs, -1, -1).reshape(-1, 4),
    ).reshape(env.num_envs, -1, 4)

    palm_pos = robot.data.body_pos_w[:, palm_id]
    palm_quat = robot.data.body_quat_w[:, palm_id]
    distance = torch.linalg.vector_norm(target_pos - palm_pos[:, None, :], dim=-1)  # (N,G)
    angle = quat_error_magnitude(
        palm_quat[:, None, :].expand(-1, target_quat.shape[1], -1).reshape(-1, 4),
        target_quat.reshape(-1, 4),
    ).reshape(env.num_envs, -1)
    best = torch.argmin(distance + angle_weight * angle, dim=1)
    rows = torch.arange(env.num_envs, device=env.device)
    return distance[rows, best], angle[rows, best], best


def guidance_reach(env, library_path):
    distance, _, _ = nearest_grasp_errors(env, library_path)
    return 0.5 * (1.0 - torch.tanh(3.0 * distance) + 1.0 - torch.tanh(20.0 * distance))


def guidance_orientation(env, library_path):
    distance, angle, _ = nearest_grasp_errors(env, library_path)
    proximity = 1.0 - torch.tanh(5.0 * distance)
    return proximity * (1.0 - torch.tanh(2.0 * angle))
