"""Observations, reward and success terms for the dinnerware task."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def object_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w), dim=-1)


def object_distance(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    return torch.linalg.vector_norm(obj.data.root_pos_w - goal.data.root_pos_w, dim=-1)


def negative_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -object_distance(env)


def task_success(env: ManagerBasedRLEnv, distance_threshold: float = 0.08) -> torch.Tensor:
    bowl: RigidObject = env.scene["bowl"]
    plate: RigidObject = env.scene["plate"]
    horizontal_distance = torch.linalg.vector_norm(bowl.data.root_pos_w[:, :2] - plate.data.root_pos_w[:, :2], dim=-1)
    height_ok = bowl.data.root_pos_w[:, 2] > 0.72
    return torch.logical_and(horizontal_distance < distance_threshold, height_ok)
