"""Task-independent observation helpers."""

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def object_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w), dim=-1)
