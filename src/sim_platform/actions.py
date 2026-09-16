"""Robot action terms."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class UpperBodyJointAction(ActionTerm):
    """Map normalized actions to selected joints while holding every other joint."""

    _asset: Articulation

    def __init__(self, cfg: "UpperBodyJointActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self._joint_ids) != len(cfg.joint_names):
            raise RuntimeError(f"Found {len(self._joint_ids)} of {len(cfg.joint_names)} configured joints")
        self._raw_actions = torch.zeros(env.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_targets = self._asset.data.default_joint_pos.clone()

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def joint_position_targets(self) -> torch.Tensor:
        """Physical position targets, ordered like ``joint_names``."""
        return self._joint_targets[:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._joint_targets[:] = self._asset.data.default_joint_pos
        targets = self._asset.data.default_joint_pos[:, self._joint_ids] + self.cfg.scale * self._processed_actions
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._joint_targets[:, self._joint_ids] = torch.clamp(targets, limits[..., 0], limits[..., 1])

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._joint_targets)

    def reset(self, env_ids=None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = self._asset.data.default_joint_pos[env_ids]


@configclass
class UpperBodyJointActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = UpperBodyJointAction
    joint_names: list[str] = MISSING
    scale: float = 0.25
