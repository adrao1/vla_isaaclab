"""Small controller interface used before a learned policy is connected."""

from __future__ import annotations

import math

import torch


class StandingUpperBodyController:
    """Generate normalized upper-body actions, optionally with a safe motion demo."""

    def __init__(self, env, demo_motion: bool = False):
        self.env = env
        self.demo_motion = demo_motion
        term = env.action_manager.get_term("upper_body")
        self.joint_names = term.joint_names
        self.name_to_index = {name: index for index, name in enumerate(self.joint_names)}

    def compute(self, step: int) -> torch.Tensor:
        action = torch.zeros((self.env.num_envs, self.env.action_manager.total_action_dim), device=self.env.device)
        if not self.demo_motion:
            return action
        phase = 2.0 * math.pi * step / 120.0
        for name, amplitude, phase_offset in (
            ("torso_joint", 0.20, 0.0),
            ("left_shoulder_pitch_joint", 0.35, 0.0),
            ("right_shoulder_pitch_joint", 0.35, math.pi),
            ("left_elbow_pitch_joint", 0.25, math.pi / 2.0),
            ("right_elbow_pitch_joint", 0.25, -math.pi / 2.0),
        ):
            action[:, self.name_to_index[name]] = amplitude * math.sin(phase + phase_offset)
        return action
