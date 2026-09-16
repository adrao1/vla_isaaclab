"""Safe joint-space controller baselines."""

import math

import torch

from ..contracts import ControllerDefinition


class StandingController:
    def __init__(self, env, robot, raise_lower=False):
        self.env = env
        self.robot = robot
        self.raise_lower = raise_lower
        term = env.action_manager.get_term("upper_body")
        self.joint_names = term.joint_names
        self.name_to_index = {name: index for index, name in enumerate(self.joint_names)}

    def compute(self, step: int) -> torch.Tensor:
        action = torch.zeros((self.env.num_envs, self.env.action_manager.total_action_dim), device=self.env.device)
        if not self.raise_lower:
            return action
        # One four-second cycle at the 30 Hz control rate: down -> raised -> down.
        lift = 0.5 * (1.0 - math.cos(2.0 * math.pi * step / 120.0))
        for name, value in (
            ("left_shoulder_pitch_joint", lift),
            ("right_shoulder_pitch_joint", lift),
            ("left_elbow_pitch_joint", 0.35 * lift),
            ("right_elbow_pitch_joint", 0.35 * lift),
        ):
            if name in self.name_to_index:
                action[:, self.name_to_index[name]] = value
        return action


def make_standing(env, robot, _unused=False):
    return StandingController(env, robot, raise_lower=False)


def make_raise_lower(env, robot, _unused=False):
    return StandingController(env, robot, raise_lower=True)


STANDING_CONTROLLER = ControllerDefinition(
    component_id="Controller-Standing-v0",
    required_robot_capabilities=frozenset({"upper_body_joint_control"}),
    factory=make_standing,
)

RAISE_LOWER_CONTROLLER = ControllerDefinition(
    component_id="Controller-RaiseLower-v0",
    required_robot_capabilities=frozenset({"upper_body_joint_control"}),
    factory=make_raise_lower,
)
