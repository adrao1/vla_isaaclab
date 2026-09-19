"""Smooth dual-arm raise/lower controller used for pipeline demonstrations."""

import math

import torch

from ..contracts import ControllerDefinition


class RaiseLowerController:
    def __init__(self, env, robot):
        self.env = env
        self.robot = robot
        term = env.action_manager.get_term("upper_body")
        self.joint_names = term.joint_names
        self.action_term = term
        self.name_to_index = {name: index for index, name in enumerate(self.joint_names)}

    def compute(self, target=None, step: int = 0) -> torch.Tensor:
        del target
        deltas = torch.zeros(
            (self.env.num_envs, self.env.action_manager.total_action_dim),
            device=self.env.device,
        )
        # One four-second cycle at the 30 Hz control rate: down -> raised -> down.
        lift = 0.5 * (1.0 - math.cos(2.0 * math.pi * step / 120.0))
        for name, value in (
            ("left_shoulder_pitch_joint", 0.75 * lift),
            ("right_shoulder_pitch_joint", 0.75 * lift),
            ("left_elbow_pitch_joint", 0.2625 * lift),
            ("right_elbow_pitch_joint", 0.2625 * lift),
        ):
            if name in self.name_to_index:
                deltas[:, self.name_to_index[name]] = value
        return self.action_term.actions_from_joint_deltas(deltas)


def make_raise_lower(env, robot, _unused=False):
    return RaiseLowerController(env, robot)


RAISE_LOWER_CONTROLLER = ControllerDefinition(
    component_id="Controller-RaiseLower-v0",
    required_robot_capabilities=frozenset({"upper_body_joint_control"}),
    factory=make_raise_lower,
    behavior_prompt="Raise and lower both arms while standing in front of the table.",
)
