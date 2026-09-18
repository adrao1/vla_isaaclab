"""Controller that holds the robot at its default standing target."""

import torch

from ..contracts import ControllerDefinition


class StandingController:
    def __init__(self, env, robot):
        self.env = env
        self.robot = robot
        term = env.action_manager.get_term("upper_body")
        self.joint_names = term.joint_names

    def compute(self, step: int) -> torch.Tensor:
        del step
        return torch.zeros(
            (self.env.num_envs, self.env.action_manager.total_action_dim),
            device=self.env.device,
        )


def make_standing(env, robot, _unused=False):
    return StandingController(env, robot)


STANDING_CONTROLLER = ControllerDefinition(
    component_id="Controller-Standing-v0",
    required_robot_capabilities=frozenset({"upper_body_joint_control"}),
    factory=make_standing,
    behavior_prompt="Stand still in the default pose.",
)
