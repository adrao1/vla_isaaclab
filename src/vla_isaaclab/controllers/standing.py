"""Controller that holds the robot at its default standing target."""

import torch

from ..contracts import ControllerDefinition


class StandingController:
    def __init__(self, env, robot):
        self.env = env
        self.robot = robot
        term = env.action_manager.get_term("upper_body")
        self.joint_names = term.joint_names

    def compute(self, target=None, step: int = 0) -> torch.Tensor:
        del target, step
        term = self.env.action_manager.get_term("upper_body")
        joint_ids, _ = self.env.scene["robot"].find_joints(term.joint_names, preserve_order=True)
        targets = self.env.scene["robot"].data.default_joint_pos[:, joint_ids]
        return term.actions_from_joint_targets(targets)


def make_standing(env, robot, _unused=False):
    return StandingController(env, robot)


STANDING_CONTROLLER = ControllerDefinition(
    component_id="Controller-Standing-v0",
    required_robot_capabilities=frozenset({"upper_body_joint_control"}),
    factory=make_standing,
    behavior_prompt="Stand still in the default pose.",
)
