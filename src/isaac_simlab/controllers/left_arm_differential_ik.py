"""Reusable task-space controller for the Unitree G1 left arm and hand."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import compute_pose_error

from .bounded_ik import bounded_dls

from ..contracts import ControllerDefinition, EndEffectorTarget


class LeftArmDifferentialIKController:
    """Convert a left-EE pose and gripper target into normalized env actions."""

    def __init__(self, env, robot_definition):
        self.env = env
        self.definition = robot_definition
        self.robot = env.scene["robot"]
        self.action_term = env.action_manager.get_term("upper_body")
        self.joint_names = self.action_term.joint_names
        self.name_to_action_index = {name: index for index, name in enumerate(self.joint_names)}
        self.action_joint_ids, found = self.robot.find_joints(self.joint_names, preserve_order=True)
        if list(found) != self.joint_names:
            raise RuntimeError(f"Action joint mismatch: expected {self.joint_names}, found {found}")

        # The G1 arm itself has five joints.  Include torso yaw so the fixed-base
        # kinematic chain has six DoF for a full 6D palm-pose command.
        self.arm_joint_names = list(
            robot_definition.waist_joint_names + robot_definition.left_arm_joint_names
        )
        self.arm_joint_ids, found = self.robot.find_joints(self.arm_joint_names, preserve_order=True)
        if list(found) != self.arm_joint_names:
            raise RuntimeError(f"Left-arm joint mismatch: expected {self.arm_joint_names}, found {found}")
        self.arm_action_indices = [self.name_to_action_index[name] for name in self.arm_joint_names]

        self.hand_joint_names = list(robot_definition.left_hand_joint_names)
        self.hand_action_indices = [self.name_to_action_index[name] for name in self.hand_joint_names]
        if len(robot_definition.left_hand_open_joint_positions) != len(self.hand_joint_names):
            raise ValueError("Left-hand open pose does not match the hand joint count")
        if len(robot_definition.left_hand_closed_joint_positions) != len(self.hand_joint_names):
            raise ValueError("Left-hand closed pose does not match the hand joint count")
        self.open_hand = torch.tensor(
            robot_definition.left_hand_open_joint_positions, device=env.device
        ).unsqueeze(0)
        self.closed_hand = torch.tensor(
            robot_definition.left_hand_closed_joint_positions, device=env.device
        ).unsqueeze(0)

        palm_ids, _ = self.robot.find_bodies([robot_definition.left_end_effector], preserve_order=True)
        self.palm_body_id = palm_ids[0]
        self.palm_jacobian_id = self.palm_body_id - 1 if self.robot.is_fixed_base else self.palm_body_id
        # Limit successive commands, not their error from measured joints.
        # A bounded tracking allowance lets PD torque build against gravity.
        self.max_joint_delta = 0.025
        self.max_tracking_error = 0.35
        self.max_torso_deviation = math.radians(25.0)
        self.ik_gain = 0.15
        self.ik_damping = 0.04
        self.orientation_weight = 0.20
        self.last_joint_targets = self.robot.data.joint_pos[:, self.arm_joint_ids].clone()
        self.last_mode = "pose"
        self.compute_count = 0
        self.debug_samples = []

    def reset(self) -> None:
        self.last_joint_targets = self.robot.data.joint_pos[:, self.arm_joint_ids].clone()
        self.compute_count = 0
        self.debug_samples = []

    def compute(self, target: EndEffectorTarget, step: int = 0) -> torch.Tensor:
        del step
        if target is None:
            raise ValueError("Controller-LeftArmDifferentialIK-v0 requires an EndEffectorTarget")

        palm_pos = self.robot.data.body_pos_w[:, self.palm_body_id]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_body_id]
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.palm_jacobian_id, :, self.arm_joint_ids
        ]
        current_arm = self.robot.data.joint_pos[:, self.arm_joint_ids]
        if target.orientation is None:
            self.last_mode = "position"
            error = target.position - palm_pos
            task_jacobian = jacobian[:, :3].clone()
            task_jacobian[:, :, 0] = 0.0
        else:
            self.last_mode = "pose"
            position_error, rotation_error = compute_pose_error(
                palm_pos, palm_quat, target.position, target.orientation,
                rot_error_type="axis_angle",
            )
            error = torch.cat((position_error, self.orientation_weight * rotation_error), dim=-1)
            task_jacobian = jacobian.clone()
            task_jacobian[:, 3:] *= self.orientation_weight

        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids]
        lower, upper = limits[..., 0].clone(), limits[..., 1].clone()
        torso_center = self.robot.data.default_joint_pos[:, self.arm_joint_ids[0]]
        lower[:, 0] = torch.maximum(lower[:, 0], torso_center - self.max_torso_deviation)
        upper[:, 0] = torch.minimum(upper[:, 0], torso_center + self.max_torso_deviation)
        # Anti-windup: project the persistent command into the tracking window.
        reference = torch.clamp(self.last_joint_targets,
                                current_arm - self.max_tracking_error,
                                current_arm + self.max_tracking_error)
        reference = torch.clamp(reference, lower, upper)
        delta_lower = torch.maximum(lower - reference, torch.full_like(reference, -self.max_joint_delta))
        delta_upper = torch.minimum(upper - reference, torch.full_like(reference, self.max_joint_delta))
        delta_lower = torch.maximum(delta_lower, current_arm - self.max_tracking_error - reference)
        delta_upper = torch.minimum(delta_upper, current_arm + self.max_tracking_error - reference)
        # Physical position bounds take precedence if physics crosses a limit.
        delta_lower = torch.minimum(delta_lower, torch.zeros_like(delta_lower))
        delta_upper = torch.maximum(delta_upper, torch.zeros_like(delta_upper))
        if target.orientation is None:
            delta_lower[:, 0] = 0.0
            delta_upper[:, 0] = 0.0
        correction = bounded_dls(task_jacobian, self.ik_gain * error,
                                 delta_lower, delta_upper, self.ik_damping)
        raw_targets = reference + correction
        arm_targets = torch.clamp(raw_targets, lower, upper)
        self.last_joint_targets = arm_targets.clone()
        self.compute_count += 1
        if self.compute_count in {1, 2, 5, 10, 25} or self.compute_count % 50 == 0:
            self.debug_samples.append(
                {
                    "step": self.compute_count,
                    "mode": self.last_mode,
                    "ee_position_m": (
                        palm_pos[0] - self.env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "target_position_m": (
                        target.position[0] - self.env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "joint_position_rad": current_arm[0].detach().cpu().tolist(),
                    "raw_joint_delta_rad": (
                        raw_targets[0] - current_arm[0]
                    ).detach().cpu().tolist(),
                    "commanded_joint_delta_rad": (
                        arm_targets[0] - current_arm[0]
                    ).detach().cpu().tolist(),
                }
            )

        targets = self.robot.data.default_joint_pos[:, self.action_joint_ids].clone()
        targets[:, self.arm_action_indices] = arm_targets
        fraction = torch.clamp(target.gripper_closed_fraction, 0.0, 1.0)
        hand_targets = self.open_hand + fraction * (self.closed_hand - self.open_hand)
        targets[:, self.hand_action_indices] = hand_targets
        return self.action_term.actions_from_joint_targets(targets)

    def diagnostics(self) -> dict:
        palm_pos = self.robot.data.body_pos_w[0, self.palm_body_id] - self.env.scene.env_origins[0]
        return {
            "arm_joint_names": self.arm_joint_names,
            "arm_joint_position_rad": self.robot.data.joint_pos[0, self.arm_joint_ids].detach().cpu().tolist(),
            "arm_joint_target_rad": self.last_joint_targets[0].detach().cpu().tolist(),
            "left_ee_position_m": palm_pos.detach().cpu().tolist(),
            "max_joint_delta_rad": self.max_joint_delta,
            "max_torso_deviation_rad": self.max_torso_deviation,
            "max_tracking_error_rad": self.max_tracking_error,
            "ik_method": "bounded_dls_accumulated_command",
            "ik_mode": self.last_mode,
            "debug_samples": self.debug_samples,
        }


def make_left_arm_differential_ik(env, robot, _unused=False):
    return LeftArmDifferentialIKController(env, robot)


LEFT_ARM_DIFFERENTIAL_IK_CONTROLLER = ControllerDefinition(
    component_id="Controller-LeftArmDifferentialIK-v0",
    required_robot_capabilities=frozenset({"fixed_base", "upper_body_joint_control", "left_end_effector"}),
    factory=make_left_arm_differential_ik,
)
