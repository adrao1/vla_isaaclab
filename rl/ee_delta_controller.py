"""EE-delta action controller for G1 RL.

Maps a low-dimensional RL action

    [dx, dy, dz, droll, dpitch, dyaw, gripper]

in normalized [-1, 1] coordinates to the existing environment's 43-D
normalized joint-position action.

The controller reuses the same bounded-DLS IK and joint-limit mapping as
the scripted sugar-box policy, but does not contain any task strategy or
phase logic.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, quat_mul

from vla_isaaclab.envs.common import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_OPEN_JOINT_POSITIONS,
    WAIST_JOINT_NAMES,
)
from vla_isaaclab.policies.bounded_ik import bounded_dls
from vla_isaaclab.policies.joint_limits import (
    joint_targets_to_normalized,
    resolved_action_joints,
)


class EEDeltaController:
    """Convert normalized EE-delta RL actions into 43-D G1 actions.

    Input action layout:

        0: dx
        1: dy
        2: dz
        3: droll
        4: dpitch
        5: dyaw
        6: gripper

    All inputs are expected in [-1, 1].

    Cartesian translation is interpreted in the world frame.

    Rotation commands are small incremental rotations applied to the
    current palm orientation.

    Gripper convention:

        -1 -> fully open
        +1 -> fully closed
    """

    ACTION_DIM = 7

    def __init__(
        self,
        env,
        position_scale: float = 0.02,
        rotation_scale: float = math.radians(5.0),
    ):
        self.env = env
        self.robot = env.scene["robot"]

        self.position_scale = position_scale
        self.rotation_scale = rotation_scale

        # Resolve the environment's complete 43-D action ordering.
        (
            self.action_term,
            self.joint_names,
            self.action_joint_ids,
        ) = resolved_action_joints(env)

        self.name_to_action_index = {
            name: index for index, name in enumerate(self.joint_names)
        }

        # Match the joint chain used by the existing scripted controller:
        # waist yaw + seven left-arm joints.
        self.arm_joint_names = [
            WAIST_JOINT_NAMES[0],
            *LEFT_ARM_JOINT_NAMES,
        ]

        self.arm_joint_ids, found = self.robot.find_joints(
            self.arm_joint_names,
            preserve_order=True,
        )

        if list(found) != self.arm_joint_names:
            raise RuntimeError(
                "Left-arm joint mismatch: "
                f"expected {self.arm_joint_names}, found {found}"
            )

        self.arm_action_indices = [
            self.name_to_action_index[name]
            for name in self.arm_joint_names
        ]

        # Dex3 left hand.
        self.hand_joint_names = list(LEFT_HAND_JOINT_NAMES)

        self.hand_joint_ids, found = self.robot.find_joints(
            self.hand_joint_names,
            preserve_order=True,
        )

        if list(found) != self.hand_joint_names:
            raise RuntimeError(
                "Left-hand joint mismatch: "
                f"expected {self.hand_joint_names}, found {found}"
            )

        self.hand_action_indices = [
            self.name_to_action_index[name]
            for name in self.hand_joint_names
        ]

        self.open_hand = torch.tensor(
            LEFT_HAND_OPEN_JOINT_POSITIONS,
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)

        nominal_closed_hand = torch.tensor(
            LEFT_HAND_CLOSED_JOINT_POSITIONS,
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)

        # Match the existing sugar-box scripted controller.
        self.grip_closure_scale = 1.10
        self.closed_hand = self.open_hand + self.grip_closure_scale * (
            nominal_closed_hand - self.open_hand
        )

        # Palm body / Jacobian indexing.
        palm_ids, _ = self.robot.find_bodies(
            [LEFT_END_EFFECTOR],
            preserve_order=True,
        )

        self.palm_body_id = palm_ids[0]

        self.palm_jacobian_id = (
            self.palm_body_id - 1
            if self.robot.is_fixed_base
            else self.palm_body_id
        )

        # Keep the same proven IK parameters as YCBSugarBoxScriptedPolicy.
        self.max_joint_delta = 0.025
        self.max_tracking_error = 0.35
        self.max_waist_yaw_deviation = math.radians(25.0)

        self.ik_gain = 0.15
        self.ik_damping = 0.04
        self.orientation_weight = 0.20

        self.last_joint_targets = self.robot.data.joint_pos[
            :, self.arm_joint_ids
        ].clone()

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Reset persistent IK targets after environment reset.

        Args:
            env_ids: Optional indices of environments that were reset.
                If None, reset controller state for all environments.
        """

        current_arm = self.robot.data.joint_pos[
            :, self.arm_joint_ids
        ]

        if env_ids is None:
            self.last_joint_targets = current_arm.clone()
            return

        env_ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.env.device,
        )

        if env_ids.numel() == 0:
            return

        self.last_joint_targets[env_ids] = current_arm[env_ids]

    def _prepare_action(self, action: torch.Tensor) -> torch.Tensor:
        """Validate and normalize the incoming RL action shape."""

        action = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=self.env.device,
        )

        # Allow a single non-batched action for num_envs == 1.
        if action.ndim == 1:
            action = action.unsqueeze(0)

        expected_shape = (self.env.num_envs, self.ACTION_DIM)

        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"Expected EE-delta action shape {expected_shape}, "
                f"got {tuple(action.shape)}"
            )

        return torch.clamp(action, -1.0, 1.0)

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        """Convert an RL EE-delta action to the environment's 43-D action."""

        action = self._prepare_action(action)

        # --------------------------------------------------------------
        # 1. Decode normalized RL action.
        # --------------------------------------------------------------

        delta_position = (
            action[:, 0:3] * self.position_scale
        )

        delta_rpy = (
            action[:, 3:6] * self.rotation_scale
        )

        # Map normalized [-1, 1] gripper action to [0, 1].
        gripper_fraction = torch.clamp(
            0.5 * (action[:, 6:7] + 1.0),
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # 2. Construct target EE pose from the measured palm pose.
        # --------------------------------------------------------------

        palm_pos = self.robot.data.body_pos_w[
            :, self.palm_body_id
        ]

        palm_quat = self.robot.data.body_quat_w[
            :, self.palm_body_id
        ]

        target_position = palm_pos + delta_position

        delta_quat = quat_from_euler_xyz(
            delta_rpy[:, 0],
            delta_rpy[:, 1],
            delta_rpy[:, 2],
        )

        target_orientation = quat_mul(
            delta_quat,
            palm_quat,
        )

        # --------------------------------------------------------------
        # 3. Compute pose error.
        # --------------------------------------------------------------

        position_error, rotation_error = compute_pose_error(
            palm_pos,
            palm_quat,
            target_position,
            target_orientation,
            rot_error_type="axis_angle",
        )

        error = torch.cat(
            (
                position_error,
                self.orientation_weight * rotation_error,
            ),
            dim=-1,
        )

        # --------------------------------------------------------------
        # 4. Extract Jacobian for waist yaw + left arm.
        # --------------------------------------------------------------

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :,
            self.palm_jacobian_id,
            :,
            self.arm_joint_ids,
        ].clone()

        # Scale orientation rows consistently with the pose error.
        jacobian[:, 3:] *= self.orientation_weight

        current_arm = self.robot.data.joint_pos[
            :, self.arm_joint_ids
        ]

        # --------------------------------------------------------------
        # 5. Compute allowed joint-target region.
        # --------------------------------------------------------------

        limits = self.robot.data.soft_joint_pos_limits[
            :, self.arm_joint_ids
        ]

        lower = limits[..., 0].clone()
        upper = limits[..., 1].clone()

        # Restrict waist yaw to +/- 25 degrees from its default pose.
        waist_yaw_center = self.robot.data.default_joint_pos[
            :, self.arm_joint_ids[0]
        ]

        lower[:, 0] = torch.maximum(
            lower[:, 0],
            waist_yaw_center - self.max_waist_yaw_deviation,
        )

        upper[:, 0] = torch.minimum(
            upper[:, 0],
            waist_yaw_center + self.max_waist_yaw_deviation,
        )

        # Anti-windup / tracking window, matching the existing controller.
        reference = torch.clamp(
            self.last_joint_targets,
            current_arm - self.max_tracking_error,
            current_arm + self.max_tracking_error,
        )

        reference = torch.clamp(
            reference,
            lower,
            upper,
        )

        delta_lower = torch.maximum(
            lower - reference,
            torch.full_like(
                reference,
                -self.max_joint_delta,
            ),
        )

        delta_upper = torch.minimum(
            upper - reference,
            torch.full_like(
                reference,
                self.max_joint_delta,
            ),
        )

        delta_lower = torch.maximum(
            delta_lower,
            current_arm
            - self.max_tracking_error
            - reference,
        )

        delta_upper = torch.minimum(
            delta_upper,
            current_arm
            + self.max_tracking_error
            - reference,
        )

        # Physical limits take precedence.
        delta_lower = torch.minimum(
            delta_lower,
            torch.zeros_like(delta_lower),
        )

        delta_upper = torch.maximum(
            delta_upper,
            torch.zeros_like(delta_upper),
        )

        # --------------------------------------------------------------
        # 6. Bounded damped-least-squares IK.
        # --------------------------------------------------------------

        correction = bounded_dls(
            jacobian,
            self.ik_gain * error,
            delta_lower,
            delta_upper,
            self.ik_damping,
        )

        raw_targets = reference + correction

        arm_targets = torch.clamp(
            raw_targets,
            lower,
            upper,
        )

        self.last_joint_targets = arm_targets.clone()

        # --------------------------------------------------------------
        # 7. Build complete physical 43-joint target.
        #
        # Anything not controlled by RL stays at the robot's default pose.
        # --------------------------------------------------------------

        targets = self.robot.data.default_joint_pos[
            :, self.action_joint_ids
        ].clone()

        targets[
            :, self.arm_action_indices
        ] = arm_targets

        hand_targets = (
            self.open_hand
            + gripper_fraction
            * (self.closed_hand - self.open_hand)
        )

        targets[
            :, self.hand_action_indices
        ] = hand_targets

        # --------------------------------------------------------------
        # 8. Convert physical joint positions to the normalized 43-D
        #    action expected by JointPositionToLimitsActionCfg.
        # --------------------------------------------------------------

        normalized_action = joint_targets_to_normalized(
            self.robot,
            self.action_joint_ids,
            targets,
        )

        return normalized_action

    def diagnostics(self) -> dict:
        """Return useful state for debugging the controller."""

        palm_pos = (
            self.robot.data.body_pos_w[
                0, self.palm_body_id
            ]
            - self.env.scene.env_origins[0]
        )

        return {
            "rl_action_dimension": self.ACTION_DIM,
            "position_scale_m": self.position_scale,
            "rotation_scale_rad": self.rotation_scale,
            "ik_joint_names": self.arm_joint_names,
            "hand_joint_names": self.hand_joint_names,
            "left_ee_position_m": (
                palm_pos.detach().cpu().tolist()
            ),
            "max_joint_delta_rad": self.max_joint_delta,
            "max_tracking_error_rad": self.max_tracking_error,
            "max_waist_yaw_deviation_rad": (
                self.max_waist_yaw_deviation
            ),
            "ik_gain": self.ik_gain,
            "ik_damping": self.ik_damping,
            "orientation_weight": self.orientation_weight,
        }
