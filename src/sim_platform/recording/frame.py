"""Scenario-independent frame capture for dataset writers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


POSE_NAMES = ("position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z")


@dataclass(frozen=True)
class FrameSnapshot:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    environment_state: np.ndarray
    rgb: np.ndarray
    control_phase: np.ndarray


class ScenarioFrameAdapter:
    """Capture one environment using stable, robot-facing data semantics."""

    def __init__(self, env, bundle):
        if env.num_envs != 1:
            raise ValueError("LeRobot recording currently supports exactly one environment")
        self.env = env
        self.bundle = bundle
        self.robot = env.scene["robot"]
        self.action_term = env.action_manager.get_term("upper_body")
        self.joint_names = list(self.action_term.joint_names)
        self.joint_ids, _ = self.robot.find_joints(self.joint_names, preserve_order=True)

        self.body_entries = []
        for role, body_name in (
            ("left_end_effector", bundle.robot.left_end_effector),
            ("right_end_effector", bundle.robot.right_end_effector),
        ):
            body_ids, _ = self.robot.find_bodies([body_name], preserve_order=True)
            self.body_entries.append((role, body_ids[0]))

        self.rigid_object_names = sorted(env.scene.rigid_objects)
        self.environment_state_names = []
        for role, _ in self.body_entries:
            self.environment_state_names.extend(f"{role}.{axis}" for axis in POSE_NAMES)
        for name in self.rigid_object_names:
            self.environment_state_names.extend(f"{name}.{axis}" for axis in POSE_NAMES)
            self.environment_state_names.extend(
                f"{name}.{axis}"
                for axis in (
                    "linear_velocity.x", "linear_velocity.y", "linear_velocity.z",
                    "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
                )
            )

        rgb = env.scene["camera"].data.output["rgb"][0, ..., :3]
        self.image_shape = tuple(int(value) for value in rgb.shape)

    @property
    def features(self) -> dict:
        state_shape = (len(self.joint_names),)
        return {
            "observation.state": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "observation.velocity": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "action": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "sim.action.normalized": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (len(self.environment_state_names),),
                "names": self.environment_state_names,
            },
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
            "next.success": {"dtype": "bool", "shape": (1,), "names": None},
            "sim.seed": {"dtype": "int64", "shape": (1,), "names": None},
            "control.phase": {"dtype": "int64", "shape": (1,), "names": ["phase_index"]},
            "observation.images.front": {
                "dtype": "video",
                "shape": (3, self.image_shape[0], self.image_shape[1]),
                "names": ["channels", "height", "width"],
            },
        }

    @staticmethod
    def _numpy(tensor, dtype=np.float32):
        return tensor.detach().cpu().numpy().astype(dtype, copy=True)

    def capture(self) -> FrameSnapshot:
        origin = self.env.scene.env_origins[0]
        state_parts = []
        for _, body_id in self.body_entries:
            position = self.robot.data.body_pos_w[0, body_id] - origin
            orientation = self.robot.data.body_quat_w[0, body_id]
            state_parts.append(self._numpy(torch.cat((position, orientation))))
        for name in self.rigid_object_names:
            rigid_object = self.env.scene[name]
            position = rigid_object.data.root_pos_w[0] - origin
            orientation = rigid_object.data.root_quat_w[0]
            state_parts.append(
                self._numpy(
                    torch.cat(
                        (
                            position,
                            orientation,
                            rigid_object.data.root_lin_vel_w[0],
                            rigid_object.data.root_ang_vel_w[0],
                        )
                    )
                )
            )

        rgb = self.env.scene["camera"].data.output["rgb"][0, ..., :3]
        return FrameSnapshot(
            joint_position=self._numpy(self.robot.data.joint_pos[0, self.joint_ids]),
            joint_velocity=self._numpy(self.robot.data.joint_vel[0, self.joint_ids]),
            environment_state=np.concatenate(state_parts).astype(np.float32, copy=False),
            rgb=self._numpy(rgb, dtype=np.uint8),
            control_phase=np.asarray(
                [int(getattr(self.env, "bowl_to_plate_phase", torch.zeros(1, device=self.env.device))[0].item())],
                dtype=np.int64,
            ),
        )

    def complete_frame(self, snapshot, reward, terminated, timed_out, horizon, task, seed) -> dict:
        success = bool(terminated[0].item())
        done = success or bool(timed_out[0].item()) or horizon
        return {
            "observation.state": snapshot.joint_position,
            "observation.velocity": snapshot.joint_velocity,
            "action": self._numpy(self.action_term.joint_position_targets[0]),
            "sim.action.normalized": self._numpy(self.action_term.raw_actions[0]),
            "observation.environment_state": snapshot.environment_state,
            "next.reward": np.asarray([reward[0].item()], dtype=np.float32),
            "next.done": np.asarray([done], dtype=np.bool_),
            "next.success": np.asarray([success], dtype=np.bool_),
            "sim.seed": np.asarray([seed], dtype=np.int64),
            "control.phase": snapshot.control_phase,
            "observation.images.front": snapshot.rgb,
            "task": task,
        }
