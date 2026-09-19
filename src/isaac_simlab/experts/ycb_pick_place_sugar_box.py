"""Deterministic side-approach strategy for the YCB sugar box."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_slerp

from ..contracts import EndEffectorTarget, ExpertDefinition


PHASES = (
    "reset",
    "move_pregrasp",
    "approach",
    "close_gripper",
    "lift",
    "move_right",
    "lower",
    "open_gripper",
    "retreat",
    "done",
    "failed",
)


@dataclass(frozen=True)
class PhaseLimit:
    minimum_steps: int
    maximum_steps: int


LIMITS = {
    # The object is spawned exactly on its support surface.  Start the
    # clearance move immediately so the arm does not settle toward the box
    # before the first commanded motion.
    "reset": PhaseLimit(0, 0),
    "move_pregrasp": PhaseLimit(100, 300),
    "approach": PhaseLimit(90, 240),
    "close_gripper": PhaseLimit(75, 75),
    "lift": PhaseLimit(90, 200),
    "move_right": PhaseLimit(60, 150),
    "lower": PhaseLimit(80, 180),
    "open_gripper": PhaseLimit(45, 45),
    "retreat": PhaseLimit(80, 180),
    "done": PhaseLimit(10_000_000, 10_000_000),
    "failed": PhaseLimit(10_000_000, 10_000_000),
}

class YCBPickPlaceSugarBoxExpert:
    """Generate side-approach EE/gripper targets without implementing control."""

    # Mean of the three open fingertips in the palm frame. Local +X is the
    # hand's approach direction.
    OPEN_FINGER_CENTER_PALM = (0.09598306, -0.01849499, 0.01195028)
    # Grip the upper half of the upright box. The previous 0.025 m offset put
    # the fingertips visibly too low; raise both pregrasp and grasp by 3 cm.
    FINGER_CENTER_ABOVE_BOX_CENTER_M = 0.055
    PREGRASP_CLEARANCE_M = 0.07
    TARGET_WORLD_DELTA = (0.02, 0.0, 0.0)

    def __init__(self, env, robot_definition, object_definition):
        self.env = env
        self.definition = robot_definition
        self.robot = env.scene["robot"]
        self.sugar_box = env.scene["object"]
        palm_ids, _ = self.robot.find_bodies(
            [robot_definition.left_end_effector], preserve_order=True
        )
        self.palm_body_id = palm_ids[0]
        self.open_finger_center_palm = torch.tensor(
            self.OPEN_FINGER_CENTER_PALM, device=env.device
        ).unsqueeze(0)
        self.sugar_box_height_m = float(
            object_definition.metadata["sugar_box_dimensions_m"][2]
        )
        self.reset()

    @property
    def phase(self) -> str:
        return PHASES[self.phase_index]

    def _palm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.robot.data.body_pos_w[:, self.palm_body_id],
            self.robot.data.body_quat_w[:, self.palm_body_id],
        )

    def reset(self) -> None:
        palm_pos, palm_quat = self._palm_pose()
        self.phase_index = 0
        self.phase_step = 0
        self.total_step = 0
        self.failed = False
        self.failure_reason = None
        self.failed_target_pos = None
        self.last_position_error = float("inf")
        self.last_orientation_error = float("inf")
        self.phase_history = []
        self.phase_start_pos = palm_pos.clone()
        self.phase_start_quat = palm_quat.clone()
        self.target_pos = palm_pos.clone()
        self.target_quat = palm_quat.clone()
        self.gripper_start = 0.0
        self.gripper_target = 0.0
        self.initial_box_pos = self.sugar_box.data.root_pos_w.clone()
        self.pregrasp_pos = None
        self.grasp_pos = None
        self.grasp_quat = None
        self.travel_quat = None
        self.approach_direction = None
        self.box_top_z = None
        self.grasp_depth_below_box_top_m = None
        self.env.expert_phase = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )

    def _compute_grasp(self) -> None:
        palm_pos, palm_quat = self._palm_pose()
        horizontal_delta = self.sugar_box.data.root_pos_w[:, :2] - palm_pos[:, :2]
        horizontal_delta = horizontal_delta / torch.clamp(
            torch.linalg.vector_norm(horizontal_delta, dim=-1, keepdim=True), min=1.0e-6
        )
        self.approach_direction = torch.cat(
            (horizontal_delta, torch.zeros_like(horizontal_delta[:, :1])), dim=-1
        )
        # Use a level side-grasp orientation: local +X points horizontally at
        # the box and local +Z stays vertical. The previous reachable-pose
        # orientation tilted +X downward by about 39 degrees, causing the
        # fingertips to contact the table before reaching the box.
        desired_yaw = torch.atan2(horizontal_delta[:, 1], horizontal_delta[:, 0])
        zeros = torch.zeros_like(desired_yaw)
        self.grasp_quat = torch.stack(
            (torch.cos(0.5 * desired_yaw), zeros, zeros, torch.sin(0.5 * desired_yaw)),
            dim=-1,
        )
        self.travel_quat = self.grasp_quat.clone()
        finger_center = self.sugar_box.data.root_pos_w.clone()
        finger_center[:, 2] += self.FINGER_CENTER_ABOVE_BOX_CENTER_M
        finger_offset = quat_apply(self.grasp_quat, self.open_finger_center_palm)
        self.box_top_z = (
            self.sugar_box.data.root_pos_w[:, 2] + 0.5 * self.sugar_box_height_m
        )
        self.grasp_depth_below_box_top_m = self.box_top_z - finger_center[:, 2]
        self.grasp_pos = finger_center - finger_offset
        self.pregrasp_pos = (
            self.grasp_pos - self.PREGRASP_CLEARANCE_M * self.approach_direction
        )
        self.initial_box_pos = self.sugar_box.data.root_pos_w.clone()

    def _set_phase(self, phase: str, position, orientation, gripper: float) -> None:
        palm_pos, palm_quat = self._palm_pose()
        self.phase_history.append(
            {
                "phase": self.phase,
                "steps": self.phase_step,
                "sugar_box_position_m": (
                    self.sugar_box.data.root_pos_w[0] - self.env.scene.env_origins[0]
                ).detach().cpu().tolist(),
            }
        )
        self.phase_index = PHASES.index(phase)
        self.phase_step = 0
        self.phase_start_pos = palm_pos.clone()
        self.phase_start_quat = palm_quat.clone()
        self.target_pos = position.clone()
        self.target_quat = orientation.clone()
        self.gripper_start = self.gripper_target
        self.gripper_target = gripper
        self.env.expert_phase.fill_(self.phase_index)
        print(f"[sugar-box-expert] phase={phase} step={self.total_step}", flush=True)

    def _fail(self, reason: str) -> None:
        if self.failed:
            return
        self.failed = True
        self.failure_reason = reason
        self.failed_target_pos = self.target_pos.clone()
        palm_pos, palm_quat = self._palm_pose()
        self._set_phase("failed", palm_pos, palm_quat, self.gripper_target)
        print(f"[sugar-box-expert] FAILED: {reason}", flush=True)

    def _pose_ready(self) -> bool:
        palm_pos, palm_quat = self._palm_pose()
        pos_error = torch.linalg.vector_norm(palm_pos - self.target_pos, dim=-1)[0]
        rot_error = quat_error_magnitude(palm_quat, self.target_quat)[0]
        self.last_position_error = pos_error.item()
        self.last_orientation_error = rot_error.item()
        orientation_ready = rot_error < math.radians(15.0)
        return bool(pos_error < 0.025 and orientation_ready)

    def _advance_if_ready(self, phase: str, position, orientation, gripper: float) -> None:
        ready = self._pose_ready()
        timed_out = self.phase_step > LIMITS[self.phase].maximum_steps
        if self.phase_step < LIMITS[self.phase].minimum_steps and not timed_out:
            return
        if not (ready or timed_out):
            return
        if timed_out and not ready:
            self._fail(
                f"{self.phase} pose timeout (position={self.last_position_error:.3f} m, "
                f"orientation={math.degrees(self.last_orientation_error):.1f} deg)"
            )
            return
        self._set_phase(phase, position, orientation, gripper)

    def _transition(self) -> None:
        if self.phase in ("done", "failed"):
            return
        if not torch.isfinite(self.sugar_box.data.root_state_w).all():
            self._fail("sugar-box state became non-finite")
            return
        if self.sugar_box.data.root_pos_w[0, 2] < 0.57:
            self._fail("sugar box fell below the tabletop")
            return

        timed_out = self.phase_step > LIMITS[self.phase].maximum_steps
        if self.phase == "reset" and timed_out:
            self._compute_grasp()
            self._set_phase("move_pregrasp", self.pregrasp_pos, self.travel_quat, 0.0)
        elif self.phase == "move_pregrasp":
            self._advance_if_ready("approach", self.grasp_pos, self.grasp_quat, 0.0)
        elif self.phase == "approach":
            self._advance_if_ready("close_gripper", self.grasp_pos, self.grasp_quat, 1.0)
        elif self.phase == "close_gripper" and timed_out:
            palm_pos, palm_quat = self._palm_pose()
            lift = palm_pos + palm_pos.new_tensor([[0.0, 0.0, 0.12]])
            self._set_phase("lift", lift, palm_quat, 1.0)
        elif self.phase == "lift":
            ready = self._pose_ready()
            if ready or timed_out:
                rise = (
                    self.sugar_box.data.root_pos_w[0, 2] - self.initial_box_pos[0, 2]
                ).item()
                if rise < 0.045:
                    self._fail(f"grasp did not lift the sugar box (rise={rise:.3f} m)")
                elif timed_out and not ready:
                    self._fail("lift pose timeout")
                else:
                    palm_pos, palm_quat = self._palm_pose()
                    delta = palm_pos.new_tensor([self.TARGET_WORLD_DELTA])
                    self._set_phase("move_right", palm_pos + delta, palm_quat, 1.0)
        elif self.phase == "move_right":
            palm_pos, palm_quat = self._palm_pose()
            lower = self.grasp_pos + self.grasp_pos.new_tensor([self.TARGET_WORLD_DELTA])
            self._advance_if_ready("lower", lower, palm_quat, 1.0)
        elif self.phase == "lower":
            self._advance_if_ready("open_gripper", self.target_pos, self.target_quat, 0.0)
        elif self.phase == "open_gripper" and timed_out:
            palm_pos, palm_quat = self._palm_pose()
            retreat = palm_pos + palm_pos.new_tensor([[0.0, 0.0, 0.16]])
            self._set_phase("retreat", retreat, palm_quat, 0.0)
        elif self.phase == "retreat":
            palm_pos, palm_quat = self._palm_pose()
            self._advance_if_ready("done", palm_pos, palm_quat, 0.0)

    def compute(self, step: int) -> EndEffectorTarget:
        del step
        duration = max(LIMITS[self.phase].minimum_steps, 1)
        alpha = min(1.0, (self.phase_step + 1) / duration)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        position = self.phase_start_pos + alpha * (self.target_pos - self.phase_start_pos)
        orientation = torch.stack(
            [
                quat_slerp(
                    self.phase_start_quat[index], self.target_quat[index].clone(), alpha
                )
                for index in range(self.env.num_envs)
            ]
        )
        gripper = self.gripper_start + alpha * (self.gripper_target - self.gripper_start)
        gripper = position.new_full((self.env.num_envs, 1), gripper)
        if self.phase not in ("done", "failed"):
            self.phase_step += 1
            self.total_step += 1
            self._transition()
        return EndEffectorTarget(position, orientation, gripper)

    def diagnostics(self) -> dict:
        origin = self.env.scene.env_origins[0]
        palm_pos, palm_quat = self._palm_pose()
        gripper_axis = quat_apply(
            self.grasp_quat if self.grasp_quat is not None else palm_quat,
            palm_pos.new_tensor([[1.0, 0.0, 0.0]]),
        )
        return {
            "phase": self.phase,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "total_steps": self.total_step,
            "sugar_box_position_m": (
                self.sugar_box.data.root_pos_w[0] - origin
            ).detach().cpu().tolist(),
            "left_ee_position_m": (palm_pos[0] - origin).detach().cpu().tolist(),
            "target_left_ee_position_m": (self.target_pos[0] - origin).detach().cpu().tolist(),
            "failed_target_left_ee_position_m": None
            if self.failed_target_pos is None
            else (self.failed_target_pos[0] - origin).detach().cpu().tolist(),
            "pregrasp_left_ee_position_m": None
            if self.pregrasp_pos is None
            else (self.pregrasp_pos[0] - origin).detach().cpu().tolist(),
            "grasp_left_ee_position_m": None
            if self.grasp_pos is None
            else (self.grasp_pos[0] - origin).detach().cpu().tolist(),
            "box_top_height_m": None
            if self.box_top_z is None
            else (self.box_top_z[0] - origin[2]).item(),
            "grasp_depth_below_box_top_m": None
            if self.grasp_depth_below_box_top_m is None
            else self.grasp_depth_below_box_top_m[0].item(),
            "pregrasp_clearance_m": self.PREGRASP_CLEARANCE_M,
            "approach_direction_world": None
            if self.approach_direction is None
            else self.approach_direction[0].detach().cpu().tolist(),
            "gripper_forward_axis_world": gripper_axis[0].detach().cpu().tolist(),
            "position_error_m": torch.linalg.vector_norm(
                palm_pos - self.target_pos, dim=-1
            )[0].item(),
            "orientation_error_rad": quat_error_magnitude(
                palm_quat, self.target_quat
            )[0].item(),
            "last_command_position_error_m": self.last_position_error,
            "last_command_orientation_error_rad": self.last_orientation_error,
            "target_displacement_world_m": list(self.TARGET_WORLD_DELTA),
            "robot_right_definition": "base -Y = world +X",
            "phase_history": self.phase_history,
        }


def make_sugar_box_expert(env, robot, objects):
    return YCBPickPlaceSugarBoxExpert(env, robot, objects)


YCB_PICK_PLACE_SUGAR_BOX_EXPERT = ExpertDefinition(
    component_id="Expert-YCBPickPlaceSugarBox-v0",
    required_world_capabilities=frozenset(
        {"support_surface", "object_spawn_region", "target_region"}
    ),
    required_robot_capabilities=frozenset(
        {"fixed_base", "upper_body_joint_control", "left_end_effector"}
    ),
    required_object_capabilities=frozenset({"manipulation_object", "sugar_box"}),
    factory=make_sugar_box_expert,
)
