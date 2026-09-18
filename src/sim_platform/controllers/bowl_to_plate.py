"""Differential-IK state machine for a physical left-hand bowl transfer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from ..contracts import ControllerDefinition


PHASES = (
    "settle",
    "pre_grasp",
    "approach",
    "close_hand",
    "lift",
    "transfer",
    "lower",
    "release",
    "retreat",
    "done",
)


@dataclass(frozen=True)
class PhaseLimits:
    minimum: int
    maximum: int
    position_tolerance: float = 0.035


LIMITS = {
    "settle": PhaseLimits(45, 45),
    "pre_grasp": PhaseLimits(60, 180, 0.045),
    "approach": PhaseLimits(35, 100, 0.035),
    "close_hand": PhaseLimits(55, 55),
    "lift": PhaseLimits(50, 140, 0.045),
    "transfer": PhaseLimits(70, 180, 0.05),
    "lower": PhaseLimits(50, 130, 0.04),
    "release": PhaseLimits(45, 45),
    "retreat": PhaseLimits(55, 140, 0.05),
    "done": PhaseLimits(10_000_000, 10_000_000),
}


class LeftHandBowlToPlateController:
    """Move the left palm with DLS IK and actuate the physical G1 fingers."""

    def __init__(self, env, robot_definition):
        self.env = env
        self.definition = robot_definition
        self.robot = env.scene["robot"]
        self.bowl = env.scene["object"]
        self.plate = env.scene["goal"]
        self.stop_after_phase = getattr(env, "controller_stop_after_phase", None)
        self.debug_continue = bool(getattr(env, "controller_debug_continue", False))
        self.action_term = env.action_manager.get_term("upper_body")
        self.joint_names = self.action_term.joint_names
        self.action_name_to_index = {name: index for index, name in enumerate(self.joint_names)}
        self.action_joint_ids, _ = self.robot.find_joints(self.joint_names, preserve_order=True)

        # Solve translation with the five arm joints.  Leaving the waist inside an
        # unconstrained position-only solve lets DLS spin it toward a singular
        # solution and turn the complete upper body away from the table.
        self.ik_joint_names = list(robot_definition.left_arm_joint_names)
        self.ik_joint_ids, found = self.robot.find_joints(self.ik_joint_names, preserve_order=True)
        if list(found) != self.ik_joint_names:
            raise RuntimeError(f"IK joint mismatch: expected {self.ik_joint_names}, found {found}")
        self.ik_action_indices = [self.action_name_to_index[name] for name in self.ik_joint_names]
        self.waist_action_index = self.action_name_to_index[robot_definition.waist_joint_names[0]]
        body_ids, _ = self.robot.find_bodies([robot_definition.left_end_effector], preserve_order=True)
        self.palm_body_id = body_ids[0]
        self.palm_jacobian_id = self.palm_body_id - 1 if self.robot.is_fixed_base else self.palm_body_id

        # Position-only differential IK is redundant for this five-joint arm.
        # A posture term keeps the elbow bent instead of allowing the minimum-
        # norm solution to straighten the arm at the edge of its workspace.
        self.ik_damping = 0.06
        self.nullspace_gain = 0.35
        # The robot definition supplies a raised reset pose on the verified IK
        # branch.  It is a posture preference only; the Cartesian target remains
        # authoritative in the DLS solve.
        self.ik_posture = self.robot.data.default_joint_pos[:, self.ik_joint_ids].clone()
        self.pregrasp_seed = self.ik_posture.clone()
        self.open_hand = {
            "left_zero_joint": 0.0,
            "left_one_joint": 1.0,
            "left_two_joint": 0.18,
            "left_three_joint": 0.10,
            "left_four_joint": -0.12,
            "left_five_joint": 0.10,
            "left_six_joint": -0.12,
        }
        self.closed_hand = {
            "left_zero_joint": 0.0,
            "left_one_joint": 0.10,
            "left_two_joint": 1.45,
            "left_three_joint": -1.35,
            "left_four_joint": -1.45,
            "left_five_joint": -1.35,
            "left_six_joint": -1.45,
        }
        self.reset()

    def reset(self) -> None:
        self.phase_index = 0
        self.phase_step = 0
        self.total_step = 0
        self.failed = False
        self.failure_reason = None
        self.debug_warnings = []
        self.phase_target_w = None
        self.phase_start_w = self._palm_position().clone()
        self.phase_start_ik = self.robot.data.joint_pos[:, self.ik_joint_ids].clone()
        self.last_target_w = self.phase_start_w.clone()
        self.last_desired_ik = self.robot.data.joint_pos[:, self.ik_joint_ids].clone()
        self.last_position_error = float("inf")
        self.waist_command = torch.zeros(self.env.num_envs, device=self.env.device)
        self.contact_force_max_by_phase = {phase: 0.0 for phase in PHASES}
        self.phase_history = []
        self.env.bowl_to_plate_phase = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        self.env.bowl_to_plate_success_counter = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        print("[bowl-controller] phase=settle", flush=True)

    @property
    def phase(self) -> str:
        return PHASES[self.phase_index]

    def _palm_position(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.palm_body_id]

    def _set_phase(self, phase: str) -> None:
        if self.phase_target_w is not None:
            self.phase_history.append(
                {
                    "phase": self.phase,
                    "steps": self.phase_step,
                    "final_palm_error_m": self.last_position_error,
                    "palm_position_m": (
                        self._palm_position()[0] - self.env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "target_position_m": (
                        self.phase_target_w[0] - self.env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "ik_joint_position_rad": self.robot.data.joint_pos[
                        0, self.ik_joint_ids
                    ].detach().cpu().tolist(),
                }
            )
        self.phase_index = PHASES.index(phase)
        self.phase_step = 0
        self.phase_start_w = self._palm_position().clone()
        self.phase_start_ik = self.robot.data.joint_pos[:, self.ik_joint_ids].clone()
        self.phase_target_w = self._target_for_phase(phase)
        self.env.bowl_to_plate_phase.fill_(self.phase_index)
        bowl_pos = (self.bowl.data.root_pos_w[0] - self.env.scene.env_origins[0]).detach().cpu().tolist()
        print(
            f"[bowl-controller] phase={phase} step={self.total_step} "
            f"bowl={[round(v, 4) for v in bowl_pos]}",
            flush=True,
        )

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.failure_reason = reason
        print(f"[bowl-controller] FAILED: {reason}", flush=True)
        self._set_phase("done")

    def _target_for_phase(self, phase: str) -> torch.Tensor:
        palm = self._palm_position().clone()
        bowl = self.bowl.data.root_pos_w.clone()
        plate = self.plate.data.root_pos_w.clone()
        if phase == "pre_grasp":
            return bowl + bowl.new_tensor((0.0, -0.26, 0.13))
        if phase == "approach":
            return bowl + bowl.new_tensor((0.0, -0.16, 0.11))
        if phase == "close_hand":
            return palm
        if phase == "lift":
            return palm + palm.new_tensor((0.0, 0.0, 0.10))
        if phase == "transfer":
            return plate + plate.new_tensor((0.0, -0.035, 0.18))
        if phase == "lower":
            return plate + plate.new_tensor((0.0, -0.025, 0.115))
        if phase == "release":
            return palm
        if phase == "retreat":
            return palm + palm.new_tensor((0.0, -0.13, 0.16))
        return palm

    def _interpolated_target(self) -> torch.Tensor:
        if self.phase_target_w is None:
            return self._palm_position()
        limits = LIMITS[self.phase]
        duration = max(limits.minimum, 1)
        alpha = min(1.0, (self.phase_step + 1) / duration)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return self.phase_start_w + alpha * (self.phase_target_w - self.phase_start_w)

    def _hand_targets(self) -> dict[str, float]:
        debug_lift_hold = self.phase == "done" and self.stop_after_phase in ("close_hand", "lift")
        if self.phase in ("close_hand", "lift", "transfer", "lower") or debug_lift_hold:
            return self.closed_hand
        return self.open_hand

    def _warn_and_continue(self, message: str, next_phase: str) -> None:
        self.debug_warnings.append(message)
        print(f"[bowl-controller] DEBUG CONTINUE: {message}", flush=True)
        self._set_phase(next_phase)

    def _complete_phase(self, next_phase: str) -> None:
        if self.stop_after_phase == self.phase:
            self._set_phase("done")
        else:
            self._set_phase(next_phase)

    def _waist_target(self) -> float:
        if self.phase in ("transfer", "lower", "release"):
            return -0.12
        return 0.0

    def _update_contact_diagnostics(self) -> None:
        phase_max = self.contact_force_max_by_phase[self.phase]
        for name, sensor in self.env.scene.sensors.items():
            if not name.startswith("left_fingertip_contact_") or sensor.data.force_matrix_w is None:
                continue
            force = torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1).max().item()
            phase_max = max(phase_max, force)
        self.contact_force_max_by_phase[self.phase] = phase_max

    def _build_action(self, target_w: torch.Tensor) -> torch.Tensor:
        self.last_target_w = target_w.clone()
        palm_position_w = self.robot.data.body_pos_w[:, self.palm_body_id]
        # The finite-difference probe in scripts/probe_g1_jacobian.py confirms
        # that this G1 PhysX view reports palm translation rows in world axes.
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.palm_jacobian_id, :3, self.ik_joint_ids
        ]
        current_ik = self.robot.data.joint_pos[:, self.ik_joint_ids]
        position_error = (target_w - palm_position_w).unsqueeze(-1)
        identity_xyz = torch.eye(3, device=self.env.device).unsqueeze(0)
        inverse = torch.linalg.solve(
            jacobian @ jacobian.transpose(1, 2) + self.ik_damping**2 * identity_xyz,
            position_error,
        )
        pseudo_inverse = jacobian.transpose(1, 2) @ inverse
        task_delta = pseudo_inverse.squeeze(-1)

        damped_pinv = jacobian.transpose(1, 2) @ torch.linalg.inv(
            jacobian @ jacobian.transpose(1, 2) + self.ik_damping**2 * identity_xyz
        )
        identity_q = torch.eye(len(self.ik_joint_ids), device=self.env.device).unsqueeze(0)
        nullspace = identity_q - damped_pinv @ jacobian
        posture_delta = (nullspace @ (self.ik_posture - current_ik).unsqueeze(-1)).squeeze(-1)
        joint_delta = task_delta + self.nullspace_gain * posture_delta
        desired_ik = current_ik + torch.clamp(joint_delta, min=-0.10, max=0.10)
        if self.phase == "pre_grasp" and self.phase_step < 90:
            alpha = min(1.0, (self.phase_step + 1) / 90.0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            desired_ik = self.phase_start_ik + alpha * (self.pregrasp_seed - self.phase_start_ik)
        self.last_desired_ik = desired_ik.clone()

        physical_targets = self.robot.data.default_joint_pos[:, self.action_joint_ids].clone()
        physical_targets[:, self.ik_action_indices] = desired_ik
        waist_goal = torch.full_like(self.waist_command, self._waist_target())
        self.waist_command += torch.clamp(waist_goal - self.waist_command, min=-0.015, max=0.015)
        physical_targets[:, self.waist_action_index] = self.waist_command
        for name, value in self._hand_targets().items():
            physical_targets[:, self.action_name_to_index[name]] = value
        return self.action_term.actions_from_joint_targets(physical_targets)

    def _maybe_transition(self) -> None:
        limits = LIMITS[self.phase]
        if self.phase_target_w is not None:
            self.last_position_error = torch.linalg.vector_norm(
                self._palm_position() - self.phase_target_w, dim=-1
            )[0].item()
        ready = self.phase_step >= limits.minimum and self.last_position_error < limits.position_tolerance
        timed_out = self.phase_step >= limits.maximum

        if self.phase == "settle" and timed_out:
            self._set_phase("pre_grasp")
        elif self.phase == "pre_grasp" and (ready or timed_out):
            if timed_out and self.last_position_error > 0.08:
                message = f"pre_grasp unreachable, palm error={self.last_position_error:.3f} m"
                if self.debug_continue:
                    self._warn_and_continue(message, "approach")
                else:
                    self._fail(message)
            else:
                self._complete_phase("approach")
        elif self.phase == "approach" and (ready or timed_out):
            if timed_out and self.last_position_error > 0.065:
                message = f"approach unreachable, palm error={self.last_position_error:.3f} m"
                if self.debug_continue:
                    self._warn_and_continue(message, "close_hand")
                else:
                    self._fail(message)
            else:
                self._complete_phase("close_hand")
        elif self.phase == "close_hand" and timed_out:
            self._complete_phase("lift")
        elif self.phase == "lift" and (ready or timed_out):
            bowl_height = self.bowl.data.root_pos_w[0, 2].item()
            support = float(self.env.cfg.scenario_metadata["support_height"])
            if bowl_height < support + 0.045:
                message = f"bowl did not lift, bowl z={bowl_height:.3f} m"
                if self.debug_continue and self.stop_after_phase == "lift":
                    self._warn_and_continue(message, "done")
                else:
                    self._fail(message)
            else:
                self._complete_phase("transfer")
        elif self.phase == "transfer" and (ready or timed_out):
            support = float(self.env.cfg.scenario_metadata["support_height"])
            if self.bowl.data.root_pos_w[0, 2].item() < support + 0.035:
                self._fail("bowl was lost during transfer")
            elif timed_out and self.last_position_error > 0.08:
                self._fail(f"transfer target unreachable, palm error={self.last_position_error:.3f} m")
            else:
                self._set_phase("lower")
        elif self.phase == "lower" and (ready or timed_out):
            self._set_phase("release")
        elif self.phase == "release" and timed_out:
            self._set_phase("retreat")
        elif self.phase == "retreat" and (ready or timed_out):
            self._set_phase("done")

    def compute(self, step: int) -> torch.Tensor:
        if step == 0 and self.total_step != 0:
            self.reset()
        self.env.bowl_to_plate_phase.fill_(self.phase_index)
        self._update_contact_diagnostics()
        target = self._interpolated_target()
        action = self._build_action(target)
        if self.phase != "settle" and self.phase_step % 30 == 0:
            palm = (self._palm_position()[0] - self.env.scene.env_origins[0]).detach().cpu().tolist()
            target_local = (target[0] - self.env.scene.env_origins[0]).detach().cpu().tolist()
            print(
                f"[bowl-controller] phase={self.phase} phase_step={self.phase_step} "
                f"palm={[round(v, 3) for v in palm]} target={[round(v, 3) for v in target_local]} "
                f"q={[round(v, 3) for v in self.robot.data.joint_pos[0, self.ik_joint_ids].detach().cpu().tolist()]} "
                f"q_target={[round(v, 3) for v in self.last_desired_ik[0].detach().cpu().tolist()]}",
                flush=True,
            )
        self.phase_step += 1
        self.total_step += 1
        self._maybe_transition()
        return action

    def diagnostics(self) -> dict:
        origin = self.env.scene.env_origins[0]
        return {
            "phase": self.phase,
            "phase_index": self.phase_index,
            "phase_step": self.phase_step,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "debug_warnings": self.debug_warnings,
            "stop_after_phase": self.stop_after_phase,
            "last_palm_position_error_m": self.last_position_error,
            "palm_position_m": (self._palm_position()[0] - origin).detach().cpu().tolist(),
            "bowl_position_m": (self.bowl.data.root_pos_w[0] - origin).detach().cpu().tolist(),
            "plate_position_m": (self.plate.data.root_pos_w[0] - origin).detach().cpu().tolist(),
            "target_position_m": (self.last_target_w[0] - origin).detach().cpu().tolist(),
            "ik_joint_names": self.ik_joint_names,
            "ik_joint_position_rad": self.robot.data.joint_pos[0, self.ik_joint_ids].detach().cpu().tolist(),
            "ik_joint_target_rad": self.last_desired_ik[0].detach().cpu().tolist(),
            "bowl_contact_force_max_n_by_phase": self.contact_force_max_by_phase,
            "phase_history": self.phase_history,
        }


def make_bowl_to_plate(env, robot, _unused=False):
    return LeftHandBowlToPlateController(env, robot)


BOWL_TO_PLATE_CONTROLLER = ControllerDefinition(
    component_id="Controller-LeftHandBowlToPlate-v0",
    required_robot_capabilities=frozenset(
        {"fixed_base", "upper_body_joint_control", "left_end_effector"}
    ),
    factory=make_bowl_to_plate,
)
