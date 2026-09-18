#!/usr/bin/env python3
"""Compare G1 palm Jacobian columns with finite-difference joint motion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(PROJECT / "configs/ycb.python.headless.kit")
args.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
app = AppLauncher(args).app

import torch

from isaaclab.utils.math import quat_apply

from sim_platform import ScenarioSelection, register_defaults
from sim_platform.runtime import ScenarioEnv
from sim_platform.scenario import compose_scenario


def main() -> None:
    register_defaults()
    bundle = compose_scenario(
        ScenarioSelection(
            world="World-Tabletop-v0",
            robot="Robot-UnitreeG1-v0",
            objects="Objects-Dinnerware-v0",
            sensors="Sensors-FixedRGBD-v0",
            task="Task-PickPlace-v0",
            controller="Controller-Standing-v0",
        ),
        enable_sensors=False,
    )
    bundle.env_cfg.sim.device = args.device
    env = ScenarioEnv(bundle.env_cfg)
    try:
        robot = env.scene["robot"]
        term = env.action_manager.get_term("upper_body")
        action_ids, _ = robot.find_joints(term.joint_names, preserve_order=True)
        arm_names = list(bundle.robot.left_arm_joint_names)
        arm_ids, _ = robot.find_joints(arm_names, preserve_order=True)
        palm_ids, _ = robot.find_bodies([bundle.robot.left_end_effector], preserve_order=True)
        jacobian_body_id = palm_ids[0] - 1 if robot.is_fixed_base else palm_ids[0]
        rows = []

        for name, joint_id in zip(arm_names, arm_ids):
            env.reset()
            neutral_action = torch.zeros((1, term.action_dim), device=env.device)
            for _ in range(20):
                env.step(neutral_action)
            q0 = robot.data.joint_pos[0, joint_id].clone()
            p0 = robot.data.body_pos_w[0, palm_ids[0]].clone()
            root_quat = robot.data.root_quat_w[0:1].clone()
            jacobian = robot.root_physx_view.get_jacobians()[0, jacobian_body_id, :3, joint_id].clone()

            targets = robot.data.default_joint_pos[:, action_ids].clone()
            local_index = term.joint_names.index(name)
            targets[:, local_index] = q0 + 0.10
            action = term.actions_from_joint_targets(targets)
            for _ in range(40):
                env.step(action)
            dq = robot.data.joint_pos[0, joint_id] - q0
            observed_w = robot.data.body_pos_w[0, palm_ids[0]] - p0
            predicted_native = jacobian * dq
            predicted_rotated = quat_apply(root_quat, predicted_native.unsqueeze(0))[0]
            rows.append(
                {
                    "joint": name,
                    "actual_delta_rad": dq.item(),
                    "observed_world_delta_m": observed_w.cpu().tolist(),
                    "jacobian_delta_native": predicted_native.cpu().tolist(),
                    "jacobian_delta_rotated_to_world": predicted_rotated.cpu().tolist(),
                }
            )
        report = {"root_quaternion_wxyz": root_quat[0].cpu().tolist(), "probes": rows}
        output = PROJECT / "outputs/g1-jacobian-probe.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()
