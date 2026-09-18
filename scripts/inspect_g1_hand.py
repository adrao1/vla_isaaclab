#!/usr/bin/env python3
"""Inspect the G1 arm/hand kinematics used by the composed scenario."""

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
args.enable_cameras = True
args.experience = str(PROJECT / "configs/ycb.python.headless.rendering.kit")
args.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
app = AppLauncher(args).app

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
        )
    )
    bundle.env_cfg.sim.device = args.device
    env = ScenarioEnv(bundle.env_cfg)
    try:
        env.reset()
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("upper_body")
        action_ids, action_names = robot.find_joints(action_term.joint_names, preserve_order=True)
        joint_rows = []
        for local_index, (joint_id, name) in enumerate(zip(action_ids, action_names)):
            limits = robot.data.soft_joint_pos_limits[0, joint_id]
            joint_rows.append(
                {
                    "action_index": local_index,
                    "joint_id": joint_id,
                    "name": name,
                    "default_rad": robot.data.default_joint_pos[0, joint_id].item(),
                    "lower_rad": limits[0].item(),
                    "upper_rad": limits[1].item(),
                }
            )

        origin = env.scene.env_origins[0]
        body_rows = []
        for body_id, name in enumerate(robot.body_names):
            if name.startswith("left_") or name in ("torso_link", "pelvis"):
                body_rows.append(
                    {
                        "body_id": body_id,
                        "name": name,
                        "position_m": (robot.data.body_pos_w[0, body_id] - origin).cpu().tolist(),
                        "quaternion_wxyz": robot.data.body_quat_w[0, body_id].cpu().tolist(),
                    }
                )

        report = {
            "fixed_base": robot.is_fixed_base,
            "num_joints": robot.num_joints,
            "num_bodies": robot.num_bodies,
            "action_joints": joint_rows,
            "left_bodies": body_rows,
        }
        output = PROJECT / "outputs/g1-hand-inspection.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
