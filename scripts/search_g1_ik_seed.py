#!/usr/bin/env python3
"""Search G1 joint configurations near a requested left-palm position."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--batches", type=int, default=40)
parser.add_argument("--target", nargs=3, type=float, default=(-0.18, -0.49, 0.82))
parser.add_argument("--center", nargs=5, type=float, help="Optional joint-space sampling center.")
parser.add_argument("--spread", nargs=5, type=float, help="Half-width around --center.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(PROJECT / "configs/ycb.python.headless.kit")
args.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
app = AppLauncher(args).app

import torch

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
    bundle.env_cfg.scene.num_envs = args.num_envs
    bundle.env_cfg.sim.device = args.device
    env = ScenarioEnv(bundle.env_cfg)
    try:
        robot = env.scene["robot"]
        names = list(bundle.robot.left_arm_joint_names)
        ids, _ = robot.find_joints(names, preserve_order=True)
        palm_ids, _ = robot.find_bodies([bundle.robot.left_end_effector], preserve_order=True)
        target = torch.tensor(args.target, device=env.device)
        limits = robot.data.soft_joint_pos_limits[:, ids]
        generator = torch.Generator(device=env.device).manual_seed(12345)
        best: list[dict] = []
        for _ in range(args.batches):
            random = torch.rand((args.num_envs, len(ids)), generator=generator, device=env.device)
            if args.center is None:
                q = limits[..., 0] + random * (limits[..., 1] - limits[..., 0])
            else:
                spread_values = args.spread or (0.8, 0.8, 0.8, 1.2, 0.8)
                center = torch.tensor(args.center, device=env.device).unsqueeze(0)
                spread = torch.tensor(spread_values, device=env.device).unsqueeze(0)
                q = center + (2.0 * random - 1.0) * spread
                q = torch.clamp(q, limits[..., 0], limits[..., 1])
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
            env.sim.forward()
            palm = robot.data.body_pos_w[:, palm_ids[0]] - env.scene.env_origins
            distance = torch.linalg.vector_norm(palm - target, dim=-1)
            values, indices = torch.topk(distance, k=min(8, args.num_envs), largest=False)
            for value, index in zip(values, indices):
                best.append(
                    {
                        "error_m": value.item(),
                        "palm_position_m": palm[index].detach().cpu().tolist(),
                        "joint_position_rad": q[index].detach().cpu().tolist(),
                    }
                )
            best = sorted(best, key=lambda row: row["error_m"])[:20]
        report = {"target_m": list(args.target), "joint_names": names, "solutions": best}
        output = PROJECT / "outputs/g1-ik-seed-search.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
