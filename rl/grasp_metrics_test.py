#!/usr/bin/env python3
"""Measure grasp metrics while running the known-working scripted policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="VLA-YCBSugarBox-G1-Grasp-v0",
)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()

ARGS.enable_cameras = False
ARGS.experience = str(PROJECT_ROOT / "configs" / "ycb.python.headless.kit")
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"

APP = AppLauncher(ARGS).app


import gymnasium as gym
import torch

import vla_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.envs.common import LEFT_END_EFFECTOR, SUPPORT_HEIGHT
from vla_isaaclab.envs.ycb_sugar_box.env_cfg import SUGAR_BOX_HALF_HEIGHT_M
from vla_isaaclab.envs.ycb_sugar_box.mdp import grasp_metrics
from vla_isaaclab.policies import YCBSugarBoxScriptedPolicy


def main():
    cfg = parse_env_cfg(
        ARGS.task,
        device=ARGS.device,
        num_envs=1,
    )

    for camera_name in (
        "camera",
        "cam_side",
        "cam_left_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ):
        if hasattr(cfg.scene, camera_name):
            setattr(cfg.scene, camera_name, None)

    env = gym.make(ARGS.task, cfg=cfg).unwrapped

    initial_box_height = SUPPORT_HEIGHT + SUGAR_BOX_HALF_HEIGHT_M

    try:
        env.reset(seed=42)
        policy = YCBSugarBoxScriptedPolicy(env)

        previous_phase = None

        with torch.inference_mode():
            for step in range(700):
                phase = policy.phase

                metrics = grasp_metrics(
                    env,
                    LEFT_END_EFFECTOR,
                    initial_box_height,
                )

                if (
                    phase != previous_phase
                    or step % 20 == 0
                    or metrics["lift_height"][0].item() > 0.02
                ):
                    print(
                        f"step={step:4d} "
                        f"phase={phase:18s} "
                        f"distance={metrics['hand_distance'][0].item():.4f} "
                        f"closure={metrics['closure_fraction'][0].item():.4f} "
                        f"lift={metrics['lift_height'][0].item():.4f}"
                    )

                previous_phase = phase

                action = policy.compute(step)
                _, _, terminated, truncated, _ = env.step(action)

                success = bool(
                    env.termination_manager.get_term("success")[0].item()
                )

                if success:
                    print(f"\nSUCCESS triggered at step {step}")
                    break

                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    print(f"\nEpisode terminated at step {step}")
                    break

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
