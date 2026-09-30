#!/usr/bin/env python3
"""Record one successful scripted sugar-box trajectory for RL waypoint extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="VLA-YCBSugarBox-G1-JointPos-v0",
    )
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl" / "sugar_box_demo.npz",
    )
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = False
    return args


ARGS = parse_args()
ARGS.experience = str(PROJECT_ROOT / "configs" / "ycb.python.headless.kit")
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"

APP = AppLauncher(ARGS).app


import gymnasium as gym
import torch

import vla_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.policies import YCBSugarBoxScriptedPolicy


def main():
    cfg = parse_env_cfg(
        ARGS.task,
        device=ARGS.device,
        num_envs=1,
    )

    # PPO training does not use camera observations, so neither does this
    # demonstration recorder.
    for camera_name in ("camera", "cam_side", "cam_left_high", "cam_left_wrist", "cam_right_wrist"):
        if hasattr(cfg.scene, camera_name):
            setattr(cfg.scene, camera_name, None)

    # Give the scripted policy enough time to complete its full sequence.
    cfg.episode_length_s = 60.0

    env = gym.make(ARGS.task, cfg=cfg).unwrapped

    try:
        env.reset(seed=42)
        policy = YCBSugarBoxScriptedPolicy(env)

        box = env.scene["object"]
        robot = env.scene["robot"]

        palm_ids, _ = robot.find_bodies(
            ["left_hand_palm_link"],
            preserve_order=True,
        )
        palm_id = palm_ids[0]

        box_poses = []
        palm_poses = []
        phases = []
        rewards = []

        success = False

        with torch.inference_mode():
            for step in range(ARGS.steps):
                # Record the state that the policy sees before taking this action.
                box_pose = torch.cat(
                    (
                        box.data.root_pos_w[0] - env.scene.env_origins[0],
                        box.data.root_quat_w[0],
                    )
                )

                palm_pose = torch.cat(
                    (
                        robot.data.body_pos_w[0, palm_id] - env.scene.env_origins[0],
                        robot.data.body_quat_w[0, palm_id],
                    )
                )

                box_poses.append(box_pose.cpu().numpy())
                palm_poses.append(palm_pose.cpu().numpy())
                phases.append(policy.phase)

                action = policy.compute(step)
                _, reward, terminated, truncated, _ = env.step(action)

                rewards.append(float(reward[0].item()))

                if "success" in env.termination_manager.active_terms:
                    step_success = bool(
                        env.termination_manager.get_term("success")[0].item()
                    )
                else:
                    step_success = False

                if step_success:
                    success = True

                if getattr(policy, "failed", False):
                    print(
                        f"Scripted policy failed at step {step}: "
                        f"{policy.strategy.failure_reason}"
                    )
                    break

                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    break

        box_poses = np.asarray(box_poses, dtype=np.float32)
        palm_poses = np.asarray(palm_poses, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        phases = np.asarray(phases)

        if not success:
            raise RuntimeError(
                "The recorded scripted episode did not satisfy the task success condition. "
                "Not saving it as an RL demonstration."
            )

        ARGS.output.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            ARGS.output,
            box_pose=box_poses,
            palm_pose=palm_poses,
            phase=phases,
            reward=rewards,
        )

        print()
        print("Saved successful demonstration:")
        print(f"  path:       {ARGS.output}")
        print(f"  steps:      {len(box_poses)}")
        print(f"  box_pose:   {box_poses.shape}")
        print(f"  palm_pose:  {palm_poses.shape}")
        print(f"  reward:     {rewards.shape}")
        print(f"  success:    {success}")

        print()
        print("Phase transitions:")
        previous = None
        for i, phase in enumerate(phases):
            if phase != previous:
                print(
                    f"  step={i:4d} "
                    f"phase={phase:20s} "
                    f"box_xyz={box_poses[i, :3]}"
                )
                previous = phase

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
