#!/usr/bin/env python3

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        default="VLA-YCBSugarBox-G1-JointPos-v0",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()

ARGS.enable_cameras = True

if ARGS.headless:
    ARGS.experience = str(
        PROJECT / "configs" / "ycb.python.headless.rendering.kit"
    )
else:
    ARGS.experience = str(
        PROJECT / "configs" / "ycb.python.rendering.kit"
    )

ARGS.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"

APP = AppLauncher(ARGS).app


import gymnasium as gym
import torch

import vla_isaaclab.envs  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def get_policy_obs(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]

        raise RuntimeError(
            f"Unexpected observation dictionary keys: {list(obs.keys())}"
        )

    return obs


def main():
    print(f"Creating environment: {ARGS.task}")

    env_cfg = parse_env_cfg(
        ARGS.task,
        device=ARGS.device,
        num_envs=1,
    )

    env = gym.make(
        ARGS.task,
        cfg=env_cfg,
    )

    print("\n=== Environment information ===")
    print("task:", ARGS.task)
    print("num_envs:", env.unwrapped.num_envs)
    print("device:", env.unwrapped.device)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    obs, info = env.reset()
    obs_tensor = get_policy_obs(obs)

    print("\n=== Reset ===")
    print("raw observation type:", type(obs))
    print("policy observation shape:", tuple(obs_tensor.shape))
    print("policy observation dtype:", obs_tensor.dtype)
    print("policy observation device:", obs_tensor.device)

    print("\n=== Random rollout ===")

    for step in range(ARGS.steps):
        action_shape = (
            env.unwrapped.num_envs,
            *env.action_space.shape,
        )

        action = torch.empty(
            action_shape,
            device=env.unwrapped.device,
            dtype=torch.float32,
        ).uniform_(-1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)
        obs_tensor = get_policy_obs(obs)

        print(
            f"step={step:03d} "
            f"obs={tuple(obs_tensor.shape)} "
            f"action={tuple(action.shape)} "
            f"reward={reward.detach().cpu().tolist()} "
            f"terminated={terminated.detach().cpu().tolist()} "
            f"truncated={truncated.detach().cpu().tolist()}"
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
