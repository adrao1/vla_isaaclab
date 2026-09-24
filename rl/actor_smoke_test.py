#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "rl"))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="VLA-YCBSugarBox-G1-JointPos-v0",
    )
    parser.add_argument("--steps", type=int, default=100)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
ARGS.enable_cameras = True
ARGS.experience = str(
    PROJECT
    / "configs"
    / (
        "ycb.python.headless.rendering.kit"
        if ARGS.headless
        else "ycb.python.rendering.kit"
    )
)
ARGS.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"

APP = AppLauncher(ARGS).app

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import vla_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from ee_delta_controller import EEDeltaController


class Actor(nn.Module):
    """X-Sim-style Gaussian actor for the 7-D EE action."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
        )

        # Learnable diagonal Gaussian standard deviation.
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, action_dim)
        )

    def forward(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)

        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)

        return mean, std

    def sample_action(self, obs: torch.Tensor):
        mean, std = self(obs)

        distribution = Normal(mean, std)

        raw_action = distribution.sample()

        # Our EE controller expects values in [-1, 1].
        action = torch.tanh(raw_action)

        return action, raw_action, mean, std


def extract_policy_obs(obs):
    """Return the flat policy observation tensor."""

    if isinstance(obs, dict):
        return obs["policy"]

    return obs


def main():
    cfg = parse_env_cfg(
        ARGS.task,
        device=ARGS.device,
        num_envs=1,
    )

    env = gym.make(
        ARGS.task,
        cfg=cfg,
    ).unwrapped

    try:
        obs, info = env.reset(seed=42)

        policy_obs = extract_policy_obs(obs)

        obs_dim = policy_obs.shape[-1]
        action_dim = EEDeltaController.ACTION_DIM

        print()
        print("=== Actor smoke test ===")
        print("policy observation shape:", tuple(policy_obs.shape))
        print("observation dimension:", obs_dim)
        print("RL action dimension:", action_dim)
        print("device:", env.device)
        print()

        actor = Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
        ).to(env.device)

        actor.eval()

        controller = EEDeltaController(env)
        controller.reset()

        palm_id = controller.palm_body_id

        start_palm = (
            env.scene["robot"].data.body_pos_w[0, palm_id]
            - env.scene.env_origins[0]
        ).clone()

        print(
            "start palm:",
            start_palm.detach().cpu().tolist(),
        )
        print()

        with torch.inference_mode():
            for step in range(ARGS.steps):
                policy_obs = extract_policy_obs(obs)

                # ------------------------------------------------------
                # Actor: 156-D observation -> 7-D action
                # ------------------------------------------------------

                (
                    rl_action,
                    raw_action,
                    mean,
                    std,
                ) = actor.sample_action(policy_obs)

                if not torch.isfinite(rl_action).all():
                    raise RuntimeError(
                        f"Non-finite RL action at step {step}"
                    )

                if tuple(rl_action.shape) != (
                    env.num_envs,
                    action_dim,
                ):
                    raise RuntimeError(
                        f"Unexpected RL action shape: "
                        f"{tuple(rl_action.shape)}"
                    )

                # ------------------------------------------------------
                # EE controller: 7-D -> 43-D
                # ------------------------------------------------------

                env_action = controller.compute(rl_action)

                if not torch.isfinite(env_action).all():
                    raise RuntimeError(
                        f"Non-finite environment action at step {step}"
                    )

                if tuple(env_action.shape) != (
                    env.num_envs,
                    env.action_manager.total_action_dim,
                ):
                    raise RuntimeError(
                        f"Unexpected env action shape: "
                        f"{tuple(env_action.shape)}"
                    )

                # ------------------------------------------------------
                # Step environment.
                # ------------------------------------------------------

                obs, reward, terminated, truncated, info = env.step(
                    env_action
                )

                policy_obs_next = extract_policy_obs(obs)

                if not torch.isfinite(policy_obs_next).all():
                    raise RuntimeError(
                        f"Non-finite observation at step {step}"
                    )

                if not torch.isfinite(reward).all():
                    raise RuntimeError(
                        f"Non-finite reward at step {step}"
                    )

                palm_pos = (
                    env.scene["robot"].data.body_pos_w[
                        0, palm_id
                    ]
                    - env.scene.env_origins[0]
                )

                print(
                    f"step={step:03d} "
                    f"obs={tuple(policy_obs.shape)} "
                    f"rl_action={tuple(rl_action.shape)} "
                    f"env_action={tuple(env_action.shape)} "
                    f"action_min={rl_action.min().item():+.3f} "
                    f"action_max={rl_action.max().item():+.3f} "
                    f"mean_abs={mean.abs().mean().item():.3f} "
                    f"std_mean={std.mean().item():.3f} "
                    f"reward={reward[0].item():.4f} "
                    f"palm={palm_pos.detach().cpu().tolist()}"
                )

                if bool(terminated[0].item()) or bool(
                    truncated[0].item()
                ):
                    print()
                    print(
                        f"episode ended at step {step}: "
                        f"terminated={terminated.detach().cpu().tolist()} "
                        f"truncated={truncated.detach().cpu().tolist()}"
                    )
                    break

        end_palm = (
            env.scene["robot"].data.body_pos_w[0, palm_id]
            - env.scene.env_origins[0]
        )

        displacement = end_palm - start_palm

        print()
        print("=== Finished ===")
        print(
            "start palm:",
            start_palm.detach().cpu().tolist(),
        )
        print(
            "end palm:",
            end_palm.detach().cpu().tolist(),
        )
        print(
            "displacement:",
            displacement.detach().cpu().tolist(),
        )

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
