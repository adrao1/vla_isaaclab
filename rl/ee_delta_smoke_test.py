#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="VLA-YCBSugarBox-G1-JointPos-v0",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT / "outputs" / "ee_delta_axes_test.mp4",
    )
    parser.add_argument(
        "--phase-steps",
        type=int,
        default=30,
        help="Number of control steps per test phase.",
    )
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

import av
import gymnasium as gym
import numpy as np
import torch

import vla_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from ee_delta_controller import EEDeltaController


def primary_camera(env):
    for name in ("cam_side", "camera", "cam_left_high"):
        if name in env.scene.sensors:
            return env.scene.sensors[name]
    raise RuntimeError("No external scene camera is configured")


def make_action(env, values):
    return torch.tensor(
        [values],
        dtype=torch.float32,
        device=env.device,
    )


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

    video_container = None
    video_stream = None

    try:
        env.reset(seed=42)

        controller = EEDeltaController(env)
        controller.reset()

        camera = primary_camera(env)
        palm_id = controller.palm_body_id

        video_path = ARGS.video.resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)

        video_container = av.open(str(video_path), mode="w")
        video_stream = video_container.add_stream("libx264", rate=30)
        video_stream.width = 640
        video_stream.height = 480
        video_stream.pix_fmt = "yuv420p"

        phases = [
            ("+X", [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("-X", [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("+Y", [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("-Y", [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("+Z", [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0]),
            ("-Z", [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -1.0]),
            ("+YAW", [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0]),
            ("-YAW", [0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -1.0]),
            ("GRIPPER CLOSE", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ("GRIPPER OPEN", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("+ROLL",  [0.0, 0.0, 0.0,  0.5, 0.0, 0.0, -1.0]),
            ("-ROLL",  [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, -1.0]),
            ("+PITCH", [0.0, 0.0, 0.0, 0.0,  0.5, 0.0, -1.0]),
            ("-PITCH", [0.0, 0.0, 0.0, 0.0, -0.5, 0.0, -1.0]),
        ]

        print(f"recording video to: {video_path}")

        global_step = 0

        with torch.inference_mode():
            for phase_name, values in phases:
                start_pos = (
                    env.scene["robot"].data.body_pos_w[0, palm_id]
                    - env.scene.env_origins[0]
                ).clone()

                print()
                print("=" * 70)
                print(f"PHASE: {phase_name}")
                print("start palm:", start_pos.detach().cpu().tolist())
                print("=" * 70)

                rl_action = make_action(env, values)

                for phase_step in range(ARGS.phase_steps):
                    env_action = controller.compute(rl_action)

                    _, reward, terminated, truncated, _ = env.step(
                        env_action
                    )

                    rgb = (
                        camera.data.output["rgb"][0, ..., :3]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    frame = av.VideoFrame.from_ndarray(
                        rgb.astype(np.uint8),
                        format="rgb24",
                    )

                    for packet in video_stream.encode(frame):
                        video_container.mux(packet)

                    palm_pos = (
                        env.scene["robot"].data.body_pos_w[0, palm_id]
                        - env.scene.env_origins[0]
                    )

                    print(
                        f"global={global_step:03d} "
                        f"phase={phase_name:14s} "
                        f"phase_step={phase_step:02d} "
                        f"palm={palm_pos.detach().cpu().tolist()} "
                        f"reward={reward.detach().cpu().tolist()}"
                    )

                    global_step += 1

                    if bool(terminated[0].item()) or bool(
                        truncated[0].item()
                    ):
                        print(
                            f"Episode ended during phase {phase_name}"
                        )
                        return

                end_pos = (
                    env.scene["robot"].data.body_pos_w[0, palm_id]
                    - env.scene.env_origins[0]
                )

                displacement = end_pos - start_pos

                print(
                    f"{phase_name} displacement:",
                    displacement.detach().cpu().tolist(),
                )

        print()
        print(f"video saved to: {video_path}")

    finally:
        if video_stream is not None:
            for packet in video_stream.encode():
                video_container.mux(packet)

        if video_container is not None:
            video_container.close()

        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
