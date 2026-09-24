#!/usr/bin/env python3
"""Measure grasp metrics and Dex3 contact forces with the scripted policy."""

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
ARGS.experience = str(
    PROJECT_ROOT / "configs" / "ycb.python.headless.kit"
)
ARGS.kit_args = (
    f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
)

APP = AppLauncher(ARGS).app


import gymnasium as gym
import torch

import vla_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.envs.common import (
    LEFT_END_EFFECTOR,
    SUPPORT_HEIGHT,
)
from vla_isaaclab.envs.ycb_sugar_box.env_cfg import (
    SUGAR_BOX_HALF_HEIGHT_M,
)
from vla_isaaclab.envs.ycb_sugar_box.mdp import (
    grasp_metrics,
)
from vla_isaaclab.policies import (
    YCBSugarBoxScriptedPolicy,
)


def filtered_contact_force(sensor) -> torch.Tensor:
    """Return sugar-box contact-force magnitude for each environment."""
    force_matrix = sensor.data.force_matrix_w

    if force_matrix is None:
        raise RuntimeError(
            "Contact sensor force_matrix_w is None. "
            "Check filter_prim_paths_expr."
        )

    # Each sensor contains:
    #   one sensing rigid body
    #   one filtered object
    #
    # Shape:
    #   (num_envs, 1, 1, 3)
    force_vector = force_matrix[:, 0, 0, :]

    return torch.linalg.vector_norm(
        force_vector,
        dim=-1,
    )


def finger_max_force(
    *forces: torch.Tensor,
) -> torch.Tensor:
    """Maximum contact magnitude across all links belonging to one finger."""
    return torch.stack(
        forces,
        dim=-1,
    ).amax(dim=-1)


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
            setattr(
                cfg.scene,
                camera_name,
                None,
            )

    env = gym.make(
        ARGS.task,
        cfg=cfg,
    ).unwrapped

    initial_box_height = (
        SUPPORT_HEIGHT
        + SUGAR_BOX_HALF_HEIGHT_M
    )

    try:
        env.reset(seed=42)

        policy = YCBSugarBoxScriptedPolicy(
            env
        )

        sensors = {
            "thumb_0": env.scene[
                "thumb_0_contact"
            ],
            "thumb_1": env.scene[
                "thumb_1_contact"
            ],
            "thumb_2": env.scene[
                "thumb_2_contact"
            ],
            "index_0": env.scene[
                "index_0_contact"
            ],
            "index_1": env.scene[
                "index_1_contact"
            ],
            "middle_0": env.scene[
                "middle_0_contact"
            ],
            "middle_1": env.scene[
                "middle_1_contact"
            ],
        }

        print("\nDex3 contact sensors:")
        print(
            "  thumb : "
            "left_hand_thumb_0_link, "
            "left_hand_thumb_1_link, "
            "left_hand_thumb_2_link"
        )
        print(
            "  index : "
            "left_hand_index_0_link, "
            "left_hand_index_1_link"
        )
        print(
            "  middle: "
            "left_hand_middle_0_link, "
            "left_hand_middle_1_link"
        )
        print()

        previous_phase = None

        with torch.inference_mode():
            for step in range(700):
                phase = policy.phase

                metrics = grasp_metrics(
                    env,
                    LEFT_END_EFFECTOR,
                    initial_box_height,
                )

                forces = {
                    name: filtered_contact_force(
                        sensor
                    )
                    for name, sensor
                    in sensors.items()
                }

                thumb_force = finger_max_force(
                    forces["thumb_0"],
                    forces["thumb_1"],
                    forces["thumb_2"],
                )

                index_force = finger_max_force(
                    forces["index_0"],
                    forces["index_1"],
                )

                middle_force = finger_max_force(
                    forces["middle_0"],
                    forces["middle_1"],
                )

                if (
                    phase != previous_phase
                    or step % 20 == 0
                    or metrics[
                        "lift_height"
                    ][0].item() > 0.02
                ):
                    print(
                        f"step={step:4d} "
                        f"phase={phase:18s} "
                        f"distance="
                        f"{metrics['hand_distance'][0].item():.4f} "
                        f"closure="
                        f"{metrics['closure_fraction'][0].item():.4f} "
                        f"lift="
                        f"{metrics['lift_height'][0].item():.4f}"
                    )

                    print(
                        "    links: "
                        f"T0={forces['thumb_0'][0].item():7.3f} "
                        f"T1={forces['thumb_1'][0].item():7.3f} "
                        f"T2={forces['thumb_2'][0].item():7.3f} | "
                        f"I0={forces['index_0'][0].item():7.3f} "
                        f"I1={forces['index_1'][0].item():7.3f} | "
                        f"M0={forces['middle_0'][0].item():7.3f} "
                        f"M1={forces['middle_1'][0].item():7.3f}"
                    )

                    print(
                        "    finger max: "
                        f"thumb={thumb_force[0].item():7.3f} "
                        f"index={index_force[0].item():7.3f} "
                        f"middle={middle_force[0].item():7.3f}"
                    )

                previous_phase = phase

                action = policy.compute(step)

                (
                    _,
                    _,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                success = bool(
                    env.termination_manager
                    .get_term("success")[0]
                    .item()
                )

                if success:
                    print(
                        f"\nSUCCESS triggered at step {step}"
                    )
                    break

                if (
                    bool(
                        terminated[0].item()
                    )
                    or bool(
                        truncated[0].item()
                    )
                ):
                    print(
                        "\nEpisode terminated "
                        f"at step {step}"
                    )
                    break

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
