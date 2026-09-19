#!/usr/bin/env python3
"""Isaac Sim half of LeRobot replay; consumes a neutral NumPy episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode_data", type=Path)
parser.add_argument("--output", type=Path, default=PROJECT / "outputs/replay/lerobot_rgb.png")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ARGS.enable_cameras = True
ARGS.experience = str(
    PROJECT / "configs" / ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit")
)
ARGS.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import numpy as np
import torch
from PIL import Image

from vla_isaaclab import ScenarioSelection, register_defaults
from vla_isaaclab.runtime import ScenarioEnv
from vla_isaaclab.scenario import compose_scenario


def main() -> int:
    with np.load(ARGS.episode_data, allow_pickle=False) as episode_data:
        actions = episode_data["actions"]
        seed = int(episode_data["seed"])
        metadata = json.loads(str(episode_data["scenario"]))
        dataset_root = str(episode_data["dataset"])
        episode_index = int(episode_data["episode"])

    register_defaults()
    selection = ScenarioSelection(
        world=metadata["world"],
        robot=metadata["robot"],
        objects=metadata["objects"],
        sensors=metadata["sensors"],
        task=metadata["task"],
        expert=metadata.get("expert"),
        controller=metadata["controller"],
    )
    bundle = compose_scenario(selection)
    bundle.env_cfg.sim.device = ARGS.device
    bundle.env_cfg.seed = seed
    bundle.env_cfg.episode_length_s = (len(actions) + 1) * bundle.env_cfg.decimation * bundle.env_cfg.sim.dt
    env = ScenarioEnv(bundle.env_cfg)
    try:
        origin = env.scene.env_origins[0]
        eye = torch.tensor([bundle.world.camera_eye], device=env.device) + origin
        target = torch.tensor([bundle.world.camera_target], device=env.device) + origin
        env.scene["camera"].set_world_poses_from_view(eye, target)
        env.reset(seed=seed)
        with torch.inference_mode():
            for action in actions:
                env.step(torch.as_tensor(action, device=env.device).unsqueeze(0))

        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        rgb = env.scene["camera"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(ARGS.output)
        report = {
            "dataset": dataset_root,
            "episode": episode_index,
            "scenario": metadata["scenario_id"],
            "seed": seed,
            "steps_replayed": len(actions),
            "output": str(ARGS.output.resolve()),
        }
        ARGS.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
