#!/usr/bin/env python3
"""Replay one episode from a scenario HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path)
parser.add_argument("--episode", default="demo_0")
parser.add_argument("--output", type=Path, default=PROJECT / "outputs/replay/rgb.png")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ARGS.enable_cameras = True
ARGS.experience = str(
    PROJECT / "configs" / ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit")
)
ARGS.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import h5py
import numpy as np
import torch
from PIL import Image

from sim_platform import ScenarioSelection, register_defaults
from sim_platform.runtime import ScenarioEnv
from sim_platform.scenario import compose_scenario


def load_tensor_tree(group, device):
    result = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            result[key] = load_tensor_tree(value, device)
        else:
            result[key] = torch.tensor(np.asarray(value), device=device)
    return result


def main() -> int:
    with h5py.File(ARGS.dataset, "r") as stream:
        metadata = json.loads(stream["data"].attrs["env_args"])
        episode = stream[f"data/{ARGS.episode}"]
        actions = np.asarray(episode["actions"])
        initial_state_numpy = None
        if "initial_state" in episode:
            initial_state_numpy = episode["initial_state"]

        register_defaults()
        selection = ScenarioSelection(
            world=metadata["world"], robot=metadata["robot"], objects=metadata["objects"],
            sensors=metadata["sensors"], task=metadata["task"], controller=metadata["controller"],
        )
        bundle = compose_scenario(selection)
        bundle.env_cfg.sim.device = ARGS.device
        bundle.env_cfg.seed = int(episode.attrs.get("seed", 42))
        bundle.env_cfg.episode_length_s = (len(actions) + 1) * bundle.env_cfg.decimation * bundle.env_cfg.sim.dt
        env = ScenarioEnv(bundle.env_cfg)
        try:
            origin = env.scene.env_origins[0]
            eye = torch.tensor([bundle.world.camera_eye], device=env.device) + origin
            target = torch.tensor([bundle.world.camera_target], device=env.device) + origin
            env.scene["camera"].set_world_poses_from_view(eye, target)
            env.reset()
            if initial_state_numpy is not None:
                state = load_tensor_tree(initial_state_numpy, env.device)
                env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
                env.scene.reset_to(state, env_ids=env_ids, is_relative=True)
            with torch.inference_mode():
                for action in actions:
                    env.step(torch.tensor(action, device=env.device).unsqueeze(0))
            ARGS.output.parent.mkdir(parents=True, exist_ok=True)
            rgb = env.scene["camera"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(ARGS.output)
            report = {
                "dataset": str(ARGS.dataset.resolve()),
                "episode": ARGS.episode,
                "scenario": metadata.get("scenario_id"),
                "steps_replayed": len(actions),
                "output": str(ARGS.output.resolve()),
            }
            report_path = ARGS.output.with_suffix(".json")
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
