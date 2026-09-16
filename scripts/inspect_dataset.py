#!/usr/bin/env python3
"""Print and validate the structure of a recorded Isaac Lab HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


REQUIRED_DATASETS = (
    "actions",
    "obs",
    "states/articulation/robot/joint_position",
    "states/rigid_object/bowl/root_pose",
    "states/rigid_object/plate/root_pose",
    "camera/frames/rgb",
    "camera/frames/depth",
    "camera/calibration/intrinsic_matrix",
    "task/bowl_to_plate_distance",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as stream:
        episodes = sorted(stream["data"].keys())
        summary = {
            "file": str(args.dataset.resolve()),
            "environment": json.loads(stream["data"].attrs["env_args"])["env_name"],
            "episodes": {},
            "valid": True,
        }
        for episode_name in episodes:
            episode = stream[f"data/{episode_name}"]
            missing = [name for name in REQUIRED_DATASETS if name not in episode]
            summary["episodes"][episode_name] = {
                "steps": int(episode.attrs["num_samples"]),
                "success": bool(episode.attrs.get("success", False)),
                "action_shape": list(episode["actions"].shape),
                "observation_shape": list(episode["obs"].shape),
                "rgb_shape": list(episode["camera/frames/rgb"].shape) if "camera/frames/rgb" in episode else None,
                "depth_shape": list(episode["camera/frames/depth"].shape) if "camera/frames/depth" in episode else None,
                "missing": missing,
            }
            summary["valid"] = summary["valid"] and not missing
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
