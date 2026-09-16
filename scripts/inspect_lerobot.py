#!/usr/bin/env python3
"""Validate a local LeRobot Dataset v3 recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REQUIRED_FEATURES = {
    "observation.state",
    "observation.velocity",
    "observation.environment_state",
    "observation.images.front",
    "action",
    "sim.action.normalized",
    "next.reward",
    "next.done",
    "next.success",
    "sim.seed",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--no-video-check", action="store_true")
    args = parser.parse_args()

    root = args.dataset.resolve()
    manifest_path = root / "meta/simulation.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = LeRobotDataset(manifest["lerobot_repo_id"], root=root, video_backend="pyav")
    tabular = dataset.hf_dataset.with_format("numpy")
    episode_indices = np.asarray(tabular["episode_index"])
    timestamps = np.asarray(tabular["timestamp"])
    actions = np.asarray(tabular["action"])
    states = np.asarray(tabular["observation.state"])

    missing = sorted(REQUIRED_FEATURES - set(dataset.features))
    timestamp_errors = []
    for episode_index in range(dataset.num_episodes):
        episode_timestamps = timestamps[episode_indices == episode_index]
        if len(episode_timestamps) == 0 or not np.all(np.diff(episode_timestamps) > 0):
            timestamp_errors.append(episode_index)

    video_samples = []
    if not args.no_video_check and len(dataset):
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            image = dataset[index]["observation.images.front"]
            video_samples.append({"index": index, "shape": list(image.shape), "finite": bool(image.isfinite().all())})

    valid = all(
        (
            manifest.get("recording_complete") is True,
            not missing,
            dataset.fps == 30,
            dataset.num_episodes == manifest.get("episodes"),
            len(dataset) == manifest.get("frames"),
            not timestamp_errors,
            bool(np.isfinite(actions).all()),
            bool(np.isfinite(states).all()),
            actions.ndim == 2 and states.ndim == 2,
            actions.shape == states.shape,
            all(sample["finite"] for sample in video_samples),
        )
    )
    report = {
        "valid": valid,
        "root": str(root),
        "format": manifest.get("format"),
        "fps": dataset.fps,
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "features": sorted(dataset.features),
        "action_shape": list(actions.shape),
        "state_shape": list(states.shape),
        "missing_features": missing,
        "timestamp_error_episodes": timestamp_errors,
        "video_samples": video_samples,
        "task_prompt": manifest.get("task_prompt"),
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
