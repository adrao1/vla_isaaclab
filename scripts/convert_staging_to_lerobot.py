#!/usr/bin/env python3
"""Convert a neutral simulation staging file to LeRobot Dataset v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaac_simlab.recording.lerobot_v3 import LeRobotV3Writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    staging_path = args.staging.resolve()
    dataset_name = staging_path.stem
    with h5py.File(staging_path, "r") as stream:
        if not bool(stream.attrs.get("complete", False)):
            raise RuntimeError(f"Staging recording is incomplete: {staging_path}")
        features = json.loads(stream.attrs["features"])
        for feature in features.values():
            feature["shape"] = tuple(feature["shape"])
        metadata = json.loads(stream.attrs["metadata"])
        writer = LeRobotV3Writer(
            root=PROJECT / "outputs/lerobot",
            dataset_name=dataset_name,
            features=features,
            metadata=metadata,
        )
        try:
            for episode_name in sorted(stream["episodes"]):
                episode = stream[f"episodes/{episode_name}"]
                task = str(episode.attrs["task"])
                for frame_index in range(int(episode.attrs["length"])):
                    frame = {key: episode[key][frame_index] for key in episode}
                    frame["task"] = task
                    writer.add_frame(frame)
                writer.save_episode()
            final_path = writer.finalize()
        except BaseException:
            writer.abort()
            raise

    if not args.keep_staging:
        staging_path.unlink()
    print(json.dumps({"dataset": str(final_path), "source": str(staging_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
