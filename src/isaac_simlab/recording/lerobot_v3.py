"""Atomic LeRobot Dataset v3 writer."""

from __future__ import annotations

import json
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotV3Writer:
    """Write episodes with the official LeRobot API and publish on finalize."""

    def __init__(
        self,
        root: Path,
        dataset_name: str,
        features: dict,
        metadata: dict,
        fps: int = 30,
        image_writer_threads: int = 4,
    ):
        self.final_root = root / dataset_name
        self.work_root = root / f".{dataset_name}.incomplete-{os.getpid()}"
        if self.final_root.exists():
            raise FileExistsError(f"Dataset already exists: {self.final_root}")
        if self.work_root.exists():
            raise FileExistsError(f"Temporary dataset already exists: {self.work_root}")
        root.mkdir(parents=True, exist_ok=True)

        self.repo_id = f"local/{dataset_name}"
        self.metadata = metadata
        self.episode_count = 0
        self.frame_count = 0
        self._pending_frames = 0
        self._closed = False
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=fps,
            features=features,
            root=self.work_root,
            robot_type="unitree_g1",
            use_videos=True,
            image_writer_threads=image_writer_threads,
            video_backend="pyav",
            vcodec="libsvtav1",
        )
        self._write_manifest(complete=False)

    def _write_manifest(self, complete: bool) -> None:
        payload = {
            "format": "LeRobotDataset-v3.0",
            "lerobot_repo_id": self.repo_id,
            "recording_complete": complete,
            "episodes": self.episode_count,
            "frames": self.frame_count,
            **self.metadata,
        }
        path = self.work_root / "meta" / "simulation.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def add_frame(self, frame: dict) -> None:
        if self._closed:
            raise RuntimeError("Cannot add a frame after finalization")
        self.dataset.add_frame(frame)
        self._pending_frames += 1

    def save_episode(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot save an episode after finalization")
        self.dataset.save_episode(parallel_encoding=False)
        self.episode_count += 1
        self.frame_count += self._pending_frames
        self._pending_frames = 0
        self._write_manifest(complete=False)

    def discard_episode(self) -> None:
        if not self._closed:
            self.dataset.clear_episode_buffer(delete_images=True)
            self._pending_frames = 0

    def finalize(self) -> Path:
        if self._closed:
            return self.final_root
        self.dataset.stop_image_writer()
        self.dataset.finalize()
        self._write_manifest(complete=True)
        self.work_root.rename(self.final_root)
        self._closed = True
        return self.final_root

    def abort(self) -> None:
        """Close writers while leaving the incomplete directory for diagnosis."""
        if self._closed:
            return
        try:
            self.dataset.clear_episode_buffer(delete_images=True)
            self.dataset.stop_image_writer()
            self.dataset.finalize()
            self._write_manifest(complete=False)
        finally:
            self._closed = True
