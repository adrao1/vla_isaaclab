"""Compressed HDF5 backend for camera-heavy demonstrations."""

import h5py

from isaaclab.utils.datasets import HDF5DatasetFileHandler


class CompressedHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """Use lightweight gzip compression for non-scalar tensors."""

    def write_episode(self, episode):
        self._raise_if_not_initialized()
        if episode.is_empty():
            return
        episode_group = self._hdf5_data_group.create_group(f"demo_{self._demo_count}")
        episode_group.attrs["num_samples"] = len(episode.data.get("actions", []))
        if episode.seed is not None:
            episode_group.attrs["seed"] = episode.seed
        if episode.success is not None:
            episode_group.attrs["success"] = episode.success

        def create(group, key, value):
            if isinstance(value, dict):
                child = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create(child, sub_key, sub_value)
                return
            array = value.detach().cpu().numpy()
            options = {"compression": "gzip", "compression_opts": 1, "shuffle": True} if array.ndim > 0 else {}
            group.create_dataset(key, data=array, **options)

        for key, value in episode.data.items():
            create(episode_group, key, value)
        self._hdf5_data_group.attrs["total"] += episode_group.attrs["num_samples"]
        self._demo_count += 1
