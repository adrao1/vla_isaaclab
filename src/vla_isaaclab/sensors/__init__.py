"""Sensor-rig definitions."""

from .fixed_rgbd import FIXED_RGBD


def register_sensor_rigs() -> None:
    from ..registry import register

    register("sensors", FIXED_RGBD)
