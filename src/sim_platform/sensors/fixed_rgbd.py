"""World-mounted RGB-D camera rig."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from ..contracts import SensorRigDefinition


def configure_fixed_rgbd(scene, world) -> None:
    scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0, focus_distance=2.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
    )


FIXED_RGBD = SensorRigDefinition(
    component_id="Sensors-FixedRGBD-v0",
    capabilities=frozenset({"rgb", "depth", "camera_calibration", "world_mount"}),
    camera_names=("camera",),
    configure_scene=configure_fixed_rgbd,
)
