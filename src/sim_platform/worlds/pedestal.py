"""Compact pedestal world used to verify scene-independent tasks."""

from isaaclab.assets import AssetBaseCfg

from ..contracts import WorldDefinition
from .common import static_box


def configure_pedestal(scene, world) -> None:
    scene.support_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportSurface",
        spawn=static_box((0.9, 0.7, 0.08), color=(0.78, 0.82, 0.88)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.08, 0.86)),
    )
    scene.support_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportBase",
        spawn=static_box((0.42, 0.42, 0.82), color=(0.55, 0.60, 0.68)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.08, 0.41)),
    )


PEDESTAL_WORLD = WorldDefinition(
    component_id="World-Pedestal-v0",
    capabilities=frozenset(
        {"robot_spawn", "workspace", "support_surface", "object_spawn_region", "target_region", "camera_mount"}
    ),
    support_height=0.90,
    robot_position=(0.0, -0.72, 0.74),
    robot_orientation_wxyz=(0.70710678, 0.0, 0.0, 0.70710678),
    camera_eye=(2.45, -2.8, 2.15),
    camera_target=(0.0, 0.05, 1.05),
    object_position=(-0.14, 0.10, 0.98),
    goal_position=(0.16, 0.08, 0.98),
    reach_target=(-0.15, -0.18, 1.18),
    configure_scene=configure_pedestal,
)
