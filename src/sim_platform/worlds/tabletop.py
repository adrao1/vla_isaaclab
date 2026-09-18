"""White table laboratory world."""

from isaaclab.assets import AssetBaseCfg

from ..contracts import WorldDefinition
from .common import static_box


def configure_tabletop(scene, world) -> None:
    scene.support_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportSurface",
        spawn=static_box((1.2, 0.8, 0.05)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, world.support_height - 0.025)),
    )
    leg_height = world.support_height - 0.05
    for index, position in enumerate(
        (
            (-0.52, -0.32, leg_height / 2.0),
            (-0.52, 0.32, leg_height / 2.0),
            (0.52, -0.32, leg_height / 2.0),
            (0.52, 0.32, leg_height / 2.0),
        )
    ):
        setattr(
            scene,
            f"support_leg_{index}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/SupportLeg_{index}",
                spawn=static_box((0.055, 0.055, leg_height)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=position),
            ),
        )


TABLETOP_WORLD = WorldDefinition(
    component_id="World-Tabletop-v0",
    capabilities=frozenset(
        {"robot_spawn", "workspace", "support_surface", "object_spawn_region", "target_region", "camera_mount"}
    ),
    support_height=0.68,
    robot_position=(0.0, -0.82, 0.74),
    robot_orientation_wxyz=(0.70710678, 0.0, 0.0, 0.70710678),
    camera_eye=(0.0, 2.20, 2.25),
    camera_target=(0.0, -0.25, 0.86),
    object_position=(-0.18, -0.24, 0.76),
    goal_position=(0.12, -0.22, 0.76),
    reach_target=(-0.20, -0.25, 1.03),
    configure_scene=configure_tabletop,
)
