"""White table laboratory world."""

from isaaclab.assets import AssetBaseCfg

from ..contracts import WorldDefinition
from .common import static_box


def configure_tabletop(scene, world) -> None:
    scene.support_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportSurface",
        spawn=static_box((1.2, 0.8, 0.05)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.725)),
    )
    for index, position in enumerate(
        ((-0.52, -0.32, 0.35), (-0.52, 0.32, 0.35), (0.52, -0.32, 0.35), (0.52, 0.32, 0.35))
    ):
        setattr(
            scene,
            f"support_leg_{index}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/SupportLeg_{index}",
                spawn=static_box((0.055, 0.055, 0.70)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=position),
            ),
        )


TABLETOP_WORLD = WorldDefinition(
    component_id="World-Tabletop-v0",
    capabilities=frozenset(
        {"robot_spawn", "workspace", "support_surface", "object_spawn_region", "target_region", "camera_mount"}
    ),
    support_height=0.75,
    robot_position=(0.0, -0.82, 0.74),
    robot_orientation_wxyz=(0.70710678, 0.0, 0.0, 0.70710678),
    camera_eye=(2.8, -3.2, 2.25),
    camera_target=(0.0, -0.05, 0.95),
    object_position=(-0.18, 0.04, 0.83),
    goal_position=(0.20, 0.02, 0.83),
    reach_target=(-0.20, -0.25, 1.10),
    configure_scene=configure_tabletop,
)
