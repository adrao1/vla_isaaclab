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
    support_height=0.62,
    # Move the initial base 12 cm closer; the robot reset pose keeps its
    # left hand above the tabletop rather than approaching below the edge.
    robot_position=(0.0, -0.64, 0.74),
    robot_orientation_wxyz=(0.70710678, 0.0, 0.0, 0.70710678),
    # Offset the camera to image-left and aim diagonally toward the lower-right
    # manipulation region for a clearer view of the hand/object interaction.
    camera_eye=(0.35, 1.90, 1.85),
    camera_target=(-0.15, -0.30, 0.72),
    object_position=(-0.18, -0.28, 0.70),
    goal_position=(0.12, -0.24, 0.70),
    reach_target=(-0.20, -0.25, 1.03),
    configure_scene=configure_tabletop,
)
