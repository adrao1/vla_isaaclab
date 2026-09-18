"""Tabletop layout for articulated appliance demonstrations."""

from ..contracts import WorldDefinition
from .tabletop import configure_tabletop


MICROWAVE_TABLETOP_WORLD = WorldDefinition(
    component_id="World-MicrowaveTabletop-v0",
    capabilities=frozenset(
        {"robot_spawn", "workspace", "support_surface", "object_spawn_region", "camera_mount"}
    ),
    support_height=0.62,
    robot_position=(0.0, -0.95, 0.74),
    robot_orientation_wxyz=(0.70710678, 0.0, 0.0, 0.70710678),
    # Oblique view from the appliance front-right side.
    camera_eye=(2.6, -3.2, 2.2),
    camera_target=(0.0, -0.30, 0.90),
    # Isaac Lab positions the imported articulation root directly; the
    # upstream MJCF's outer showcase offset is outside that articulation.
    object_position=(0.0, 0.0, 0.62),
    goal_position=(0.0, 0.0, 0.62),
    reach_target=(0.0, -0.25, 1.0),
    configure_scene=configure_tabletop,
)
