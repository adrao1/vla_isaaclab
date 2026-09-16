"""Shared world construction helpers."""

import isaaclab.sim as sim_utils


def static_box(size, color=(0.92, 0.92, 0.92)) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75, metallic=0.0),
    )
