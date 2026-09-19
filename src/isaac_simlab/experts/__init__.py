"""Scripted task experts."""

from .ycb_pick_place_sugar_box import YCB_PICK_PLACE_SUGAR_BOX_EXPERT


def register_experts() -> None:
    from ..registry import register

    register("experts", YCB_PICK_PLACE_SUGAR_BOX_EXPERT)
