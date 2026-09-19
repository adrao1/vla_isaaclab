"""Task definitions."""

from .pick_place import PICK_PLACE_TASK
from .reach import REACH_TASK
from .scene_preview import SCENE_PREVIEW_TASK
from .ycb_pick_place_sugar_box import YCB_PICK_PLACE_SUGAR_BOX_TASK


def register_tasks() -> None:
    from ..registry import register

    register("tasks", REACH_TASK)
    register("tasks", PICK_PLACE_TASK)
    register("tasks", SCENE_PREVIEW_TASK)
    register("tasks", YCB_PICK_PLACE_SUGAR_BOX_TASK)
