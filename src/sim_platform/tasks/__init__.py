"""Task definitions."""

from .bowl_to_plate import BOWL_TO_PLATE_TASK
from .pick_place import PICK_PLACE_TASK
from .reach import REACH_TASK
from .scene_preview import SCENE_PREVIEW_TASK


def register_tasks() -> None:
    from ..registry import register

    register("tasks", REACH_TASK)
    register("tasks", PICK_PLACE_TASK)
    register("tasks", BOWL_TO_PLATE_TASK)
    register("tasks", SCENE_PREVIEW_TASK)
