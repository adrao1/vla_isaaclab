"""Task definitions."""

from .pick_place import PICK_PLACE_TASK
from .reach import REACH_TASK


def register_tasks() -> None:
    from ..registry import register

    register("tasks", REACH_TASK)
    register("tasks", PICK_PLACE_TASK)
