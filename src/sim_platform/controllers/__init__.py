"""Controller definitions."""

from .standing import RAISE_LOWER_CONTROLLER, STANDING_CONTROLLER


def register_controllers() -> None:
    from ..registry import register

    register("controllers", STANDING_CONTROLLER)
    register("controllers", RAISE_LOWER_CONTROLLER)
