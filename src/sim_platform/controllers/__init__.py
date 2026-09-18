"""Controller definitions."""

from .raise_lower import RAISE_LOWER_CONTROLLER
from .standing import STANDING_CONTROLLER


def register_controllers() -> None:
    from ..registry import register

    register("controllers", STANDING_CONTROLLER)
    register("controllers", RAISE_LOWER_CONTROLLER)
