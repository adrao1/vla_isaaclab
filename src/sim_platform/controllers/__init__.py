"""Controller definitions."""

from .bowl_to_plate import BOWL_TO_PLATE_CONTROLLER
from .raise_lower import RAISE_LOWER_CONTROLLER
from .standing import STANDING_CONTROLLER


def register_controllers() -> None:
    from ..registry import register

    register("controllers", STANDING_CONTROLLER)
    register("controllers", RAISE_LOWER_CONTROLLER)
    register("controllers", BOWL_TO_PLATE_CONTROLLER)
