"""Controller definitions."""

from .left_arm_differential_ik import LEFT_ARM_DIFFERENTIAL_IK_CONTROLLER
from .raise_lower import RAISE_LOWER_CONTROLLER
from .standing import STANDING_CONTROLLER


def register_controllers() -> None:
    from ..registry import register

    register("controllers", STANDING_CONTROLLER)
    register("controllers", RAISE_LOWER_CONTROLLER)
    register("controllers", LEFT_ARM_DIFFERENTIAL_IK_CONTROLLER)
