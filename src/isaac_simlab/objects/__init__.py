"""Object-set definitions."""

from .dinnerware import DINNERWARE_OBJECTS
from .empty import EMPTY_OBJECTS
from .microwave import MICROWAVE_OBJECTS
from .ycb import YCB_BASIC_OBJECTS, YCB_SUGAR_BOX_OBJECTS


def register_object_sets() -> None:
    from ..registry import register

    register("objects", EMPTY_OBJECTS)
    register("objects", DINNERWARE_OBJECTS)
    register("objects", YCB_BASIC_OBJECTS)
    register("objects", YCB_SUGAR_BOX_OBJECTS)
    register("objects", MICROWAVE_OBJECTS)
