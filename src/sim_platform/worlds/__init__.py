"""World definitions."""

from .pedestal import PEDESTAL_WORLD
from .tabletop import TABLETOP_WORLD


def register_worlds() -> None:
    from ..registry import register

    register("worlds", TABLETOP_WORLD)
    register("worlds", PEDESTAL_WORLD)
