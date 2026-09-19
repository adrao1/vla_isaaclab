"""Robot definitions."""

from .unitree_g1 import UNITREE_G1


def register_robots() -> None:
    from ..registry import register

    register("robots", UNITREE_G1)
