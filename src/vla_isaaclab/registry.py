"""Registries for independently selectable simulation components."""

from __future__ import annotations

from typing import Any


_REGISTRIES: dict[str, dict[str, Any]] = {
    "worlds": {},
    "robots": {},
    "objects": {},
    "sensors": {},
    "tasks": {},
    "experts": {},
    "controllers": {},
}
_DEFAULTS_REGISTERED = False


def register(kind: str, component: Any) -> None:
    registry = _REGISTRIES[kind]
    component_id = component.component_id
    if component_id in registry:
        raise KeyError(f"Duplicate {kind} component ID: {component_id}")
    registry[component_id] = component


def get(kind: str, component_id: str) -> Any:
    try:
        return _REGISTRIES[kind][component_id]
    except KeyError as error:
        available = ", ".join(sorted(_REGISTRIES[kind]))
        raise KeyError(f"Unknown {kind} component '{component_id}'. Available: {available}") from error


def list_components() -> dict[str, list[str]]:
    return {kind: sorted(registry) for kind, registry in _REGISTRIES.items()}


def register_defaults() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    from .controllers import register_controllers
    from .experts import register_experts
    from .objects import register_object_sets
    from .robots import register_robots
    from .sensors import register_sensor_rigs
    from .tasks import register_tasks
    from .worlds import register_worlds

    register_worlds()
    register_robots()
    register_object_sets()
    register_sensor_rigs()
    register_tasks()
    register_experts()
    register_controllers()
    _DEFAULTS_REGISTERED = True
