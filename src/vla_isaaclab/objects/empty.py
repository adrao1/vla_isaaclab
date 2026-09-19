from ..contracts import ObjectSetDefinition


def configure_empty(scene, world) -> None:
    return None


EMPTY_OBJECTS = ObjectSetDefinition(
    component_id="Objects-None-v0",
    capabilities=frozenset(),
    entity_roles={},
    configure_scene=configure_empty,
)
