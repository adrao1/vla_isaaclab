"""Component contracts used by the scenario composer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


SceneBuilder = Callable[[Any, Any], None]
ManagerBuilder = Callable[[Any, Any, Any], tuple[Any, Any, Any]]


@dataclass(frozen=True)
class WorldDefinition:
    component_id: str
    capabilities: frozenset[str]
    support_height: float
    robot_position: tuple[float, float, float]
    robot_orientation_wxyz: tuple[float, float, float, float]
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    object_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]
    reach_target: tuple[float, float, float]
    configure_scene: SceneBuilder = field(compare=False, repr=False)


@dataclass(frozen=True)
class RobotDefinition:
    component_id: str
    capabilities: frozenset[str]
    action_joint_names: tuple[str, ...]
    lower_body_joint_names: tuple[str, ...]
    waist_joint_names: tuple[str, ...]
    left_arm_joint_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    left_hand_joint_names: tuple[str, ...]
    right_hand_joint_names: tuple[str, ...]
    left_fingertip_body_names: tuple[str, ...]
    right_fingertip_body_names: tuple[str, ...]
    left_end_effector: str
    right_end_effector: str
    left_hand_open_joint_positions: tuple[float, ...]
    left_hand_closed_joint_positions: tuple[float, ...]
    configure_scene: SceneBuilder = field(compare=False, repr=False)


@dataclass(frozen=True)
class ObjectSetDefinition:
    component_id: str
    capabilities: frozenset[str]
    entity_roles: dict[str, str]
    configure_scene: Callable[[Any, WorldDefinition, RobotDefinition, ObjectSetDefinition], None] = field(
        compare=False, repr=False
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorRigDefinition:
    component_id: str
    capabilities: frozenset[str]
    camera_names: tuple[str, ...]
    configure_scene: Callable[[Any, WorldDefinition], None] = field(compare=False, repr=False)


@dataclass(frozen=True)
class TaskDefinition:
    component_id: str
    instruction: str
    required_world_capabilities: frozenset[str]
    required_robot_capabilities: frozenset[str]
    required_object_capabilities: frozenset[str]
    required_sensor_capabilities: frozenset[str]
    configure_scene: Callable[
        [Any, WorldDefinition, RobotDefinition, ObjectSetDefinition], None
    ] = field(compare=False, repr=False)
    build_managers: ManagerBuilder = field(compare=False, repr=False)


@dataclass(frozen=True)
class ControllerDefinition:
    component_id: str
    required_robot_capabilities: frozenset[str]
    factory: Callable[[Any, RobotDefinition, bool], Any] = field(compare=False, repr=False)
    behavior_prompt: str | None = None


@dataclass(frozen=True)
class EndEffectorTarget:
    """Task-space target produced by an Expert and consumed by a Controller."""

    position: torch.Tensor
    orientation: torch.Tensor | None
    gripper_closed_fraction: torch.Tensor


@dataclass(frozen=True)
class ExpertDefinition:
    component_id: str
    required_world_capabilities: frozenset[str]
    required_robot_capabilities: frozenset[str]
    required_object_capabilities: frozenset[str]
    factory: Callable[[Any, Any, Any], Any] = field(compare=False, repr=False)


@dataclass(frozen=True)
class ScenarioSelection:
    world: str
    robot: str
    objects: str
    sensors: str
    task: str
    controller: str
    expert: str | None = None

    @property
    def scenario_id(self) -> str:
        parts = (self.world, self.robot, self.objects, self.sensors, self.task, self.expert, self.controller)
        parts = tuple(part for part in parts if part is not None)
        return "__".join(part.replace("-v0", "") for part in parts)
