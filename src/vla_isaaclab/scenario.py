"""Compose registered scenario components."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .actions import UpperBodyJointActionCfg
from .contracts import ScenarioSelection
from .registry import get


@configclass
class ComposedSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.25)),
        collision_group=-1,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1100.0, color=(0.85, 0.88, 1.0)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88), angle=0.8),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.3827, 0.0, 0.0)),
    )


@configclass
class ScenarioEnvCfg(ManagerBasedRLEnvCfg):
    env_name: str = "ComposableScenario-v0"
    scene: ComposedSceneCfg = ComposedSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    observations: object = None
    actions: object = None
    rewards: object = None
    terminations: object = None
    episode_length_s: float = 10.0
    decimation: int = 4
    scenario_metadata: dict = {}

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 0.8
        self.sim.physics_material.dynamic_friction = 0.65
        self.sim.physics_material.restitution = 0.02
        self.sim.physics_material.friction_combine_mode = "max"
        self.rerender_on_reset = True
        self.wait_for_textures = True


@dataclass
class ScenarioBundle:
    selection: ScenarioSelection
    env_cfg: ScenarioEnvCfg
    world: object
    robot: object
    objects: object
    sensors: object
    task: object
    expert: object | None
    controller: object


def require(component_id, actual, required, kind):
    missing = set(required) - set(actual)
    if missing:
        raise ValueError(f"{component_id} lacks required {kind} capabilities: {sorted(missing)}")


def compose_scenario(selection: ScenarioSelection, enable_sensors: bool = True) -> ScenarioBundle:
    world = get("worlds", selection.world)
    robot = get("robots", selection.robot)
    objects = get("objects", selection.objects)
    sensors = get("sensors", selection.sensors)
    task = get("tasks", selection.task)
    expert = get("experts", selection.expert) if selection.expert is not None else None
    controller = get("controllers", selection.controller)

    require(world.component_id, world.capabilities, task.required_world_capabilities, "world")
    require(robot.component_id, robot.capabilities, task.required_robot_capabilities, "robot")
    require(objects.component_id, objects.capabilities, task.required_object_capabilities, "object")
    require(sensors.component_id, sensors.capabilities, task.required_sensor_capabilities, "sensor")
    require(robot.component_id, robot.capabilities, controller.required_robot_capabilities, "controller")
    if expert is not None:
        require(world.component_id, world.capabilities, expert.required_world_capabilities, "expert world")
        require(robot.component_id, robot.capabilities, expert.required_robot_capabilities, "expert robot")
        require(objects.component_id, objects.capabilities, expert.required_object_capabilities, "expert object")

    cfg = ScenarioEnvCfg()
    cfg.env_name = selection.scenario_id
    world.configure_scene(cfg.scene, world)
    robot.configure_scene(cfg.scene, world)
    objects.configure_scene(cfg.scene, world)
    if enable_sensors:
        sensors.configure_scene(cfg.scene, world)
    task.configure_scene(cfg.scene, world, robot, objects)

    @configclass
    class ActionsCfg:
        upper_body = UpperBodyJointActionCfg(
            asset_name="robot", joint_names=list(robot.action_joint_names), use_joint_limits=True
        )

    cfg.actions = ActionsCfg()
    cfg.observations, cfg.rewards, cfg.terminations = task.build_managers(world, robot, objects)
    cfg.viewer.eye = world.camera_eye
    cfg.viewer.lookat = world.camera_target
    cfg.scenario_metadata = {
        "scenario_id": selection.scenario_id,
        "world": selection.world,
        "robot": selection.robot,
        "objects": selection.objects,
        "object_set_metadata": objects.metadata,
        "sensors": selection.sensors,
        "sensors_enabled": enable_sensors,
        "task": selection.task,
        "task_instruction": task.instruction,
        "expert": selection.expert,
        "controller": selection.controller,
        "support_height": world.support_height,
        "reach_target": world.reach_target,
        "left_end_effector": robot.left_end_effector,
    }
    return ScenarioBundle(selection, cfg, world, robot, objects, sensors, task, expert, controller)
