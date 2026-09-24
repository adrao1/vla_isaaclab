"""Sugar-box success and failure termination terms."""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from ...common import (
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_OPEN_JOINT_POSITIONS,
)


def _filtered_contact_force(
    env: ManagerBasedRLEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Return sugar-box contact-force magnitude for one contact sensor."""
    sensor = env.scene[sensor_name]
    force_matrix = sensor.data.force_matrix_w

    if force_matrix is None:
        raise RuntimeError(
            f"Contact sensor {sensor_name!r} has no filtered contact-force data."
        )

    # Every Dex3 contact sensor monitors one robot link and filters against
    # one object body:
    #
    #     (num_envs, 1, 1, 3)
    #
    # Select the world-frame XYZ force vector for every environment.
    force = force_matrix[:, 0, 0, :]

    return torch.linalg.vector_norm(
        force,
        dim=-1,
    )


def dex3_grasp_contacts(
    env: ManagerBasedRLEnv,
    min_force: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return three-finger Dex3 contact metrics for the sugar box.

    Each logical finger may contact the object through any rigid link that
    belongs to that finger. A finger therefore counts as contacting when the
    maximum filtered contact force across its links is at least ``min_force``.

    A grasp requires simultaneous thumb, index, and middle-finger contact.
    """

    thumb_0 = _filtered_contact_force(
        env,
        "thumb_0_contact",
    )
    thumb_1 = _filtered_contact_force(
        env,
        "thumb_1_contact",
    )
    thumb_2 = _filtered_contact_force(
        env,
        "thumb_2_contact",
    )

    index_0 = _filtered_contact_force(
        env,
        "index_0_contact",
    )
    index_1 = _filtered_contact_force(
        env,
        "index_1_contact",
    )

    middle_0 = _filtered_contact_force(
        env,
        "middle_0_contact",
    )
    middle_1 = _filtered_contact_force(
        env,
        "middle_1_contact",
    )

    thumb_force = torch.stack(
        (
            thumb_0,
            thumb_1,
            thumb_2,
        ),
        dim=-1,
    ).amax(dim=-1)

    index_force = torch.stack(
        (
            index_0,
            index_1,
        ),
        dim=-1,
    ).amax(dim=-1)

    middle_force = torch.stack(
        (
            middle_0,
            middle_1,
        ),
        dim=-1,
    ).amax(dim=-1)

    thumb_contact = thumb_force >= min_force
    index_contact = index_force >= min_force
    middle_contact = middle_force >= min_force

    is_grasping = (
        thumb_contact
        & index_contact
        & middle_contact
    )

    return {
        "thumb_force": thumb_force,
        "index_force": index_force,
        "middle_force": middle_force,
        "thumb_contact": thumb_contact,
        "index_contact": index_contact,
        "middle_contact": middle_contact,
        "is_grasping": is_grasping,
    }


def task_metrics(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    command_name: str,
) -> dict[str, torch.Tensor]:
    """Metrics for the original pick-and-place task."""
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]

    palm_ids, _ = robot.find_bodies(
        [palm_body_name],
        preserve_order=True,
    )

    target = (
        env.command_manager.get_command(
            command_name
        )[:, :3]
        + env.scene.env_origins
    )

    return {
        "xy_error": torch.linalg.vector_norm(
            sugar_box.data.root_pos_w[:, :2]
            - target[:, :2],
            dim=-1,
        ),
        "height_error": torch.abs(
            sugar_box.data.root_pos_w[:, 2]
            - target[:, 2]
        ),
        "linear_speed": torch.linalg.vector_norm(
            sugar_box.data.root_lin_vel_w,
            dim=-1,
        ),
        "angular_speed": torch.linalg.vector_norm(
            sugar_box.data.root_ang_vel_w,
            dim=-1,
        ),
        "hand_distance": torch.linalg.vector_norm(
            robot.data.body_pos_w[:, palm_ids[0]]
            - sugar_box.data.root_pos_w,
            dim=-1,
        ),
    }


def grasp_metrics(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    initial_box_height: float,
    min_contact_force: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Metrics used by the grasp RL task."""
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]

    # Cache IDs because these reward/termination functions run every step.
    palm_id = getattr(
        env,
        "_grasp_reward_palm_id",
        None,
    )

    if palm_id is None:
        palm_ids, found = robot.find_bodies(
            [palm_body_name],
            preserve_order=True,
        )

        if list(found) != [palm_body_name]:
            raise RuntimeError(
                f"Could not resolve palm body "
                f"{palm_body_name!r}: {found}"
            )

        palm_id = palm_ids[0]
        env._grasp_reward_palm_id = palm_id

    hand_joint_ids = getattr(
        env,
        "_grasp_reward_hand_joint_ids",
        None,
    )

    if hand_joint_ids is None:
        hand_joint_ids, found = robot.find_joints(
            list(LEFT_HAND_JOINT_NAMES),
            preserve_order=True,
        )

        if list(found) != list(
            LEFT_HAND_JOINT_NAMES
        ):
            raise RuntimeError(
                "Left-hand joint mismatch: "
                f"expected "
                f"{list(LEFT_HAND_JOINT_NAMES)}, "
                f"found {found}"
            )

        env._grasp_reward_hand_joint_ids = (
            hand_joint_ids
        )

    palm_pos = robot.data.body_pos_w[
        :,
        palm_id,
    ]
    box_pos = sugar_box.data.root_pos_w

    hand_distance = torch.linalg.vector_norm(
        palm_pos - box_pos,
        dim=-1,
    )

    hand_pos = robot.data.joint_pos[
        :,
        hand_joint_ids,
    ]

    open_pos = torch.tensor(
        LEFT_HAND_OPEN_JOINT_POSITIONS,
        device=env.device,
        dtype=hand_pos.dtype,
    ).unsqueeze(0)

    closed_pos = torch.tensor(
        LEFT_HAND_CLOSED_JOINT_POSITIONS,
        device=env.device,
        dtype=hand_pos.dtype,
    ).unsqueeze(0)

    closure_per_joint = (
        (hand_pos - open_pos)
        / (
            closed_pos - open_pos
        ).clamp_min(1.0e-6)
    )

    closure_fraction = torch.clamp(
        closure_per_joint,
        0.0,
        1.0,
    ).mean(dim=-1)

    lift_height = (
        sugar_box.data.root_pos_w[:, 2]
        - initial_box_height
    )

    contacts = dex3_grasp_contacts(
        env,
        min_force=min_contact_force,
    )

    return {
        "hand_distance": hand_distance,
        "closure_fraction": closure_fraction,
        "lift_height": lift_height,
        "linear_speed": torch.linalg.vector_norm(
            sugar_box.data.root_lin_vel_w,
            dim=-1,
        ),
        "thumb_force": contacts["thumb_force"],
        "index_force": contacts["index_force"],
        "middle_force": contacts["middle_force"],
        "thumb_contact": contacts[
            "thumb_contact"
        ],
        "index_contact": contacts[
            "index_contact"
        ],
        "middle_contact": contacts[
            "middle_contact"
        ],
        "is_grasping": contacts[
            "is_grasping"
        ],
    }


def task_success(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    command_name: str,
    hold_steps: int = 15,
) -> torch.Tensor:
    """Original pick-and-place success condition."""
    metrics = task_metrics(
        env,
        palm_body_name,
        command_name,
    )

    instantaneous = (
        (metrics["xy_error"] < 0.015)
        & (metrics["height_error"] < 0.015)
        & (metrics["linear_speed"] < 0.04)
        & (metrics["angular_speed"] < 0.30)
        & (metrics["hand_distance"] > 0.20)
    )

    counter = getattr(
        env,
        "task_success_counter",
        None,
    )

    if (
        counter is None
        or counter.shape[0] != env.num_envs
    ):
        counter = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        env.task_success_counter = counter

    counter[:] = torch.where(
        instantaneous,
        counter + 1,
        torch.zeros_like(counter),
    )

    return counter >= hold_steps


def grasp_success(
    env: ManagerBasedRLEnv,
    min_contact_force: float = 0.5,
    hold_steps: int = 90,
) -> torch.Tensor:
    """Success when a physical three-finger Dex3 grasp is held."""
    contacts = dex3_grasp_contacts(
        env,
        min_force=min_contact_force,
    )

    # Success for this simplified experiment depends only on the same
    # physical grasp definition used by the grasp reward:
    #
    #     thumb AND index AND middle contact.
    #
    # There is deliberately no hand-distance or lift requirement here.
    instantaneous = contacts[
        "is_grasping"
    ]

    counter = getattr(
        env,
        "grasp_success_counter",
        None,
    )

    if (
        counter is None
        or counter.shape[0] != env.num_envs
    ):
        counter = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        env.grasp_success_counter = counter

    counter[:] = torch.where(
        instantaneous,
        counter + 1,
        torch.zeros_like(counter),
    )

    return counter >= hold_steps


def object_fallen(
    env: ManagerBasedRLEnv,
    support_height: float,
) -> torch.Tensor:
    sugar_box: RigidObject = env.scene[
        "object"
    ]

    return (
        sugar_box.data.root_pos_w[:, 2]
        < support_height - 0.05
    )


def invalid_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene[
        "object"
    ]

    return ~(
        torch.isfinite(
            robot.data.joint_pos
        ).all(dim=-1)
        & torch.isfinite(
            sugar_box.data.root_state_w
        ).all(dim=-1)
    )
