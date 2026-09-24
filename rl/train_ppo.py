#!/usr/bin/env python3
"""Train PPO with 7-D end-effector delta actions on VLA Isaac Lab tasks."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "rl"))

from isaaclab.app import AppLauncher


# =====================================================================
# Arguments
# =====================================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--task",
        default="VLA-YCBSugarBox-G1-Grasp-v0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000_000,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--update-epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--norm-adv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
    )

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    # RL training is state-based. We do not need RGB sensors.
    args.enable_cameras = False

    return args


ARGS = parse_args()

ARGS.experience = str(
    PROJECT_ROOT
    / "configs"
    / "ycb.python.headless.kit"
)

ARGS.kit_args = (
    f"--portable-root "
    f"{PROJECT_ROOT}/outputs/runtime/kit"
)

APP = AppLauncher(ARGS).app


# =====================================================================
# Imports that require Isaac Sim to be running
# =====================================================================

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import vla_isaaclab  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

from ee_delta_controller import EEDeltaController

from vla_isaaclab.envs.common import (
    LEFT_END_EFFECTOR,
    SUPPORT_HEIGHT,
)

from vla_isaaclab.envs.ycb_sugar_box.env_cfg import (
    SUGAR_BOX_HALF_HEIGHT_M,
)

from vla_isaaclab.envs.ycb_sugar_box.mdp import (
    grasp_metrics,
)


GRASP_TASK = "VLA-YCBSugarBox-G1-Grasp-v0"


# =====================================================================
# Neural network
# =====================================================================


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
):
    """X-Sim / CleanRL-style orthogonal initialization."""
    torch.nn.init.orthogonal_(
        layer.weight,
        std,
    )

    torch.nn.init.constant_(
        layer.bias,
        bias_const,
    )

    return layer


class Agent(nn.Module):
    """X-Sim-style PPO actor/critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    obs_dim,
                    256,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    256,
                    256,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    256,
                    256,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    256,
                    1,
                )
            ),
        )

        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    obs_dim,
                    256,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    256,
                    256,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    256,
                    256,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    256,
                    action_dim,
                ),
                std=0.01 * np.sqrt(2),
            ),
        )

        # X-Sim initializes log std to -0.5.
        self.actor_logstd = nn.Parameter(
            torch.ones(
                1,
                action_dim,
            )
            * -0.5
        )

    def get_value(
        self,
        x: torch.Tensor,
    ):
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        action_mean = self.actor_mean(x)

        action_logstd = (
            self.actor_logstd.expand_as(
                action_mean
            )
        )

        action_std = torch.exp(
            action_logstd
        )

        probs = Normal(
            action_mean,
            action_std,
        )

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# =====================================================================
# Helpers
# =====================================================================


def get_policy_obs(
    obs_dict,
) -> torch.Tensor:
    """Return the concatenated policy observation tensor."""

    if isinstance(
        obs_dict,
        dict,
    ):
        if "policy" not in obs_dict:
            raise RuntimeError(
                "Observation dictionary does not contain "
                "'policy'. "
                f"Available keys: {list(obs_dict.keys())}"
            )

        return obs_dict["policy"]

    return obs_dict


def make_writer(
    run_dir: Path,
):
    """TensorBoard is optional."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print(
            "TensorBoard not installed; "
            "continuing without SummaryWriter."
        )
        return None

    return SummaryWriter(
        str(run_dir)
    )


# =====================================================================
# Main
# =====================================================================


def main():
    # -----------------------------------------------------------------
    # Seeds
    # -----------------------------------------------------------------

    random.seed(
        ARGS.seed
    )

    np.random.seed(
        ARGS.seed
    )

    torch.manual_seed(
        ARGS.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            ARGS.seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # -----------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------

    cfg = parse_env_cfg(
        ARGS.task,
        device=ARGS.device,
        num_envs=ARGS.num_envs,
    )

    # PPO does not use RGB observations.
    for camera_name in (
        "camera",
        "cam_side",
        "cam_left_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ):
        if hasattr(
            cfg.scene,
            camera_name,
        ):
            setattr(
                cfg.scene,
                camera_name,
                None,
            )

    env = gym.make(
        ARGS.task,
        cfg=cfg,
    ).unwrapped

    obs_dict, _ = env.reset(
        seed=ARGS.seed
    )

    next_obs = get_policy_obs(
        obs_dict
    )


    # -----------------------------------------------------------------
    # Dimensions
    # -----------------------------------------------------------------

    device = torch.device(
        ARGS.device
    )

    num_envs = ARGS.num_envs

    obs_dim = int(
        next_obs.shape[-1]
    )

    # Our RL interface is:
    #
    # [dx, dy, dz, droll, dpitch, dyaw, gripper]
    action_dim = 7


    # -----------------------------------------------------------------
    # Controller
    # -----------------------------------------------------------------

    controller = EEDeltaController(
        env
    )

    controller.reset()


    # -----------------------------------------------------------------
    # PPO
    # -----------------------------------------------------------------

    agent = Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=ARGS.learning_rate,
        eps=1e-5,
    )


    # -----------------------------------------------------------------
    # Batch configuration
    # -----------------------------------------------------------------

    batch_size = (
        num_envs
        * ARGS.num_steps
    )

    if (
        batch_size
        % ARGS.num_minibatches
        != 0
    ):
        raise ValueError(
            f"batch_size={batch_size} must be divisible "
            f"by num_minibatches={ARGS.num_minibatches}"
        )

    minibatch_size = (
        batch_size
        // ARGS.num_minibatches
    )

    num_iterations = (
        ARGS.total_timesteps
        // batch_size
    )

    if num_iterations < 1:
        raise ValueError(
            "total_timesteps must be at least one PPO batch. "
            f"batch_size={batch_size}, "
            f"total_timesteps={ARGS.total_timesteps}"
        )


    # -----------------------------------------------------------------
    # Rollout storage
    # -----------------------------------------------------------------

    observations = torch.zeros(
        (
            ARGS.num_steps,
            num_envs,
            obs_dim,
        ),
        dtype=torch.float32,
        device=device,
    )

    actions = torch.zeros(
        (
            ARGS.num_steps,
            num_envs,
            action_dim,
        ),
        dtype=torch.float32,
        device=device,
    )

    logprobs = torch.zeros(
        (
            ARGS.num_steps,
            num_envs,
        ),
        dtype=torch.float32,
        device=device,
    )

    rewards = torch.zeros(
        (
            ARGS.num_steps,
            num_envs,
        ),
        dtype=torch.float32,
        device=device,
    )

    dones = torch.zeros(
        (
            ARGS.num_steps,
            num_envs,
        ),
        dtype=torch.float32,
        device=device,
    )

    values = torch.zeros(
        (
            ARGS.num_steps,
            num_envs,
        ),
        dtype=torch.float32,
        device=device,
    )


    # -----------------------------------------------------------------
    # Initial state
    # -----------------------------------------------------------------

    next_obs = next_obs.to(
        device
    )

    next_done = torch.zeros(
        num_envs,
        dtype=torch.float32,
        device=device,
    )


    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    run_name = (
        f"{ARGS.task}"
        f"__ppo"
        f"__{ARGS.seed}"
        f"__{int(time.time())}"
    )

    run_dir = (
        PROJECT_ROOT
        / "runs"
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = make_writer(
        run_dir
    )

    global_step = 0
    start_time = time.time()


    # -----------------------------------------------------------------
    # Grasp diagnostic constants
    # -----------------------------------------------------------------

    is_grasp_task = (
        ARGS.task == GRASP_TASK
    )

    initial_box_height = (
        SUPPORT_HEIGHT
        + SUGAR_BOX_HALF_HEIGHT_M
    )


    # =================================================================
    # Configuration summary
    # =================================================================

    print()
    print("=" * 72)
    print("PPO TRAINING")
    print("=" * 72)

    print(
        "task:",
        ARGS.task,
    )

    print(
        "device:",
        ARGS.device,
    )

    print(
        "num_envs:",
        num_envs,
    )

    print(
        "obs_dim:",
        obs_dim,
    )

    print(
        "RL action_dim:",
        action_dim,
    )

    print(
        "Isaac action_dim:",
        env.action_manager.total_action_dim,
    )

    print(
        "num_steps:",
        ARGS.num_steps,
    )

    print(
        "batch_size:",
        batch_size,
    )

    print(
        "num_minibatches:",
        ARGS.num_minibatches,
    )

    print(
        "minibatch_size:",
        minibatch_size,
    )

    print(
        "update_epochs:",
        ARGS.update_epochs,
    )

    print(
        "num_iterations:",
        num_iterations,
    )

    print(
        "initial actor std:",
        torch.exp(
            agent.actor_logstd
        )
        .mean()
        .item(),
    )

    print(
        "run_dir:",
        run_dir,
    )

    print("=" * 72)
    print()


    # =================================================================
    # PPO training loop
    # =================================================================

    for iteration in range(
        1,
        num_iterations + 1,
    ):
        iteration_start = time.time()


        # -------------------------------------------------------------
        # Grasp-task diagnostics for this rollout
        # -------------------------------------------------------------

        diag_distance_sum = 0.0
        diag_closure_sum = 0.0
        diag_lift_sum = 0.0
        diag_samples = 0

        diag_min_distance = float("inf")
        diag_max_closure = 0.0
        diag_max_lift = float("-inf")

        diag_successes = 0
        diag_done_episodes = 0


        # -------------------------------------------------------------
        # Learning-rate annealing
        # -------------------------------------------------------------

        if ARGS.anneal_lr:
            fraction = (
                1.0
                - (
                    iteration - 1.0
                )
                / num_iterations
            )

            current_lr = (
                fraction
                * ARGS.learning_rate
            )

            optimizer.param_groups[
                0
            ]["lr"] = current_lr


        # =============================================================
        # Collect rollout
        # =============================================================

        agent.eval()

        rollout_start = time.time()


        for step in range(
            ARGS.num_steps
        ):
            global_step += num_envs


            observations[
                step
            ] = next_obs

            dones[
                step
            ] = next_done


            # ---------------------------------------------------------
            # Grasp physical state BEFORE env.step()
            #
            # We do this before env.step because Isaac Lab may
            # automatically reset environments that terminate.
            # ---------------------------------------------------------

            if is_grasp_task:
                with torch.no_grad():
                    metrics = grasp_metrics(
                        env,
                        LEFT_END_EFFECTOR,
                        initial_box_height,
                    )

                    distance = metrics[
                        "hand_distance"
                    ]

                    closure = metrics[
                        "closure_fraction"
                    ]

                    lift = metrics[
                        "lift_height"
                    ]

                    diag_distance_sum += (
                        distance.sum().item()
                    )

                    diag_closure_sum += (
                        closure.sum().item()
                    )

                    diag_lift_sum += (
                        lift.sum().item()
                    )

                    diag_samples += num_envs

                    diag_min_distance = min(
                        diag_min_distance,
                        distance.min().item(),
                    )

                    diag_max_closure = max(
                        diag_max_closure,
                        closure.max().item(),
                    )

                    diag_max_lift = max(
                        diag_max_lift,
                        lift.max().item(),
                    )


            # ---------------------------------------------------------
            # Actor + critic
            #
            # Store the RAW Gaussian sample for PPO.
            #
            # This matches X-Sim.
            # ---------------------------------------------------------

            with torch.no_grad():
                (
                    raw_action,
                    logprob,
                    _,
                    value,
                ) = agent.get_action_and_value(
                    next_obs
                )


                values[
                    step
                ] = value.flatten()


            actions[
                step
            ] = raw_action

            logprobs[
                step
            ] = logprob


            # ---------------------------------------------------------
            # X-Sim clips the sampled Gaussian action before env.step().
            #
            # Our action space is the normalized 7-D EE command:
            #
            # [-1, 1]
            # ---------------------------------------------------------

            executed_action = torch.clamp(
                raw_action,
                -1.0,
                1.0,
            )


            # ---------------------------------------------------------
            # Convert:
            #
            # 7-D EE action
            #
            #       ->
            #
            # 43-D Isaac joint action
            # ---------------------------------------------------------

            env_action = (
                controller.compute(
                    executed_action
                )
            )


            # ---------------------------------------------------------
            # Sanity check before stepping
            # ---------------------------------------------------------

            if not torch.isfinite(
                env_action
            ).all():
                raise RuntimeError(
                    f"Non-finite environment action "
                    f"at rollout step {step}"
                )


            # ---------------------------------------------------------
            # Environment step
            # ---------------------------------------------------------

            (
                next_obs_dict,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                env_action
            )


            # ---------------------------------------------------------
            # Grasp success / episode diagnostics
            # ---------------------------------------------------------

            if is_grasp_task:
                success = (
                    env.termination_manager
                    .get_term(
                        "success"
                    )
                )

                diag_successes += int(
                    success.sum().item()
                )

                diag_done_episodes += int(
                    torch.logical_or(
                        terminated,
                        truncated,
                    )
                    .sum()
                    .item()
                )


            next_obs = get_policy_obs(
                next_obs_dict
            )


            next_done = (
                torch.logical_or(
                    terminated,
                    truncated,
                )
                .float()
            )


            rewards[
                step
            ] = reward


            # ---------------------------------------------------------
            # Runtime checks
            # ---------------------------------------------------------

            if not torch.isfinite(
                next_obs
            ).all():
                raise RuntimeError(
                    f"Non-finite observation "
                    f"at rollout step {step}"
                )


            if not torch.isfinite(
                reward
            ).all():
                raise RuntimeError(
                    f"Non-finite reward "
                    f"at rollout step {step}"
                )


        rollout_time = (
            time.time()
            - rollout_start
        )


        # =============================================================
        # Bootstrap final critic value
        # =============================================================

        with torch.no_grad():
            next_value = (
                agent
                .get_value(
                    next_obs
                )
                .reshape(-1)
            )


        # =============================================================
        # GAE
        #
        # Current implementation:
        #
        # terminated and truncated are both treated as done.
        #
        # This is not yet X-Sim's finite-horizon GAE handling.
        # =============================================================

        advantages = torch.zeros_like(
            rewards
        )


        last_gae = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=device,
        )


        for t in reversed(
            range(
                ARGS.num_steps
            )
        ):

            if (
                t
                == ARGS.num_steps - 1
            ):
                next_non_terminal = (
                    1.0
                    - next_done
                )

                next_values = (
                    next_value
                )

            else:
                next_non_terminal = (
                    1.0
                    - dones[
                        t + 1
                    ]
                )

                next_values = (
                    values[
                        t + 1
                    ]
                )


            delta = (
                rewards[t]
                + ARGS.gamma
                * next_values
                * next_non_terminal
                - values[t]
            )


            last_gae = (
                delta
                + ARGS.gamma
                * ARGS.gae_lambda
                * next_non_terminal
                * last_gae
            )


            advantages[
                t
            ] = last_gae


        returns = (
            advantages
            + values
        )


        # =============================================================
        # Flatten rollout
        # =============================================================

        batch_obs = observations.reshape(
            -1,
            obs_dim,
        )

        batch_actions = actions.reshape(
            -1,
            action_dim,
        )

        batch_logprobs = logprobs.reshape(
            -1
        )

        batch_advantages = advantages.reshape(
            -1
        )

        batch_returns = returns.reshape(
            -1
        )

        batch_values = values.reshape(
            -1
        )


        # =================================================================
        # PPO update
        # =================================================================

        agent.train()


        batch_indices = np.arange(
            batch_size
        )

        clip_fractions = []


        approx_kl = torch.tensor(
            0.0,
            device=device,
        )

        old_approx_kl = torch.tensor(
            0.0,
            device=device,
        )

        pg_loss = torch.tensor(
            0.0,
            device=device,
        )

        value_loss = torch.tensor(
            0.0,
            device=device,
        )

        entropy_loss = torch.tensor(
            0.0,
            device=device,
        )


        update_start = time.time()

        stop_for_kl = False


        for epoch in range(
            ARGS.update_epochs
        ):

            np.random.shuffle(
                batch_indices
            )


            for start in range(
                0,
                batch_size,
                minibatch_size,
            ):

                end = (
                    start
                    + minibatch_size
                )


                minibatch_indices = (
                    batch_indices[
                        start:end
                    ]
                )


                # -----------------------------------------------------
                # Re-evaluate stored RAW actions
                # -----------------------------------------------------

                (
                    _,
                    new_logprob,
                    entropy,
                    new_value,
                ) = agent.get_action_and_value(
                    batch_obs[
                        minibatch_indices
                    ],
                    batch_actions[
                        minibatch_indices
                    ],
                )


                # -----------------------------------------------------
                # PPO probability ratio
                # -----------------------------------------------------

                log_ratio = (
                    new_logprob
                    - batch_logprobs[
                        minibatch_indices
                    ]
                )


                ratio = torch.exp(
                    log_ratio
                )


                # -----------------------------------------------------
                # KL diagnostics
                # -----------------------------------------------------

                with torch.no_grad():
                    old_approx_kl = (
                        -log_ratio
                    ).mean()


                    approx_kl = (
                        (
                            ratio
                            - 1.0
                        )
                        - log_ratio
                    ).mean()


                    clip_fraction = (
                        (
                            (
                                ratio
                                - 1.0
                            ).abs()
                            > ARGS.clip_coef
                        )
                        .float()
                        .mean()
                        .item()
                    )


                    clip_fractions.append(
                        clip_fraction
                    )


                # -----------------------------------------------------
                # Optional KL early stopping
                # -----------------------------------------------------

                if (
                    ARGS.target_kl
                    is not None
                    and approx_kl.item()
                    > ARGS.target_kl
                ):
                    stop_for_kl = True
                    break


                # -----------------------------------------------------
                # Advantage normalization
                # -----------------------------------------------------

                minibatch_advantages = (
                    batch_advantages[
                        minibatch_indices
                    ]
                )


                if ARGS.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages
                        - minibatch_advantages.mean()
                    ) / (
                        minibatch_advantages.std()
                        + 1e-8
                    )


                # -----------------------------------------------------
                # PPO clipped policy loss
                # -----------------------------------------------------

                policy_loss_1 = (
                    -minibatch_advantages
                    * ratio
                )


                policy_loss_2 = (
                    -minibatch_advantages
                    * torch.clamp(
                        ratio,
                        1.0
                        - ARGS.clip_coef,
                        1.0
                        + ARGS.clip_coef,
                    )
                )


                pg_loss = (
                    torch.max(
                        policy_loss_1,
                        policy_loss_2,
                    )
                    .mean()
                )


                # -----------------------------------------------------
                # Critic loss
                # -----------------------------------------------------

                new_value = (
                    new_value.view(-1)
                )


                value_loss = (
                    0.5
                    * (
                        (
                            new_value
                            - batch_returns[
                                minibatch_indices
                            ]
                        )
                        ** 2
                    )
                    .mean()
                )


                # -----------------------------------------------------
                # Entropy
                # -----------------------------------------------------

                entropy_loss = (
                    entropy.mean()
                )


                # -----------------------------------------------------
                # Combined loss
                # -----------------------------------------------------

                loss = (
                    pg_loss
                    - ARGS.ent_coef
                    * entropy_loss
                    + ARGS.vf_coef
                    * value_loss
                )


                # -----------------------------------------------------
                # Gradient update
                # -----------------------------------------------------

                optimizer.zero_grad()

                loss.backward()


                nn.utils.clip_grad_norm_(
                    agent.parameters(),
                    ARGS.max_grad_norm,
                )


                optimizer.step()


            if stop_for_kl:
                break


        update_time = (
            time.time()
            - update_start
        )


        # =================================================================
        # PPO diagnostics
        # =================================================================

        with torch.no_grad():

            y_pred = (
                batch_values
                .detach()
                .cpu()
                .numpy()
            )


            y_true = (
                batch_returns
                .detach()
                .cpu()
                .numpy()
            )


            variance_y = np.var(
                y_true
            )


            if variance_y == 0:
                explained_variance = np.nan

            else:
                explained_variance = (
                    1.0
                    - np.var(
                        y_true
                        - y_pred
                    )
                    / variance_y
                )


        elapsed = (
            time.time()
            - start_time
        )


        sps = int(
            global_step
            / elapsed
        )


        mean_reward = (
            rewards.mean().item()
        )


        mean_advantage = (
            advantages.mean().item()
        )


        actor_std = (
            torch.exp(
                agent.actor_logstd
            )
            .mean()
            .item()
        )


        if clip_fractions:
            mean_clip_fraction = float(
                np.mean(
                    clip_fractions
                )
            )
        else:
            mean_clip_fraction = 0.0


        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )


        iteration_time = (
            time.time()
            - iteration_start
        )


        # -----------------------------------------------------------------
        # Grasp diagnostics
        # -----------------------------------------------------------------

        mean_hand_distance = 0.0
        mean_closure = 0.0
        mean_lift = 0.0
        success_rate = 0.0


        if (
            is_grasp_task
            and diag_samples > 0
        ):
            mean_hand_distance = (
                diag_distance_sum
                / diag_samples
            )

            mean_closure = (
                diag_closure_sum
                / diag_samples
            )

            mean_lift = (
                diag_lift_sum
                / diag_samples
            )

            if diag_done_episodes > 0:
                success_rate = (
                    diag_successes
                    / diag_done_episodes
                )


        # -----------------------------------------------------------------
        # Console output
        # -----------------------------------------------------------------

        print(
            f"iter={iteration:04d}/{num_iterations} "
            f"step={global_step:09d} "
            f"reward={mean_reward:+.5f} "
            f"pg={pg_loss.item():+.5f} "
            f"v={value_loss.item():.5f} "
            f"entropy={entropy_loss.item():.5f} "
            f"kl={approx_kl.item():.6f} "
            f"clipfrac={mean_clip_fraction:.3f} "
            f"std={actor_std:.3f} "
            f"sps={sps}"
        )


        if is_grasp_task:
            print(
                "    grasp: "
                f"dist_mean={mean_hand_distance:.4f} "
                f"dist_min={diag_min_distance:.4f} "
                f"closure_mean={mean_closure:.4f} "
                f"closure_max={diag_max_closure:.4f} "
                f"lift_mean={mean_lift:.4f} "
                f"lift_max={diag_max_lift:.4f} "
                f"successes={diag_successes} "
                f"episodes={diag_done_episodes} "
                f"success_rate={success_rate:.3f}"
            )


        # =================================================================
        # TensorBoard
        # =================================================================

        if writer is not None:

            writer.add_scalar(
                "charts/learning_rate",
                current_lr,
                global_step,
            )

            writer.add_scalar(
                "charts/SPS",
                sps,
                global_step,
            )

            writer.add_scalar(
                "charts/actor_std",
                actor_std,
                global_step,
            )

            writer.add_scalar(
                "train/mean_reward",
                mean_reward,
                global_step,
            )

            writer.add_scalar(
                "train/mean_advantage",
                mean_advantage,
                global_step,
            )

            writer.add_scalar(
                "losses/policy_loss",
                pg_loss.item(),
                global_step,
            )

            writer.add_scalar(
                "losses/value_loss",
                value_loss.item(),
                global_step,
            )

            writer.add_scalar(
                "losses/entropy",
                entropy_loss.item(),
                global_step,
            )

            writer.add_scalar(
                "losses/old_approx_kl",
                old_approx_kl.item(),
                global_step,
            )

            writer.add_scalar(
                "losses/approx_kl",
                approx_kl.item(),
                global_step,
            )

            writer.add_scalar(
                "losses/clip_fraction",
                mean_clip_fraction,
                global_step,
            )

            writer.add_scalar(
                "losses/explained_variance",
                explained_variance,
                global_step,
            )

            writer.add_scalar(
                "time/rollout_time",
                rollout_time,
                global_step,
            )

            writer.add_scalar(
                "time/update_time",
                update_time,
                global_step,
            )

            writer.add_scalar(
                "time/iteration_time",
                iteration_time,
                global_step,
            )


            if is_grasp_task:

                writer.add_scalar(
                    "grasp/mean_hand_distance",
                    mean_hand_distance,
                    global_step,
                )

                writer.add_scalar(
                    "grasp/min_hand_distance",
                    diag_min_distance,
                    global_step,
                )

                writer.add_scalar(
                    "grasp/mean_closure",
                    mean_closure,
                    global_step,
                )

                writer.add_scalar(
                    "grasp/max_closure",
                    diag_max_closure,
                    global_step,
                )

                writer.add_scalar(
                    "grasp/mean_lift",
                    mean_lift,
                    global_step,
                )

                writer.add_scalar(
                    "grasp/max_lift",
                    diag_max_lift,
                    global_step,
                )

                writer.add_scalar(
                    "grasp/successes",
                    diag_successes,
                    global_step,
                )

                writer.add_scalar(
                    "grasp/success_rate",
                    success_rate,
                    global_step,
                )


        # =================================================================
        # Periodic checkpoint
        # =================================================================

        if (
            ARGS.save_every > 0
            and iteration
            % ARGS.save_every
            == 0
        ):

            checkpoint_path = (
                run_dir
                / f"ckpt_{iteration}.pt"
            )


            torch.save(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(ARGS),
                },
                checkpoint_path,
            )


            print(
                f"saved: "
                f"{checkpoint_path}"
            )


    # =================================================================
    # Final checkpoint
    # =================================================================

    final_path = (
        run_dir
        / "final_ckpt.pt"
    )


    torch.save(
        {
            "iteration": num_iterations,
            "global_step": global_step,
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(ARGS),
        },
        final_path,
    )


    print()
    print(
        "final model saved to:",
        final_path,
    )


    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    if writer is not None:
        writer.close()


    env.close()


# =====================================================================
# Entry point
# =====================================================================


if __name__ == "__main__":
    try:
        main()

    finally:
        APP.close()
