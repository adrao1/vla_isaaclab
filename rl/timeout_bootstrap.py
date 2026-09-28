"""Timeout bootstrapping for this project's state-only RL environment.

Captures final observations before Isaac Lab's automatic reset.
Installed on one environment instance; shared Isaac Lab is unchanged.
"""
from types import MethodType
import torch


def install_timeout_bootstrap(env, agent, gamma):
    if getattr(env, "_vla_timeout_bootstrap_installed", False):
        raise RuntimeError("Timeout bootstrap already installed")

    original_reset = env._reset_idx
    original_step = env.step
    pending = {}

    def reset_with_capture(self, env_ids):
        if pending.get("stepping", False):
            # These flags belong to the transition that triggered reset.
            timeout_only = (
                self.reset_time_outs[env_ids]
                & ~self.reset_terminated[env_ids]
            )
            selected = env_ids[timeout_only]

            if selected.numel():
                # Current policy observations have no history or corruption.
                # Compute while the final physical state still exists.
                with torch.no_grad():
                    observations = self.observation_manager.compute()
                    final_obs = observations["policy"][selected].clone()
                    values = agent.get_value(final_obs).reshape(-1)
                    if not torch.isfinite(values).all():
                        raise RuntimeError("Non-finite timeout bootstrap value")
                    pending["bonus"][selected] = gamma * values
                    pending["mask"][selected] = True

        return original_reset(env_ids)

    def step_with_bootstrap(self, action):
        pending["bonus"] = torch.zeros(
            self.num_envs, device=self.device
        )
        pending["mask"] = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        pending["stepping"] = True
        try:
            obs, reward, terminated, truncated, info = original_step(action)
        finally:
            pending["stepping"] = False

        expected = truncated.bool() & ~terminated.bool()
        if not torch.equal(pending["mask"], expected):
            raise RuntimeError("Timeout capture mask does not match returned flags")

        # Preserve raw reward for diagnostics; do not alter reward_buf in place.
        info = dict(info)
        info["raw_reward"] = reward.detach().clone()
        info["timeout_bootstrap_bonus"] = pending["bonus"].clone()
        info["timeout_bootstrap_mask"] = pending["mask"].clone()
        corrected_reward = reward + pending["bonus"]

        if expected.any() and not pending.get("reported", False):
            print(
                "[timeout-bootstrap] Captured pre-reset values for "
                f"{expected.sum().item()} timeout(s).",
                flush=True,
            )
            pending["reported"] = True

        return obs, corrected_reward, terminated, truncated, info

    env._reset_idx = MethodType(reset_with_capture, env)
    env.step = MethodType(step_with_bootstrap, env)
    env._vla_timeout_bootstrap_installed = True
