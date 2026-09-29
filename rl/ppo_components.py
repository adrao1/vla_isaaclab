from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

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

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ):
        """Return either the actor mean or a sampled Gaussian action."""

        action_mean = self.actor_mean(
            x
        )

        if deterministic:
            return action_mean

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

        return probs.sample()

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


class RolloutVideoRecorder:
    """Encode RGB frames to H.264 using the repo's existing PyAV pattern."""

    def __init__(
        self,
        path: Path,
        fps: int,
    ):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "Evaluation video recording requires PyAV. "
                "The repo's preview-video path also uses the 'av' package."
            ) from exc

        self.av = av
        self.path = path

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.container = av.open(
            str(self.path),
            mode="w",
        )

        self.stream = None
        self.fps = fps

    def add_rgb(
        self,
        rgb: np.ndarray,
    ):
        """Append one uint8 RGB frame."""

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(
                np.uint8
            )

        if self.stream is None:
            height, width = rgb.shape[:2]

            self.stream = self.container.add_stream(
                "libx264",
                rate=self.fps,
            )

            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"

        frame = self.av.VideoFrame.from_ndarray(
            rgb,
            format="rgb24",
        )

        for packet in self.stream.encode(
            frame
        ):
            self.container.mux(
                packet
            )

    def close(self):
        if self.container is None:
            return

        if self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(
                    packet
                )

        self.container.close()
        self.container = None
        self.stream = None


def load_agent_checkpoint(agent, path, device):
    """Load model weights only. Optimizer and counters are intentionally fresh."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("agent", checkpoint)
    agent.load_state_dict(state, strict=True)
    return {
        "iteration": checkpoint.get("iteration"),
        "global_step": checkpoint.get("global_step"),
        "source_task": checkpoint.get("args", {}).get("task"),
    }
