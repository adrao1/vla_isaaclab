"""Convert a 13-D dex13 checkpoint into a 7-D grip7 checkpoint (weights only).

Our shortcut, NOT X-Sim. Arm rows, hidden layers and critic are copied. The new
grip output starts with weight 0 and bias -1 (hand open), and its log-std is the
mean of the arm log-stds. The optimizer is not carried over.

Usage: python rl/warmstart_grip.py SRC.pt DST.pt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_components import Agent  # noqa: E402


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    state = dict(ckpt.get("agent", ckpt))

    w_key, b_key = "actor_mean.6.weight", "actor_mean.6.bias"
    if state[w_key].shape[0] != 13:
        raise SystemExit(f"Expected a 13-D source, got {state[w_key].shape[0]}-D")
    obs_dim = state["actor_mean.0.weight"].shape[1]

    new = dict(state)
    new[w_key] = torch.cat([state[w_key][:6], torch.zeros(1, state[w_key].shape[1])], 0)
    new[b_key] = torch.cat([state[b_key][:6], torch.full((1,), -1.0)], 0)
    logstd = state["actor_logstd"]
    new["actor_logstd"] = torch.cat([logstd[:, :6], logstd[:, :6].mean(1, keepdim=True)], 1)

    # Self-check: loads strictly, arm outputs identical, grip output is exactly -1.
    old_agent = Agent(obs_dim, 13)
    old_agent.load_state_dict(state, strict=True)
    agent = Agent(obs_dim, 7)
    agent.load_state_dict(new, strict=True)
    x = torch.randn(64, obs_dim)
    with torch.no_grad():
        a13, a7 = old_agent.actor_mean(x), agent.actor_mean(x)
        v13, v7 = old_agent.get_value(x), agent.get_value(x)
    assert torch.allclose(a13[:, :6], a7[:, :6], atol=1e-6), "arm outputs differ"
    assert torch.allclose(a7[:, 6], torch.full((64,), -1.0)), "grip not open"
    assert torch.allclose(v13, v7), "critic differs"

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"agent": new, "iteration": 0, "global_step": 0,
         "args": {"task": "warmstart", "source": str(src)}},
        dst,
    )
    print(f"OK: wrote {dst} (obs_dim={obs_dim}, action_dim=7, grip mean=-1, "
          f"grip logstd={new['actor_logstd'][0, 6].item():.3f})")


if __name__ == "__main__":
    main()
