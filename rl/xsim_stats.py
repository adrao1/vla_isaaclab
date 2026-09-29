"""Per-rollout X-Sim diagnostics for XSimLift training (printed, not used by PPO)."""


class XSimRolloutStats:
    KEYS = ("tcp_dist", "reach", "is_grasped", "lift", "success", "static_reward",
            "wp_angle", "toppled")

    def __init__(self):
        self.update = 0
        self._clear()

    def _clear(self):
        self.n = 0
        self.sums = {k: 0.0 for k in self.KEYS}
        self.idx_sum = 0.0
        self.idx1_frac_sum = 0.0
        self.lift_max = float("-inf")

    def add(self, env):
        st = getattr(env, "_xsim_state", None)
        if st is None or not st.get("metrics"):
            return
        m = st["metrics"]
        for k in self.KEYS:
            if k in m:
                self.sums[k] += float(m[k].float().mean())
        idx = m["wp_idx_before"].float()
        self.idx_sum += float(idx.mean())
        self.idx1_frac_sum += float((idx >= 1).float().mean())
        self.lift_max = max(self.lift_max, float(m["lift"].max()))
        self.n += 1

    def report_line(self):
        self.update += 1
        if self.n == 0:
            line = f"XSIM update={self.update} (no metrics)"
        else:
            a = {k: v / self.n for k, v in self.sums.items()}
            line = (
                f"XSIM update={self.update} tcp_dist={a['tcp_dist']:.4f} reach={a['reach']:.3f} "
                f"grasp_frac={a['is_grasped']:.3f} lift_mean_cm={100 * a['lift']:.2f} "
                f"lift_max_cm={100 * self.lift_max:.2f} wp_idx_mean={self.idx_sum / self.n:.3f} "
                f"wp_idx1_frac={self.idx1_frac_sum / self.n:.3f} success_frac={a['success']:.3f} "
                f"static_r={a['static_reward']:.3f} tilt_deg={57.2958 * a['wp_angle']:.1f} "
                f"toppled_frac={a['toppled']:.3f}"
            )
        self._clear()
        return line
