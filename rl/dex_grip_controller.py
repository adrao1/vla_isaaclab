"""6-D end-effector deltas + 1 absolute grip action for the left Dex3 hand.

Same interface shape as X-Sim's Panda (6 EE + 1 gripper), a G1 adaptation:
grip action a in [-1, 1] -> fraction f = (a + 1) / 2 of the straight path from
the open pose (just inside the soft limits) to the closing-side soft limits.
Closing directions (thumb_1/2 +, index/middle -) were verified by
rl/legacy/probe_dex_closure.py. thumb_0 stays at its open value.
"""
import torch

from ee_delta_controller import EEDeltaController

GRIP_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
CLOSE_SIGN = (0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0)
OPEN_POS = (0.0, 0.0, 0.09, -0.08, -0.09, -0.08, -0.09)


class DexGripEEDeltaController(EEDeltaController):
    ACTION_DIM = 7

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        if list(self.hand_joint_names) != list(GRIP_JOINT_NAMES):
            raise RuntimeError(f"Unexpected hand joint order: {self.hand_joint_names}")
        soft = self.robot.data.soft_joint_pos_limits[0, self.hand_joint_ids]
        sign = torch.tensor(CLOSE_SIGN, device=soft.device)
        self.grip_open = torch.tensor(OPEN_POS, device=soft.device)
        self.grip_closed = torch.where(
            sign > 0, soft[:, 1], torch.where(sign < 0, soft[:, 0], self.grip_open)
        )

    def _hand_targets(self, action):
        fraction = (0.5 * (action[:, 6:7] + 1.0)).clamp(0.0, 1.0)
        return self.grip_open + fraction * (self.grip_closed - self.grip_open)

    def diagnostics(self):
        result = super().diagnostics()
        result["hand_action_mode"] = "absolute_grip_synergy"
        result["grip_open"] = self.grip_open.tolist()
        result["grip_closed"] = self.grip_closed.tolist()
        return result
