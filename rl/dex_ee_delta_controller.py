"""13-D G1 adaptation: six palm deltas and seven independent joint deltas.

This is a Dex3 adaptation, not X-Sim's original Panda action space.
No object-specific closed-hand target is used by this controller.
"""
import torch
from ee_delta_controller import EEDeltaController

class DexEEDeltaController(EEDeltaController):
    ACTION_DIM = 13

    def __init__(self, env, hand_delta_scale=0.02, **kwargs):
        super().__init__(env, **kwargs)
        if hand_delta_scale <= 0:
            raise ValueError('hand_delta_scale must be positive')
        self.hand_delta_scale = hand_delta_scale

    def _hand_targets(self, action):
        # Positive action means positive joint angle, independent of curl sign.
        current = self.robot.data.joint_pos[:, self.hand_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self.hand_joint_ids]
        return torch.clamp(current + self.hand_delta_scale * action[:, 6:13],
                           limits[..., 0], limits[..., 1])

    def diagnostics(self):
        result = super().diagnostics()
        result['hand_delta_scale_rad'] = self.hand_delta_scale
        result['hand_action_mode'] = 'measured_joint_position_plus_delta'
        return result
