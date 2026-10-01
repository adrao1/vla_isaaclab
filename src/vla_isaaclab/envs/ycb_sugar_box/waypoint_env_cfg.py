"""Physical lift/transfer/place task, initialized from the guided grasp policy."""
import math
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg, TerminationTermCfg, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude
from .guided_grasp_env_cfg import GuidedGraspEnvCfg, GuidedRewardsCfg, contact_reward, completion_reward
from .grasp_env_cfg import GraspTerminationsCfg, INITIAL_BOX_HEIGHT
from .env_cfg import INITIAL_XY, SUGAR_BOX_ORIENTATION_WXYZ, TARGET_DISPLACEMENT_M
from .mdp.terminations import dex3_grasp_contacts
from ..common import LEFT_HAND_JOINT_NAMES, LEFT_HAND_OPEN_JOINT_POSITIONS

STAGES = ('lift', 'transfer_mid', 'transfer', 'lower', 'release', 'done')
LIFT_HEIGHT_M = 0.05
XY_TOL_M = 0.04
TRANSFER_XY_TOL_M = 0.04
# Half the transfer (lift→mid or mid→hover). tanh(error / this) stays nonzero across a hop.
TRANSPORT_LENGTH_M = 0.5 * abs(TARGET_DISPLACEMENT_M)

class WaypointCommand(CommandTerm):
    """Per-environment stages. Only the success term advances state, once per step.

    Commands are local object poses. Release uses the opposite quaternion sign
    (same physical orientation) to distinguish it from lower without adding dims.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.held = torch.zeros_like(self.stage)
        self.last_step = torch.full_like(self.stage, -1)
        self.bonus = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros((self.num_envs, 7), device=self.device)
        self.start = torch.tensor((*INITIAL_XY, INITIAL_BOX_HEIGHT), device=self.device)
        self.destination = self.start + self.start.new_tensor((TARGET_DISPLACEMENT_M, 0., 0.))
        self.mid = self.start + self.start.new_tensor((0.5 * TARGET_DISPLACEMENT_M, 0., LIFT_HEIGHT_M))
        self.lift = self.start + self.start.new_tensor((0., 0., LIFT_HEIGHT_M))
        self.hover = self.destination + self.start.new_tensor((0., 0., LIFT_HEIGHT_M))
        self.orientation = self.start.new_tensor(SUGAR_BOX_ORIENTATION_WXYZ)
        self.hand_ids, _ = env.scene['robot'].find_joints(list(LEFT_HAND_JOINT_NAMES), preserve_order=True)
        self.open_hand = self.start.new_tensor(LEFT_HAND_OPEN_JOINT_POSITIONS)
        self._update_command()

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        self.stage[env_ids] = 0
        self.held[env_ids] = 0
        self.last_step[env_ids] = -1
        self.bonus[env_ids] = 0
        self._update_command()

    def _update_metrics(self):
        pass

    def _update_command(self):
        self._command[:, :3] = self.destination
        self._command[self.stage == 0, :3] = self.lift
        self._command[self.stage == 1, :3] = self.mid
        self._command[self.stage == 2, :3] = self.hover
        self._command[:, 3:] = self.orientation
        self._command[self.stage >= 4, 3:] *= -1

    def measure(self):
        obj = self._env.scene['object'].data
        pos = obj.root_pos_w - self._env.scene.env_origins
        contacts = dex3_grasp_contacts(self._env, min_force=0.5)
        angle = quat_error_magnitude(obj.root_quat_w, self.orientation.expand(self.num_envs, -1))
        return pos, contacts, angle, obj

    def advance(self):
        tick = self._env.common_step_counter
        fresh = self.last_step != tick
        pos, contacts, angle, obj = self.measure()
        grasp = contacts['is_grasping']
        rise = pos[:, 2] - self.start[2]
        xy_dest = torch.linalg.vector_norm(pos[:, :2] - self.destination[:2], dim=-1)
        xy_mid = torch.linalg.vector_norm(pos[:, :2] - self.mid[:2], dim=-1)
        placed = (xy_dest < XY_TOL_M) & ((pos[:, 2] - self.destination[2]).abs() < XY_TOL_M) & (angle < math.radians(10))
        slow = (torch.linalg.vector_norm(obj.root_lin_vel_w, dim=-1) < 0.03) & (torch.linalg.vector_norm(obj.root_ang_vel_w, dim=-1) < 0.2)
        opened = (self._env.scene['robot'].data.joint_pos[:, self.hand_ids] - self.open_hand).abs().amax(dim=-1) < 0.15
        free = opened & ~(contacts['thumb_contact'] | contacts['index_contact'] | contacts['middle_contact'])
        up = grasp & (rise >= LIFT_HEIGHT_M)
        good = (((self.stage == 0) & up)
                | ((self.stage == 1) & up & (xy_mid < TRANSFER_XY_TOL_M))
                | ((self.stage == 2) & up & (xy_dest < TRANSFER_XY_TOL_M))
                | ((self.stage == 3) & grasp & placed & slow)
                | ((self.stage == 4) & free & placed & slow))
        self.held[fresh] = torch.where(good, self.held + 1, 0)[fresh]
        transfer_or_place = (self.stage == 1) | (self.stage == 2) | (self.stage == 3)
        needed = torch.where(transfer_or_place, 5, math.ceil(0.5 / self._env.step_dt))
        passed = fresh & (self.stage < 5) & (self.held >= needed)
        self.bonus[fresh] = 0
        self.bonus[passed & (self.stage < 4)] = 1
        self.stage[passed] += 1
        self.held[passed] = 0
        self.last_step[fresh] = tick
        self._update_command()
        return self.stage == 5

@configclass
class WaypointCommandCfg(CommandTermCfg):
    class_type: type = WaypointCommand
    resampling_time_range: tuple = (1.e9, 1.e9)
    debug_vis: bool = False

@configclass
class WaypointCommandsCfg:
    target_pose = WaypointCommandCfg()

def command(env):
    return env.command_manager.get_term('target_pose')

def success(env):
    return command(env).advance()

def transport_reward(env):
    cmd = command(env)
    pos, contacts, angle, _ = cmd.measure()
    error = torch.linalg.vector_norm(pos - cmd.command[:, :3], dim=-1)
    # Only physically held objects earn transport shaping.
    return (cmd.stage < 4).float() * contacts['is_grasping'].float() * (
        1 - torch.tanh(error / TRANSPORT_LENGTH_M)
    ) * (1 - torch.tanh(angle))

def staged_contact_reward(env):
    return contact_reward(env) * (command(env).stage < 4).float()

def lower_reward(env):
    cmd = command(env)
    pos, contacts, angle, _ = cmd.measure()
    z_err = (pos[:, 2] - cmd.destination[2]).abs()
    xy_err = torch.linalg.vector_norm(pos[:, :2] - cmd.destination[:2], dim=-1)
    return (cmd.stage == 3).float() * contacts['is_grasping'].float() * (
        1 - torch.tanh(z_err / 0.20)
    ) * (1 - torch.tanh(xy_err / 0.04)) * (1 - torch.tanh(angle))

def release_reward(env):
    cmd = command(env)
    pos, contacts, angle, _ = cmd.measure()
    error = torch.linalg.vector_norm(pos - cmd.destination, dim=-1)
    free_fraction = 1 - (contacts['thumb_contact'].float() + contacts['index_contact'].float() + contacts['middle_contact'].float()) / 3
    return (cmd.stage == 4).float() * free_fraction * (1 - torch.tanh(30 * error)) * (1 - torch.tanh(angle))

def milestone_reward(env):
    return command(env).bonus / env.step_dt

def staged_reach(env):
    from .guided_grasp_env_cfg import approach_reward
    return approach_reward(env) * (command(env).stage < 4).float()

def staged_orientation(env):
    from .guided_grasp_env_cfg import orientation_reward
    return orientation_reward(env) * (command(env).stage < 4).float()

@configclass
class WaypointRewardsCfg(GuidedRewardsCfg):
    object_motion = None
    reach = RewardTermCfg(func=staged_reach, weight=1.)
    orientation = RewardTermCfg(func=staged_orientation, weight=0.5)
    grasp = RewardTermCfg(func=staged_contact_reward, weight=0.25)
    transport = RewardTermCfg(func=transport_reward, weight=4.)
    lower = RewardTermCfg(func=lower_reward, weight=4.)
    release = RewardTermCfg(func=release_reward, weight=4.)
    milestone = RewardTermCfg(func=milestone_reward, weight=2.)
    completion = RewardTermCfg(func=completion_reward, weight=10.)

@configclass
class WaypointTerminationsCfg(GraspTerminationsCfg):
    success = TerminationTermCfg(func=success)

@configclass
class WaypointEnvCfg(GuidedGraspEnvCfg):
    commands: WaypointCommandsCfg = WaypointCommandsCfg()
    rewards: WaypointRewardsCfg = WaypointRewardsCfg()
    terminations: WaypointTerminationsCfg = WaypointTerminationsCfg()
    episode_length_s: float = 30.
    task_instruction: str = 'Lift the sugar box 5 cm, transfer it 30 cm robot-left, lower it and release it stably.'
