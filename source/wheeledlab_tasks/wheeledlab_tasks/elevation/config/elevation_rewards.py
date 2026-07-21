"""Reward functions and ``ElevationRewardsCfg``."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from . import goal_dist


def forward_vel(env):
    lin_vel = mdp.base_lin_vel(env)
    return torch.clamp(lin_vel[..., 0], max=1.2)


def forward_wheel_spin(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    throttle_joints = asset.find_joints(".*_throttle")[0]
    throttle_joint_vel = mdp.joint_vel(env)[..., throttle_joints]
    sum_vels = torch.sum(throttle_joint_vel, dim=-1)
    return torch.clamp(sum_vels, max=200)


def higher_elevation(env):
    pos = mdp.root_pos_w(env)
    z_value = pos[..., 2] - 0.19
    vel = mdp.base_lin_vel(env)[..., 0]
    condition = (z_value > 0.1) & (vel > 0.1)
    rew = torch.where(condition, z_value, torch.zeros_like(z_value))
    return torch.clip(rew, min=0, max=1)


def change_in_elevation(env):
    vel = mdp.root_lin_vel_w(env)
    change_in_z = vel[..., 2]
    return torch.where(change_in_z > 0, change_in_z, torch.zeros_like(change_in_z))


def steep_penalty(env, thresh_pitch):
    orient = mdp.root_quat_w(env)
    euler_xyz = mdp.euler_xyz_from_quat(orient)
    euler_xyz = torch.stack(euler_xyz, dim=-1)
    pitch = euler_xyz[:, 1]
    steep_ramp = torch.clamp(pitch - thresh_pitch, min=0)
    return steep_ramp


def elevation_continuity(env, threshold_elev):
    pos = mdp.root_pos_w(env)
    z_value = pos[..., 2] - 0.19
    if not hasattr(elevation_continuity, "prev_elevation"):
        elevation_continuity.prev_elevation = z_value.clone()
    delta_z = z_value - elevation_continuity.prev_elevation
    on_ramp = z_value > threshold_elev
    ascending = delta_z > 0
    descending = delta_z < 0
    rew_ascend = torch.where(on_ramp & ascending, 50 * delta_z, torch.zeros_like(z_value))
    rew_descend = torch.where(on_ramp & descending, -50 * delta_z, torch.zeros_like(z_value))
    rew = rew_ascend + rew_descend
    elevation_continuity.prev_elevation = z_value.clone()
    return rew


def yaw_change_onElev(env, threshold_yaw, threshold_z):
    pos = mdp.root_pos_w(env)
    z_value = pos[..., 2] - 0.19
    ang_vel_yaw = mdp.base_ang_vel(env)[..., 2]
    condition = (z_value > threshold_z) & (abs(ang_vel_yaw) > threshold_yaw)
    rew = torch.where(condition, 2 * abs(ang_vel_yaw) ** 2, torch.zeros_like(ang_vel_yaw))
    return rew


def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.0)
    return penalty


def roll_on_elev(env, z_start, roll_rate_thresh):
    ang_vel_roll = mdp.base_ang_vel(env)[..., 0]
    pos = mdp.root_pos_w(env)
    z_value = pos[..., 2] - 0.19
    condition = (z_value > z_start) & (abs(ang_vel_roll) > roll_rate_thresh)
    rew = torch.where(condition, abs(ang_vel_roll) * 2.0, torch.zeros_like(ang_vel_roll))
    return rew


def goal_progress_rate(env):
    pos = mdp.root_pos_w(env)
    vel = mdp.root_lin_vel_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]

    vel_vector = vel[:, :2]
    goal_vector = goal_pos - pos[:, :2]
    proj_scal = torch.sum(vel_vector * goal_vector, dim=-1) / torch.norm(goal_vector, dim=-1)

    return proj_scal


def goal_progress(env):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")[:, :2]
    dist = torch.norm(goal_pos - pos[:, :2], dim=-1)
    vel = mdp.root_lin_vel_w(env)[:, :2]
    dt = env.step_dt
    next_pos = pos[:, :2] + vel * dt
    next_dist = torch.norm(goal_pos - next_pos, dim=-1)

    return dist - next_dist


def is_falling_penalty(env, max_body_z_vel: float = 0.10):
    lin_vel = mdp.base_lin_vel(env)
    is_falling = lin_vel[..., 2] > max_body_z_vel
    return is_falling


def ascending(env):
    vel_w = mdp.root_lin_vel_w(env)
    rew = torch.clamp(vel_w[..., 2], min=0.0)
    return rew


def low_vel_penalty(env, min_vel: float = 0.1):
    lin_vel = mdp.base_lin_vel(env)
    vel = lin_vel[..., 0]
    penalty = torch.where(vel < min_vel, 1.0, 0.0)
    return penalty

def goal_dist_lin_with_threshold(env, threshold: float = 3.0):
    '''Linearly scaled distance from goal penalty, becomes positive at threshold distance or closer'''
    dist = goal_dist(env)[..., 0]
    rew = -torch.clamp(dist, max=5.0) + threshold
    return rew

@configclass
class ElevationRewardsCfg:
    goal_reached = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["at_goal"]},
        weight=200.0,
    )

    vel_towards_goal = RewTerm(
        func=goal_progress_rate,
        weight=5.0,
    )

    dist_to_goal = RewTerm(
        func=goal_dist_lin_with_threshold,
        weight=0.5
    )

    stuck_penalty = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["stuck"]},
        weight=-100.0,
    )

#    rollover_penalty = RewTerm(
#        func=mdp.rewards.is_terminated_term,
#        params={"term_keys": ["rollover"]},
#        weight=-1000.0,
#    )

#    time_penalty = RewTerm(
#        func = mdp.is_alive,
#        weight = -.5
#    )
