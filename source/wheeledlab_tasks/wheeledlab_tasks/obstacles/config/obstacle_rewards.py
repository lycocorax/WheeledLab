"""Reward functions for obstacle avoidance."""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .obstacle_observations import goal_dist
from .obstacle_pillar_registry import get_pillar_centers_local_m

def is_stuck(env, min_vel, wheel_spin_thr):
    not_moving = forward_vel(env) < min_vel
    throttle_joints_asset = SceneEntityCfg("robot", joint_names=".*throttle")
    joint_vels = mdp.joint_vel(env, asset_cfg=throttle_joints_asset)
    spinning_wheels = torch.sum(joint_vels, dim=-1) > wheel_spin_thr
    return torch.logical_and(not_moving, spinning_wheels)


def forward_vel(env):
    lin_vel = mdp.base_lin_vel(env)
    return torch.clamp(lin_vel[..., 0], max=1.2)


def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    return torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.0)


def goal_progress_rate(env):
    pos = mdp.root_pos_w(env)
    vel = mdp.root_lin_vel_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]

    vel_vector = vel[:, :2]
    goal_vector = goal_pos - pos[:, :2]
    proj_scal = torch.sum(vel_vector * goal_vector, dim=-1) / torch.norm(goal_vector, dim=-1)

    return proj_scal


def goal_dist_lin_with_threshold(env, threshold: float = 0.0):
    dist = goal_dist(env)[..., 0]
    return -torch.clamp(dist, max=10.0) + threshold

def projected_progress_potential_shaping(
    env,
    dt=0.1,
    gamma=0.99,
):
    pos = mdp.root_pos_w(env)[:, :2]
    vel = mdp.root_lin_vel_w(env)[:, :2]
    goal = mdp.generated_commands(env, "goal_pose")[:, :2]

    dist_now = torch.norm(goal - pos, dim=-1)

    projected_pos = pos + vel * dt
    dist_next = torch.norm(goal - projected_pos, dim=-1)

    # gamma * Potential(next) - Potential(now)
    # where Potential = -dist
    return (-gamma * dist_next) - (-dist_now)


def pillar_proximity_rbf(env, sigma: float = 0.45):
    """Sum of Gaussian kernels at pillar centers (world XY). Larger when close to any pillar."""
    device = env.device
    n = env.num_envs
    c_np = get_pillar_centers_local_m()
    if c_np is None or c_np.size == 0:
        return torch.zeros(n, device=device, dtype=torch.float32)

    centers = torch.as_tensor(c_np, dtype=torch.float32, device=device)
    terrain = env.scene.terrain
    origins = getattr(terrain, "env_origins", None)
    if origins is None:
        origins = getattr(terrain, "terrain_origins", None)
    if origins is None:
        return torch.zeros(n, device=device, dtype=torch.float32)

    origins_xy = origins[:, :2].to(device=device, dtype=torch.float32)
    pillar_w = origins_xy.unsqueeze(1) + centers.unsqueeze(0)
    pos = mdp.root_pos_w(env)[:, :2]
    diff = pos[:, None, :] - pillar_w
    d2 = (diff * diff).sum(dim=-1).clamp(min=0.0)
    sigma_t = torch.tensor(float(sigma), device=device, dtype=torch.float32)
    return torch.exp(-d2 / (2.0 * sigma_t * sigma_t)).sum(dim=-1)


@configclass
class ObstacleRewardsCfg:
    """Shaping for goal progress and explicit penalty near procedural pillars (RBF)."""

    goal_reached = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["at_goal"]},
        weight=100.0,
    )

    #vel_towards_goal = RewTerm(
    #    func=goal_progress_rate,
    #    weight=0.2,
    #)

    dist_to_goal = RewTerm(
        func=goal_dist_lin_with_threshold,
        weight=0.1,
    )

    #goal_potential = RewTerm(
    #    func=projected_progress_potential_shaping,
    #    weight=10.,
    #)

    #stuck_penalty = RewTerm(
    #    func=is_stuck,
    #    params={
    #        "min_vel": 0.02,
    #        "wheel_spin_thr": 5.0,
    #    },
    #    weight=-2.0,
    #)

    #version with stuck as a termination
    stuck_penalty = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["stuck"]},
        weight=-100,
    )

    #pillar_rbf_penalty = RewTerm(
    #    func=pillar_proximity_rbf,
    #    params={"sigma": 0.45},
    #    weight=-0.5,
    #)

    #time_penalty = RewTerm(
    #    func=mdp.is_alive,
    #    weight=-.5
    #)

    #version with time out as a termination penalty
    time_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["time_out"]},
        weight=-100.
    )
