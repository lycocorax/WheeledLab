"""Termination functions and configs."""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .elevation_rewards import forward_vel, upright_penalty


def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0


def is_stuck(env, min_vel, wheel_spin_thr):
    not_moving = forward_vel(env) < min_vel
    throttle_joints_asset = SceneEntityCfg("robot", joint_names=".*throttle")
    joint_vels = mdp.joint_vel(env, asset_cfg=throttle_joints_asset)
    spinning_wheels = torch.sum(joint_vels, dim=-1) > wheel_spin_thr
    return torch.logical_and(not_moving, spinning_wheels)


def close_to_goal(env, dist):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]
    curr_dist = torch.norm(goal_pos - pos[:, :2], dim=-1)
    return curr_dist < dist


@configclass
class ElevationTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cart_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -1.0},
    )

    stuck = DoneTerm(
        func=is_stuck,
        params={
            "min_vel": 0.02,
            "wheel_spin_thr": 5.0,
        },
    )

    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 60.0},
    )

    at_goal = DoneTerm(
        func=close_to_goal,
        params={"dist": 0.5},
    )


@configclass
class ElevationPlayTerminationsCfg:
    """Terminations for play cfg."""

    at_goal = DoneTerm(
        func=close_to_goal,
        params={"dist": 0.5},
    )
