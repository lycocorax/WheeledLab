"""Curriculum for obstacle task (terrain difficulty / env origins)."""

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from wheeledlab.envs.mdp.curriculums import decrease_reward_weight_over_time

from .obstacle_rewards import is_stuck


def obstacle_terrain_curriculum(env, env_ids: torch.Tensor):
    """Promote or demote envs across terrain difficulty tiers based on episode outcome."""
    move_up = env.termination_manager.get_term("at_goal")[env_ids]

    # Stuck is shaped by reward ``is_stuck``; there is no ``stuck`` termination term.
    # got_stuck = is_stuck(env, min_vel=0.02, wheel_spin_thr=5.0)[env_ids]
    rolled_over = env.termination_manager.get_term("rollover")[env_ids]
    timed_out = env.termination_manager.get_term("time_out")[env_ids]
    move_down = rolled_over | timed_out

    if hasattr(env.scene, "terrain") and env.scene.terrain.terrain_origins is not None:
        env.scene.terrain.update_env_origins(env_ids, move_up, move_down)


@configclass
class ObstacleCurriculumCfg:
    """Curriculum terms for obstacle avoidance training."""
    #more_stuck_penalty = CurrTerm(
    #    func=decrease_reward_weight_over_time,
    #    params={
    #        "reward_term_name": "stuck_penalty",
    #        "decrease": 0.2,
    #        "episodes_per_decrease": 50,
    #        "max_decreases": 15,
    #        "lower_limit": -2.0
    #    },
    #)

    #curriculum_difficulty_update = CurrTerm(
    #    func=obstacle_terrain_curriculum,
    #)
