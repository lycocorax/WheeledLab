"""Curriculum terms for elevation."""
import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from wheeledlab.envs.mdp import increase_reward_weight_over_time

def terrain_curriculum(env, env_ids: torch.Tensor):
    """Evaluates robot performance and teleports them to different difficulty tiers."""
    
    # Success
    move_up = env.termination_manager.get_term("at_goal")[env_ids]
    
    # Failure
    got_stuck = env.termination_manager.get_term("stuck")[env_ids]
    rolled_over = env.termination_manager.get_term("rollover")[env_ids]
    timed_out = env.termination_manager.get_term("time_out")[env_ids]
    move_down = got_stuck | rolled_over | timed_out
    
    # This updates self.terrain_levels and self.env_origins for the specific env_ids
    if hasattr(env.scene, "terrain") and env.scene.terrain.terrain_origins is not None:
        env.scene.terrain.update_env_origins(env_ids, move_up, move_down)

@configclass
class ElevationCurriculumCfg:
    """Configuration for the elevation policy curriculum."""

    #more_vel_reward = CurrTerm(
    #    func=increase_reward_weight_over_time,
    #    params={
    #        "reward_term_name": "vel_towards_goal",
    #        "increase": 2.0,
    #        "episodes_per_increase": 50,
    #        "max_increases": 10,
    #    },
    #)

    curriculum_difficulty_update = CurrTerm(
        func=terrain_curriculum
    )
