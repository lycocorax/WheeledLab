"""Obstacle task configuration."""

from .obstacle_commands import ObstacleCommandCfg
from .obstacle_curriculum import ObstacleCurriculumCfg
from .obstacle_events import ObstacleSceneEventsCfg
from .obstacle_observations import ObstacleHeightmapObsCfg, ObstacleDepthObsCfg, apply_policy_obs_spec_to_observations
from .obstacle_policy_obs_spec import ObstaclePolicyObsSpec
from .obstacle_rewards import ObstacleRewardsCfg
from .obstacle_scene_cfg import ObstacleDepthSceneCfg, ObstacleHeightmapSceneCfg
from .obstacle_terminations import ObstacleTerminationsCfg

__all__ = [
    "ObstacleCommandCfg",
    "ObstacleCurriculumCfg",
    "ObstacleDepthObsCfg",
    "ObstacleHeightmapObsCfg",
    "ObstaclePolicyObsSpec",
    "ObstacleRewardsCfg",
    "ObstacleDepthSceneCfg",
    "ObstacleHeightmapSceneCfg"
    "ObstacleSceneEventsCfg",
    "ObstacleTerminationsCfg",
    "apply_policy_obs_spec_to_observations",
]
