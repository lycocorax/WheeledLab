"""Mushr obstacle RL env: pillar terrain, depth + mask CNN, and RBF proximity shaping."""

from enum import Enum

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from wheeledlab_tasks.common import Mushr4WDActionCfg

from wheeledlab_tasks.obstacles.config.obstacle_commands import ObstacleCommandCfg
from wheeledlab_tasks.obstacles.config.obstacle_curriculum import ObstacleCurriculumCfg
from wheeledlab_tasks.obstacles.config.obstacle_events import ObstacleSceneEventsCfg
from wheeledlab_tasks.obstacles.config.obstacle_observations import (
    ObstacleDepthObsCfg,
    ObstacleHeightmapObsCfg,
    apply_policy_obs_spec_to_observations,
    camera_data_depth,
    camera_data_binary_mask,
)
from wheeledlab_tasks.obstacles.config.obstacle_policy_obs_spec import ObstaclePolicyObsSpec
from wheeledlab_tasks.obstacles.config.obstacle_rewards import ObstacleRewardsCfg
from wheeledlab_tasks.obstacles.config.obstacle_scene_cfg import ObstacleDepthSceneCfg, ObstacleHeightmapSceneCfg
from wheeledlab_tasks.obstacles.config.obstacle_terminations import ObstacleTerminationsCfg

__all__ = ["MushrObstaclesRLEnvCfg"]

class TestType(Enum):
    DEPTH_CNN = 1
    DEPTH_MLP = 2
    HEIGHTMAP_MLP = 3

@configclass
class MushrObstaclesRLEnvCfg(ManagerBasedRLEnvCfg):
    """Simplified navigation on pillar terrain with extra shaping (mask + RBF); elevation stays general."""

    seed: int = 42
    num_envs: int = 512
    env_spacing: float = 0.0

    # TODO: test CNN or noCNN
    type: TestType = TestType.HEIGHTMAP_MLP

    policy_obs_spec: ObstaclePolicyObsSpec | None = ObstaclePolicyObsSpec()

    observations: ObstacleDepthObsCfg | ObstacleHeightmapObsCfg = ObstacleDepthObsCfg()
    actions: Mushr4WDActionCfg = Mushr4WDActionCfg()

    events: ObstacleSceneEventsCfg = ObstacleSceneEventsCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()
    rewards: ObstacleRewardsCfg = ObstacleRewardsCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()

    commands: ObstacleCommandCfg = ObstacleCommandCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = [20.0, -20.0, 20.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        self.sim.dt = 0.01
        self.decimation = 10
        self.actions.throttle_steer.scale = (3.0, 0.488)
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.24

        match self.type:
            case TestType.DEPTH_CNN:
                self.observations = ObstacleDepthObsCfg()
                self.scene = ObstacleDepthSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing) 
                apply_policy_obs_spec_to_observations(self.observations, self.policy_obs_spec)
            case TestType.DEPTH_MLP:
                self.observations = ObstacleDepthObsCfg()
                self.observations.ConcatObs.obstacle_mask = None
                self.policy_obs_spec = None
                self.scene = ObstacleDepthSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing) 
            case TestType.HEIGHTMAP_MLP:
                self.observations = ObstacleHeightmapObsCfg()
                self.policy_obs_spec = None
                self.scene = ObstacleHeightmapSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing) 
