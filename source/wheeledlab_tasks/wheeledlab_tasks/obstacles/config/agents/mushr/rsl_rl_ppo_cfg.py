"""RSL-RL PPO defaults for the obstacle (pillar) task (depth + mask CNN)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from wheeledlab_tasks.obstacles.config.obstacle_policy_obs_spec import ObstaclePolicyObsSpec

_DEFAULT_OBSTACLE_POLICY_OBS = ObstaclePolicyObsSpec()


@configclass
class MushrObstacleActorCriticCfg(RslRlPpoActorCriticCfg):
    """CNN/RNN kwargs aligned with MushrObstaclesRLEnvCfg.policy_obs_spec (mask on by default)."""

    img_h: int = _DEFAULT_OBSTACLE_POLICY_OBS.depth_downsample_hw[0]
    img_w: int = _DEFAULT_OBSTACLE_POLICY_OBS.depth_downsample_hw[1]
    cnn_in_channels: int = int(_DEFAULT_OBSTACLE_POLICY_OBS.cnn_in_channels())
    cnn_base_channels: int = _DEFAULT_OBSTACLE_POLICY_OBS.cnn_base_channels
    state_mlp_hidden_dim: int = _DEFAULT_OBSTACLE_POLICY_OBS.state_mlp_hidden_dim
    cnn_mlp_hidden_dim: int = _DEFAULT_OBSTACLE_POLICY_OBS.cnn_mlp_hidden_dim
    rnn_hidden_dim = 256
    rnn_num_layers = 1


@configclass
class MushrObstaclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 4000
    save_interval = 50
    experiment_name = "ppo_mushr_obstacle"

    empirical_normalization = False
    policy = MushrObstacleActorCriticCfg(
        #TODO: change this when testing CNN
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="relu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
