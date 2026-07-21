"""Policy-facing depth preprocessing and CNN widths for the elevation task.

Contract with ActorCriticRecurrentCNN:

- The policy observation vector must place the flattened CNN input first: one channel
  per map (depth, and optionally binary obstacle mask), then all other (proprio) terms.
  The network splits as obs[:, :C*H*W] (image stack) and obs[:, C*H*W:] (proprio).
- depth_downsample_hw (img_h, img_w) must match the env's preprocessing
  (see apply_policy_obs_spec_to_observations).
  Proprio width is not configured here: it is num_actor_obs - depth_flat_dim() at policy init.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ElevationPolicyObsSpec:
    """Depth grid + CNN hyperparameters. Proprio length is inferred from num_actor_obs."""

    depth_downsample_hw: tuple[int, int] = (32, 32)
    depth_max_range_m: float = 4.0
    cnn_base_channels: int = 32
    state_mlp_hidden_dim: int = 64
    use_obstacle_mask: bool = True
    """If True, concatenate camera_data_binary_mask after depth (same downsample_hw)."""

    obstacle_mask_danger_threshold_m: float = 0.45
    """Distance below which the binary mask is 1 (meters), when use_obstacle_mask is True."""

    def cnn_in_channels(self) -> int:
        return 2 if self.use_obstacle_mask else 1

    def depth_flat_dim(self) -> int:
        h, w = self.depth_downsample_hw
        return int(h * w * self.cnn_in_channels())

    def proprio_dim_from_total_obs(self, num_actor_obs: int) -> int:
        """Width of the proprio tail given total policy obs dim (CNN prefix = :meth:`depth_flat_dim`)."""
        d = self.depth_flat_dim()
        n = int(num_actor_obs)
        if n < d:
            raise ValueError(
                f"num_actor_obs ({n}) must be >= depth_flat_dim ({d}) from depth_downsample_hw "
                f"and use_obstacle_mask."
            )
        return n - d

    def to_rsl_policy_kwargs(self) -> dict:
        h, w = self.depth_downsample_hw
        return {
            "img_h": int(h),
            "img_w": int(w),
            "cnn_in_channels": int(self.cnn_in_channels()),
            "cnn_base_channels": int(self.cnn_base_channels),
            "state_mlp_hidden_dim": int(self.state_mlp_hidden_dim),
        }
