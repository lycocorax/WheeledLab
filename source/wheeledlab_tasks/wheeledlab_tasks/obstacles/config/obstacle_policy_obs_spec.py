"""Policy-facing depth + optional obstacle mask and CNN widths for the obstacle task."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ObstaclePolicyObsSpec:
    """Depth + binary mask in front of the CNN; proprio width inferred from total obs dim."""

    depth_downsample_hw: tuple[int, int] = (40, 40)
    depth_max_range_m: float = 4.0
    cnn_base_channels: int = 16
    state_mlp_hidden_dim: int = 64
    cnn_mlp_hidden_dim: int = 64
    use_obstacle_mask: bool = False
    """If True, concatenate ``camera_data_binary_mask`` after depth (same ``downsample_hw``)."""

    obstacle_mask_danger_threshold_m: float = 0.45

    def cnn_in_channels(self) -> int:
        return 2 if self.use_obstacle_mask else 1

    def depth_flat_dim(self) -> int:
        h, w = self.depth_downsample_hw
        return int(h * w * self.cnn_in_channels())

    def proprio_dim_from_total_obs(self, num_actor_obs: int) -> int:
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
            "cnn_mlp_hidden_dim": int(self.cnn_mlp_hidden_dim),
        }
