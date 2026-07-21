"""Observation callables and ElevationObsCfg for the elevation task."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

from wheeledlab.envs.mdp.observations import root_euler_xyz

from .elevation_policy_obs_spec import ElevationPolicyObsSpec

gaussian_blur = transforms.GaussianBlur(3, sigma=(0.1, 0.5))

# =========================================
# ====== Camera Data Augmentation =========
# =========================================

def camera_data_depth(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    downsample_hw: tuple[int, int] = (32, 32),
    max_range_m: float = 4.0,
) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"]
    B = depth.shape[0]
    depth = depth.permute(0, 3, 1, 2).float()
    dh, dw = int(downsample_hw[0]), int(downsample_hw[1])
    depth = F.interpolate(depth, size=(dh, dw), mode="bilinear", align_corners=False)
    depth = torch.clamp(depth / max_range_m, 0.0, 1.0)
    return depth.reshape(B, -1)


def camera_data_depth_flattened_aug(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    downsample_hw: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]

    depth = sensor.data.output["distance_to_image_plane"]
    B, H, W, C = depth.shape

    depth[torch.isinf(depth)] = 0.0

    depth = depth.permute(0, 3, 1, 2).float()

    dh, dw = int(downsample_hw[0]), int(downsample_hw[1])
    depth = F.interpolate(depth, size=(dh, dw), mode="bilinear", align_corners=False)

    base_noise_std = 0.005
    depth_dependent_std = 0.005 * torch.pow(depth, 2)

    total_noise_std = base_noise_std + depth_dependent_std
    noise = torch.randn_like(depth) * total_noise_std
    depth = depth + noise

    hole_prob = 0.01 + (0.09 * (depth == 1.0).float())
    mask = torch.rand_like(depth) > hole_prob
    depth = depth * mask

    return depth.reshape(B, -1)

def camera_data_binary_mask(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    downsample_hw: tuple[int, int] = (32, 32),
    danger_threshold_m: float = 0.45, # Objects closer than 45 cm are "1"
) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"] # (B, H, W, 1)
    
    depth = depth.permute(0, 3, 1, 2).float()
    dh, dw = int(downsample_hw[0]), int(downsample_hw[1])
    depth = F.interpolate(depth, size=(dh, dw), mode="bilinear", align_corners=False)

    # 1 if object is too close, 0 otherwise
    binary_mask = (depth < danger_threshold_m).float()
    
    return binary_mask.reshape(depth.shape[0], -1)

# =================================
# ========= Proprio obs ===========
# =================================

def world_height_map(env, sensor_cfg: SceneEntityCfg, offset: int, plane_init_value: int):
    height_scan = -mdp.height_scan(env, sensor_cfg, offset)
    world_pos_z = mdp.root_pos_w(env)[..., 2] - plane_init_value
    corr_height_scan = height_scan + world_pos_z.unsqueeze(-1)
    return corr_height_scan


def goal_relative_xyz(env: ManagerBasedEnv):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]
    rel_pos = goal_pos - pos[:, :2]
    return torch.nan_to_num(rel_pos, nan=0)


def goal_relative_root(env: ManagerBasedEnv):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")

    dx_dy = goal_pos[:, :2] - pos[:, :2]

    euler = root_euler_xyz(env)
    yaw = euler[:, 2]

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    dx_body = cos_yaw * dx_dy[:, 0] + sin_yaw * dx_dy[:, 1]
    dy_body = -sin_yaw * dx_dy[:, 0] + cos_yaw * dx_dy[:, 1]

    rel_body = torch.stack([dx_body, dy_body], dim=-1)

    return torch.nan_to_num(rel_body, nan=0.0)


def goal_relative_polar(env: ManagerBasedEnv):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")

    dx_dy = goal_pos[:, :2] - pos[:, :2]
    dist = torch.norm(dx_dy, dim=-1, keepdim=True)

    euler = root_euler_xyz(env)
    yaw = euler[:, 2]

    goal_yaw_w = torch.atan2(dx_dy[:, 1], dx_dy[:, 0])

    rel_yaw = goal_yaw_w - yaw
    rel_yaw = (rel_yaw + torch.pi) % (2 * torch.pi) - torch.pi

    return torch.cat([dist, rel_yaw.unsqueeze(-1)], dim=-1)

def goal_dist(env: ManagerBasedEnv):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")

    dx_dy = goal_pos[:, :2] - pos[:, :2]
    dist = torch.norm(dx_dy, dim=-1)

    return dist


@configclass
class ElevationObsCfg:
    """Observation groups (see MushrElevationRLEnvCfg.policy_obs_spec for depth/CNN contract)."""

    @configclass
    class ConcatObs(ObsGroup):
        # depth_data must stay first; optional obstacle_mask (see policy_obs_spec) must
        # immediately follow depth so ActorCriticRecurrentCNN sees [flat_depth | flat_mask? | proprio].
        depth_data = ObsTerm(
            func=camera_data_depth,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )

        obstacle_mask: ObsTerm | None = None

        goal_polar_coords_relative = ObsTerm(
            func=goal_relative_polar
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-10.0, 10.0),
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-10.0, 10.0),
        )

        #projected_gravity = ObsTerm(
        #    func=mdp.projected_gravity,
        #    scale=1.0,
        #)

        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ConcatObs = ConcatObs()


def apply_policy_obs_spec_to_observations(obs_cfg: ElevationObsCfg, spec: ElevationPolicyObsSpec) -> None:
    """Copy depth (and optional mask) preprocessing from ElevationPolicyObsSpec into the policy group."""
    h, w = spec.depth_downsample_hw
    obs_cfg.policy.depth_data.params = dict(obs_cfg.policy.depth_data.params)
    obs_cfg.policy.depth_data.params["downsample_hw"] = (int(h), int(w))
    obs_cfg.policy.depth_data.params["max_range_m"] = float(spec.depth_max_range_m)

    if spec.use_obstacle_mask:
        obs_cfg.policy.obstacle_mask = ObsTerm(
            func=camera_data_binary_mask,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "downsample_hw": (int(h), int(w)),
                "danger_threshold_m": float(spec.obstacle_mask_danger_threshold_m),
            },
        )
    else:
        obs_cfg.policy.obstacle_mask = None
