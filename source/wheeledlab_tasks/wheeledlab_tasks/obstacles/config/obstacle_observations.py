"""Observation callables and ``Obstacle[CNN/Heightmap]ObsCfg`` for the obstacle (pillar) task."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

from wheeledlab.envs.mdp.observations import root_euler_xyz

from .obstacle_policy_obs_spec import ObstaclePolicyObsSpec

gaussian_blur = transforms.GaussianBlur(3, sigma=(0.1, 0.5))


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
    downsample_hw: tuple[int, int] = (40, 40),
    max_range_m: float = 4.0,
) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"]
    B = depth.shape[0]

    depth = depth.permute(0, 3, 1, 2).float()

    dh, dw = int(downsample_hw[0]), int(downsample_hw[1])
    depth = F.interpolate(depth, size=(dh, dw), mode="bilinear", align_corners=False)

    base_noise_std = 0.01
    depth_dependent_std = 0.005 * torch.pow(depth, 2)

    total_noise_std = base_noise_std + depth_dependent_std
    noise = torch.randn_like(depth) * total_noise_std
    depth = depth + noise

    hole_prob = 0.05 + (0.09 * (depth == 1.0).float())
    mask = torch.rand_like(depth) > hole_prob
    depth = depth * mask
    depth = torch.clamp(depth / max_range_m, 0.0, 1.0)

    return depth.reshape(B, -1)



def camera_data_binary_mask(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    downsample_hw: tuple[int, int] = (32, 32),
    danger_threshold_m: float = 0.45,
) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"]

    depth = depth.permute(0, 3, 1, 2).float()
    dh, dw = int(downsample_hw[0]), int(downsample_hw[1])
    depth = F.interpolate(depth, size=(dh, dw), mode="bilinear", align_corners=False)

    binary_mask = (depth < danger_threshold_m).float()

    return binary_mask.reshape(depth.shape[0], -1)


def world_height_map(env, sensor_cfg:SceneEntityCfg, offset:float, plane_init_value:float):
    height_scan = - mdp.height_scan(env, sensor_cfg, offset)
    world_pos_z = mdp.root_pos_w(env)[..., 2]
    corr_height_scan = height_scan + world_pos_z.unsqueeze(-1)
    return corr_height_scan


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
    return torch.norm(dx_dy, dim=-1, keepdim=True)



@configclass
class ObstacleDepthObsCfg:
    """Policy observations: depth, optional mask, goal-relative polar, proprio."""

    @configclass
    class ConcatObs(ObsGroup):
        depth_data = ObsTerm(
            func=camera_data_depth_flattened_aug,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )

        obstacle_mask: ObsTerm | None = None

        goal_polar_coords_relative = ObsTerm(
            func=goal_relative_polar,
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-10.0, 10.0),
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-10.0, 10.0),
        )

        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ConcatObs = ConcatObs()

@configclass
class ObstacleHeightmapObsCfg:
    """Policy observations: depth, goal-relative polar, proprio."""

    @configclass
    class ConcatObs(ObsGroup):
        elevation_map = ObsTerm(
            func=world_height_map,
            params={
                    "sensor_cfg":SceneEntityCfg("height_scanner"),
                    "offset": 0.084,
                    "plane_init_value": 0.19
                },
            clip=(-10., 10.),
        )

        #depth_data = ObsTerm(
        #    func=camera_data_depth,
        #    params={"sensor_cfg": SceneEntityCfg("camera")},
        #)

        goal_polar_coords_relative = ObsTerm(
            func=goal_relative_polar,
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-10.0, 10.0),
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-10.0, 10.0),
        )

        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ConcatObs = ConcatObs()


def apply_policy_obs_spec_to_observations(obs_cfg: ObstacleDepthObsCfg, spec: ObstaclePolicyObsSpec) -> None:
    """Wire depth/mask preprocessing from ``ObstaclePolicyObsSpec`` into the policy group."""
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
