"""Obstacle task: pillar height-field, terrain generator, and scene (camera, terrain, robot)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from wheeledlab_assets.mushr import MUSHR_SUS_CFG

from .obstacle_pillar_registry import set_pillar_centers_local_m

import numpy as np

@configclass
class ObstacleRectHfCfg(HfTerrainBaseCfg):
    """Random diverse rectangular obstacle placements on a flat patch.
    
    Counts and dimensions scale with curriculum difficulty and are randomized.
    """

    obstacle_height_range_m: tuple[float, float] = (0.1, 0.9)
    """Min and max height of the obstacles in meters."""
    
    obstacle_width_range_m: tuple[float, float] = (0.025, 0.25)
    """Min and max width (x-axis footprint) of the obstacles in meters."""
    
    obstacle_length_range_m: tuple[float, float] = (0.025, .25)
    """Min and max length (y-axis footprint) of the obstacles in meters."""

    num_obstacles_difficulty_range: tuple[int, int] = (5, 90)
    """At difficulty 0 -> limits maximum spawn count to the first value; at difficulty 1 -> limits to the second value."""

    apply_spacing: bool = True

    min_center_separation_m: float = 0.3
    """Minimum distance between pillar centers (keeps random layouts navigable)."""

    max_place_attempts_per_object: int = 500
    """Rejection-sampling attempts per pillar before giving up on that slot."""

    flat_patch_sampling = {
        "init_pos": FlatPatchSamplingCfg(
            num_patches=512,
            patch_radius=0.16,
            max_height_diff=0.01,
            x_range=(-4.0, 4.0),
            y_range=(-4.0, 4.0),
        ),
        "target": FlatPatchSamplingCfg(
            num_patches=512,
            patch_radius=0.16,
            max_height_diff=0.1,
            x_range=(-4.5, 4.5),
            y_range=(-4.5, 4.5),
        ),
    }

@configclass
class ObstaclesPlaygroundCfg(HfTerrainBaseCfg):  
    """Config for basic object-based terrain generation."""

    # == Wall generation only ==
    generate_walls: bool = True
    num_walls_range: tuple = (10, 150)  # Higher difficulty = more walls
    length_range: tuple = (0.1, 0.1)  # Range for random wall length in m. Not affected by difficulty
    thickness_range: tuple = (0.1, 0.1)  # Range for random wall thickness in m. Not affected by difficulty
    height_range: tuple = (0.2, 0.2)  # Range for random wall heights in m. Not affected by difficulty


    flat_patch_sampling = {
        "init_pos":  # reset
        FlatPatchSamplingCfg(
            num_patches=512,
            patch_radius=0.16,
            max_height_diff=0.01,  # 1cm height variation
            x_range=(-4.0, 4.0),
            y_range=(-4.0, 4.0),
        ),
        "target":  # goal pose sampling
        FlatPatchSamplingCfg(
            num_patches=512,
            patch_radius=0.16,
            max_height_diff=0.1,
            x_range=(-4.5, 4.5),
            y_range=(-4.5, 4.5),
        ),
    }


@height_field_to_mesh
def create_playground_hf(difficulty: float, cfg: ObstaclesPlaygroundCfg):
    """Obstacle avoidance playground."""
    rows = int(cfg.size[0] / cfg.horizontal_scale)
    cols = int(cfg.size[1] / cfg.horizontal_scale)

    hf = np.zeros((rows, cols))

    if cfg.generate_walls:
        hf = add_walls(hf, difficulty, cfg)

    return hf.astype(np.float32)


def add_walls(hf, difficulty, cfg: ObstaclesPlaygroundCfg):
    """Adds random rectangular walls to a height field."""
    rows, cols = hf.shape
    x_grid, y_grid = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    num_walls = int(cfg.num_walls_range[0] + (cfg.num_walls_range[0] + cfg.num_walls_range[1]) * difficulty)
    thickness_adj = (np.random.uniform(cfg.thickness_range[0], cfg.thickness_range[1])) / cfg.horizontal_scale
    length_adj = (np.random.uniform(cfg.length_range[0], cfg.length_range[1])) / cfg.horizontal_scale
    wall_height_adj = (np.random.uniform(cfg.height_range[0], cfg.height_range[1])) / cfg.vertical_scale

    for _ in range(num_walls):
        angle = np.random.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)

        margin = max(cfg.length_range[1], thickness_adj)
        x0 = np.random.uniform(margin, rows - margin)
        y0 = np.random.uniform(margin, cols - margin)

        max_len_x = (rows - x0) / dx if dx > 0 else (-x0 / dx if dx < 0 else np.inf)
        max_len_y = (cols - y0) / dy if dy > 0 else (-y0 / dy if dy < 0 else np.inf)
        max_length = min(abs(max_len_x), abs(max_len_y))

        length_adj = min(length_adj, max_length)

        px = -dy
        py = dx

        rx = x_grid - x0
        ry = y_grid - y0

        proj_length = rx * dx + ry * dy
        proj_width = rx * px + ry * py

        mask = (proj_length >= 0) & (proj_length <= length_adj) & (np.abs(proj_width) <= thickness_adj / 2)

        hf[mask] = hf[int(x0), int(y0)] + wall_height_adj

    return hf

def add_obstacles(hf: np.ndarray, difficulty: float, cfg: ObstacleRectHfCfg) -> np.ndarray:
    """Randomly stamps rectangular obstacles of varying dimensions.
    """
    rows, cols = hf.shape
    hs = float(cfg.horizontal_scale)
    vs = float(cfg.vertical_scale)
    size_x, size_y = float(cfg.size[0]), float(cfg.size[1])

    diff = float(np.clip(difficulty, 0.0, 1.0))

    # Determine randomized object count based on the difficulty ceiling
    n_lo, n_hi = int(cfg.num_obstacles_difficulty_range[0]), int(cfg.num_obstacles_difficulty_range[1])
    num_obstacles = int(np.round(n_lo + (n_hi - n_lo) * diff))
    
    if(cfg.apply_spacing):
        min_sep = float(cfg.min_center_separation_m)
        min_sep_sq = min_sep * min_sep
        max_attempts = int(cfg.max_place_attempts_per_object)
        centers: list[tuple[float, float]] = []

        for _ in range(num_obstacles):
            placed = False
            for _attempt in range(max_attempts):
                # 1. Sample dimensions within the scaled ranges
                h_m = np.random.uniform(cfg.obstacle_height_range_m[0], cfg.obstacle_height_range_m[1])
                w_m = np.random.uniform(cfg.obstacle_width_range_m[0], cfg.obstacle_width_range_m[1])
                l_m = np.random.uniform(cfg.obstacle_length_range_m[0], cfg.obstacle_length_range_m[1])
                
                # 2. Sample center point (kept within patch bounds to prevent indexing errors)
                x_m = float(np.random.uniform(w_m / 2, size_x - w_m / 2))
                y_m = float(np.random.uniform(l_m / 2, size_y - l_m / 2))
                
                # Spacing logic check
                ok = True
                for px, py in centers:
                    dx = x_m - px
                    dy = y_m - py
                    if dx * dx + dy * dy < min_sep_sq:
                        ok = False
                        break
                
                if not ok:
                    continue  # Rejection sample, try again
                
                # Valid placement found
                centers.append((x_m, y_m))
                
                # 3. Convert locations to grid pixels
                start_x_px = int(np.clip(round((x_m - w_m / 2) / hs), 0, rows - 1))
                end_x_px = int(np.clip(round((x_m + w_m / 2) / hs), start_x_px + 1, rows))
                
                start_y_px = int(np.clip(round((y_m - l_m / 2) / hs), 0, cols - 1))
                end_y_px = int(np.clip(round((y_m + l_m / 2) / hs), start_y_px + 1, cols))
                
                # Center indices for reading the base height
                center_x = int(np.clip(round(x_m / hs), 0, rows - 1))
                center_y = int(np.clip(round(y_m / hs), 0, cols - 1))
                
                # 4. Stamp the obstacle onto the height field
                h_adj = h_m / vs
                base_height = float(hf[center_x, center_y])
                
                hf_slice = hf[start_x_px:end_x_px, start_y_px:end_y_px]
                hf[start_x_px:end_x_px, start_y_px:end_y_px] = np.maximum(hf_slice, base_height + h_adj)
                
                placed = True
                break  # Successfully placed, move to next obstacle

            if not placed:
                break  # If max attempts reached without placement, patch is likely too full; stop adding
    else:
        for _ in range(num_obstacles):
            # 1. Sample dimensions within the scaled ranges
            h_m = np.random.uniform(cfg.obstacle_height_range_m[0], cfg.obstacle_height_range_m[1])
            w_m = np.random.uniform(cfg.obstacle_width_range_m[0], cfg.obstacle_width_range_m[1])
            l_m = np.random.uniform(cfg.obstacle_length_range_m[0], cfg.obstacle_length_range_m[1])
            
            # 2. Sample center point (kept within patch bounds to prevent indexing errors)
            x_m = np.random.uniform(w_m / 2, size_x - w_m / 2)
            y_m = np.random.uniform(l_m / 2, size_y - l_m / 2)
            
            # 3. Convert locations to grid pixels
            start_x_px = int(np.clip(round((x_m - w_m / 2) / hs), 0, rows - 1))
            # Ensure the slice is at least 1 pixel wide and capped at array bounds
            end_x_px = int(np.clip(round((x_m + w_m / 2) / hs), start_x_px + 1, rows))
            
            start_y_px = int(np.clip(round((y_m - l_m / 2) / hs), 0, cols - 1))
            end_y_px = int(np.clip(round((y_m + l_m / 2) / hs), start_y_px + 1, cols))
            
            # Center indices for reading the base height
            center_x = int(np.clip(round(x_m / hs), 0, rows - 1))
            center_y = int(np.clip(round(y_m / hs), 0, cols - 1))
            
            # 4. Stamp the obstacle onto the height field
            h_adj = h_m / vs
            base_height = float(hf[center_x, center_y])
            
            hf_slice = hf[start_x_px:end_x_px, start_y_px:end_y_px]
            hf[start_x_px:end_x_px, start_y_px:end_y_px] = np.maximum(hf_slice, base_height + h_adj)
            
    return hf

@configclass
class ObstaclePillarHfCfg(HfTerrainBaseCfg):
    """Random pillar placements on a flat patch; count scales with curriculum difficulty.

    Centers are registered for RBF rewards.
    """

    pillar_radius_m: float = 0.1
    """Footprint radius in meters."""

    pillar_height_m: float = 0.8
    pillar_margin_m: float = 0.2
    """Inset from each patch edge when sampling pillar centers (meters)."""

    num_pillars_difficulty_range: tuple[int, int] = (0, 60)
    """At difficulty 0 → first count (almost none); at difficulty 1 → second count (dense)."""

    min_center_separation_m: float = 0.38
    """Minimum distance between pillar centers (keeps random layouts navigable)."""

    max_place_attempts_per_pillar: int = 500
    """Rejection-sampling attempts per pillar before giving up on that slot."""

    flat_patch_sampling = {
        "init_pos": FlatPatchSamplingCfg(
            num_patches=512,
            patch_radius=0.16,
            max_height_diff=0.01,
            x_range=(-4.0, 4.0),
            y_range=(-4.0, 4.0),
        ),
        "target": FlatPatchSamplingCfg(
            num_patches=512,
            patch_radius=0.16,
            max_height_diff=0.1,
            x_range=(-4.5, 4.5),
            y_range=(-4.5, 4.5),
        ),
    }


def add_random_pillars(hf: np.ndarray, difficulty: float, cfg: ObstaclePillarHfCfg) -> tuple[np.ndarray, np.ndarray]:
    """Random non-overlapping pillar footprints; density scales with ``difficulty``."""
    rows, cols = hf.shape
    hs = float(cfg.horizontal_scale)
    vs = float(cfg.vertical_scale)
    size_x, size_y = float(cfg.size[0]), float(cfg.size[1])

    height_m = float(cfg.pillar_height_m) * (1.0 + 0.0 * float(difficulty))
    h_adj = height_m / vs
    rad_m = float(cfg.pillar_radius_m)
    rad_px = max(1.0, rad_m / hs)

    margin = float(cfg.pillar_margin_m)
    min_sep = float(cfg.min_center_separation_m)
    min_sep_sq = min_sep * min_sep
    n_lo, n_hi = int(cfg.num_pillars_difficulty_range[0]), int(cfg.num_pillars_difficulty_range[1])
    num_target = int(np.round(n_lo + (n_hi - n_lo) * float(np.clip(difficulty, 0.0, 1.0))))

    centers: list[list[float]] = []
    ii, jj = np.ogrid[0:rows, 0:cols]

    inner_lo_x = margin + rad_m
    inner_hi_x = size_x - margin - rad_m
    inner_lo_y = margin + rad_m
    inner_hi_y = size_y - margin - rad_m
    if inner_lo_x >= inner_hi_x or inner_lo_y >= inner_hi_y:
        centers_m = np.zeros((0, 2), dtype=np.float64)
        return hf, centers_m

    max_attempts = int(cfg.max_place_attempts_per_pillar)

    for _ in range(num_target):
        placed = False
        for _attempt in range(max_attempts):
            x_m = float(np.random.uniform(inner_lo_x, inner_hi_x))
            y_m = float(np.random.uniform(inner_lo_y, inner_hi_y))
            ok = True
            for px, py in centers:
                dx = x_m - px
                dy = y_m - py
                if dx * dx + dy * dy < min_sep_sq:
                    ok = False
                    break
            if not ok:
                continue
            centers.append([x_m, y_m])
            ci = int(np.clip(round(x_m / hs), 0, rows - 1))
            cj = int(np.clip(round(y_m / hs), 0, cols - 1))
            dist2 = (ii.astype(np.float64) - ci) ** 2 + (jj.astype(np.float64) - cj) ** 2
            mask = dist2 <= rad_px**2
            base = float(hf[ci, cj])
            hf[mask] = np.maximum(hf[mask], base + h_adj)
            placed = True
            break
        if not placed:
            break

    if not centers:
        centers_m = np.zeros((0, 2), dtype=np.float64)
    else:
        centers_m = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    return hf, centers_m


@height_field_to_mesh
def create_obstacle_pillar_hf(difficulty: float, cfg: ObstaclePillarHfCfg):
    """Flat patch with randomly placed pillars; count scales with ``difficulty``; registers centers for RBF."""
    rows = int(cfg.size[0] / cfg.horizontal_scale)
    cols = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((rows, cols), dtype=np.float64)
    hf, centers_m = add_random_pillars(hf, difficulty, cfg)
    set_pillar_centers_local_m(centers_m)
    return hf.astype(np.float32)

@height_field_to_mesh
def create_rect_obstacles_hf(difficulty: float, cfg: ObstacleRectHfCfg):
    """Flat patch with randomly placed pillars; count scales with ``difficulty``; registers centers for RBF."""
    rows = int(cfg.size[0] / cfg.horizontal_scale)
    cols = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((rows, cols), dtype=np.float64)
    hf = add_obstacles(hf, difficulty, cfg)
    return hf.astype(np.float32)


@configclass
class ObstacleTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Curriculum grid over pillar-only sub-terrain patches."""

    vertical_scale = 0.1
    horizontal_scale = 0.05
    border_width = 1.0
    border_height = -20
    slope_threshold = 0.5

    size = (10.0, 10.0)
    num_rows = 4
    num_cols = 4

    curriculum = True

    sub_terrains = {
        #"obstacle_pillars": ObstaclePillarHfCfg(
        #    proportion=0.0,
        #    function=create_obstacle_pillar_hf,
        #    vertical_scale=vertical_scale,
        #    horizontal_scale=horizontal_scale,
        #),
        "obstacle_random": ObstacleRectHfCfg(
            proportion=1.0,
            function=create_rect_obstacles_hf,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale
        ),
        #"obstacle_walls": ObstaclesPlaygroundCfg(
        #    proportion=1.0,
        #    function=create_playground_hf,
        #    vertical_scale=vertical_scale,
        #    horizontal_scale=horizontal_scale,
        #)
    }


@configclass
class ObstacleTerrainImporterCfg(TerrainImporterCfg):
    prim_path = "/World/elevation"
    terrain_type = "generator"
    terrain_generator = ObstacleTerrainGeneratorCfg()
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis = False


@configclass
class ObstacleDepthSceneCfg(InteractiveSceneCfg):
    """Raycaster camera, pillar terrain, lighting, and Mushr robot."""

    camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/camera_link",
        update_period=0.1,
        data_types=["distance_to_image_plane"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.9299999475479126,
            horizontal_aperture=3.8959999084472656,
            vertical_aperture=2.453000068664551,
            height=60, # real camera default: 480 by 848
            width=106,
        ),
        max_distance=10.0,
        depth_clipping_behavior="max",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.0),
            rot=tuple(R.from_euler("xyz", [-90.0, 0.0, -90.0], degrees=True).as_quat().tolist()),
            convention="ros",
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/elevation/terrain"],
    )

    terrain = ObstacleTerrainImporterCfg()

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ObstacleHeightmapSceneCfg(InteractiveSceneCfg):
    """Top-down raycaster, pillar terrain, lighting, and Mushr robot."""

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 20.0),
            # rot=(0.0, 1.0, 0.0, 0.0),
        ),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(size=[2.5, 2.5], resolution=0.1),
        debug_vis=False,
        mesh_prim_paths=["/World/elevation/terrain"],
    )

    terrain = ObstacleTerrainImporterCfg()

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
