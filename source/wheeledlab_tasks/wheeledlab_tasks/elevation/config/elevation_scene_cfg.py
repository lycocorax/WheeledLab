"""Elevation scene: camera, robot, terrain importers, and procedural height-field terrain."""

import numpy as np
import noise
from scipy.interpolate import splprep, splev, griddata
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets.mushr import MUSHR_SUS_CFG


# --- Height-field subterrain configs ---

@configclass
class PlaygroundHfCfg(HfTerrainBaseCfg):  # TODO
    """Config for basic terrain generation."""

    # == Wall generation ==
    generate_walls: bool = True
    num_walls_range: tuple = (0, 35)  # Higher difficulty = more walls
    length_range: tuple = (0.1, 0.2)  # Range for random wall length in m. Not affected by difficulty
    thickness_range: tuple = (0.1, 0.2)  # Range for random wall thickness in m. Not affected by difficulty
    height_range: tuple = (0.5, 0.9)  # Range for random wall heights in m. Not affected by difficulty


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


@configclass
class NoiseHfCfg(HfTerrainBaseCfg):
    """Config for noise-based terrain generation."""

    generate_roads: bool = True
    road_num_nodes: int = 10  # increase for gnarlier roads
    road_width_range: tuple[float, float] = (0.2, 0.5)  # width is narrower at high difficulty
    octaves: int = 3  # layers of noise; increase for more complex terrain
    freq: float = 250.0  # higher = smoother/wider hills

    upper_clip_prop: float = 0.25
    lower_clip_prop: float = 0.25
    vary_ht: bool = False
    ht_range: tuple[float, float] = (0.3, 1.4)

    flat_patch_sampling = {
        "init_pos": FlatPatchSamplingCfg(
            num_patches=256,
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


def add_roads(hf, difficulty, horizontal_scale, num_nodes, road_width_range):
    rows, cols = hf.shape
    road_width_px = (road_width_range[1] - difficulty * (road_width_range[1] - road_width_range[0])) / horizontal_scale
    shoulder_width_px = 10.0

    current_row, current_col = rows / 2, cols / 2
    heading = np.random.uniform(0, 2 * np.pi)
    step_dist = rows * 0.8 / num_nodes
    key_pts = [[current_row, current_col]]

    for _ in range(num_nodes - 1):
        best_heading = heading
        min_effort = float("inf")
        candidates = np.linspace(-np.pi / 4, np.pi / 4, 5)

        for angle_off in candidates:
            test_heading = heading + angle_off
            test_row = np.clip(current_row + step_dist * np.sin(test_heading), 0, rows - 1)
            test_col = np.clip(current_col + step_dist * np.cos(test_heading), 0, cols - 1)

            effort = abs(hf[int(test_row), int(test_col)] - hf[int(current_row), int(current_col)])
            effort += np.sqrt((test_row - rows / 2) ** 2 + (test_col - cols / 2) ** 2) * 0.01

            if effort < min_effort:
                min_effort, best_heading = effort, test_heading

        heading = best_heading

        current_row += step_dist * np.sin(heading)
        current_col += step_dist * np.cos(heading)
        if current_col >= cols or current_row >= rows:
            break
        key_pts.append([current_row, current_col])

    key_pts = np.array(key_pts)

    tck, _ = splprep([key_pts[:, 0], key_pts[:, 1]], s=0)
    u = np.linspace(0, 1, 500)
    path_row, path_col = splev(u, tck)
    path = np.stack([np.clip(path_row, 0, rows - 1), np.clip(path_col, 0, cols - 1)], axis=1)

    road_z = hf[path[:, 0].astype(int), path[:, 1].astype(int)]
    road_z = gaussian_filter1d(road_z, sigma=10)

    road_mask = np.zeros((rows, cols), dtype=bool)
    road_mask[path[:, 0].astype(int), path[:, 1].astype(int)] = True
    distance_field = distance_transform_edt(~road_mask)

    height_map = griddata(path, road_z, (np.indices((rows, cols)).transpose(1, 2, 0)), method="nearest")

    new_hf = hf.copy()
    core = distance_field <= road_width_px
    shoulder = (distance_field > road_width_px) & (distance_field <= road_width_px + shoulder_width_px)

    new_hf[core] = height_map[core]

    t = (distance_field[shoulder] - road_width_px) / shoulder_width_px
    smooth = 3 * t**2 - 2 * t**3
    new_hf[shoulder] = height_map[shoulder] * (1 - smooth) + hf[shoulder] * smooth

    return new_hf



def add_walls(hf, difficulty, cfg: PlaygroundHfCfg):
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


@height_field_to_mesh
def create_noise_hf(difficulty: float, cfg: NoiseHfCfg):
    """Procedurally generates height field using simplex noise."""
    rows = int(cfg.size[0] / cfg.horizontal_scale)
    cols = int(cfg.size[1] / cfg.horizontal_scale)
    curr_ht_mult = cfg.ht_range[0] + (cfg.ht_range[1] - cfg.ht_range[0]) * difficulty
    curr_max_height = curr_ht_mult * (1.0 - cfg.upper_clip_prop)
    curr_floor_clip = curr_ht_mult * cfg.lower_clip_prop
    max_ht_adj = curr_max_height / cfg.vertical_scale
    ht_mult_adj = curr_ht_mult / cfg.vertical_scale
    min_ht_adj = curr_floor_clip / cfg.vertical_scale
    offset = 10000.0 * np.random.rand()

    hf = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            hf[i, j] = noise.snoise2((i + offset) / cfg.freq, (j + offset) / cfg.freq, octaves=cfg.octaves)

    hf = (hf + 1.0) / 2.0

    rx = np.linspace(0.2, 1.0, rows)
    ry = np.linspace(0.2, 1.0, cols)
    rxv, ryv = np.meshgrid(rx, ry, indexing="ij")
    ramp = (rxv + ryv) * 0.5

    if cfg.vary_ht:
        hf = hf * ht_mult_adj * ramp
    else:
        hf = hf * ht_mult_adj

    hf = np.clip(hf, min_ht_adj, max_ht_adj)

    if cfg.generate_roads:
        hf = add_roads(hf, difficulty, cfg.horizontal_scale, cfg.road_num_nodes, cfg.road_width_range)

    hf = add_walls(
        hf,
        difficulty,
        PlaygroundHfCfg(vertical_scale=cfg.vertical_scale, horizontal_scale=cfg.horizontal_scale),
    )
    return (hf - min_ht_adj).astype(np.float32)


@height_field_to_mesh
def create_playground_hf(difficulty: float, cfg: PlaygroundHfCfg):
    """Obstacle avoidance playground."""
    rows = int(cfg.size[0] / cfg.horizontal_scale)
    cols = int(cfg.size[1] / cfg.horizontal_scale)

    hf = np.zeros((rows, cols))

    if cfg.generate_walls:
        hf = add_walls(hf, difficulty, cfg)

    return hf.astype(np.float32)


# --- Terrain generator / importers ---


@configclass
class ElevationTerrainGeneratorCfg(TerrainGeneratorCfg):  # noqa: D101
    vertical_scale = 0.01
    horizontal_scale = 0.02
    border_width = 1.0
    border_height = -7
    slope_threshold = 0.5

    size = (10.0, 10.0)
    num_rows = 4
    num_cols = 4

    curriculum = True

    sub_terrains = {
        "noise_terrain": NoiseHfCfg(
            proportion=0.0,
            function=create_noise_hf,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
        ),
        "playground_terrain": PlaygroundHfCfg(
            proportion=1.0,
            function=create_playground_hf,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
        ),
    }


@configclass
class ElevationTerrainImporterCfg(TerrainImporterCfg):
    prim_path = "/World/elevation"
    terrain_type = "generator"
    terrain_generator = ElevationTerrainGeneratorCfg()
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis = False


@configclass
class ElevationUSDTerrainImporterCfg(TerrainImporterCfg):  # old
    height = 0.25
    prim_path = "/World/elevation"
    terrain_type = "usd"
    usd_path = f"{WHEELEDLAB_ASSETS_DATA_DIR}/Terrains/huge_compact.usd"
    collision_group = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis = False


# --- Scene (camera, terrain, robot) ---


@configclass
class ElevationSceneCfg(InteractiveSceneCfg):
    camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/camera_link",
        update_period=0.1,
        data_types=["distance_to_image_plane"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.9299999475479126,
            horizontal_aperture=3.8959999084472656,
            vertical_aperture=2.453000068664551,
            height=36,
            width=64,
        ),
        max_distance=50.0,
        depth_clipping_behavior="max",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.0),
            rot=tuple(R.from_euler("xyz", [-90.0, 0.0, -90.0], degrees=True).as_quat().tolist()),
            convention="ros",
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/elevation/terrain"],
    )

    terrain = ElevationTerrainImporterCfg()

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ElevationUSDSceneCfg(InteractiveSceneCfg):  # old
    terrain = ElevationUSDTerrainImporterCfg()
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn=sim_utils.GroundPlaneCfg(
            size=(1600.0, 1200.0),
            color=(3, 3, 3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        ),
    )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, self.terrain.height))
