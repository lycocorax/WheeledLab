"""Reset and startup events for the obstacle task."""

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.events import reset_root_state_from_terrain
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class ObstacleSceneEventsCfg:
    """Randomization at startup; terrain-based reset for the robot."""

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 5,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.2, 0.5),
            "operation": "add",
        },
    )

    set_goal = EventTerm(
        func=reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"x": (-9.0, 9.0), "y": (-9.0, 9.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.1, 0.2),
                "y": (0.1, 0.2),
            },
        },
    )
