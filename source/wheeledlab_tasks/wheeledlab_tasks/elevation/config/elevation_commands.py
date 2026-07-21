"""Command generators for elevation. Mainly used for goal generation."""

from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommandCfg
from isaaclab.utils import configclass


@configclass
class ElevationCommandCfg:
    """Configuration for the elevation commands."""

    goal_pose = TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        ranges=TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(10.0, 10.0),
        simple_heading=True,
        debug_vis=True,
    )
