"""Runtime storage for pillar centers (local XY meters) used by RBF obstacle shaping.

Populated when the pillar height-field is generated; read during reward computation.

CURRENTLY UNUSED! Helps with reward shaping though.
"""

from __future__ import annotations

import numpy as np

_pillar_centers_local_m: np.ndarray | None = None


def set_pillar_centers_local_m(centers_xy_m: np.ndarray) -> None:
    """Replace stored pillar centers (Nx2) in patch-local meters."""
    global _pillar_centers_local_m
    if centers_xy_m is None or centers_xy_m.size == 0:
        _pillar_centers_local_m = None
    else:
        _pillar_centers_local_m = np.asarray(centers_xy_m, dtype=np.float64).reshape(-1, 2)


def get_pillar_centers_local_m() -> np.ndarray | None:
    """Return a copy of pillar centers, or None if not generated."""
    if _pillar_centers_local_m is None:
        return None
    return _pillar_centers_local_m.copy()
