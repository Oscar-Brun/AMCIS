"""Shared wall / MC plot styling helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.collections import LineCollection

from adjoint_mc.geometry.wall import WallGeometry

REGION_COLORS = {
    "wall": "#4C72B0",
    "puff": "#DD8452",
    "pump": "#8172B3",
    "main_wall": "#4C72B0",
    "inner_divertor": "#55A868",
    "outer_divertor": "#C44E52",
}


def region_color(name: str) -> str:
    return REGION_COLORS.get(name, "gray")


def log_weights(values: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(values, 1.0))


def wall_outline_segments(wall: WallGeometry) -> LineCollection:
    lines = [[(seg.r0, seg.z0), (seg.r1, seg.z1)] for seg in wall.segments]
    return LineCollection(lines, colors="#BBBBBB", linewidths=0.6, alpha=0.8, zorder=1)


def log_color_limits(data: np.ndarray, field: str) -> tuple[float, float] | None:
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        return None
    if field == "S_ion":
        vmin = float(np.percentile(positive, 5))
        vmax = float(np.percentile(positive, 99))
    else:
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return max(vmin, 1.0e-30), vmax


# Backward-compatible aliases used during extraction
_region_color = region_color
_log_weights = log_weights
_wall_outline_segments = wall_outline_segments
