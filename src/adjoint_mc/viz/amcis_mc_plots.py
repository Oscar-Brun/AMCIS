"""AMCIS MC diagnostics — wall hit scatter (single target Ω)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.viz.wall_style import log_weights, region_color, wall_outline_segments

if TYPE_CHECKING:
    from adjoint_mc.tracker.amcis_backward import AmcisMcResult


def plot_amcis_wall_hits(
    wall: WallGeometry,
    mc_result: "AmcisMcResult",
    *,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """Wall hit locations coloured by region; marker size ~ log10(W). No birth scatter (fixed Ω)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(wall_outline_segments(wall))

    wall_hits = [s for s in mc_result.scores if s.termination == "wall" and s.hit_r is not None]
    lost = [s for s in mc_result.scores if s.termination != "wall"]

    if wall_hits:
        for region in sorted({h.region_name for h in wall_hits if h.region_name}):
            group = [h for h in wall_hits if h.region_name == region]
            rs = [h.hit_r for h in group]
            zs = [h.hit_z for h in group]
            sizes = [20.0 + 8.0 * log_weights(np.array([h.weight]))[0] for h in group]
            ax.scatter(
                rs,
                zs,
                s=sizes,
                c=region_color(region or "wall"),
                edgecolors="k",
                linewidths=0.3,
                alpha=0.85,
                label=f"Wall hit — {region} ({len(group)})",
                zorder=3,
            )

    ax.scatter(
        [mc_result.target_r],
        [mc_result.target_z],
        s=120,
        c="red",
        marker="*",
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
        label=r"target $\Omega$",
    )

    if lost:
        ax.scatter([], [], s=0, label=f"Lost / no wall ({len(lost)})")

    rs = [s.r0 for s in wall.segments] + [s.r1 for s in wall.segments]
    zs = [s.z0 for s in wall.segments] + [s.z1 for s in wall.segments]
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.02)
    ax.set_ylim(min(zs) - 0.02, max(zs) + 0.02)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(
        "Wall hits (marker size ~ log10 W)\n"
        f"All histories born at ({mc_result.target_r:.3f}, {mc_result.target_z:.3f}) m"
    )
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def iter_amcis_mc_plot_tabs(
    wall: WallGeometry,
    mc_result: "AmcisMcResult",
) -> Iterator[tuple[str, Figure]]:
    """Single MC diagnostic tab for AMCIS."""
    yield "Wall hits", plot_amcis_wall_hits(wall, mc_result)
