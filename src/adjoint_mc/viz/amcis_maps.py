"""AMCIS provenance wall maps (survival weights) and HDG field context."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.io.wall_flux import WallNeutralFluxResult
from adjoint_mc.scoring.amcis_provenance import AmcisProvenanceResult
from adjoint_mc.viz.plasma_fields import (
    neutral_perpendicular_flux_spec,
    physical_field_array,
    physical_field_specs,
    plot_physical_field_with_target,
)
from adjoint_mc.viz.provenance import smooth_segment_values

if TYPE_CHECKING:
    from adjoint_mc.pipeline.amcis_run import AmcisRunResult
    from adjoint_mc.tracker.amcis_backward import AmcisMcResult


def _overlay_separatrix(
    ax,
    solution: Any | None,
    *,
    color: str = "black",
    linestyle: str = "--",
) -> None:
    if solution is None:
        return
    from adjoint_mc.viz.separatrix import overlay_separatrix

    overlay_separatrix(ax, solution, color=color, linestyle=linestyle, linewidth=1.4)


def plot_amcis_wall_map(
    wall: WallGeometry,
    provenance: AmcisProvenanceResult,
    *,
    solution: Any | None = None,
    values: str = "probability",
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    """Colour wall segments by f_k(Ω) or raw visibility C_k."""
    if values == "contribution":
        data = provenance.segment_contribution
        title = r"Wall visibility $C_k(\Omega)$"
        cbar_label = r"$\log_{10}$ visibility weight $C_k$"
    elif values == "emission_probability":
        if not provenance.has_emission_weighting or provenance.segment_emission_probability is None:
            raise ValueError("emission_probability requires SOLEDGE wall flux on provenance")
        data = provenance.segment_emission_probability
        title = r"Emission-weighted provenance $f_k^{\mathrm{flux}}(\Omega)$"
        cbar_label = r"$\log_{10}$ $f_k^{\mathrm{flux}}$ ($C_k\,\Gamma_k^{\mathrm{wall}}$)"
    elif values == "emission_weight":
        if not provenance.has_emission_weighting or provenance.segment_emission_weight is None:
            raise ValueError("emission_weight requires SOLEDGE wall flux on provenance")
        data = provenance.segment_emission_weight
        title = r"Cross-term $D_k = C_k(\Omega)\,\Gamma_k^{\mathrm{wall}}$"
        cbar_label = r"$\log_{10}$ $D_k$"
    elif values == "attributed_flux":
        if not provenance.has_emission_weighting or provenance.segment_attributed_flux is None:
            raise ValueError("attributed_flux requires SOLEDGE wall flux on provenance")
        data = provenance.segment_attributed_flux
        title = r"Attributed wall emission $\Phi_{k\to\Omega} = f_k(\Omega)\,\Gamma_k^{\mathrm{wall}}$"
        cbar_label = r"$\log_{10}$ $\Phi_{k\to\Omega}$"
    else:
        data = provenance.segment_probability
        title = r"Wall provenance $f_k(\Omega)$"
        cbar_label = r"$\log_{10}$ provenance fraction $f_k$"

    display = smooth_segment_values(wall, data, sigma_m, log_domain=True)
    title += (
        f"\ntarget ({provenance.target_r:.3f}, {provenance.target_z:.3f}) m"
        f" — contour-smoothed display"
    )

    fig, ax = plt.subplots(figsize=figsize)
    raw = np.asarray(data, dtype=float)
    disp = np.asarray(display, dtype=float)

    lines_pos: list = []
    colors_pos: list[float] = []
    lines_zero: list = []
    for seg in wall.segments:
        line = [(seg.r0, seg.z0), (seg.r1, seg.z1)]
        value = float(disp[seg.segment_index])
        if value > 0.0 or raw[seg.segment_index] > 0.0:
            lines_pos.append(line)
            colors_pos.append(float(np.log10(max(value, 1e-30))))
        else:
            lines_zero.append(line)

    if lines_zero:
        ax.add_collection(LineCollection(lines_zero, colors="#D0D0D0", linewidths=2.0, zorder=1))
    if lines_pos:
        color_array = np.asarray(colors_pos)
        vmin, vmax = np.percentile(color_array, (5.0, 95.0)) if color_array.size > 2 else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        lc = LineCollection(lines_pos, array=color_array, cmap="viridis", linewidths=2.8, zorder=2)
        lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    rs = [s.r0 for s in wall.segments] + [s.r1 for s in wall.segments]
    zs = [s.z0 for s in wall.segments] + [s.z1 for s in wall.segments]
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.02)
    ax.set_ylim(min(zs) - 0.02, max(zs) + 0.02)
    _overlay_separatrix(ax, solution, color="black", linestyle="-")
    ax.scatter(
        [provenance.target_r],
        [provenance.target_z],
        s=80,
        c="red",
        marker="*",
        zorder=5,
        label="target Ω",
    )
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_amcis_target_on_grid(
    grid,
    provenance: AmcisProvenanceResult,
    *,
    solution: Any | None = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
) -> Figure:
    """Show target point on the plasma mask (context plot)."""
    from adjoint_mc.fields.pretabulate import PretabulatedGrid

    if not isinstance(grid, PretabulatedGrid):
        raise TypeError("grid must be PretabulatedGrid")
    fig, ax = plt.subplots(figsize=figsize)
    rr, zz = np.meshgrid(grid.r_coords, grid.z_coords)
    mask = np.ma.masked_where(~grid.mask, np.ones_like(grid.mask, dtype=float))
    ax.pcolormesh(rr, zz, mask, cmap="Greys", alpha=0.35, shading="auto")
    _overlay_separatrix(ax, solution, color="black", linestyle="-")
    ax.scatter(
        [provenance.target_r],
        [provenance.target_z],
        s=100,
        c="red",
        marker="*",
        zorder=3,
        label="target Ω",
    )
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title("AMCIS target on plasma grid")
    ax.legend()
    fig.tight_layout()
    return fig


def _unavailable_field_figure(message: str, *, figsize: Tuple[float, float] = (9.0, 8.0)) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_amcis_nn_field(
    solution: Any,
    provenance: AmcisProvenanceResult,
    *,
    n_levels: int = 50,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """Neutral density n_n on the HDG simple mesh (HDG_postprocess)."""
    try:
        spec = next(s for s in physical_field_specs(solution) if s.key == "n_n")
    except StopIteration:
        return _unavailable_field_figure("Neutral density n_n unavailable for this solution.")
    data = physical_field_array(solution, "n_n")
    return plot_physical_field_with_target(
        solution,
        spec,
        data,
        target_r=provenance.target_r,
        target_z=provenance.target_z,
        n_levels=n_levels,
        figsize=figsize,
        suptitle=(
            f"SOLEDGE-HDG neutral density — AMCIS target "
            f"({provenance.target_r:.3f}, {provenance.target_z:.3f}) m"
        ),
    )


def plot_amcis_gamma_n_field(
    solution: Any,
    provenance: AmcisProvenanceResult,
    *,
    n_levels: int = 50,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """Perpendicular neutral flux Γ_n on the HDG simple mesh."""
    spec = neutral_perpendicular_flux_spec(solution)
    if spec is None:
        return _unavailable_field_figure(
            "Perpendicular neutral flux unavailable (need Neq ≥ 6 with Gamman)."
        )
    data = physical_field_array(solution, spec.key)
    return plot_physical_field_with_target(
        solution,
        spec,
        data,
        target_r=provenance.target_r,
        target_z=provenance.target_z,
        n_levels=n_levels,
        figsize=figsize,
        suptitle=(
            f"SOLEDGE-HDG {spec.tab_title} — AMCIS target "
            f"({provenance.target_r:.3f}, {provenance.target_z:.3f}) m"
        ),
    )


def plot_amcis_ionization_source(
    solution: Any,
    provenance: AmcisProvenanceResult,
    *,
    n_levels: int = 50,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """Volume ionization source S_ion on the HDG simple mesh."""
    from adjoint_mc.viz.plasma_fields import plot_ionization_source

    fig, ax = plot_ionization_source(solution, n_levels=n_levels)
    fig.set_size_inches(figsize[0], figsize[1])
    _overlay_separatrix(ax, solution, color="white", linestyle="-")
    ax.scatter(
        [provenance.target_r],
        [provenance.target_z],
        s=120,
        c="red",
        marker="*",
        zorder=10,
        edgecolors="white",
        linewidths=0.6,
        label=r"target $\Omega$",
    )
    ax.legend(loc="upper right")
    stats_line = fig._suptitle.get_text() if fig._suptitle is not None else "Volume ionization source"
    fig.suptitle(
        f"{stats_line}\n"
        f"AMCIS target ({provenance.target_r:.3f}, {provenance.target_z:.3f}) m",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_amcis_wall_neutral_flux(
    wall: WallGeometry,
    wall_flux: WallNeutralFluxResult,
    provenance: AmcisProvenanceResult,
    *,
    solution: Any | None = None,
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    """SOLEDGE boundary neutral flux |Γ_n,⊥| mapped onto wall segments."""
    from adjoint_mc.viz.provenance import _plot_segment_scalar_map

    fig = _plot_segment_scalar_map(
        wall,
        wall_flux.segment_flux,
        title=(
            "SOLEDGE wall neutral flux (|Γ_n,⊥|, ring integral)\n"
            f"AMCIS target ({provenance.target_r:.3f}, {provenance.target_z:.3f}) m"
            " — compare shape with f_k(Ω), not same quantity"
        ),
        cbar_label=r"$\log_{10}$ ring flux  $\int |\Gamma_{n,\perp}|\, ds\, 2\pi R$",
        cmap="plasma",
        figsize=figsize,
        sigma_m=sigma_m,
    )
    _overlay_separatrix(fig.axes[0], solution, color="black", linestyle="-")
    fig.axes[0].scatter(
        [provenance.target_r],
        [provenance.target_z],
        s=80,
        c="red",
        marker="*",
        zorder=9,
        label="target Ω",
    )
    fig.axes[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_amcis_wall_hits(
    wall: WallGeometry,
    result: AmcisMcResult,
    *,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """AMCIS individual wall hit points coloured by region; marker size ~ log10(W)."""
    from adjoint_mc.viz.wall_style import wall_outline_segments, region_color, log_weights

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(wall_outline_segments(wall))

    wall_hits = [s for s in result.scores if s.termination == "wall" and s.hit_r is not None]
    lost = [s for s in result.scores if s.termination != "wall"]

    # Target point (birth)
    ax.scatter(
        [result.target_r],
        [result.target_z],
        s=120,
        c="red",
        marker="*",
        zorder=5,
        edgecolors="white",
        linewidths=0.6,
        label=r"target $\Omega$",
    )

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

    if lost:
        ax.scatter([], [], s=0, label=f"Lost / no wall ({len(lost)})")

    rs = [s.r0 for s in wall.segments] + [s.r1 for s in wall.segments]
    zs = [s.z0 for s in wall.segments] + [s.z1 for s in wall.segments]
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.02)
    ax.set_ylim(min(zs) - 0.02, max(zs) + 0.02)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("AMCIS wall hits (marker size ~ log10 W)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def iter_amcis_plot_tabs(result: "AmcisRunResult") -> Iterator[tuple[str, Figure]]:
    """All AMCIS GUI / export figures: provenance maps + HDG neutral fields."""
    yield (
        "Wall f_k(Ω)",
        plot_amcis_wall_map(
            result.wall, result.provenance, solution=result.solution, values="probability"
        ),
    )
    yield (
        "Wall hits (W)",
        plot_amcis_wall_hits(result.wall, result.mc_result),
    )
    yield (
        "Visibility C_k",
        plot_amcis_wall_map(
            result.wall, result.provenance, solution=result.solution, values="contribution"
        ),
    )
    yield "n_n", plot_amcis_nn_field(result.solution, result.provenance)
    yield "S_ion", plot_amcis_ionization_source(result.solution, result.provenance)
    flux_spec = neutral_perpendicular_flux_spec(result.solution)
    flux_tab = flux_spec.tab_title if flux_spec is not None else r"Γ_n,⊥"
    yield flux_tab, plot_amcis_gamma_n_field(result.solution, result.provenance)
    if result.wall_flux is not None:
        yield (
            "Wall |Γ_n,⊥|",
            plot_amcis_wall_neutral_flux(
                result.wall, result.wall_flux, result.provenance, solution=result.solution
            ),
        )
    if result.provenance.has_emission_weighting:
        yield (
            r"f_k^flux(Ω)",
            plot_amcis_wall_map(
                result.wall,
                result.provenance,
                solution=result.solution,
                values="emission_probability",
            ),
        )
        yield (
            r"D_k = C_k Γ_wall",
            plot_amcis_wall_map(
                result.wall,
                result.provenance,
                solution=result.solution,
                values="emission_weight",
            ),
        )
        yield (
            r"Φ_k→Ω",
            plot_amcis_wall_map(
                result.wall,
                result.provenance,
                solution=result.solution,
                values="attributed_flux",
            ),
        )
    yield "Target Ω", plot_amcis_target_on_grid(
        result.grid, result.provenance, solution=result.solution
    )
