"""Core fueling GUI maps — f_k, f_k^flux, and HDG context plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.io.wall_flux import WallNeutralFluxResult
from adjoint_mc.scoring.amcis_provenance import emission_weighted_segment_probability
from adjoint_mc.scoring.provenance import ProvenanceResult
from adjoint_mc.tracker.backward_full import BackwardFullResult
from adjoint_mc.viz.wall_style import log_color_limits
from adjoint_mc.viz.plasma_fields import physical_field_array, physical_field_specs, plot_ionization_source
from adjoint_mc.viz.provenance import _plot_segment_scalar_map, smooth_segment_values

if TYPE_CHECKING:
    from adjoint_mc.pipeline.core_fueling_run import CoreFuelingRunResult

_REGION_COLORS = {
    "wall": "#4C72B0",
    "puff": "#DD8452",
    "pump": "#8172B3",
    "main_wall": "#4C72B0",
    "inner_divertor": "#55A868",
    "outer_divertor": "#C44E52",
}


def _region_color(name: str) -> str:
    return _REGION_COLORS.get(name, "gray")


def _overlay_separatrix(ax, solution: Any | None, *, color: str = "white", linestyle: str = "-") -> None:
    if solution is None:
        return
    from adjoint_mc.viz.separatrix import overlay_separatrix

    overlay_separatrix(ax, solution, color=color, linestyle=linestyle, linewidth=1.4)


def _unavailable_figure(message: str, *, figsize: Tuple[float, float] = (9.0, 8.0)) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_core_fueling_wall_fraction(
    wall: WallGeometry,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    """Wall segment map without separatrix (f_k / f_k^flux display)."""
    data = np.asarray(values, dtype=float)
    display = smooth_segment_values(wall, data, sigma_m, log_domain=True)
    fig, ax = plt.subplots(figsize=figsize)
    raw = data
    disp = display

    outline = LineCollection(
        [[(seg.r0, seg.z0), (seg.r1, seg.z1)] for seg in wall.segments],
        colors="#888888",
        linewidths=1.6,
        zorder=1,
    )
    ax.add_collection(outline)

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
        ax.add_collection(LineCollection(lines_zero, colors="#BBBBBB", linewidths=2.4, zorder=2))
    if lines_pos:
        color_array = np.asarray(colors_pos)
        vmin, vmax = np.percentile(color_array, (5.0, 95.0)) if color_array.size > 2 else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        lc = LineCollection(lines_pos, array=color_array, cmap="viridis", linewidths=2.8, zorder=3)
        lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    rs = [s.r0 for s in wall.segments] + [s.r1 for s in wall.segments]
    zs = [s.z0 for s in wall.segments] + [s.z1 for s in wall.segments]
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.02)
    ax.set_ylim(min(zs) - 0.02, max(zs) + 0.02)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title(title + "\n(contour-smoothed display)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_core_fueling_f_k(wall: WallGeometry, provenance: ProvenanceResult) -> Figure:
    return plot_core_fueling_wall_fraction(
        wall,
        provenance.segment_probability,
        title=r"Wall provenance $f_k$ (MC connectivity only)",
        cbar_label=r"$\log_{10}$ provenance fraction $f_k$",
    )


def plot_core_fueling_f_k_flux(
    wall: WallGeometry,
    provenance: ProvenanceResult,
) -> Figure:
    if provenance.wall_flux is None:
        return _unavailable_figure(
            "Emission-weighted provenance f_k^flux requires SOLEDGE wall neutral flux\n"
            "(|Γ_n,⊥| on boundary segments)."
        )
    f_k_flux = emission_weighted_segment_probability(
        wall,
        provenance.segment_contribution,
        provenance.wall_flux,
    )
    return plot_core_fueling_wall_fraction(
        wall,
        f_k_flux,
        title=r"Emission-weighted provenance $f_k^{\mathrm{flux}} = C_k\,\Gamma_k^{\mathrm{wall}} / \Sigma_j C_j\,\Gamma_j^{\mathrm{wall}}$",
        cbar_label=r"$\log_{10}$ $f_k^{\mathrm{flux}}$",
        sigma_m=None,
    )


def plot_core_fueling_s_ion(solution: Any, *, figsize: Tuple[float, float] = (9.0, 8.0)) -> Figure:
    fig, ax = plot_ionization_source(solution, n_levels=50)
    fig.set_size_inches(figsize[0], figsize[1])
    _overlay_separatrix(ax, solution, color="white")
    stats_line = fig._suptitle.get_text() if fig._suptitle is not None else "Volume ionization source"
    fig.suptitle(f"{stats_line}\nCore fueling — S_ion on HDG mesh", fontsize=11)
    fig.tight_layout()
    return fig


def plot_core_birth_zone_map(
    grid: PretabulatedGrid,
    birth_mask: np.ndarray,
    solution: object | None = None,
) -> Figure:
    """S_ion on the pre-tabulated grid with core birth mask (inside separatrix)."""
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    limits = log_color_limits(grid.fields["S_ion"], "S_ion")
    mesh_kwargs: dict = {"shading": "auto", "cmap": "magma"}
    if limits is not None:
        from matplotlib.colors import LogNorm

        mesh_kwargs["norm"] = LogNorm(vmin=limits[0], vmax=limits[1])
    ax.pcolormesh(grid.r_coords, grid.z_coords, grid.fields["S_ion"], alpha=0.35, **mesh_kwargs)
    overlay = ax.pcolormesh(
        grid.r_coords,
        grid.z_coords,
        np.where(birth_mask, 1.0, np.nan),
        cmap="Greens",
        alpha=0.55,
        shading="auto",
    )
    _overlay_separatrix(ax, solution, color="white")
    n_core = int(np.count_nonzero(birth_mask))
    n_plasma = int(np.count_nonzero(grid.mask))
    ax.set_title(
        "Core birth zone — S_ion-weighted seeds inside separatrix (ψ = 1)\n"
        f"{n_core} / {n_plasma} plasma cells ({100.0 * n_core / max(n_plasma, 1):.1f} %)",
        fontsize=11,
    )
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    fig.colorbar(overlay, ax=ax, label="core birth mask")
    fig.tight_layout()
    return fig


def plot_core_fueling_wall_hits(
    wall: WallGeometry,
    mc_result: BackwardFullResult,
    solution: Any | None = None,
    *,
    max_display: int = 500,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """Poloidal birth points and wall impacts for backward MC histories."""
    wall_scores = [
        score
        for score in mc_result.scores
        if score.termination == "wall" and score.hit_r is not None and score.hit_z is not None
    ]
    if len(wall_scores) > max_display:
        wall_scores.sort(key=lambda s: s.weight, reverse=True)
        wall_scores = wall_scores[:max_display]
        subtitle = f"Top {max_display} wall hits by weight (of {mc_result.tallies.n_wall} total)"
    else:
        subtitle = f"{len(wall_scores)} wall hits"

    fig, ax = plt.subplots(figsize=figsize)
    outline = LineCollection(
        [[(seg.r0, seg.z0), (seg.r1, seg.z1)] for seg in wall.segments],
        colors="#CCCCCC",
        linewidths=1.0,
        zorder=1,
    )
    ax.add_collection(outline)
    _overlay_separatrix(ax, solution, color="black", linestyle="--")

    if wall_scores:
        weights = np.asarray([s.weight for s in wall_scores], dtype=float)
        log_w = np.log10(np.maximum(weights, 1e-30))
        for score in wall_scores:
            color = _region_color(score.region_name or "wall")
            ax.plot(
                [score.seed_r, score.hit_r],
                [score.seed_z, score.hit_z],
                color=color,
                alpha=0.25,
                linewidth=0.6,
                zorder=2,
            )
            ax.scatter(score.seed_r, score.seed_z, s=10, color=color, alpha=0.55, zorder=3)
            ax.scatter(
                score.hit_r,
                score.hit_z,
                s=16,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                alpha=0.85,
                zorder=4,
            )
        ax.text(
            0.02,
            0.98,
            f"{subtitle}\nlog10(W): [{log_w.min():.1f}, {log_w.max():.1f}]\n"
            "small dots = core birth  •  large dots = wall impact",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    rs = [s.r0 for s in wall.segments] + [s.r1 for s in wall.segments]
    zs = [s.z0 for s in wall.segments] + [s.z1 for s in wall.segments]
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.02)
    ax.set_ylim(min(zs) - 0.02, max(zs) + 0.02)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title("Backward MC — core births and wall impacts")
    handles = [
        Patch(facecolor=_region_color(name), edgecolor="none", label=name)
        for name in sorted({s.region_name for s in wall_scores if s.region_name})
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_core_fueling_wall_neutral_flux(
    wall: WallGeometry,
    wall_flux: WallNeutralFluxResult,
    solution: Any | None = None,
    *,
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    """Single map of SOLEDGE ring-integrated |Γ_n,⊥| on wall segments."""
    fig = _plot_segment_scalar_map(
        wall,
        wall_flux.segment_flux,
        title="SOLEDGE wall neutral flux (|Γ_n,⊥|, ring integral per segment)",
        cbar_label=r"$\log_{10}$ ring flux  $\int |\Gamma_{n,\perp}|\, ds\, 2\pi R$",
        cmap="plasma",
        figsize=figsize,
        sigma_m=sigma_m,
    )
    _overlay_separatrix(fig.axes[0], solution, color="black", linestyle="--")
    fig.tight_layout()
    return fig


def plot_core_fueling_nn(solution: Any, *, figsize: Tuple[float, float] = (9.0, 8.0)) -> Figure:
    """Neutral density n_n on the HDG simple mesh."""
    try:
        spec = next(s for s in physical_field_specs(solution) if s.key == "n_n")
    except StopIteration:
        return _unavailable_figure("Neutral density n_n unavailable for this solution.")
    from adjoint_mc.viz.plasma_fields import plot_physical_field

    data = physical_field_array(solution, "n_n")
    fig = plot_physical_field(solution, spec, data, figsize=figsize)
    _overlay_separatrix(fig.axes[0], solution, color="white")
    fig.axes[0].set_title(r"Neutral density $n_n$ — core fueling context", fontsize=12)
    fig.tight_layout()
    return fig


def iter_core_fueling_plot_tabs(
    result: "CoreFuelingRunResult",
) -> list[tuple[str, str, Figure]]:
    """GUI plot tabs for the core fueling tab (f_k^flux-focused set)."""
    wall = result.wall
    provenance = result.provenance
    solution = result.solution
    tabs: list[tuple[str, str, Figure]] = [
        ("fk", "f_k", plot_core_fueling_f_k(wall, provenance)),
        ("fk_flux", r"f_k^flux", plot_core_fueling_f_k_flux(wall, provenance)),
        ("s_ion", "S_ion", plot_core_fueling_s_ion(solution)),
        (
            "core_birth",
            "Core birth zone",
            plot_core_birth_zone_map(result.grid, result.birth_mask, solution=solution),
        ),
        (
            "wall_hits",
            "Wall hits",
            plot_core_fueling_wall_hits(wall, result.mc_result, solution=solution),
        ),
    ]
    if provenance.wall_flux is not None:
        tabs.append(
            (
                "wall_flux",
                r"Wall |Γ_n,⊥|",
                plot_core_fueling_wall_neutral_flux(wall, provenance.wall_flux, solution=solution),
            )
        )
    else:
        tabs.append(
            (
                "wall_flux",
                r"Wall |Γ_n,⊥|",
                _unavailable_figure("SOLEDGE wall neutral flux unavailable for this case."),
            )
        )
    tabs.append(("n_n", r"n_n", plot_core_fueling_nn(solution)))
    return tabs


def format_core_fueling_summary_text(result: "CoreFuelingRunResult", *, plots_s: float = 0.0) -> str:
    """Text summary emphasising f_k and f_k^flux for the core fueling tab."""
    from adjoint_mc.viz.mc_summary import format_backward_full_summary_text

    prov = result.provenance
    timing = result.timing
    mc_summary = format_backward_full_summary_text(
        result.mc_result,
        grid=result.grid,
        grid_build_s=timing.grid_build_s,
        mc_s=timing.mc_s,
        mc_label="Cython MC (core births)",
        kernel_s=timing.kernel_s,
        pack_s=timing.pack_s,
        n_threads=result.n_threads,
        plots_s=plots_s,
        provenance_s=timing.provenance_s,
        total_s=timing.total_s + plots_s,
        header="Core fueling — S_ion births inside separatrix (ψ = 1)",
    )
    lines = [
        mc_summary,
        "",
        f"Core cells (birth mask) : {int(np.count_nonzero(result.birth_mask))}",
        f"∫ S_ion dV (core)       : {result.core_fueling_rate_s:.6g} s⁻¹  (validation closure only)",
        "",
        f"Wall hits : {prov.n_wall_hits} / {prov.n_histories} histories",
        "",
        "Wall provenance f_k [%] — MC connectivity (conditional on wall hits):",
    ]
    for name in sorted(prov.region_probability):
        frac = 100.0 * prov.region_probability[name]
        lines.append(f"  {name:20s}  f_k={frac:6.2f} %")

    if prov.wall_flux is not None:
        f_k_flux = emission_weighted_segment_probability(
            result.wall, prov.segment_contribution, prov.wall_flux
        )
        region_flux: dict[str, float] = {}
        for seg in result.wall.segments:
            region_flux[seg.region_name] = region_flux.get(seg.region_name, 0.0) + float(
                f_k_flux[seg.segment_index]
            )
        lines.extend(
            [
                "",
                "Emission-weighted provenance f_k^flux [%] — C_k × Γ_k^wall, normalised:",
            ]
        )
        for name in sorted(region_flux):
            lines.append(f"  {name:20s}  f_k^flux={100.0 * region_flux[name]:6.2f} %")
    else:
        lines.extend(["", "f_k^flux unavailable (SOLEDGE wall neutral flux missing)."])

    return "\n".join(lines)
