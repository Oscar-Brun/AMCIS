""" provenance maps, uncertainties, and plasma-zone analysis."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.scoring.provenance import (
    ProvenanceResult,
    _contributions_from_scores,
    _normalize_contributions,
    _seed_in_zone,
)
from adjoint_mc.tracker.backward_full import BackwardFullResult
from adjoint_mc.viz.separatrix import overlay_separatrix

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


def default_smooth_sigma_m(wall: WallGeometry) -> float:
    """Gaussian smoothing scale along the wall contour (~15 mesh edges)."""
    lengths = np.asarray([seg.length for seg in wall.segments], dtype=float)
    if lengths.size == 0:
        return 0.02
    median_ds = float(np.median(lengths[lengths > 0.0])) if np.any(lengths > 0.0) else 0.002
    return float(np.clip(5.0 * median_ds, 0.005, 0.06))


def _wall_poloidal_chains(wall: WallGeometry, *, endpoint_tol: float = 1e-8) -> list[list[int]]:
    """Group segment indices into contiguous poloidal chains (mesh topology)."""
    n = wall.n_segments
    if n == 0:
        return []
    chains: list[list[int]] = [[0]]
    for i in range(1, n):
        prev = wall.segments[i - 1]
        seg = wall.segments[i]
        connected = (
            prev.region_name == seg.region_name
            and abs(prev.r1 - seg.r0) <= endpoint_tol
            and abs(prev.z1 - seg.z0) <= endpoint_tol
        )
        if connected:
            chains[-1].append(i)
        else:
            chains.append([i])
    return chains


def _chain_arc_midpoints(wall: WallGeometry, chain: list[int]) -> np.ndarray:
    """Arc length [m] at the midpoint of each segment in a chain."""
    arc = 0.0
    points: list[float] = []
    for index in chain:
        seg = wall.segments[index]
        points.append(arc + 0.5 * seg.length)
        arc += seg.length
    return np.asarray(points, dtype=float)


def _gaussian_smooth_along_arc(
    values: np.ndarray,
    arc_s: np.ndarray,
    sigma_m: float,
) -> np.ndarray:
    """Length-weighted Gaussian filter along a 1D contour (all segments kept)."""
    n = len(values)
    if n <= 1 or sigma_m <= 0.0:
        return np.asarray(values, dtype=float).copy()

    data = np.asarray(values, dtype=float)
    ds = arc_s[:, None] - arc_s[None, :]
    weights = np.exp(-0.5 * (ds / sigma_m) ** 2)
    row_sum = weights.sum(axis=1)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    return (weights @ data) / row_sum


def smooth_segment_values(
    wall: WallGeometry,
    values: np.ndarray,
    sigma_m: float | None = None,
    *,
    log_domain: bool = False,
) -> np.ndarray:
    """
    Gaussian smooth along the physical wall contour, keeping every segment.

    Smoothing follows mesh connectivity (not polar angle), separately on each
    contiguous poloidal chain. Optional log-domain filtering reduces salt-and-
    pepper on log-scale maps while preserving all 1373+ segment boundaries.
    """
    n = wall.n_segments
    if n == 0:
        return np.asarray(values, dtype=float).copy()
    sigma = default_smooth_sigma_m(wall) if sigma_m is None else float(sigma_m)
    if sigma <= 0.0:
        return np.asarray(values, dtype=float).copy()

    data = np.asarray(values, dtype=float)
    out = np.zeros(n, dtype=float)

    for chain in _wall_poloidal_chains(wall):
        idx = np.asarray(chain, dtype=int)
        chain_vals = data[idx]
        arc_s = _chain_arc_midpoints(wall, chain)

        if log_domain:
            positive = chain_vals > 0.0
            if not np.any(positive):
                out[idx] = chain_vals
                continue
            floor = float(np.min(chain_vals[positive]))
            log_vals = np.log10(np.where(positive, chain_vals, floor))
            smoothed_log = _gaussian_smooth_along_arc(log_vals, arc_s, sigma)
            out[idx] = np.power(10.0, smoothed_log)
        else:
            out[idx] = _gaussian_smooth_along_arc(chain_vals, arc_s, sigma)

    return out


def _segment_probabilities_from_scores(
    wall: WallGeometry,
    scores: list,
) -> np.ndarray:
    segment_c, region_c = _contributions_from_scores(wall, scores)
    segment_p, _, _, _ = _normalize_contributions(segment_c, region_c, 1.0)
    return segment_p


def _color_limits(
    color_values: np.ndarray,
    *,
    log_scale: bool,
    percentile_scale: bool,
) -> tuple[float, float]:
    if color_values.size == 0:
        return 0.0, 1.0
    if percentile_scale and color_values.size > 2:
        lo, hi = np.percentile(color_values, (5.0, 95.0))
    else:
        lo, hi = float(color_values.min()), float(color_values.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _draw_segment_scalar_on_ax(
    ax,
    wall: WallGeometry,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "inferno",
    log_scale: bool = True,
    zero_label: str = "zero / unmapped",
    percentile_scale: bool = True,
    show_cbar: bool = True,
    fig=None,
) -> None:
    """Colour wall segments by a per-segment scalar."""
    data = np.asarray(values, dtype=float)

    lines_pos: list = []
    colors_pos: list[float] = []
    lines_zero: list = []
    for seg in wall.segments:
        line = [(seg.r0, seg.z0), (seg.r1, seg.z1)]
        value = float(data[seg.segment_index])
        if value > 0.0:
            lines_pos.append(line)
            colors_pos.append(float(np.log10(value)) if log_scale else value)
        else:
            lines_zero.append(line)

    if lines_zero:
        ax.add_collection(
            LineCollection(lines_zero, colors="#D0D0D0", linewidths=2.0, zorder=1, label=zero_label)
        )

    if lines_pos:
        color_array = np.asarray(colors_pos, dtype=float)
        vmin, vmax = _color_limits(color_array, log_scale=log_scale, percentile_scale=percentile_scale)
        lc = LineCollection(lines_pos, array=color_array, cmap=cmap, linewidths=2.8, zorder=2)
        lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
        if show_cbar and fig is not None:
            cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)
    else:
        ax.text(0.5, 0.5, "No positive values", ha="center", va="center", transform=ax.transAxes)

    _style_wall_axes(ax, wall, title)


def _plot_segment_scalar_map(
    wall: WallGeometry,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "inferno",
    log_scale: bool = True,
    zero_label: str = "zero / unmapped",
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
    percentile_scale: bool = True,
) -> Figure:
    """Colour wall segments by a per-segment scalar (contour-smoothed, log scale on positives)."""
    display = smooth_segment_values(
        wall,
        values,
        sigma_m,
        log_domain=log_scale,
    )

    fig, ax = plt.subplots(figsize=figsize)
    _draw_segment_scalar_on_ax(
        ax,
        wall,
        display,
        title=title,
        cbar_label=cbar_label,
        cmap=cmap,
        log_scale=log_scale,
        zero_label=zero_label,
        percentile_scale=percentile_scale,
        show_cbar=True,
        fig=fig,
    )
    fig.tight_layout()
    return fig


def _style_wall_axes(ax, wall: WallGeometry, title: str) -> None:
    rs = [s.r0 for s in wall.segments] + [s.r1 for s in wall.segments]
    zs = [s.z0 for s in wall.segments] + [s.z1 for s in wall.segments]
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.02)
    ax.set_ylim(min(zs) - 0.02, max(zs) + 0.02)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)


def _annotate_wall_regions(wall: WallGeometry, ax) -> None:
    """Label each wall region once at its poloidal centroid."""
    by_region: dict[str, list[tuple[float, float]]] = {}
    for seg in wall.segments:
        by_region.setdefault(seg.region_name, []).append(
            (0.5 * (seg.r0 + seg.r1), 0.5 * (seg.z0 + seg.z1))
        )
    for name, points in by_region.items():
        xs, ys = zip(*points)
        ax.text(
            float(np.mean(xs)),
            float(np.mean(ys)),
            name,
            color=_region_color(name),
            fontsize=9,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            zorder=5,
        )


def plot_segment_provenance_map(
    wall: WallGeometry,
    result: ProvenanceResult,
    *,
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    fig = _plot_segment_scalar_map(
        wall,
        result.segment_probability,
        title="Wall provenance f_k (contour-smoothed)",
        cbar_label=r"$\log_{10}$ provenance probability $f_k$",
        cmap="viridis",
        figsize=figsize,
        sigma_m=sigma_m,
    )
    _annotate_wall_regions(wall, fig.axes[0])
    return fig


def plot_segment_flux_density_map(
    wall: WallGeometry,
    result: ProvenanceResult,
    *,
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    return _plot_segment_scalar_map(
        wall,
        result.segment_flux_density_s,
        title=r"Attributed flux density $\Gamma_{\mathrm{prov},k} / (2\pi R\, ds)$ (contour-smoothed)",
        cbar_label=r"$\log_{10}$ flux density [s$^{-1}$ m$^{-2}$]",
        cmap="magma",
        figsize=figsize,
        sigma_m=sigma_m,
    )


def plot_segment_soledge_neutral_map(
    wall: WallGeometry,
    result: ProvenanceResult,
    *,
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    if result.wall_flux is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "SOLEDGE boundary flux unavailable", ha="center", va="center")
        ax.axis("off")
        return fig
    return _plot_segment_scalar_map(
        wall,
        result.wall_flux.segment_flux,
        title="SOLEDGE neutral wall flux (|Γ_n,⊥|, ring integral)",
        cbar_label=r"$\log_{10}$ ring flux  $\int |\Gamma_{n,\perp}|\, ds\, 2\pi R$",
        cmap="plasma",
        figsize=figsize,
        sigma_m=sigma_m,
    )


def plot_segment_soledge_parallel_map(
    wall: WallGeometry,
    result: ProvenanceResult,
    *,
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure:
    if result.wall_flux is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "SOLEDGE parallel flux unavailable", ha="center", va="center")
        ax.axis("off")
        return fig
    return _plot_segment_scalar_map(
        wall,
        result.wall_flux.segment_parallel_flux,
        title="SOLEDGE parallel recycling flux (|Γ_∥,wall|, ring integral)",
        cbar_label=r"$\log_{10}$ ring flux  $\int |\Gamma_{\parallel,\mathrm{wall}}|\, ds\, 2\pi R$",
        cmap="cividis",
        figsize=figsize,
        sigma_m=sigma_m,
    )


def plot_plasma_zone_wall_maps(
    wall: WallGeometry,
    mc_result: BackwardFullResult,
    provenance: ProvenanceResult,
    *,
    max_panels: int = 4,
    figsize: Tuple[float, float] = (12.0, 9.0),
    sigma_m: float | None = None,
) -> Figure:
    """
    Smoothed wall f_k maps for each plasma birth zone (where do seeds from zone Z hit?).
    """
    zones = [
        z
        for z in provenance.plasma_zone_provenance
        if z.zone.name != "full" and z.n_wall_hits > 0
    ]
    zones = sorted(zones, key=lambda z: z.n_wall_hits, reverse=True)[:max_panels]

    if not zones:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No plasma-zone wall hits to display", ha="center", va="center")
        ax.axis("off")
        return fig

    sigma = default_smooth_sigma_m(wall) if sigma_m is None else float(sigma_m)

    n_panels = len(zones)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    shared_pos: list[float] = []
    for zone in zones:
        zone_scores = [score for score in mc_result.scores if _seed_in_zone(score, zone.zone)]
        seg_p = smooth_segment_values(
            wall,
            _segment_probabilities_from_scores(wall, zone_scores),
            sigma,
            log_domain=True,
        )
        shared_pos.extend(float(v) for v in seg_p if v > 0.0)

    if shared_pos:
        shared_log = np.log10(np.asarray(shared_pos, dtype=float))
        shared_vmin, shared_vmax = _color_limits(shared_log, log_scale=True, percentile_scale=True)
    else:
        shared_vmin, shared_vmax = 0.0, 1.0

    for ax, zone in zip(flat_axes, zones):
        zone_scores = [score for score in mc_result.scores if _seed_in_zone(score, zone.zone)]
        seg_p = smooth_segment_values(
            wall,
            _segment_probabilities_from_scores(wall, zone_scores),
            sigma,
            log_domain=True,
        )
        data = np.asarray(seg_p, dtype=float)

        lines_pos: list = []
        colors_pos: list[float] = []
        lines_zero: list = []
        for seg in wall.segments:
            line = [(seg.r0, seg.z0), (seg.r1, seg.z1)]
            value = float(data[seg.segment_index])
            if value > 0.0:
                lines_pos.append(line)
                colors_pos.append(float(np.log10(value)))
            else:
                lines_zero.append(line)

        if lines_zero:
            ax.add_collection(LineCollection(lines_zero, colors="#D0D0D0", linewidths=1.5, zorder=1))
        if lines_pos:
            lc = LineCollection(
                lines_pos,
                array=np.asarray(colors_pos),
                cmap="viridis",
                linewidths=2.2,
                zorder=2,
            )
            lc.set_clim(shared_vmin, shared_vmax)
            ax.add_collection(lc)

        pct_wall = 100.0 * zone.region_probability.get("wall", 0.0)
        ax.set_title(
            f"{zone.zone.label}\n{zone.n_wall_hits} hits, wall f={pct_wall:.0f}%",
            fontsize=10,
        )
        _style_wall_axes(ax, wall, "")

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    if shared_pos:
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(shared_vmin, shared_vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(r"$\log_{10}$ segment provenance $f_k$ (shared scale)")

    fig.suptitle(
        "Wall hit pattern by plasma birth zone (contour-smoothed f_k)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_zone_contrast_map(
    wall: WallGeometry,
    mc_result: BackwardFullResult,
    provenance: ProvenanceResult,
    *,
    zone_a: str = "lower_half",
    zone_b: str = "upper_half",
    figsize: Tuple[float, float] = (9.0, 8.0),
    sigma_m: float | None = None,
) -> Figure | None:
    """log10(f_k zone A / f_k zone B) on the wall — highlights upper/lower asymmetry."""
    by_name = {z.zone.name: z for z in provenance.plasma_zone_provenance}
    if zone_a not in by_name or zone_b not in by_name:
        return None
    if by_name[zone_a].n_wall_hits < 2 or by_name[zone_b].n_wall_hits < 2:
        return None

    sigma = default_smooth_sigma_m(wall) if sigma_m is None else float(sigma_m)

    eps = 1e-30
    scores_a = [s for s in mc_result.scores if _seed_in_zone(s, by_name[zone_a].zone)]
    scores_b = [s for s in mc_result.scores if _seed_in_zone(s, by_name[zone_b].zone)]
    fa = smooth_segment_values(
        wall, _segment_probabilities_from_scores(wall, scores_a), sigma, log_domain=True
    )
    fb = smooth_segment_values(
        wall, _segment_probabilities_from_scores(wall, scores_b), sigma, log_domain=True
    )
    ratio = np.log10((fa + eps) / (fb + eps))

    fig, ax = plt.subplots(figsize=figsize)
    lines: list = []
    colors: list[float] = []
    for seg in wall.segments:
        val = float(ratio[seg.segment_index])
        if fa[seg.segment_index] <= 0.0 and fb[seg.segment_index] <= 0.0:
            continue
        lines.append([(seg.r0, seg.z0), (seg.r1, seg.z1)])
        colors.append(val)

    if not lines:
        ax.text(0.5, 0.5, "No contrast to display", ha="center", va="center")
        ax.axis("off")
        return fig

    color_array = np.asarray(colors)
    vmax = float(np.percentile(np.abs(color_array), 95)) if color_array.size > 2 else 1.0
    vmax = max(vmax, 0.5)
    lc = LineCollection(lines, array=color_array, cmap="RdBu_r", linewidths=2.8, zorder=2)
    lc.set_clim(-vmax, vmax)
    ax.add_collection(lc)
    outline = LineCollection(
        [[(seg.r0, seg.z0), (seg.r1, seg.z1)] for seg in wall.segments],
        colors="#E8E8E8",
        linewidths=0.8,
        zorder=1,
    )
    ax.add_collection(outline)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(f_k^{\mathrm{A}} / f_k^{\mathrm{B}})$")
    label_a = by_name[zone_a].zone.label
    label_b = by_name[zone_b].zone.label
    _style_wall_axes(
        ax,
        wall,
        f"Birth-zone contrast: {label_a} vs {label_b}",
    )
    fig.tight_layout()
    return fig


def plot_high_weight_trajectories(
    wall: WallGeometry,
    mc_result: BackwardFullResult,
    *,
    max_traces: int = 80,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """Poloidal seeds and wall hits for the highest-weight wall histories."""
    wall_scores = [
        score
        for score in mc_result.scores
        if score.termination == "wall" and score.hit_r is not None and score.hit_z is not None
    ]
    wall_scores.sort(key=lambda s: s.weight, reverse=True)
    wall_scores = wall_scores[:max_traces]

    fig, ax = plt.subplots(figsize=figsize)
    outline = LineCollection(
        [[(seg.r0, seg.z0), (seg.r1, seg.z1)] for seg in wall.segments],
        colors="#CCCCCC",
        linewidths=1.0,
        zorder=1,
    )
    ax.add_collection(outline)

    if wall_scores:
        weights = np.asarray([s.weight for s in wall_scores], dtype=float)
        log_w = np.log10(np.maximum(weights, 1e-30))
        for score in wall_scores:
            color = _region_color(score.region_name or "wall")
            ax.plot(
                [score.seed_r, score.hit_r],
                [score.seed_z, score.hit_z],
                color=color,
                alpha=0.35,
                linewidth=0.8,
                zorder=2,
            )
            ax.scatter(score.seed_r, score.seed_z, s=12, color=color, alpha=0.7, zorder=3)
            ax.scatter(score.hit_r, score.hit_z, s=18, color=color, edgecolor="black", linewidth=0.3, zorder=4)

        ax.text(
            0.02,
            0.98,
            f"Top {len(wall_scores)} wall histories by weight\n"
            f"log10(W): [{log_w.min():.1f}, {log_w.max():.1f}]",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    _style_wall_axes(ax, wall, "High-weight backward trajectories (seed → wall)")
    handles = [
        Patch(facecolor=_region_color(name), edgecolor="none", label=name)
        for name in sorted({s.region_name for s in wall_scores if s.region_name})
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def iter_provenance_plot_tabs(
    wall: WallGeometry,
    provenance: ProvenanceResult,
    mc_result: BackwardFullResult | None = None,
) -> list[tuple[str, str, Figure]]:
    """(key, tab_title, figure) for the production GUI notebook."""
    tabs: list[tuple[str, str, Figure]] = [
        ("map_fk", "Provenance f_k", plot_segment_provenance_map(wall, provenance)),
        ("map_density", "Γ_prov density", plot_segment_flux_density_map(wall, provenance)),
    ]
    if mc_result is not None:
        tabs.append(
            (
                "zones",
                "Birth zones → wall",
                plot_plasma_zone_wall_maps(wall, mc_result, provenance),
            )
        )
        contrast = plot_zone_contrast_map(wall, mc_result, provenance)
        if contrast is not None:
            tabs.append(("zone_contrast", "Lower vs upper", contrast))
        tabs.append(
            ("trajectories", "Top trajectories", plot_high_weight_trajectories(wall, mc_result))
        )
    if provenance.wall_flux is not None:
        tabs.append(
            ("soledge_n", "SOLEDGE Γ_n⊥", plot_segment_soledge_neutral_map(wall, provenance))
        )
        tabs.append(
            ("soledge_par", "SOLEDGE Γ_∥", plot_segment_soledge_parallel_map(wall, provenance))
        )
    return tabs
