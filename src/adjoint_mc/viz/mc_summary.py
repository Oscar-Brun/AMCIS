"""Backward MC run summary text (used by core fueling tab)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from adjoint_mc.tracker.backward_full import BackwardFullResult
from adjoint_mc.viz.termination_summary import format_termination_breakdown_lines

if TYPE_CHECKING:
    from adjoint_mc.fields.pretabulate import PretabulatedGrid


def _format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)} min {remainder:.1f} s"


def format_backward_full_summary_text(
    result: BackwardFullResult,
    *,
    grid: PretabulatedGrid | None = None,
    grid_build_s: float | None = None,
    mc_s: float | None = None,
    mc_label: str = "Backward MC",
    kernel_s: float | None = None,
    pack_s: float | None = None,
    n_threads: int | None = None,
    plots_s: float | None = None,
    provenance_s: float | None = None,
    total_s: float | None = None,
    header: str | None = None,
) -> str:
    tallies = result.tallies
    hit_counts = tallies.region_hit_counts(result.scores)
    n_wall = tallies.n_wall
    cx_counts = [s.n_cx_events for s in result.scores]
    lines = [
        header or "Backward adjoint MC (ionization + charge exchange)",
        "",
        "Ionization: W *= exp(-Sigma_ion ds) along the path.",
        "CX: discrete rejection events; W unchanged, velocity resampled.",
        "",
        f"Histories       : {result.n_histories}",
        f"Wall hits       : {n_wall} ({100.0 * tallies.wall_fraction:.1f} %)",
        f"Lost / other    : {tallies.n_lost}",
        f"CX events total : {result.total_cx_events}",
        f"CX mean / hist  : {result.mean_cx_events:.2f}  (max {max(cx_counts) if cx_counts else 0})",
        f"tau_max         : {result.config.tau_max}",
        f"max path        : {result.config.max_path_m} m",
        f"vacuum search   : {result.config.vacuum_wall_search_m} m",
    ]
    lines.extend(format_termination_breakdown_lines(result.scores, grid=grid))
    if n_threads is not None:
        lines.append(f"OpenMP threads  : {n_threads}")
    lines.extend(["", "Hits by region (count):"])
    for name in sorted(hit_counts):
        count = hit_counts[name]
        pct = 100.0 * count / n_wall if n_wall else 0.0
        lines.append(f"  {name:8s}  {count:5d}  ({pct:5.1f} % of hits)")
    lines.extend(["", "Weight share among wall hits:"])
    for name, frac in sorted(tallies.region_fractions().items()):
        lines.append(f"  {name:8s}  {100.0 * frac:6.2f} % of total W")
    timing_parts: list[str] = []
    if grid_build_s is not None:
        timing_parts.append(f"grid build {_format_duration(grid_build_s)}")
    if pack_s is not None:
        timing_parts.append(f"pack {_format_duration(pack_s)}")
    if mc_s is not None:
        per_hist = mc_s / result.n_histories if result.n_histories else 0.0
        timing_parts.append(f"{mc_label} {_format_duration(mc_s)} ({_format_duration(per_hist)}/hist)")
    if kernel_s is not None:
        timing_parts.append(f"Cython kernel {_format_duration(kernel_s)}")
    if plots_s is not None:
        timing_parts.append(f"plots {_format_duration(plots_s)}")
    if provenance_s is not None:
        timing_parts.append(f"provenance {_format_duration(provenance_s)}")
    if total_s is not None:
        timing_parts.append(f"total {_format_duration(total_s)}")
    if timing_parts:
        lines.extend(["", "Timing:"])
        for part in timing_parts:
            lines.append(f"  {part}")
    return "\n".join(lines)
