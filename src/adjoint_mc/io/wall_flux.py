"""Neutral and parallel wall flux from SOLEDGE-HDG boundary diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adjoint_mc.geometry.wall import WallGeometry


@dataclass(frozen=True)
class WallNeutralFluxResult:
    """Axisymmetric ring-integrated flux proxies on wall segments."""

    segment_flux: np.ndarray
    segment_parallel_flux: np.ndarray
    region_flux: dict[str, float] = field(default_factory=dict)
    region_parallel_flux: dict[str, float] = field(default_factory=dict)
    region_fraction: dict[str, float] = field(default_factory=dict)
    region_parallel_fraction: dict[str, float] = field(default_factory=dict)
    total_flux: float = 0.0
    total_parallel_flux: float = 0.0
    source: str = "hdg_boundary_flux"


def _segment_midpoints(wall: WallGeometry) -> np.ndarray:
    return np.array(
        [[0.5 * (s.r0 + s.r1), 0.5 * (s.z0 + s.z1)] for s in wall.segments],
        dtype=float,
    )


def _segments_by_region(wall: WallGeometry) -> dict[int, list[int]]:
    by_region: dict[int, list[int]] = {}
    for seg in wall.segments:
        by_region.setdefault(seg.region_id, []).append(seg.segment_index)
    return by_region


def _nearest_segment_index(
    wall: WallGeometry,
    mids: np.ndarray,
    r_g: float,
    z_g: float,
    *,
    candidates: list[int] | None = None,
) -> int:
    if candidates is None:
        indices = np.arange(wall.n_segments, dtype=int)
    else:
        indices = np.asarray(candidates, dtype=int)
    dist2 = (mids[indices, 0] - r_g) ** 2 + (mids[indices, 1] - z_g) ** 2
    return int(indices[int(np.argmin(dist2))])


def _map_boundary_variable_to_segments(
    wall: WallGeometry,
    r_pts: np.ndarray,
    z_pts: np.ndarray,
    values: np.ndarray,
    ds_pts: np.ndarray,
    *,
    boundary_flags_pts: np.ndarray | None = None,
    abs_value: bool = True,
) -> np.ndarray:
    """
    Accumulate ring-integrated boundary Gauss values onto wall segments.

    When ``boundary_flags_pts`` is supplied, each point is mapped to the nearest
    segment in the same mesh boundary region first (avoids puff flux leaking to
    the wrong wall chain on WEST).
    """
    mids = _segment_midpoints(wall)
    by_region = _segments_by_region(wall)
    segment_flux = np.zeros(wall.n_segments, dtype=float)

    if boundary_flags_pts is None:
        flag_iter: list[int | None] = [None] * len(r_pts)
    else:
        flag_iter = [int(v) for v in np.asarray(boundary_flags_pts, dtype=int).reshape(-1)]

    for r_g, z_g, raw_value, ds, region_id in zip(r_pts, z_pts, values, ds_pts, flag_iter):
        if not np.isfinite(r_g) or not np.isfinite(z_g) or r_g <= 0.0:
            continue
        value = abs(float(raw_value)) if abs_value else float(raw_value)
        if value <= 0.0:
            continue
        candidates = by_region.get(region_id) if region_id is not None else None
        seg_idx = _nearest_segment_index(wall, mids, float(r_g), float(z_g), candidates=candidates)
        segment_flux[seg_idx] += value * float(ds) * 2.0 * np.pi * float(r_g)
    return segment_flux


def _region_aggregate(wall: WallGeometry, segment_values: np.ndarray) -> dict[str, float]:
    region: dict[str, float] = {}
    for seg, value in zip(wall.segments, segment_values):
        region[seg.region_name] = region.get(seg.region_name, 0.0) + float(value)
    return region


def _region_fractions(region_flux: dict[str, float]) -> dict[str, float]:
    total = float(sum(region_flux.values()))
    if total <= 0.0:
        return dict(region_flux)
    return {name: float(v / total) for name, v in region_flux.items()}


def effective_boundary_emission_flux(
    neutral_flux: np.ndarray,
    gamma_puff_wall: np.ndarray,
) -> np.ndarray:
    """
    Per-Gauss emission proxy for provenance weighting.

    ``neutral_flux`` from HDG excludes gas puff injection; ``gamma_puff_wall``
    is added on BC code 56 faces (see HDG_postprocess ``_calculate_gamma_puff_wall``).
    """
    neutral = np.abs(np.asarray(neutral_flux, dtype=float).reshape(-1))
    puff = np.maximum(np.asarray(gamma_puff_wall, dtype=float).reshape(-1), 0.0)
    return neutral + puff


def extract_wall_neutral_flux(
    solution: Any,
    wall: WallGeometry,
) -> WallNeutralFluxResult:
    """
    Map HDG boundary fluxes onto wall segments.

    - ``segment_flux`` : |neutral_flux| + gamma_puff_wall (ring-integrated)
    - ``segment_parallel_flux`` : parallel recycling flux proxy
    """
    from hdg_postprocess.core.solution.boundary import calculate_boundary_summary

    summary = calculate_boundary_summary(
        solution,
        variables=[
            "neutral_flux",
            "gamma_puff_wall",
            "gamma_parallel_wall",
            "boundary_flag",
            "r",
            "z",
            "ds",
        ],
    )
    r_pts = np.asarray(summary["r"], dtype=float).reshape(-1)
    z_pts = np.asarray(summary["z"], dtype=float).reshape(-1)
    ds_pts = np.asarray(summary["ds"], dtype=float).reshape(-1)
    boundary_flags = np.asarray(summary["boundary_flag"], dtype=int).reshape(-1)
    neutral_pts = np.asarray(summary["neutral_flux"], dtype=float).reshape(-1)
    puff_pts = np.asarray(summary["gamma_puff_wall"], dtype=float).reshape(-1)
    parallel_pts = np.asarray(summary["gamma_parallel_wall"], dtype=float).reshape(-1)
    emission_pts = effective_boundary_emission_flux(neutral_pts, puff_pts)

    segment_neutral = _map_boundary_variable_to_segments(
        wall,
        r_pts,
        z_pts,
        emission_pts,
        ds_pts,
        boundary_flags_pts=boundary_flags,
        abs_value=False,
    )
    segment_parallel = _map_boundary_variable_to_segments(
        wall,
        r_pts,
        z_pts,
        parallel_pts,
        ds_pts,
        boundary_flags_pts=boundary_flags,
        abs_value=True,
    )

    region_neutral = _region_aggregate(wall, segment_neutral)
    region_parallel = _region_aggregate(wall, segment_parallel)
    return WallNeutralFluxResult(
        segment_flux=segment_neutral,
        segment_parallel_flux=segment_parallel,
        region_flux=region_neutral,
        region_parallel_flux=region_parallel,
        region_fraction=_region_fractions(region_neutral),
        region_parallel_fraction=_region_fractions(region_parallel),
        total_flux=float(np.sum(segment_neutral)),
        total_parallel_flux=float(np.sum(segment_parallel)),
        source="hdg_neutral_flux_plus_puff",
    )
