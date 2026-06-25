"""Adjoint tallies → provenance probabilities and attributed fueling flux."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.fields.grid_interp import grid_spacing
from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.io.wall_flux import WallNeutralFluxResult, extract_wall_neutral_flux
from adjoint_mc.scoring.tallies import HistoryScore, WallTallyResult
from adjoint_mc.tracker.backward_full import BackwardFullResult


@dataclass(frozen=True)
class RegionUncertainty:
    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PlasmaZoneSpec:
    name: str
    label: str
    r_min: float | None = None
    r_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None


@dataclass(frozen=True)
class PlasmaZoneProvenance:
    zone: PlasmaZoneSpec
    n_seeds: int
    n_wall_hits: int
    fueling_rate_s: float
    region_contribution: dict[str, float]
    region_probability: dict[str, float]
    region_attributed_flux_s: dict[str, float]


@dataclass(frozen=True)
class ProvenanceResult:
    """Physical provenance map derived from backward adjoint MC tallies."""

    n_histories: int
    n_wall_hits: int
    total_adjoint_weight: float
    fueling_rate_total_s: float
    segment_contribution: np.ndarray
    segment_probability: np.ndarray
    segment_attributed_flux_s: np.ndarray
    segment_ring_area_m2: np.ndarray
    segment_flux_density_s: np.ndarray
    region_contribution: dict[str, float]
    region_probability: dict[str, float]
    region_attributed_flux_s: dict[str, float]
    region_probability_uncertainty: dict[str, RegionUncertainty] = field(default_factory=dict)
    plasma_zone_provenance: tuple[PlasmaZoneProvenance, ...] = ()
    convergence_checkpoints: tuple[int, ...] = ()
    convergence_region_fractions: dict[str, np.ndarray] = field(default_factory=dict)
    wall_flux: WallNeutralFluxResult | None = None
    region_hit_fraction: dict[str, float] = field(default_factory=dict)


def segment_ring_areas(wall: WallGeometry) -> np.ndarray:
    """Poloidal arc length × 2πR for each wall segment [m²]."""
    areas = np.zeros(wall.n_segments, dtype=float)
    for seg in wall.segments:
        r_mid = 0.5 * (seg.r0 + seg.r1)
        areas[seg.segment_index] = seg.length * 2.0 * np.pi * r_mid
    return areas


def integrated_ionization_rate(grid: PretabulatedGrid) -> float:
    """Total plasma ionization rate ∫ S_ion dV [s⁻¹] from the pre-tabulated grid."""
    return integrated_ionization_rate_in_zone(grid, PlasmaZoneSpec("full", "full"))


def integrated_ionization_rate_in_mask(grid: PretabulatedGrid, birth_mask: np.ndarray) -> float:
    """Total ionization rate ∫ S_ion dV over a masked subset of the plasma grid."""
    if birth_mask.shape != grid.mask.shape:
        raise ValueError("birth_mask must match grid.mask shape")
    dr, dz = grid_spacing(grid)
    s_ion = grid.fields["S_ion"]
    total = 0.0
    for j, z in enumerate(grid.z_coords):
        for i, r in enumerate(grid.r_coords):
            if not grid.mask[j, i] or not birth_mask[j, i]:
                continue
            value = float(s_ion[j, i])
            if not np.isfinite(value) or value <= 0.0:
                continue
            dV = 2.0 * np.pi * float(r) * dr * dz
            total += value * dV
    return float(total)


def integrated_ionization_rate_in_zone(grid: PretabulatedGrid, zone: PlasmaZoneSpec) -> float:
    dr, dz = grid_spacing(grid)
    s_ion = grid.fields["S_ion"]
    total = 0.0
    for j, z in enumerate(grid.z_coords):
        if zone.z_min is not None and float(z) < zone.z_min:
            continue
        if zone.z_max is not None and float(z) > zone.z_max:
            continue
        for i, r in enumerate(grid.r_coords):
            if not grid.mask[j, i]:
                continue
            if zone.r_min is not None and float(r) < zone.r_min:
                continue
            if zone.r_max is not None and float(r) > zone.r_max:
                continue
            dV = 2.0 * np.pi * float(r) * dr * dz
            total += float(s_ion[j, i]) * dV
    return float(total)


def default_plasma_zones(grid: PretabulatedGrid) -> list[PlasmaZoneSpec]:
    r_mid = 0.5 * (grid.r_min + grid.r_max)
    z_span = grid.z_max - grid.z_min
    z_band = max(0.25 * z_span, 1e-3)
    return [
        PlasmaZoneSpec("full", "Full plasma"),
        PlasmaZoneSpec("lower_half", "Lower half (Z < 0)", z_max=0.0),
        PlasmaZoneSpec("upper_half", "Upper half (Z ≥ 0)", z_min=0.0),
        PlasmaZoneSpec("outer_r", f"Outer R (R ≥ {r_mid:.2f} m)", r_min=r_mid),
        PlasmaZoneSpec(
            "near_midplane",
            f"Near midplane (|Z| ≤ {z_band:.2f} m)",
            z_min=-z_band,
            z_max=z_band,
        ),
    ]


def _seed_in_zone(score: HistoryScore, zone: PlasmaZoneSpec) -> bool:
    if zone.r_min is not None and score.seed_r < zone.r_min:
        return False
    if zone.r_max is not None and score.seed_r > zone.r_max:
        return False
    if zone.z_min is not None and score.seed_z < zone.z_min:
        return False
    if zone.z_max is not None and score.seed_z > zone.z_max:
        return False
    return True


def _wall_scores(scores: list[HistoryScore]) -> list[HistoryScore]:
    return [
        score
        for score in scores
        if score.termination == "wall" and score.segment_index is not None and score.region_name
    ]


def _region_names(wall: WallGeometry) -> list[str]:
    return sorted({seg.region_name for seg in wall.segments})


def _contributions_from_scores(
    wall: WallGeometry,
    scores: list[HistoryScore],
) -> tuple[np.ndarray, dict[str, float]]:
    segment_c = np.zeros(wall.n_segments, dtype=float)
    region_c = {name: 0.0 for name in _region_names(wall)}
    for score in scores:
        if score.termination != "wall" or score.segment_index is None or not score.region_name:
            continue
        segment_c[score.segment_index] += score.weight
        region_c[score.region_name] += score.weight
    return segment_c, region_c


def _normalize_contributions(
    segment_c: np.ndarray,
    region_c: dict[str, float],
    fueling_rate_s: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, float], np.ndarray]:
    total_c = float(np.sum(segment_c))
    if total_c <= 0.0:
        segment_p = np.zeros_like(segment_c)
        region_p = {name: 0.0 for name in region_c}
    else:
        segment_p = segment_c / total_c
        region_p = {name: float(w / total_c) for name, w in region_c.items()}
    region_flux = {name: float(f * fueling_rate_s) for name, f in region_p.items()}
    segment_flux = segment_p * fueling_rate_s
    return segment_p, region_p, region_flux, segment_flux


def bootstrap_region_probabilities(
    wall: WallGeometry,
    scores: list[HistoryScore],
    *,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, RegionUncertainty]:
    wall_hits = _wall_scores(scores)
    names = _region_names(wall)
    if len(wall_hits) < 2 or n_bootstrap < 1:
        return {
            name: RegionUncertainty(0.0, 0.0, 0.0, 0.0)
            for name in names
        }

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(wall_hits), size=len(wall_hits))
        region_c = {name: 0.0 for name in names}
        for i in idx:
            region_c[wall_hits[i].region_name] += wall_hits[i].weight
        total = float(sum(region_c.values()))
        if total <= 0.0:
            continue
        for name in names:
            samples[name].append(region_c[name] / total)

    out: dict[str, RegionUncertainty] = {}
    for name in names:
        arr = np.asarray(samples[name], dtype=float)
        if arr.size == 0:
            out[name] = RegionUncertainty(0.0, 0.0, 0.0, 0.0)
        else:
            out[name] = RegionUncertainty(
                mean=float(np.mean(arr)),
                std=float(np.std(arr)),
                ci_low=float(np.percentile(arr, 2.5)),
                ci_high=float(np.percentile(arr, 97.5)),
            )
    return out


def compute_convergence_curves(
    wall: WallGeometry,
    scores: list[HistoryScore],
    *,
    n_checkpoints: int = 8,
) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    names = _region_names(wall)
    n_hist = len(scores)
    if n_hist < 2:
        return (), {}

    checkpoints = tuple(
        sorted(
            {
                max(2, int(round(x)))
                for x in np.linspace(max(2, n_hist // 20), n_hist, num=n_checkpoints)
            }
        )
    )
    curves: dict[str, list[float]] = {name: [] for name in names}
    for n in checkpoints:
        _, region_c = _contributions_from_scores(wall, scores[:n])
        total = float(sum(region_c.values()))
        for name in names:
            curves[name].append(region_c[name] / total if total > 0.0 else 0.0)
    return checkpoints, {name: np.asarray(values, dtype=float) for name, values in curves.items()}


def compute_plasma_zone_provenance(
    wall: WallGeometry,
    scores: list[HistoryScore],
    grid: PretabulatedGrid,
    zones: list[PlasmaZoneSpec] | None = None,
) -> tuple[PlasmaZoneProvenance, ...]:
    zones = zones or default_plasma_zones(grid)
    results: list[PlasmaZoneProvenance] = []
    for zone in zones:
        zone_scores = [score for score in scores if _seed_in_zone(score, zone)]
        fueling = integrated_ionization_rate_in_zone(grid, zone)
        _, region_c = _contributions_from_scores(wall, zone_scores)
        _, region_p, region_flux, _ = _normalize_contributions(
            np.zeros(wall.n_segments, dtype=float),
            region_c,
            fueling,
        )
        n_wall = sum(
            1
            for score in zone_scores
            if score.termination == "wall" and score.region_name
        )
        results.append(
            PlasmaZoneProvenance(
                zone=zone,
                n_seeds=len(zone_scores),
                n_wall_hits=n_wall,
                fueling_rate_s=fueling,
                region_contribution=region_c,
                region_probability=region_p,
                region_attributed_flux_s=region_flux,
            )
        )
    return tuple(results)


def compute_provenance(
    wall: WallGeometry,
    mc_result: BackwardFullResult,
    grid: PretabulatedGrid,
    *,
    solution: Any | None = None,
    include_wall_flux: bool = True,
    bootstrap_samples: int = 200,
    fueling_rate_s: float | None = None,
) -> ProvenanceResult:
    """
    Convert adjoint wall tallies into provenance probabilities and attributed flux.

    Normalisation (conditional on wall hits):
      f_k = C_k / Σ_j C_j,  C_k = Σ_i W_i 1{hit ∈ k}

    Attributed fueling flux (steady-state closure):
      Γ_prov,k = f_k × ∫ S_ion dV
    """
    tallies = mc_result.tallies
    scores = mc_result.scores
    segment_c, region_c = _contributions_from_scores(wall, scores)

    q_fuel = (
        float(fueling_rate_s)
        if fueling_rate_s is not None
        else integrated_ionization_rate(grid)
    )
    segment_p, region_p, region_flux, segment_flux = _normalize_contributions(
        segment_c, region_c, q_fuel
    )

    ring_areas = segment_ring_areas(wall)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_density = np.divide(
            segment_flux,
            ring_areas,
            out=np.zeros_like(segment_flux),
            where=ring_areas > 0.0,
        )

    hit_counts = tallies.region_hit_counts(scores)
    n_wall = max(tallies.n_wall, 1)
    region_hit_frac = {name: float(hit_counts.get(name, 0) / n_wall) for name in region_c}

    wall_flux = None
    if include_wall_flux and solution is not None:
        try:
            wall_flux = extract_wall_neutral_flux(solution, wall)
        except Exception:
            wall_flux = None

    uncertainty = bootstrap_region_probabilities(
        wall,
        scores,
        n_bootstrap=bootstrap_samples,
        seed=mc_result.seed,
    )
    checkpoints, convergence = compute_convergence_curves(wall, scores)
    zone_provenance = compute_plasma_zone_provenance(wall, scores, grid)

    return ProvenanceResult(
        n_histories=mc_result.n_histories,
        n_wall_hits=tallies.n_wall,
        total_adjoint_weight=float(tallies.total_weight),
        fueling_rate_total_s=q_fuel,
        segment_contribution=segment_c,
        segment_probability=segment_p,
        segment_attributed_flux_s=segment_flux,
        segment_ring_area_m2=ring_areas,
        segment_flux_density_s=flux_density,
        region_contribution=region_c,
        region_probability=region_p,
        region_attributed_flux_s=region_flux,
        region_probability_uncertainty=uncertainty,
        plasma_zone_provenance=zone_provenance,
        convergence_checkpoints=checkpoints,
        convergence_region_fractions=convergence,
        wall_flux=wall_flux,
        region_hit_fraction=region_hit_frac,
    )


def export_provenance_csv(path: Path | str, result: ProvenanceResult, wall: WallGeometry) -> None:
    """Write region and segment provenance tables to CSV files (``*_regions.csv``, ``*_segments.csv``)."""
    path = Path(path)
    stem = path.with_suffix("")
    region_path = Path(f"{stem}_regions.csv")
    seg_path = Path(f"{stem}_segments.csv")
    zone_path = Path(f"{stem}_zones.csv")

    with region_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "region",
                "adjoint_contribution_C",
                "provenance_probability_f",
                "f_ci_low",
                "f_ci_high",
                "attributed_flux_s-1",
                "mc_hit_fraction",
                "soledge_neutral_flux_fraction",
                "soledge_parallel_flux_fraction",
            ]
        )
        for name in sorted(result.region_contribution.keys()):
            unc = result.region_probability_uncertainty.get(name)
            neutral_f = ""
            parallel_f = ""
            if result.wall_flux is not None:
                neutral_f = result.wall_flux.region_fraction.get(name, 0.0)
                parallel_f = result.wall_flux.region_parallel_fraction.get(name, 0.0)
            writer.writerow(
                [
                    name,
                    result.region_contribution[name],
                    result.region_probability[name],
                    unc.ci_low if unc else "",
                    unc.ci_high if unc else "",
                    result.region_attributed_flux_s[name],
                    result.region_hit_fraction.get(name, 0.0),
                    neutral_f,
                    parallel_f,
                ]
            )

    with seg_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "segment_index",
                "region",
                "r_mid",
                "z_mid",
                "ring_area_m2",
                "adjoint_contribution_C",
                "provenance_probability_f",
                "attributed_flux_s-1",
                "flux_density_s-1_m-2",
            ]
        )
        for seg in wall.segments:
            idx = seg.segment_index
            writer.writerow(
                [
                    idx,
                    seg.region_name,
                    0.5 * (seg.r0 + seg.r1),
                    0.5 * (seg.z0 + seg.z1),
                    result.segment_ring_area_m2[idx],
                    result.segment_contribution[idx],
                    result.segment_probability[idx],
                    result.segment_attributed_flux_s[idx],
                    result.segment_flux_density_s[idx],
                ]
            )

    with zone_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "plasma_zone",
                "zone_label",
                "n_seeds",
                "n_wall_hits",
                "fueling_rate_s-1",
                "wall_region",
                "provenance_probability_f",
                "attributed_flux_s-1",
            ]
        )
        for zone_result in result.plasma_zone_provenance:
            for name in sorted(zone_result.region_probability.keys()):
                writer.writerow(
                    [
                        zone_result.zone.name,
                        zone_result.zone.label,
                        zone_result.n_seeds,
                        zone_result.n_wall_hits,
                        zone_result.fueling_rate_s,
                        name,
                        zone_result.region_probability[name],
                        zone_result.region_attributed_flux_s[name],
                    ]
                )


def format_provenance_details_text(result: ProvenanceResult) -> str:
    """Provenance-only summary (MC diagnostics are expected separately)."""
    lines = [
        "",
        "--- Provenance (adjoint → f_k → Γ_prov,k) ---",
        "",
        "C_k = Σ W on wall region k;  f_k = C_k / Σ C_j;  Γ_prov,k = f_k × ∫ S_ion dV",
        "",
        f"Total adjoint ΣW   : {result.total_adjoint_weight:.6g}",
        f"∫ S_ion dV         : {result.fueling_rate_total_s:.6g} s⁻¹",
        "",
        "Provenance f_k [95% bootstrap CI] and Γ_prov,k by region:",
    ]
    for name in sorted(result.region_probability.keys()):
        f_k = result.region_probability[name]
        gamma = result.region_attributed_flux_s[name]
        hits = result.region_hit_fraction.get(name, 0.0)
        unc = result.region_probability_uncertainty.get(name)
        ci_text = ""
        if unc and unc.ci_high > unc.ci_low:
            ci_text = f"  [{100.0 * unc.ci_low:.1f}–{100.0 * unc.ci_high:.1f} %]"
        lines.append(
            f"  {name:12s}  f={100.0 * f_k:6.2f}%{ci_text}  "
            f"Γ_prov={gamma:.4g} s⁻¹  (MC hits {100.0 * hits:5.1f} %)"
        )

    if result.plasma_zone_provenance:
        lines.extend(["", "Provenance by plasma birth zone (which wall feeds each region):"])
        for zone_result in result.plasma_zone_provenance:
            if zone_result.zone.name == "full":
                continue
            top = max(zone_result.region_probability.items(), key=lambda item: item[1])
            lines.append(
                f"  {zone_result.zone.label:28s}  "
                f"∫S_ion={zone_result.fueling_rate_s:.3g} s⁻¹  "
                f"top wall source: {top[0]} ({100.0 * top[1]:.1f} %)"
            )

    if result.wall_flux is not None:
        lines.extend(["", "SOLEDGE boundary flux shape (normalised by region):"])
        for name in sorted(result.wall_flux.region_fraction.keys()):
            neutral = result.wall_flux.region_fraction.get(name, 0.0)
            parallel = result.wall_flux.region_parallel_fraction.get(name, 0.0)
            lines.append(
                f"  {name:12s}  neutral ⊥ {100.0 * neutral:6.2f} %  "
                f"parallel Γ {100.0 * parallel:6.2f} %"
            )
    return "\n".join(lines)


def format_provenance_summary_text(result: ProvenanceResult) -> str:
    """Standalone provenance summary (notebook / export)."""
    header = [
        " Neutral provenance (adjoint → probabilities → flux)",
        "",
        f"Histories          : {result.n_histories}",
        f"Wall hits          : {result.n_wall_hits}",
    ]
    return "\n".join(header) + format_provenance_details_text(result)


def format_provenance_run_summary_text(
    mc_result: BackwardFullResult,
    provenance: ProvenanceResult,
    *,
    grid: PretabulatedGrid,
    grid_build_s: float,
    mc_s: float,
    provenance_s: float,
    total_s: float,
    kernel_s: float,
    pack_s: float,
    n_threads: int,
    plots_s: float = 0.0,
    header: str | None = None,
) -> str:
    """Combined MC diagnostics + provenance block for the GUI."""
    from adjoint_mc.viz.plots import format_backward_full_summary_text

    mc_summary = format_backward_full_summary_text(
        mc_result,
        grid=grid,
        grid_build_s=grid_build_s,
        mc_s=mc_s,
        mc_label="Cython MC",
        kernel_s=kernel_s,
        pack_s=pack_s,
        n_threads=n_threads,
        plots_s=plots_s,
        provenance_s=provenance_s,
        total_s=total_s,
        header=header or " Full backward MC + provenance (Cython + OpenMP)",
    )
    return mc_summary + format_provenance_details_text(provenance)
