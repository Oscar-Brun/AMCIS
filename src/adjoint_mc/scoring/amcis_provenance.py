"""AMCIS wall→target provenance from survival-weighted backward MC."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.io.wall_flux import WallNeutralFluxResult
from adjoint_mc.scoring.tallies import HistoryScore, count_terminations


class AmcisMcLike(Protocol):
    target_r: float
    target_z: float
    n_histories: int
    seed: int
    scores: list[HistoryScore]

    @property
    def total_cx_events(self) -> int: ...

    @property
    def tallies(self): ...


@dataclass(frozen=True)
class AmcisProvenanceResult:
    """Wall visibility map for a single plasma target (R, Z)."""

    target_r: float
    target_z: float
    n_histories: int
    n_wall_hits: int
    total_visibility_weight: float
    segment_contribution: np.ndarray
    segment_probability: np.ndarray
    region_contribution: dict[str, float]
    region_probability: dict[str, float]
    region_hit_fraction: dict[str, float] = field(default_factory=dict)
    termination_counts: dict[str, int] = field(default_factory=dict)
    has_emission_weighting: bool = False
    segment_wall_flux: np.ndarray | None = None
    segment_emission_weight: np.ndarray | None = None
    segment_emission_probability: np.ndarray | None = None
    segment_attributed_flux: np.ndarray | None = None
    region_emission_weight: dict[str, float] | None = None
    region_emission_probability: dict[str, float] | None = None
    region_attributed_flux: dict[str, float] | None = None
    total_emission_weight: float = 0.0
    total_attributed_flux: float = 0.0


def _contributions_from_scores(
    wall: WallGeometry,
    scores: list[HistoryScore],
) -> tuple[np.ndarray, dict[str, float]]:
    segment_c = np.zeros(wall.n_segments, dtype=float)
    region_names = sorted({seg.region_name for seg in wall.segments})
    region_c = {name: 0.0 for name in region_names}
    for score in scores:
        if score.termination != "wall" or score.segment_index is None or not score.region_name:
            continue
        segment_c[score.segment_index] += score.weight
        region_c[score.region_name] += score.weight
    return segment_c, region_c


def _normalize(segment_c: np.ndarray, region_c: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
    total = float(np.sum(segment_c))
    if total <= 0.0:
        return np.zeros_like(segment_c), {name: 0.0 for name in region_c}
    segment_p = segment_c / total
    region_p = {name: float(w / total) for name, w in region_c.items()}
    return segment_p, region_p


def emission_weighted_segment_probability(
    wall: WallGeometry,
    segment_contribution: np.ndarray,
    wall_flux: WallNeutralFluxResult,
) -> np.ndarray:
    """
    Emission-weighted wall provenance fractions:

    f_k^flux = C_k Γ_k^wall / Σ_j C_j Γ_j^wall
    """
    segment_c = np.asarray(segment_contribution, dtype=float)
    total_c = float(np.sum(segment_c))
    if total_c <= 0.0:
        return np.zeros(wall.n_segments, dtype=float)
    segment_p = segment_c / total_c
    region_c = _aggregate_segments_by_region(wall, segment_c)
    _, _, segment_p_flux, _, _, _, _, _, _ = _emission_weighting(
        wall, segment_c, segment_p, region_c, wall_flux
    )
    return segment_p_flux


def _aggregate_segments_by_region(
    wall: WallGeometry,
    segment_values: np.ndarray,
) -> dict[str, float]:
    region: dict[str, float] = {}
    for seg in wall.segments:
        region[seg.region_name] = region.get(seg.region_name, 0.0) + float(
            segment_values[seg.segment_index]
        )
    return region


def _emission_weighting(
    wall: WallGeometry,
    segment_c: np.ndarray,
    segment_p: np.ndarray,
    region_c: dict[str, float],
    wall_flux: WallNeutralFluxResult,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float,
    float,
]:
    """
    Combine backward visibility with SOLEDGE wall neutral flux.

    D_k = C_k Γ_k^wall ;  f_k^flux = D_k / Σ_j D_j
    Φ_k = (C_k / Σ_j C_j) Γ_k^wall  (attributed emission rate to target Ω)
    """
    gamma = np.asarray(wall_flux.segment_flux, dtype=float)
    if gamma.shape[0] != wall.n_segments:
        raise ValueError(
            f"wall_flux has {gamma.shape[0]} segments, expected {wall.n_segments}"
        )

    segment_d = segment_c * gamma
    segment_attributed = segment_p * gamma
    region_d = _aggregate_segments_by_region(wall, segment_d)
    region_attributed = _aggregate_segments_by_region(wall, segment_attributed)
    segment_p_flux, region_p_flux = _normalize(segment_d, region_d)
    return (
        gamma,
        segment_d,
        segment_p_flux,
        segment_attributed,
        region_d,
        region_p_flux,
        region_attributed,
        float(np.sum(segment_d)),
        float(np.sum(segment_attributed)),
    )


def compute_amcis_provenance(
    wall: WallGeometry,
    mc_result: AmcisMcLike,
    *,
    wall_flux: WallNeutralFluxResult | None = None,
) -> AmcisProvenanceResult:
    """
    Normalise wall survival weights into provenance fractions f_k(Ω).

    C_k = Σ W_i on segment k (wall hits only);  f_k = C_k / Σ_j C_j.

    When ``wall_flux`` is supplied, also compute emission-weighted maps using
    SOLEDGE ring-integrated |Γ_n,⊥| on each segment:

    D_k = C_k Γ_k^wall ;  f_k^flux(Ω) = D_k / Σ_j D_j
    Φ_k→Ω = f_k(Ω) Γ_k^wall = (C_k / Σ_j C_j) Γ_k^wall
    """
    scores = mc_result.scores
    segment_c, region_c = _contributions_from_scores(wall, scores)
    segment_p, region_p = _normalize(segment_c, region_c)

    n_wall = sum(1 for s in scores if s.termination == "wall")
    region_hits = {name: 0 for name in region_c}
    for score in scores:
        if score.termination == "wall" and score.region_name:
            region_hits[score.region_name] = region_hits.get(score.region_name, 0) + 1
    hit_frac = {
        name: float(region_hits.get(name, 0) / max(n_wall, 1)) for name in region_c
    }

    emission_kwargs: dict[str, object] = {}
    if wall_flux is not None:
        (
            gamma,
            segment_d,
            segment_p_flux,
            segment_attributed,
            region_d,
            region_p_flux,
            region_attributed,
            total_d,
            total_attributed,
        ) = _emission_weighting(wall, segment_c, segment_p, region_c, wall_flux)
        emission_kwargs = {
            "has_emission_weighting": True,
            "segment_wall_flux": gamma,
            "segment_emission_weight": segment_d,
            "segment_emission_probability": segment_p_flux,
            "segment_attributed_flux": segment_attributed,
            "region_emission_weight": region_d,
            "region_emission_probability": region_p_flux,
            "region_attributed_flux": region_attributed,
            "total_emission_weight": total_d,
            "total_attributed_flux": total_attributed,
        }

    return AmcisProvenanceResult(
        target_r=mc_result.target_r,
        target_z=mc_result.target_z,
        n_histories=mc_result.n_histories,
        n_wall_hits=n_wall,
        total_visibility_weight=float(np.sum(segment_c)),
        segment_contribution=segment_c,
        segment_probability=segment_p,
        region_contribution=region_c,
        region_probability=region_p,
        region_hit_fraction=hit_frac,
        termination_counts=count_terminations(scores),
        **emission_kwargs,
    )


def format_amcis_summary_text(
    mc_result: AmcisMcLike,
    provenance: AmcisProvenanceResult,
    *,
    mc_s: float,
    total_s: float,
    mc_engine: str | None = None,
) -> str:
    lines = [
        "AMCIS — wall provenance to single plasma target",
        f"Target (R, Z): ({provenance.target_r:.4f}, {provenance.target_z:.4f}) m",
        f"Histories: {provenance.n_histories}  seed: {mc_result.seed}",
    ]
    if mc_engine:
        lines.append(f"MC engine: {mc_engine}")
    lines.extend(
        [
        f"Wall hits: {provenance.n_wall_hits} ({100.0 * provenance.n_wall_hits / max(provenance.n_histories, 1):.1f} %)",
        f"Total visibility weight Σ W: {provenance.total_visibility_weight:.4g}",
        f"Mean W (wall hits): {mc_result.tallies.mean_weight:.4g}",
        f"CX events (total): {mc_result.total_cx_events}",
        "",
        "Termination counts:",
        ]
    )
    for term, count in sorted(provenance.termination_counts.items()):
        lines.append(f"  {term}: {count}")
    lines.extend(
        [
            "",
            "Weight model: W <- exp(-integral Sigma_ion ds); CX via rejection, W unchanged",
            "",
            "Wall provenance f_k(Ω) [%] — conditional on wall hits:",
        ]
    )
    for name in sorted(provenance.region_probability):
        frac = 100.0 * provenance.region_probability[name]
        hits = 100.0 * provenance.region_hit_fraction.get(name, 0.0)
        lines.append(f"  {name:20s}  f_k={frac:6.2f} %   hits={hits:5.1f} %")
    if provenance.has_emission_weighting and provenance.region_emission_probability:
        lines.extend(
            [
                "",
                "Emission-weighted provenance f_k^flux(Ω) [%] — C_k × Γ_k^wall, normalised:",
                f"  Σ D_k = {provenance.total_emission_weight:.4g}"
                f"   Σ Φ_k→Ω = {provenance.total_attributed_flux:.4g}",
            ]
        )
        for name in sorted(provenance.region_emission_probability):
            frac = 100.0 * provenance.region_emission_probability[name]
            attributed = provenance.region_attributed_flux or {}
            lines.append(
                f"  {name:20s}  f_k^flux={frac:6.2f} %"
                f"   Φ→Ω={attributed.get(name, 0.0):.4g}"
            )
    lines.extend(
        [
            "",
            f"Timing: MC {mc_s:.2f} s  total {total_s:.2f} s",
        ]
    )
    return "\n".join(lines)


def export_amcis_csv(path: Path | str, provenance: AmcisProvenanceResult, wall: WallGeometry) -> None:
    """Write ``*_regions.csv`` and ``*_segments.csv`` for one AMCIS run."""
    path = Path(path)
    stem = path.with_suffix("")
    region_path = Path(f"{stem}_regions.csv")
    seg_path = Path(f"{stem}_segments.csv")

    region_header = ["region", "C_k", "f_k", "hit_fraction", "target_r_m", "target_z_m"]
    if provenance.has_emission_weighting:
        region_header[3:3] = ["Gamma_k_wall", "D_k", "f_k_flux", "Phi_to_target"]

    with region_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(region_header)
        for name in sorted(provenance.region_probability):
            row = [
                name,
                provenance.region_contribution[name],
                provenance.region_probability[name],
                provenance.region_hit_fraction.get(name, 0.0),
                provenance.target_r,
                provenance.target_z,
            ]
            if provenance.has_emission_weighting and provenance.segment_wall_flux is not None:
                region_gamma = _aggregate_segments_by_region(wall, provenance.segment_wall_flux)
                row[3:3] = [
                    region_gamma.get(name, 0.0),
                    (provenance.region_emission_weight or {}).get(name, 0.0),
                    (provenance.region_emission_probability or {}).get(name, 0.0),
                    (provenance.region_attributed_flux or {}).get(name, 0.0),
                ]
            writer.writerow(row)

    seg_header = [
        "segment_index",
        "region",
        "r0",
        "z0",
        "r1",
        "z1",
        "C_k",
        "f_k",
        "target_r_m",
        "target_z_m",
    ]
    if provenance.has_emission_weighting:
        seg_header[8:8] = ["Gamma_k_wall", "D_k", "f_k_flux", "Phi_to_target"]

    with seg_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(seg_header)
        for seg in wall.segments:
            idx = seg.segment_index
            row = [
                idx,
                seg.region_name,
                seg.r0,
                seg.z0,
                seg.r1,
                seg.z1,
                provenance.segment_contribution[idx],
                provenance.segment_probability[idx],
                provenance.target_r,
                provenance.target_z,
            ]
            if provenance.has_emission_weighting and provenance.segment_wall_flux is not None:
                row[8:8] = [
                    provenance.segment_wall_flux[idx],
                    (provenance.segment_emission_weight or np.zeros(1))[idx],
                    (provenance.segment_emission_probability or np.zeros(1))[idx],
                    (provenance.segment_attributed_flux or np.zeros(1))[idx],
                ]
            writer.writerow(row)
