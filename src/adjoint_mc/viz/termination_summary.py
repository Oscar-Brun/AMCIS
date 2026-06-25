"""Format non-wall termination breakdowns for GUI summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING

from adjoint_mc.scoring.tallies import (
    NON_WALL_TERMINATIONS,
    HistoryScore,
    count_terminations,
    is_lost_at_birth,
)

if TYPE_CHECKING:
    from adjoint_mc.fields.pretabulate import PretabulatedGrid

_TERMINATION_LABELS: dict[str, str] = {
    "lost": "lost (no wall hit)",
    "max_path": "max backward path",
    "max_steps": "max step count",
}


@dataclass(frozen=True)
class LostHistoryDiagnostic:
    """Split ``lost`` outcomes into birth vs in-flight, with optional grid check."""

    at_birth: int
    in_flight: int
    at_birth_outside_plasma: int | None = None
    at_birth_inside_plasma: int | None = None
    in_flight_mean_path_m: float | None = None
    in_flight_mean_steps: float | None = None
    in_flight_mean_cx: float | None = None

    @property
    def total_lost(self) -> int:
        return self.at_birth + self.in_flight


def analyze_lost_histories(
    scores: list[HistoryScore],
    grid: PretabulatedGrid | None = None,
) -> LostHistoryDiagnostic:
    """Classify lost histories (ghost birth vs flight) and verify n_e at seeds."""
    birth_scores: list[HistoryScore] = []
    flight_scores: list[HistoryScore] = []
    for score in scores:
        if score.termination != "lost":
            continue
        if is_lost_at_birth(score):
            birth_scores.append(score)
        else:
            flight_scores.append(score)

    outside: int | None = None
    inside: int | None = None
    if grid is not None:
        from adjoint_mc.fields.grid_interp import in_plasma

        outside = 0
        inside = 0
        for score in birth_scores:
            if in_plasma(grid, score.seed_r, score.seed_z):
                inside += 1
            else:
                outside += 1

    return LostHistoryDiagnostic(
        at_birth=len(birth_scores),
        in_flight=len(flight_scores),
        at_birth_outside_plasma=outside,
        at_birth_inside_plasma=inside,
        in_flight_mean_path_m=float(mean(s.path_m for s in flight_scores)) if flight_scores else None,
        in_flight_mean_steps=float(mean(s.n_steps for s in flight_scores)) if flight_scores else None,
        in_flight_mean_cx=float(mean(s.n_cx_events for s in flight_scores)) if flight_scores else None,
    )


def format_lost_diagnostic_lines(
    scores: list[HistoryScore],
    grid: PretabulatedGrid | None = None,
) -> list[str]:
    """Detail lines for ``lost`` histories (ghost births vs in-flight)."""
    diag = analyze_lost_histories(scores, grid)
    if diag.total_lost == 0:
        return []

    lines = ["  lost detail:"]
    birth_note = ""
    if diag.at_birth_outside_plasma is not None:
        birth_note = f"  ({diag.at_birth_outside_plasma} with n_e≤0 at seed"
        if diag.at_birth_inside_plasma:
            birth_note += f", {diag.at_birth_inside_plasma} anomalous n_e>0"
        birth_note += ")"
    lines.append(f"    at birth (path≈0)     {diag.at_birth:5d}{birth_note}")

    flight_suffix = ""
    if diag.in_flight and diag.in_flight_mean_path_m is not None:
        flight_suffix = (
            f"  mean path {diag.in_flight_mean_path_m:.2f} m"
            f", steps {diag.in_flight_mean_steps:.0f}"
            f", CX {diag.in_flight_mean_cx:.1f}"
        )
    lines.append(f"    in flight (path>0)   {diag.in_flight:5d}{flight_suffix}")
    return lines


def format_termination_breakdown_lines(
    scores: list[HistoryScore],
    *,
    grid: PretabulatedGrid | None = None,
) -> list[str]:
    """Human-readable lines for non-wall termination counts."""
    counts = count_terminations(scores)
    non_wall = {key: counts[key] for key in counts if key != "wall"}
    if not non_wall:
        return []

    lines = ["Non-wall termination breakdown:"]
    for key in NON_WALL_TERMINATIONS:
        value = non_wall.get(key, 0)
        if value:
            label = _TERMINATION_LABELS.get(key, key)
            lines.append(f"  {label:22s}  {value:5d}")
    for key in sorted(non_wall):
        if key in NON_WALL_TERMINATIONS:
            continue
        lines.append(f"  {key:22s}  {non_wall[key]:5d}")

    if non_wall.get("lost", 0):
        lines.extend(format_lost_diagnostic_lines(scores, grid))
    return lines
