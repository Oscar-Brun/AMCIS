"""Wall scoring tallies for backward adjoint MC."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adjoint_mc.geometry.wall import WallGeometry


@dataclass
class WallTallyResult:
    """Aggregated wall scores from a backward MC batch."""

    n_histories: int
    n_wall: int
    n_lost: int
    total_weight: float
    mean_weight: float
    region_weights: dict[str, float] = field(default_factory=dict)
    segment_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))

    @property
    def wall_fraction(self) -> float:
        if self.n_histories == 0:
            return 0.0
        return float(self.n_wall / self.n_histories)

    def region_fractions(self) -> dict[str, float]:
        if self.total_weight <= 0.0:
            return {name: 0.0 for name in self.region_weights}
        return {name: float(w / self.total_weight) for name, w in self.region_weights.items()}

    def region_hit_counts(self, scores: list[HistoryScore]) -> dict[str, int]:
        counts = {name: 0 for name in self.region_weights}
        for score in scores:
            if score.termination == "wall" and score.region_name:
                counts[score.region_name] = counts.get(score.region_name, 0) + 1
        return counts


NON_WALL_TERMINATIONS: tuple[str, ...] = ("lost", "max_path", "max_steps")

_LOST_AT_BIRTH_PATH_EPS_M = 1e-9


def is_lost_at_birth(score: HistoryScore, *, path_eps: float = _LOST_AT_BIRTH_PATH_EPS_M) -> bool:
    """True when a lost history never left the birth point (immediate termination)."""
    return (
        score.termination == "lost"
        and score.n_steps == 0
        and score.path_m <= path_eps
    )


def count_terminations(scores: list[HistoryScore]) -> dict[str, int]:
    """Count histories by terminal status (wall, lost, max_path, max_steps, …)."""
    counts: dict[str, int] = {}
    for score in scores:
        term = score.termination or "unknown"
        counts[term] = counts.get(term, 0) + 1
    return counts


@dataclass(frozen=True)
class HistoryScore:
    """Outcome of one backward history."""

    weight: float
    termination: str
    region_name: str | None = None
    segment_index: int | None = None
    path_m: float = 0.0
    n_steps: int = 0
    seed_r: float = 0.0
    seed_z: float = 0.0
    hit_r: float | None = None
    hit_z: float | None = None
    n_cx_events: int = 0


def empty_wall_tallies(wall: WallGeometry, n_histories: int) -> WallTallyResult:
    return WallTallyResult(
        n_histories=n_histories,
        n_wall=0,
        n_lost=0,
        total_weight=0.0,
        mean_weight=0.0,
        region_weights={name: 0.0 for name in sorted({s.region_name for s in wall.segments})},
        segment_weights=np.zeros(wall.n_segments, dtype=float),
    )


def accumulate_wall_scores(
    wall: WallGeometry,
    n_histories: int,
    scores: list[HistoryScore],
) -> WallTallyResult:
    """Sum terminal weights on wall segments and regions."""
    tally = empty_wall_tallies(wall, n_histories)
    n_wall = 0
    n_lost = 0
    total_weight = 0.0

    for score in scores:
        if score.termination == "wall" and score.segment_index is not None:
            n_wall += 1
            total_weight += score.weight
            tally.segment_weights[score.segment_index] += score.weight
            region = score.region_name or "unknown"
            tally.region_weights[region] = tally.region_weights.get(region, 0.0) + score.weight
        else:
            n_lost += 1

    tally.n_wall = n_wall
    tally.n_lost = n_lost
    tally.total_weight = total_weight
    tally.mean_weight = float(total_weight / n_wall) if n_wall else 0.0
    return tally
