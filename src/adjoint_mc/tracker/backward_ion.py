"""
Backward adjoint tracker — ionization only.

Seeds in zones of strong S_ion, backward flight with W *= exp(Sigma_ion ds),
score W on the wall at termination. No charge exchange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from adjoint_mc.config import (
    DEFAULT_MAX_PATH_M,
    DEFAULT_MAX_STEP_M,
    DEFAULT_NEUTRAL_SPEED_M_S,
    DEFAULT_TAU_MAX,
    DEFAULT_VACUUM_WALL_SEARCH_M,
)
from adjoint_mc.fields.grid_interp import in_plasma, macroscopic_ionization_rate
from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import WallGeometry, intersect_ray
from adjoint_mc.sampling.seeds import IonizationSeed, sample_ionization_seeds
from adjoint_mc.scoring.tallies import HistoryScore, WallTallyResult, accumulate_wall_scores
from adjoint_mc.tracker.backward_trace import BackwardTraceFrame, append_trace_frame


@dataclass(frozen=True)
class BackwardIonConfig:
    """Numerical parameters for the ionization-only backward tracker."""

    speed_m_s: float = DEFAULT_NEUTRAL_SPEED_M_S
    tau_max: float = DEFAULT_TAU_MAX
    max_step_m: float = DEFAULT_MAX_STEP_M
    vacuum_wall_search_m: float = DEFAULT_VACUUM_WALL_SEARCH_M
    max_path_m: float = DEFAULT_MAX_PATH_M
    max_steps: int = 20_000


@dataclass(frozen=True)
class BackwardIonResult:
    """Batch outcome of backward ionization-only MC."""

    config: BackwardIonConfig
    n_histories: int
    seed: int
    tallies: WallTallyResult
    scores: list[HistoryScore]


def _random_velocity(rng: np.random.Generator, speed_m_s: float) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-15:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        norm = 1.0
    return direction / norm * speed_m_s


def _history_score(
    seed: IonizationSeed,
    *,
    weight: float,
    termination: str,
    region_name: str | None = None,
    segment_index: int | None = None,
    path_m: float = 0.0,
    n_steps: int = 0,
    hit_r: float | None = None,
    hit_z: float | None = None,
) -> HistoryScore:
    return HistoryScore(
        weight=weight,
        termination=termination,
        region_name=region_name,
        segment_index=segment_index,
        path_m=path_m,
        n_steps=n_steps,
        seed_r=seed.r,
        seed_z=seed.z,
        hit_r=hit_r,
        hit_z=hit_z,
    )


def track_backward_ionization(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    seed: IonizationSeed,
    rng: np.random.Generator,
    *,
    config: BackwardIonConfig | None = None,
    trace: list[BackwardTraceFrame] | None = None,
) -> HistoryScore:
    """Track one backward history from an ionization seed to wall or loss."""
    cfg = config or BackwardIonConfig()
    if cfg.speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")

    position = np.asarray(seed.position, dtype=float)
    velocity = _random_velocity(rng, cfg.speed_m_s)
    direction = velocity / cfg.speed_m_s
    log_weight = 0.0
    path_m = 0.0
    last_sigma = macroscopic_ionization_rate(grid, seed.r, seed.z, cfg.speed_m_s)

    append_trace_frame(
        trace,
        r=seed.r,
        z=seed.z,
        log_weight=log_weight,
        path_m=path_m,
        event="birth",
    )

    if not in_plasma(grid, seed.r, seed.z):
        append_trace_frame(
            trace,
            r=seed.r,
            z=seed.z,
            log_weight=log_weight,
            path_m=path_m,
            event="lost",
        )
        return _history_score(seed, weight=1.0, termination="lost")

    for step_index in range(cfg.max_steps):
        r = float(math.hypot(position[0], position[1]))
        z = float(position[2])
        backward = -direction

        if not in_plasma(grid, r, z):
            hit = intersect_ray(
                position, backward, wall, t_min=1e-12, t_max=cfg.vacuum_wall_search_m
            )
            if hit is not None:
                log_weight += last_sigma * hit.t
                append_trace_frame(
                    trace,
                    r=hit.r,
                    z=hit.z,
                    log_weight=log_weight,
                    path_m=path_m + hit.t,
                    event="wall",
                    region_name=hit.region_name,
                )
                return _history_score(
                    seed,
                    weight=float(math.exp(log_weight)),
                    termination="wall",
                    region_name=hit.region_name,
                    segment_index=hit.segment_index,
                    path_m=path_m + hit.t,
                    n_steps=step_index,
                    hit_r=hit.r,
                    hit_z=hit.z,
                )
            append_trace_frame(
                trace,
                r=r,
                z=z,
                log_weight=log_weight,
                path_m=path_m,
                event="lost",
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="lost",
                path_m=path_m,
                n_steps=step_index,
            )

        sigma = macroscopic_ionization_rate(grid, r, z, cfg.speed_m_s)
        last_sigma = sigma
        if sigma > 0.0:
            ds_optical = cfg.tau_max / sigma
        else:
            ds_optical = cfg.max_step_m
        ds = min(cfg.max_step_m, ds_optical, cfg.max_path_m - path_m)
        if ds <= 0.0:
            append_trace_frame(
                trace,
                r=r,
                z=z,
                log_weight=log_weight,
                path_m=path_m,
                event="max_path",
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="max_path",
                path_m=path_m,
                n_steps=step_index,
            )

        hit = intersect_ray(position, backward, wall, t_min=1e-12, t_max=ds)
        if hit is not None:
            log_weight += sigma * hit.t
            append_trace_frame(
                trace,
                r=hit.r,
                z=hit.z,
                log_weight=log_weight,
                path_m=path_m + hit.t,
                event="wall",
                region_name=hit.region_name,
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="wall",
                region_name=hit.region_name,
                segment_index=hit.segment_index,
                path_m=path_m + hit.t,
                n_steps=step_index + 1,
                hit_r=hit.r,
                hit_z=hit.z,
            )

        log_weight += sigma * ds
        position = position + backward * ds
        path_m += ds
        r_new = float(math.hypot(position[0], position[1]))
        z_new = float(position[2])
        append_trace_frame(
            trace,
            r=r_new,
            z=z_new,
            log_weight=log_weight,
            path_m=path_m,
            event="step",
        )
        if path_m >= cfg.max_path_m - 1e-12:
            append_trace_frame(
                trace,
                r=r_new,
                z=z_new,
                log_weight=log_weight,
                path_m=path_m,
                event="max_path",
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="max_path",
                path_m=path_m,
                n_steps=step_index + 1,
            )

    append_trace_frame(
        trace,
        r=float(math.hypot(position[0], position[1])),
        z=float(position[2]),
        log_weight=log_weight,
        path_m=path_m,
        event="max_steps",
    )
    return _history_score(
        seed,
        weight=float(math.exp(log_weight)),
        termination="max_steps",
        path_m=path_m,
        n_steps=cfg.max_steps,
    )


def run_backward_ionization_mc(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    *,
    n_histories: int,
    seed: int = 42,
    config: BackwardIonConfig | None = None,
    traces: list[list[BackwardTraceFrame]] | None = None,
) -> BackwardIonResult:
    """Run a batch of backward ionization-only histories."""
    if n_histories < 1:
        raise ValueError("n_histories must be >= 1")

    cfg = config or BackwardIonConfig()
    rng = np.random.default_rng(seed)
    ion_seeds = sample_ionization_seeds(grid, n_histories, rng)
    scores: list[HistoryScore] = []
    for ion_seed in ion_seeds:
        hist_trace: list[BackwardTraceFrame] | None = [] if traces is not None else None
        scores.append(
            track_backward_ionization(
                grid,
                wall,
                ion_seed,
                rng,
                config=cfg,
                trace=hist_trace,
            )
        )
        if traces is not None and hist_trace is not None:
            traces.append(hist_trace)
    tallies = accumulate_wall_scores(wall, n_histories, scores)
    return BackwardIonResult(
        config=cfg,
        n_histories=n_histories,
        seed=seed,
        tallies=tallies,
        scores=scores,
    )
