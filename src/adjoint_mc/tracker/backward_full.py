"""
Backward adjoint tracker — ionization + charge exchange.

Extends ionization-only tracking with discrete CX events (rejection kernel) and Maxwellian
neutral birth at local T_i (T_n ≡ T_i).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from adjoint_mc.atomic.cx_rejection import CxRejectionConfig, sample_cx_velocity
from adjoint_mc.config import (
    DEFAULT_MAX_PATH_M,
    DEFAULT_MAX_STEP_M,
    DEFAULT_NEUTRAL_SPEED_M_S,
    DEFAULT_TAU_MAX,
    DEFAULT_VACUUM_WALL_SEARCH_M,
)
from adjoint_mc.fields.grid_interp import (
    in_plasma,
    local_ti_ev,
    macroscopic_cx_rate,
    macroscopic_ionization_rate,
    sample_neutral_maxwellian_velocity,
)
from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import WallGeometry, intersect_ray
from adjoint_mc.sampling.seeds import IonizationSeed, sample_ionization_seeds
from adjoint_mc.scoring.tallies import HistoryScore, WallTallyResult, accumulate_wall_scores
from adjoint_mc.tracker.backward_trace import BackwardTraceFrame, append_trace_frame


@dataclass(frozen=True)
class BackwardFullConfig:
    """Numerical parameters for the full backward tracker."""

    tau_max: float = DEFAULT_TAU_MAX
    max_step_m: float = DEFAULT_MAX_STEP_M
    vacuum_wall_search_m: float = DEFAULT_VACUUM_WALL_SEARCH_M
    max_path_m: float = DEFAULT_MAX_PATH_M
    max_steps: int = 20_000
    fallback_speed_m_s: float = DEFAULT_NEUTRAL_SPEED_M_S
    cx_rejection: CxRejectionConfig = field(default_factory=CxRejectionConfig)


@dataclass(frozen=True)
class BackwardFullResult:
    """Batch outcome of full backward MC (ionization + CX)."""

    config: BackwardFullConfig
    n_histories: int
    seed: int
    tallies: WallTallyResult
    scores: list[HistoryScore]

    @property
    def total_cx_events(self) -> int:
        return sum(s.n_cx_events for s in self.scores)

    @property
    def mean_cx_events(self) -> float:
        if not self.scores:
            return 0.0
        return float(self.total_cx_events / len(self.scores))


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
    n_cx_events: int = 0,
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
        n_cx_events=n_cx_events,
    )


def _maybe_apply_cx(
    velocity: np.ndarray,
    r: float,
    z: float,
    ds: float,
    sigma_cx: float,
    grid: PretabulatedGrid,
    rng: np.random.Generator,
    config: BackwardFullConfig,
) -> tuple[np.ndarray, int]:
    """Apply one backward CX event over path length ds with probability 1 - exp(-Sigma_cx ds)."""
    if sigma_cx <= 0.0 or ds <= 0.0:
        return velocity, 0
    p_cx = 1.0 - math.exp(-sigma_cx * ds)
    if rng.random() >= p_cx:
        return velocity, 0

    ti = local_ti_ev(grid, r, z)
    sample = sample_cx_velocity(
        velocity,
        ti,
        rng,
        config=config.cx_rejection,
    )
    if sample is None:
        return velocity, 0
    return sample.v_n_after.copy(), 1


def track_backward_full(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    seed: IonizationSeed,
    rng: np.random.Generator,
    *,
    config: BackwardFullConfig | None = None,
    trace: list[BackwardTraceFrame] | None = None,
) -> HistoryScore:
    """Track one backward history with ionization weighting and CX velocity exchanges."""
    cfg = config or BackwardFullConfig()

    position = np.asarray(seed.position, dtype=float)
    velocity = sample_neutral_maxwellian_velocity(
        grid,
        seed.r,
        seed.z,
        rng,
        fallback_speed_m_s=cfg.fallback_speed_m_s,
    )
    log_weight = 0.0
    path_m = 0.0
    n_cx = 0

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

    speed = float(np.linalg.norm(velocity))
    if speed <= 0.0:
        velocity = sample_neutral_maxwellian_velocity(
            grid, seed.r, seed.z, rng, fallback_speed_m_s=cfg.fallback_speed_m_s
        )
        speed = float(np.linalg.norm(velocity))
    direction = velocity / speed

    for step_index in range(cfg.max_steps):
        r = float(math.hypot(position[0], position[1]))
        z = float(position[2])
        backward = -direction
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            append_trace_frame(
                trace,
                r=r,
                z=z,
                log_weight=log_weight,
                path_m=path_m,
                event="lost",
                n_cx=n_cx,
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="lost",
                path_m=path_m,
                n_steps=step_index,
                n_cx_events=n_cx,
            )
        direction = velocity / speed

        if not in_plasma(grid, r, z):
            hit = intersect_ray(
                position, backward, wall, t_min=1e-12, t_max=cfg.vacuum_wall_search_m
            )
            if hit is not None:
                sigma_ion = macroscopic_ionization_rate(grid, r, z, speed)
                log_weight += sigma_ion * hit.t
                append_trace_frame(
                    trace,
                    r=hit.r,
                    z=hit.z,
                    log_weight=log_weight,
                    path_m=path_m + hit.t,
                    event="wall",
                    n_cx=n_cx,
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
                    n_cx_events=n_cx,
                )
            append_trace_frame(
                trace,
                r=r,
                z=z,
                log_weight=log_weight,
                path_m=path_m,
                event="lost",
                n_cx=n_cx,
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="lost",
                path_m=path_m,
                n_steps=step_index,
                n_cx_events=n_cx,
            )

        sigma_ion = macroscopic_ionization_rate(grid, r, z, speed)
        sigma_cx = macroscopic_cx_rate(grid, r, z, speed)
        sigma_eff = sigma_ion + sigma_cx
        if sigma_eff > 0.0:
            ds_optical = cfg.tau_max / sigma_eff
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
                n_cx=n_cx,
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="max_path",
                path_m=path_m,
                n_steps=step_index,
                n_cx_events=n_cx,
            )

        hit = intersect_ray(position, backward, wall, t_min=1e-12, t_max=ds)
        if hit is not None:
            log_weight += sigma_ion * hit.t
            velocity, n_new = _maybe_apply_cx(
                velocity, r, z, hit.t, sigma_cx, grid, rng, cfg
            )
            n_cx += n_new
            if n_new:
                append_trace_frame(
                    trace,
                    r=r,
                    z=z,
                    log_weight=log_weight,
                    path_m=path_m + hit.t,
                    event="cx",
                    n_cx=n_cx,
                )
            speed = float(np.linalg.norm(velocity))
            if speed > 0.0:
                direction = velocity / speed
            append_trace_frame(
                trace,
                r=hit.r,
                z=hit.z,
                log_weight=log_weight,
                path_m=path_m + hit.t,
                event="wall",
                n_cx=n_cx,
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
                n_cx_events=n_cx,
            )

        log_weight += sigma_ion * ds
        velocity, n_new = _maybe_apply_cx(velocity, r, z, ds, sigma_cx, grid, rng, cfg)
        n_cx += n_new
        if n_new:
            append_trace_frame(
                trace,
                r=r,
                z=z,
                log_weight=log_weight,
                path_m=path_m + ds,
                event="cx",
                n_cx=n_cx,
            )
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            append_trace_frame(
                trace,
                r=r,
                z=z,
                log_weight=log_weight,
                path_m=path_m + ds,
                event="lost",
                n_cx=n_cx,
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="lost",
                path_m=path_m + ds,
                n_steps=step_index + 1,
                n_cx_events=n_cx,
            )
        direction = velocity / speed
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
            n_cx=n_cx,
        )
        if path_m >= cfg.max_path_m - 1e-12:
            append_trace_frame(
                trace,
                r=r_new,
                z=z_new,
                log_weight=log_weight,
                path_m=path_m,
                event="max_path",
                n_cx=n_cx,
            )
            return _history_score(
                seed,
                weight=float(math.exp(log_weight)),
                termination="max_path",
                path_m=path_m,
                n_steps=step_index + 1,
                n_cx_events=n_cx,
            )

    append_trace_frame(
        trace,
        r=float(math.hypot(position[0], position[1])),
        z=float(position[2]),
        log_weight=log_weight,
        path_m=path_m,
        event="max_steps",
        n_cx=n_cx,
    )
    return _history_score(
        seed,
        weight=float(math.exp(log_weight)),
        termination="max_steps",
        path_m=path_m,
        n_steps=cfg.max_steps,
        n_cx_events=n_cx,
    )


def run_backward_full_mc(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    *,
    n_histories: int,
    seed: int = 42,
    config: BackwardFullConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval: int | None = None,
    traces: list[list[BackwardTraceFrame]] | None = None,
) -> BackwardFullResult:
    """Run a batch of full backward histories (ionization + CX)."""
    if n_histories < 1:
        raise ValueError("n_histories must be >= 1")

    cfg = config or BackwardFullConfig()
    rng = np.random.default_rng(seed)
    ion_seeds = sample_ionization_seeds(grid, n_histories, rng)
    report_every = progress_interval
    if progress_callback is not None:
        if report_every is None:
            report_every = 1 if n_histories <= 500 else max(1, n_histories // 100)
        progress_callback(0, n_histories)

    scores: list[HistoryScore] = []
    for i, ion_seed in enumerate(ion_seeds):
        hist_trace: list[BackwardTraceFrame] | None = [] if traces is not None else None
        scores.append(
            track_backward_full(grid, wall, ion_seed, rng, config=cfg, trace=hist_trace)
        )
        if traces is not None and hist_trace is not None:
            traces.append(hist_trace)
        if progress_callback is not None:
            done = i + 1
            if done % report_every == 0 or done == n_histories:
                progress_callback(done, n_histories)

    tallies = accumulate_wall_scores(wall, n_histories, scores)
    return BackwardFullResult(
        config=cfg,
        n_histories=n_histories,
        seed=seed,
        tallies=tallies,
        scores=scores,
    )
