"""
AMCIS backward MC — single target point, survival (attenuation) weights.

Ionization: W <- W * exp(-Sigma_ion * ds)  (forward survival along the path)
CX: unchanged from the full backward tracker (rejection on sigma(E_rel), W unchanged)
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
from adjoint_mc.sampling.point_seed import sample_point_seeds
from adjoint_mc.sampling.seeds import IonizationSeed
from adjoint_mc.scoring.tallies import HistoryScore, WallTallyResult, accumulate_wall_scores
from adjoint_mc.tracker.backward_trace import BackwardTraceFrame, append_trace_frame


def _apply_cx(
    velocity: np.ndarray,
    r: float,
    z: float,
    ds: float,
    sigma_cx: float,
    grid: PretabulatedGrid,
    rng: np.random.Generator,
    config: AmcisConfig,
) -> tuple[np.ndarray, int]:
    """One backward CX event over ds; rejection on sigma(E_rel), W unchanged."""
    if not config.enable_cx or sigma_cx <= 0.0 or ds <= 0.0:
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


@dataclass(frozen=True)
class AmcisConfig:
    """Numerical parameters for AMCIS backward tracking."""

    tau_max: float = DEFAULT_TAU_MAX
    max_step_m: float = DEFAULT_MAX_STEP_M
    vacuum_wall_search_m: float = DEFAULT_VACUUM_WALL_SEARCH_M
    max_path_m: float = DEFAULT_MAX_PATH_M
    max_steps: int = 20_000
    fallback_speed_m_s: float = DEFAULT_NEUTRAL_SPEED_M_S
    cx_rejection: CxRejectionConfig = field(default_factory=CxRejectionConfig)
    enable_cx: bool = True


@dataclass(frozen=True)
class AmcisMcResult:
    """Batch outcome of AMCIS backward MC."""

    config: AmcisConfig
    target_r: float
    target_z: float
    n_histories: int
    seed: int
    tallies: WallTallyResult
    scores: list[HistoryScore]

    @property
    def total_cx_events(self) -> int:
        return sum(s.n_cx_events for s in self.scores)


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


def track_amcis_backward(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    seed: IonizationSeed,
    rng: np.random.Generator,
    *,
    config: AmcisConfig | None = None,
    trace: list[BackwardTraceFrame] | None = None,
) -> HistoryScore:
    """Track one backward history with survival weights exp(-integral Sigma_ion ds)."""
    cfg = config or AmcisConfig()

    position = np.asarray(seed.position, dtype=float)
    velocity = sample_neutral_maxwellian_velocity(
        grid,
        seed.r,
        seed.z,
        rng,
        fallback_speed_m_s=cfg.fallback_speed_m_s,
    )
    log_survival = 0.0
    path_m = 0.0
    n_cx = 0

    append_trace_frame(
        trace,
        r=seed.r,
        z=seed.z,
        log_weight=log_survival,
        path_m=path_m,
        event="birth",
    )

    if not in_plasma(grid, seed.r, seed.z):
        append_trace_frame(trace, r=seed.r, z=seed.z, log_weight=log_survival, path_m=path_m, event="lost")
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
            return _history_score(
                seed,
                weight=float(math.exp(log_survival)),
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
                log_survival -= sigma_ion * hit.t
                return _history_score(
                    seed,
                    weight=float(math.exp(log_survival)),
                    termination="wall",
                    region_name=hit.region_name,
                    segment_index=hit.segment_index,
                    path_m=path_m + hit.t,
                    n_steps=step_index,
                    hit_r=hit.r,
                    hit_z=hit.z,
                    n_cx_events=n_cx,
                )
            return _history_score(
                seed,
                weight=float(math.exp(log_survival)),
                termination="lost",
                path_m=path_m,
                n_steps=step_index,
                n_cx_events=n_cx,
            )

        sigma_ion = macroscopic_ionization_rate(grid, r, z, speed)
        sigma_cx = macroscopic_cx_rate(grid, r, z, speed) if cfg.enable_cx else 0.0
        sigma_eff = sigma_ion + sigma_cx
        if sigma_eff > 0.0:
            ds_optical = cfg.tau_max / sigma_eff
        else:
            ds_optical = cfg.max_step_m
        ds = min(cfg.max_step_m, ds_optical, cfg.max_path_m - path_m)
        if ds <= 0.0:
            return _history_score(
                seed,
                weight=float(math.exp(log_survival)),
                termination="max_path",
                path_m=path_m,
                n_steps=step_index,
                n_cx_events=n_cx,
            )

        hit = intersect_ray(position, backward, wall, t_min=1e-12, t_max=ds)
        if hit is not None:
            log_survival -= sigma_ion * hit.t
            velocity, n_new = _apply_cx(velocity, r, z, hit.t, sigma_cx, grid, rng, cfg)
            n_cx += n_new
            return _history_score(
                seed,
                weight=float(math.exp(log_survival)),
                termination="wall",
                region_name=hit.region_name,
                segment_index=hit.segment_index,
                path_m=path_m + hit.t,
                n_steps=step_index + 1,
                hit_r=hit.r,
                hit_z=hit.z,
                n_cx_events=n_cx,
            )

        log_survival -= sigma_ion * ds
        velocity, n_new = _apply_cx(velocity, r, z, ds, sigma_cx, grid, rng, cfg)
        n_cx += n_new
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            return _history_score(
                seed,
                weight=float(math.exp(log_survival)),
                termination="lost",
                path_m=path_m + ds,
                n_steps=step_index + 1,
                n_cx_events=n_cx,
            )
        direction = velocity / speed
        position = position + backward * ds
        path_m += ds
        if path_m >= cfg.max_path_m - 1e-12:
            return _history_score(
                seed,
                weight=float(math.exp(log_survival)),
                termination="max_path",
                path_m=path_m,
                n_steps=step_index + 1,
                n_cx_events=n_cx,
            )

    return _history_score(
        seed,
        weight=float(math.exp(log_survival)),
        termination="max_steps",
        path_m=path_m,
        n_steps=cfg.max_steps,
        n_cx_events=n_cx,
    )


def _run_amcis_mc_python(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    *,
    target_r: float,
    target_z: float,
    n_histories: int,
    seed: int = 42,
    config: AmcisConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AmcisMcResult:
    """Python reference kernel for AMCIS."""
    if n_histories < 1:
        raise ValueError("n_histories must be >= 1")

    cfg = config or AmcisConfig()
    rng = np.random.default_rng(seed)
    point_seeds = sample_point_seeds(grid, target_r, target_z, n_histories, rng)

    if progress_callback is not None:
        progress_callback(0, n_histories)

    scores: list[HistoryScore] = []
    for i, point_seed in enumerate(point_seeds):
        scores.append(track_amcis_backward(grid, wall, point_seed, rng, config=cfg))
        if progress_callback is not None:
            done = i + 1
            if done == n_histories or done % max(1, n_histories // 100) == 0:
                progress_callback(done, n_histories)

    tallies = accumulate_wall_scores(wall, n_histories, scores)
    return AmcisMcResult(
        config=cfg,
        target_r=float(target_r),
        target_z=float(target_z),
        n_histories=n_histories,
        seed=seed,
        tallies=tallies,
        scores=scores,
    )


def run_amcis_mc(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    *,
    target_r: float,
    target_z: float,
    n_histories: int,
    seed: int = 42,
    config: AmcisConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    use_cython: bool = True,
    n_threads: int = 0,
) -> AmcisMcResult:
    """Run AMCIS: all histories born at (target_r, target_z), survival weights to wall."""
    if use_cython:
        from adjoint_mc.tracker.amcis_cython import cython_available, run_amcis_mc_cython

        if cython_available():
            return run_amcis_mc_cython(
                grid,
                wall,
                target_r=target_r,
                target_z=target_z,
                n_histories=n_histories,
                seed=seed,
                config=config,
                progress_callback=progress_callback,
                n_threads=n_threads,
            )

    return _run_amcis_mc_python(
        grid,
        wall,
        target_r=target_r,
        target_z=target_z,
        n_histories=n_histories,
        seed=seed,
        config=config,
        progress_callback=progress_callback,
    )
