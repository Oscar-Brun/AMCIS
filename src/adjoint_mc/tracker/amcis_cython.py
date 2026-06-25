"""Cython + OpenMP wrapper for AMCIS backward MC (survival weights)."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.sampling.point_seed import sample_point_seeds
from adjoint_mc.scoring.tallies import accumulate_wall_scores
from adjoint_mc.tracker.amcis_backward import AmcisConfig, AmcisMcResult
from adjoint_mc.tracker.backward_cython import (
    CythonRunTiming,
    _chunk_size_for_progress,
    _scores_from_cython_batch,
    cython_available,
    default_cython_thread_count,
    pack_cx_calibration_for_cython,
    pack_grid_for_cython,
    pack_seeds_for_cython,
    pack_wall_for_cython,
)


def run_amcis_mc_cython(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    *,
    target_r: float,
    target_z: float,
    n_histories: int,
    seed: int = 42,
    config: AmcisConfig | None = None,
    return_timing: bool = False,
    sync_numpy_rng: bool = False,
    cx_rejection: bool = True,
    calibrate_hdg: bool = True,
    n_threads: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
    chunk_size: int | None = None,
) -> AmcisMcResult | tuple[AmcisMcResult, CythonRunTiming]:
    """
    Run AMCIS via the Cython kernel with survival weights (W = exp(-∫ Σ_ion ds)).

    ``n_threads=0`` uses all CPU cores via OpenMP; ``n_threads=1`` disables OpenMP.
    """
    if not cython_available():
        raise ImportError(
            "Cython extension not built. Run: pip install -e . from the project root."
        )

    from adjoint_mc.core._tracker import run_backward_batch_cython

    if n_histories < 1:
        raise ValueError("n_histories must be >= 1")

    cfg = config or AmcisConfig()
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    point_seeds = sample_point_seeds(grid, target_r, target_z, n_histories, rng)
    grid_arrays = pack_grid_for_cython(grid)
    wall_arrays = pack_wall_for_cython(wall)
    seed_arrays = pack_seeds_for_cython(point_seeds)
    cx_calibration = pack_cx_calibration_for_cython(calibrate_hdg=calibrate_hdg)
    pack_s = time.perf_counter() - t0

    rng_state = 0
    rng_inc = 0
    if sync_numpy_rng:
        inner = rng.bit_generator.state.get("state", {})
        if isinstance(inner, dict):
            rng_state = int(inner.get("state", 0)) & ((1 << 64) - 1)
            rng_inc = int(inner.get("inc", 0)) | 1
        else:
            rng_state = int(inner) & ((1 << 64) - 1)

    threads = int(n_threads)
    if threads == 0:
        threads = default_cython_thread_count()

    batch_kwargs = dict(
        grid_arrays=grid_arrays,
        wall_arrays=wall_arrays,
        seed=seed,
        rng_state=rng_state,
        rng_inc=rng_inc,
        sync_numpy_rng=sync_numpy_rng,
        tau_max=float(cfg.tau_max),
        max_step_m=float(cfg.max_step_m),
        vacuum_wall_search_m=float(cfg.vacuum_wall_search_m),
        max_path_m=float(cfg.max_path_m),
        max_steps=int(cfg.max_steps),
        fallback_speed_m_s=float(cfg.fallback_speed_m_s),
        enable_cx=bool(cfg.enable_cx),
        cx_rejection=cx_rejection,
        calibrate_hdg=calibrate_hdg,
        cx_calibration=cx_calibration,
        cx_max_trials=int(cfg.cx_rejection.max_trials_per_sample),
        n_threads=threads,
        survival_weight=True,
    )

    t1 = time.perf_counter()
    if progress_callback is None:
        batch = run_backward_batch_cython(seed_arrays, **batch_kwargs)
        scores = _scores_from_cython_batch(batch, point_seeds, wall)
    else:
        chunk = chunk_size or _chunk_size_for_progress(n_histories)
        scores = []
        progress_callback(0, n_histories)
        for start in range(0, n_histories, chunk):
            end = min(start + chunk, n_histories)
            chunk_seeds = point_seeds[start:end]
            chunk_arrays = pack_seeds_for_cython(chunk_seeds)
            batch = run_backward_batch_cython(
                chunk_arrays,
                history_start_index=start,
                **batch_kwargs,
            )
            scores.extend(_scores_from_cython_batch(batch, chunk_seeds, wall))
            progress_callback(end, n_histories)
    kernel_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    tallies = accumulate_wall_scores(wall, n_histories, scores)
    result = AmcisMcResult(
        config=cfg,
        target_r=float(target_r),
        target_z=float(target_z),
        n_histories=n_histories,
        seed=seed,
        tallies=tallies,
        scores=scores,
    )
    unpack_s = time.perf_counter() - t2

    timing = CythonRunTiming(pack_seconds=pack_s, kernel_seconds=kernel_s, unpack_seconds=unpack_s)
    if return_timing:
        return result, timing
    return result
