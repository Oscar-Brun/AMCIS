"""Cython backward tracker wrapper."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from adjoint_mc.atomic.cx_cross_section import _T_CALIB_GRID, calibration_factor
from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import WallGeometry
from adjoint_mc.sampling.seeds import IonizationSeed, sample_ionization_seeds
from adjoint_mc.scoring.tallies import HistoryScore, WallTallyResult, accumulate_wall_scores
from adjoint_mc.tracker.backward_full import BackwardFullConfig, BackwardFullResult

_TERM_NAMES = ("wall", "lost", "max_path", "max_steps")


def _chunk_size_for_progress(n_histories: int) -> int:
    """Histories per kernel chunk for periodic GUI updates."""
    if n_histories <= 20:
        return 1
    desired_updates = min(100, max(10, n_histories // 50))
    return max(1, (n_histories + desired_updates - 1) // desired_updates)


def cython_available() -> bool:
    try:
        from adjoint_mc.core._tracker import CYTHON_AVAILABLE

        return bool(CYTHON_AVAILABLE)
    except ImportError:
        return False


def pack_cx_calibration_for_cython(*, calibrate_hdg: bool = True) -> dict | None:
    """Pack log-log OpenADAS calibration table for the Cython Janev kernel."""
    if not calibrate_hdg:
        return None
    factors = np.array([calibration_factor(float(t)) for t in _T_CALIB_GRID], dtype=np.float64)
    return {
        "log_t": np.ascontiguousarray(np.log(_T_CALIB_GRID), dtype=np.float64),
        "log_f": np.ascontiguousarray(np.log(np.maximum(factors, 1.0e-12)), dtype=np.float64),
    }


def default_cython_thread_count() -> int:
    """Default OpenMP thread count (all cores)."""
    return max(1, os.cpu_count() or 1)


def pack_grid_for_cython(grid: PretabulatedGrid) -> dict:
    """Pack PretabulatedGrid arrays for the Cython extension."""
    return {
        "r_min": float(grid.r_min),
        "r_max": float(grid.r_max),
        "z_min": float(grid.z_min),
        "z_max": float(grid.z_max),
        "n_r": int(grid.n_r),
        "n_z": int(grid.n_z),
        "r_coords": np.ascontiguousarray(grid.r_coords, dtype=np.float64),
        "z_coords": np.ascontiguousarray(grid.z_coords, dtype=np.float64),
        "mask": np.ascontiguousarray(grid.mask.astype(np.uint8)),
        "n": np.ascontiguousarray(grid.fields["n"], dtype=np.float64),
        "ti": np.ascontiguousarray(grid.fields["ti"], dtype=np.float64),
        "iz_rate": np.ascontiguousarray(grid.fields["iz_rate"], dtype=np.float64),
        "cx_rate": np.ascontiguousarray(grid.fields["cx_rate"], dtype=np.float64),
    }


def pack_wall_for_cython(wall: WallGeometry) -> dict:
    """Pack wall segments as contiguous float64 arrays."""
    if not wall.segments:
        raise ValueError("Wall geometry has no segments")
    return {
        "r0": np.ascontiguousarray([s.r0 for s in wall.segments], dtype=np.float64),
        "z0": np.ascontiguousarray([s.z0 for s in wall.segments], dtype=np.float64),
        "r1": np.ascontiguousarray([s.r1 for s in wall.segments], dtype=np.float64),
        "z1": np.ascontiguousarray([s.z1 for s in wall.segments], dtype=np.float64),
        "region_names": [s.region_name for s in wall.segments],
    }


def pack_seeds_for_cython(seeds: list[IonizationSeed]) -> dict:
    """Pack ionization seeds for the Cython batch kernel."""
    return {
        "x": np.ascontiguousarray([s.position[0] for s in seeds], dtype=np.float64),
        "y": np.ascontiguousarray([s.position[1] for s in seeds], dtype=np.float64),
        "z": np.ascontiguousarray([s.position[2] for s in seeds], dtype=np.float64),
        "r": np.ascontiguousarray([s.r for s in seeds], dtype=np.float64),
        "z_plane": np.ascontiguousarray([s.z for s in seeds], dtype=np.float64),
    }


def _scores_from_cython_batch(
    batch: dict,
    seeds: list[IonizationSeed],
    wall: WallGeometry,
) -> list[HistoryScore]:
    region_names = pack_wall_for_cython(wall)["region_names"]
    scores: list[HistoryScore] = []
    n = len(seeds)
    for i in range(n):
        term_code = int(batch["termination_codes"][i])
        termination = _TERM_NAMES[term_code] if 0 <= term_code < len(_TERM_NAMES) else "lost"
        seg = int(batch["segment_index"][i])
        region_name = region_names[seg] if seg >= 0 else None
        hit_r = float(batch["hit_r"][i])
        hit_z = float(batch["hit_z"][i])
        scores.append(
            HistoryScore(
                weight=float(batch["weights"][i]),
                termination=termination,
                region_name=region_name,
                segment_index=seg if seg >= 0 else None,
                path_m=float(batch["path_m"][i]),
                n_steps=int(batch["n_steps"][i]),
                seed_r=float(batch["seed_r"][i]),
                seed_z=float(batch["seed_z"][i]),
                hit_r=hit_r if np.isfinite(hit_r) else None,
                hit_z=hit_z if np.isfinite(hit_z) else None,
                n_cx_events=int(batch["n_cx_events"][i]),
            )
        )
    return scores


@dataclass(frozen=True)
class CythonRunTiming:
    """Timing breakdown for a Cython batch run."""

    pack_seconds: float
    kernel_seconds: float
    unpack_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.pack_seconds + self.kernel_seconds + self.unpack_seconds


def run_backward_full_mc_cython(
    grid: PretabulatedGrid,
    wall: WallGeometry,
    *,
    n_histories: int,
    seed: int = 42,
    config: BackwardFullConfig | None = None,
    return_timing: bool = False,
    sync_numpy_rng: bool = False,
    enable_cx: bool = True,
    cx_rejection: bool = True,
    calibrate_hdg: bool = True,
    n_threads: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
    chunk_size: int | None = None,
    ion_seeds: list[IonizationSeed] | None = None,
) -> BackwardFullResult | tuple[BackwardFullResult, CythonRunTiming]:
    """
    Run backward MC via the Cython kernel.

    ``cx_rejection=True`` (default) uses Janev + rejection.
    ``n_threads=0`` uses all CPU cores via OpenMP; ``n_threads=1`` disables OpenMP.
    """
    if not cython_available():
        raise ImportError(
            "Cython extension not built. Run: pip install -e . from the project root."
        )

    from adjoint_mc.core._tracker import run_backward_batch_cython

    if n_histories < 1:
        raise ValueError("n_histories must be >= 1")

    cfg = config or BackwardFullConfig()
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    if ion_seeds is None:
        ion_seeds = sample_ionization_seeds(grid, n_histories, rng)
    elif len(ion_seeds) != n_histories:
        raise ValueError(
            f"ion_seeds length {len(ion_seeds)} does not match n_histories={n_histories}"
        )
    grid_arrays = pack_grid_for_cython(grid)
    wall_arrays = pack_wall_for_cython(wall)
    seed_arrays = pack_seeds_for_cython(ion_seeds)
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
        enable_cx=enable_cx,
        cx_rejection=cx_rejection,
        calibrate_hdg=calibrate_hdg,
        cx_calibration=cx_calibration,
        cx_max_trials=int(cfg.cx_rejection.max_trials_per_sample),
        n_threads=threads,
    )

    t1 = time.perf_counter()
    if progress_callback is None:
        batch = run_backward_batch_cython(
            seed_arrays,
            **batch_kwargs,
        )
        scores = _scores_from_cython_batch(batch, ion_seeds, wall)
    else:
        chunk = chunk_size or _chunk_size_for_progress(n_histories)
        scores = []
        progress_callback(0, n_histories)
        for start in range(0, n_histories, chunk):
            end = min(start + chunk, n_histories)
            chunk_seeds = ion_seeds[start:end]
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
    result = BackwardFullResult(
        config=cfg,
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


def compare_tally_relative_errors(
    python_result: BackwardFullResult,
    cython_result: BackwardFullResult,
    *,
    rtol: float = 1e-3,
) -> dict[str, float | bool]:
    """Compare key tallies between Python and Cython runs."""
    pt, ct = python_result.tallies, cython_result.tallies

    def rel(a: float, b: float) -> float:
        denom = max(abs(a), 1e-30)
        return float(abs(a - b) / denom)

    wall_frac_err = rel(pt.wall_fraction, ct.wall_fraction)
    weight_err = rel(pt.total_weight, ct.total_weight)
    n_wall_err = rel(float(pt.n_wall), float(ct.n_wall))

    return {
        "wall_fraction_error": wall_frac_err,
        "total_weight_error": weight_err,
        "n_wall_error": n_wall_err,
        "within_rtol": (
            wall_frac_err <= rtol and weight_err <= rtol and n_wall_err <= rtol
        ),
        "rtol": rtol,
    }
