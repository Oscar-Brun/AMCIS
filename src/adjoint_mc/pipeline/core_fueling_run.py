"""Core fueling provenance: S_ion births inside separatrix → wall provenance."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adjoint_mc.config import (
    DEFAULT_GRID_N_R,
    DEFAULT_GRID_N_Z,
    DEFAULT_MAX_PATH_M,
    DEFAULT_N_HISTORIES,
    DEFAULT_NEUTRAL_SPEED_M_S,
    DEFAULT_SEED,
    DEFAULT_TAU_MAX,
    DEFAULT_VACUUM_WALL_SEARCH_M,
)
from adjoint_mc.fields.pretabulate import PretabulatedGrid, build_pretabulated_grid
from adjoint_mc.geometry.wall import WallGeometry, extract_wall_geometry
from adjoint_mc.io.hdg_loader import load_hdg_solution_cached
from adjoint_mc.pipeline.timing import RunTiming
from adjoint_mc.sampling.seeds import sample_ionization_seeds
from adjoint_mc.scoring.provenance import (
    ProvenanceResult,
    compute_provenance,
    integrated_ionization_rate_in_mask,
)
from adjoint_mc.tracker.backward_cython import (
    cython_available,
    default_cython_thread_count,
    run_backward_full_mc_cython,
)
from adjoint_mc.tracker.backward_full import BackwardFullConfig, BackwardFullResult
from adjoint_mc.viz.separatrix import core_birth_mask_on_grid

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class CoreFuelingRunConfig:
    """Parameters for core fueling provenance (inside separatrix)."""

    solution_path: Path
    n_histories: int = DEFAULT_N_HISTORIES
    seed: int = DEFAULT_SEED
    grid_n_r: int = DEFAULT_GRID_N_R
    grid_n_z: int = DEFAULT_GRID_N_Z
    tau_max: float = DEFAULT_TAU_MAX
    neutral_speed_m_s: float = DEFAULT_NEUTRAL_SPEED_M_S
    max_path_m: float = DEFAULT_MAX_PATH_M
    vacuum_wall_search_m: float = DEFAULT_VACUUM_WALL_SEARCH_M


@dataclass(frozen=True)
class CoreFuelingRunResult:
    config: CoreFuelingRunConfig
    mc_result: BackwardFullResult
    provenance: ProvenanceResult
    wall: WallGeometry
    grid: PretabulatedGrid
    timing: RunTiming
    n_threads: int
    birth_mask: np.ndarray
    core_fueling_rate_s: float
    solution: object


def run_core_fueling_pipeline(
    config: CoreFuelingRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> CoreFuelingRunResult:
    """Backward MC with S_ion births restricted to the plasma core (ψ = 1)."""
    if not cython_available():
        raise ImportError("Cython extension not built. From the project root: pip install -e .")

    t_total = time.perf_counter()
    n_threads = default_cython_thread_count()
    solution_path = config.solution_path.resolve()

    loaded = load_hdg_solution_cached(str(solution_path))
    solution = loaded.solution

    t_grid = time.perf_counter()
    grid = build_pretabulated_grid(
        solution,
        n_r=int(config.grid_n_r),
        n_z=int(config.grid_n_z),
    )
    wall = extract_wall_geometry(solution, solution_path=str(solution_path))
    birth_mask = core_birth_mask_on_grid(grid, solution)
    core_fueling_rate_s = integrated_ionization_rate_in_mask(grid, birth_mask)
    grid_build_s = time.perf_counter() - t_grid

    full_config = BackwardFullConfig(
        tau_max=float(config.tau_max),
        max_path_m=float(config.max_path_m),
        vacuum_wall_search_m=float(config.vacuum_wall_search_m),
        fallback_speed_m_s=float(config.neutral_speed_m_s),
    )

    rng = np.random.default_rng(int(config.seed))
    ion_seeds = sample_ionization_seeds(
        grid,
        int(config.n_histories),
        rng,
        birth_mask=birth_mask,
    )

    t_mc = time.perf_counter()
    mc_result, timing = run_backward_full_mc_cython(
        grid,
        wall,
        n_histories=int(config.n_histories),
        seed=int(config.seed),
        config=full_config,
        return_timing=True,
        cx_rejection=True,
        n_threads=n_threads,
        progress_callback=progress_callback,
        ion_seeds=ion_seeds,
    )
    mc_s = time.perf_counter() - t_mc

    t_prov = time.perf_counter()
    provenance = compute_provenance(
        wall,
        mc_result,
        grid,
        solution=solution,
        fueling_rate_s=core_fueling_rate_s,
    )
    prov_s = time.perf_counter() - t_prov
    total_s = time.perf_counter() - t_total

    return CoreFuelingRunResult(
        config=config,
        mc_result=mc_result,
        provenance=provenance,
        wall=wall,
        grid=grid,
        timing=RunTiming(
            grid_build_s=grid_build_s,
            mc_s=mc_s,
            provenance_s=prov_s,
            kernel_s=timing.kernel_seconds,
            pack_s=timing.pack_seconds,
            total_s=total_s,
        ),
        n_threads=n_threads,
        birth_mask=birth_mask,
        core_fueling_rate_s=core_fueling_rate_s,
        solution=solution,
    )
