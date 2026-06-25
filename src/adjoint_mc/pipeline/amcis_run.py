"""AMCIS pipeline: HDG solution → grid → point-target backward MC → wall map."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from adjoint_mc.io.wall_flux import WallNeutralFluxResult, extract_wall_neutral_flux
from adjoint_mc.scoring.amcis_provenance import AmcisProvenanceResult, compute_amcis_provenance
from adjoint_mc.tracker.amcis_backward import AmcisConfig, AmcisMcResult, run_amcis_mc
from adjoint_mc.tracker.backward_cython import cython_available, default_cython_thread_count

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class AmcisRunConfig:
    solution_path: Path
    target_r: float
    target_z: float
    n_histories: int = DEFAULT_N_HISTORIES
    seed: int = DEFAULT_SEED
    grid_n_r: int = DEFAULT_GRID_N_R
    grid_n_z: int = DEFAULT_GRID_N_Z
    tau_max: float = DEFAULT_TAU_MAX
    neutral_speed_m_s: float = DEFAULT_NEUTRAL_SPEED_M_S
    max_path_m: float = DEFAULT_MAX_PATH_M
    vacuum_wall_search_m: float = DEFAULT_VACUUM_WALL_SEARCH_M
    enable_cx: bool = True
    use_cython: bool = True
    n_threads: int = 0


@dataclass(frozen=True)
class AmcisRunTiming:
    grid_build_s: float
    mc_s: float
    provenance_s: float
    total_s: float
    mc_engine: str = "python"


@dataclass(frozen=True)
class AmcisRunResult:
    config: AmcisRunConfig
    mc_result: AmcisMcResult
    provenance: AmcisProvenanceResult
    wall: WallGeometry
    grid: PretabulatedGrid
    solution: Any
    wall_flux: WallNeutralFluxResult | None
    timing: AmcisRunTiming


def run_amcis_pipeline(
    config: AmcisRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> AmcisRunResult:
    t_total = time.perf_counter()
    solution_path = config.solution_path.resolve()

    loaded = load_hdg_solution_cached(str(solution_path))
    t_grid = time.perf_counter()
    grid = build_pretabulated_grid(
        loaded.solution,
        n_r=int(config.grid_n_r),
        n_z=int(config.grid_n_z),
    )
    wall = extract_wall_geometry(loaded.solution, solution_path=str(solution_path))
    try:
        wall_flux = extract_wall_neutral_flux(loaded.solution, wall)
    except Exception:
        wall_flux = None
    grid_build_s = time.perf_counter() - t_grid

    mc_config = AmcisConfig(
        tau_max=float(config.tau_max),
        max_path_m=float(config.max_path_m),
        vacuum_wall_search_m=float(config.vacuum_wall_search_m),
        fallback_speed_m_s=float(config.neutral_speed_m_s),
        enable_cx=bool(config.enable_cx),
    )
    mc_engine = (
        f"cython ({default_cython_thread_count() if config.n_threads == 0 else config.n_threads} threads)"
        if config.use_cython and cython_available()
        else "python"
    )

    t_mc = time.perf_counter()
    mc_result = run_amcis_mc(
        grid,
        wall,
        target_r=float(config.target_r),
        target_z=float(config.target_z),
        n_histories=int(config.n_histories),
        seed=int(config.seed),
        config=mc_config,
        progress_callback=progress_callback,
        use_cython=bool(config.use_cython),
        n_threads=int(config.n_threads),
    )
    mc_s = time.perf_counter() - t_mc

    t_prov = time.perf_counter()
    provenance = compute_amcis_provenance(wall, mc_result, wall_flux=wall_flux)
    prov_s = time.perf_counter() - t_prov

    return AmcisRunResult(
        config=config,
        mc_result=mc_result,
        provenance=provenance,
        wall=wall,
        grid=grid,
        solution=loaded.solution,
        wall_flux=wall_flux,
        timing=AmcisRunTiming(
            grid_build_s=grid_build_s,
            mc_s=mc_s,
            provenance_s=prov_s,
            total_s=time.perf_counter() - t_total,
            mc_engine=mc_engine,
        ),
    )
