"""Build uniform (R, Z) field tables from HDG pointwise evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

from hdg_postprocess.core.solution import batched_sampling as batched_sampling_ops
from hdg_postprocess.core.solution import preparation as prep_ops
from hdg_postprocess.routines.atomic import (
    calculate_cx_rate,
    calculate_iz_rate_cons,
    calculate_iz_source_cons,
)
from hdg_postprocess.routines.plasma import calculate_Ti_cons

PRETAB_FIELD_NAMES: Tuple[str, ...] = (
    "n",
    "ti",
    "te",
    "nn",
    "S_ion",
    "iz_rate",
    "cx_rate",
)


@dataclass
class PretabulatedGrid:
    """Uniform Cartesian grid of plasma fields for fast bilinear lookup."""

    r_min: float
    r_max: float
    z_min: float
    z_max: float
    n_r: int
    n_z: int
    r_coords: np.ndarray
    z_coords: np.ndarray
    mask: np.ndarray
    fields: Dict[str, np.ndarray]
    build_seconds: float = 0.0

    @property
    def coverage_fraction(self) -> float:
        if self.mask.size == 0:
            return 0.0
        return float(np.count_nonzero(self.mask) / self.mask.size)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "r_min": self.r_min,
            "r_max": self.r_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "n_r": self.n_r,
            "n_z": self.n_z,
            "r_coords": self.r_coords,
            "z_coords": self.z_coords,
            "mask": self.mask,
            "build_seconds": self.build_seconds,
        }
        for name in PRETAB_FIELD_NAMES:
            payload[f"field_{name}"] = self.fields[name]
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path | str) -> PretabulatedGrid:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            fields = {name: np.asarray(data[f"field_{name}"]) for name in PRETAB_FIELD_NAMES}
            return cls(
                r_min=float(data["r_min"]),
                r_max=float(data["r_max"]),
                z_min=float(data["z_min"]),
                z_max=float(data["z_max"]),
                n_r=int(data["n_r"]),
                n_z=int(data["n_z"]),
                r_coords=np.asarray(data["r_coords"]),
                z_coords=np.asarray(data["z_coords"]),
                mask=np.asarray(data["mask"], dtype=bool),
                fields=fields,
                build_seconds=float(data.get("build_seconds", 0.0)),
            )


@dataclass
class FieldErrorStats:
    name: str
    median_rel: float
    p95_rel: float
    max_rel: float
    n_samples: int


@dataclass
class GridErrorReport:
    n_r: int
    n_z: int
    coverage_fraction: float
    build_seconds: float
    by_field: Dict[str, FieldErrorStats] = field(default_factory=dict)


def mesh_extent(solution: Any) -> Dict[str, float]:
    """Mesh bounding box in (R, Z) from HDG metadata."""
    extent = solution.mesh.metadata.extent
    return {
        "r_min": float(extent["minr"]),
        "r_max": float(extent["maxr"]),
        "z_min": float(extent["minz"]),
        "z_max": float(extent["maxz"]),
    }


def ensure_hdg_interpolators(solution: Any) -> None:
    """Ensure HDG pointwise / batched sampling is ready."""
    prep_ops.ensure_simple_physical(solution)
    prep_ops.ensure_interpolators(solution)
    _ = solution.mesh.geometry.element_locator


def _as_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _plasma_locator_mask(solution: Any, r_values: np.ndarray, z_values: np.ndarray) -> np.ndarray:
    locator = solution.mesh.geometry.element_locator
    element_ids = np.fromiter(
        (int(locator(float(r), float(z))) for r, z in zip(r_values, z_values)),
        dtype=np.int32,
        count=r_values.size,
    )
    return element_ids >= 0


def _sample_atomic_fields(
    solution: Any,
    state: np.ndarray,
    *,
    active: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    atomic = solution.additional_parameters.atomic
    if atomic is None or "iz" not in atomic or "cx" not in atomic:
        raise ValueError("Atomic data (iz, cx) must be configured before pre-tabulation")

    adim = solution.parameters["adimensionalization"]
    physics = solution.parameters["physics"]
    t0 = float(adim["temperature_scale"])
    n0 = float(adim["density_scale"])
    mref = physics["Mref"]

    n_points = state.shape[0]
    s_ion = np.zeros(n_points, dtype=float)
    iz_rate = np.zeros(n_points, dtype=float)
    cx_rate = np.zeros(n_points, dtype=float)

    if active is None:
        active = np.ones(n_points, dtype=bool)
    else:
        active = np.asarray(active, dtype=bool).reshape(-1)

    rho_idx = solution._cons_idx[b"rho"]
    active &= state[:, rho_idx] > 0.0
    if not np.any(active):
        return {"S_ion": s_ion, "iz_rate": iz_rate, "cx_rate": cx_rate}

    plasma_state = state[active]
    s_ion[active] = np.asarray(
        calculate_iz_source_cons(plasma_state, atomic["iz"], t0, n0, mref, solution._cons_idx),
        dtype=float,
    ).reshape(-1)
    iz_rate[active] = np.asarray(
        calculate_iz_rate_cons(plasma_state, atomic["iz"], t0, n0, mref),
        dtype=float,
    ).reshape(-1)
    # CX rate at T_i (SOLEDGE-HDG convention T_n ≡ T_i), not T_e from cons form.
    ti = np.asarray(
        calculate_Ti_cons(plasma_state, t0, mref, solution._cons_idx),
        dtype=float,
    ).reshape(-1)
    cx_rate[active] = np.asarray(calculate_cx_rate(ti, atomic["cx"]), dtype=float).reshape(-1)
    return {"S_ion": s_ion, "iz_rate": iz_rate, "cx_rate": cx_rate}


def build_pretabulated_grid(
    solution: Any,
    *,
    n_r: int,
    n_z: int,
    r_bounds: Tuple[float, float] | None = None,
    z_bounds: Tuple[float, float] | None = None,
) -> PretabulatedGrid:
    """
    Sample plasma fields on a uniform (R, Z) grid using HDG batched interpolators.

    Values outside the plasma domain are masked out (NaN in field arrays).
    """
    if n_r < 2 or n_z < 2:
        raise ValueError("Grid must have at least 2 nodes per axis")

    ensure_hdg_interpolators(solution)
    extent = mesh_extent(solution)
    r_min, r_max = r_bounds or (extent["r_min"], extent["r_max"])
    z_min, z_max = z_bounds or (extent["z_min"], extent["z_max"])

    r_coords = np.linspace(r_min, r_max, n_r, dtype=np.float64)
    z_coords = np.linspace(z_min, z_max, n_z, dtype=np.float64)
    r_grid, z_grid = np.meshgrid(r_coords, z_coords, indexing="xy")
    r_flat = r_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    t0 = time.perf_counter()
    mask_flat = _plasma_locator_mask(solution, r_flat, z_flat)
    mask = mask_flat.reshape(n_z, n_r)
    plasma_indices = np.flatnonzero(mask_flat)

    fields: Dict[str, np.ndarray] = {
        name: np.full(r_flat.size, np.nan, dtype=np.float64) for name in PRETAB_FIELD_NAMES
    }

    if plasma_indices.size:
        r_plasma = r_flat[plasma_indices]
        z_plasma = z_flat[plasma_indices]

        plasma = batched_sampling_ops.sample_variables(
            solution,
            r_plasma,
            z_plasma,
            ["n", "ti", "te", "nn"],
        )

        state = np.zeros((plasma_indices.size, solution.neq), dtype=np.float64)
        for index in range(solution.neq):
            state[:, index] = solution.interpolators.solution[index].evaluate_many(
                r_plasma, z_plasma
            )

        atomic = _sample_atomic_fields(solution, state, active=np.ones(plasma_indices.size, dtype=bool))

        for name in ("n", "ti", "te", "nn"):
            fields[name][plasma_indices] = np.asarray(plasma[name], dtype=float).reshape(-1)
        for name in ("S_ion", "iz_rate", "cx_rate"):
            fields[name][plasma_indices] = atomic[name]

    shaped_fields: Dict[str, np.ndarray] = {}
    for name in PRETAB_FIELD_NAMES:
        shaped_fields[name] = fields[name].reshape(n_z, n_r)

    build_seconds = time.perf_counter() - t0
    return PretabulatedGrid(
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
        n_r=n_r,
        n_z=n_z,
        r_coords=r_coords,
        z_coords=z_coords,
        mask=mask,
        fields=shaped_fields,
        build_seconds=build_seconds,
    )


def _exact_evaluators(solution: Any) -> Dict[str, Callable[[float, float], float]]:
    pw_plasma = solution.pointwise.plasma
    pw_sources = solution.pointwise.sources

    return {
        "n": lambda r, z: _as_scalar(pw_plasma.n(r, z)),
        "ti": lambda r, z: _as_scalar(pw_plasma.ti(r, z)),
        "te": lambda r, z: _as_scalar(pw_plasma.te(r, z)),
        "nn": lambda r, z: _as_scalar(pw_plasma.nn(r, z)),
        "S_ion": lambda r, z: _as_scalar(pw_sources.ionization_source(r, z)),
        "iz_rate": lambda r, z: _as_scalar(pw_sources.ionization_rate(r, z)),
        "cx_rate": lambda r, z: _as_scalar(pw_sources.cx_rate(r, z)),
    }


def _random_plasma_points(
    solution: Any,
    extent: Dict[str, float],
    n_samples: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
    locator = solution.mesh.geometry.element_locator
    r_vals: List[float] = []
    z_vals: List[float] = []
    attempts = 0
    while len(r_vals) < n_samples and attempts < max_attempts:
        attempts += 1
        r = rng.uniform(extent["r_min"], extent["r_max"])
        z = rng.uniform(extent["z_min"], extent["z_max"])
        if int(locator(r, z)) >= 0:
            r_vals.append(r)
            z_vals.append(z)
    if len(r_vals) < n_samples:
        raise RuntimeError(
            f"Could only sample {len(r_vals)} plasma points out of {n_samples} requested"
        )
    return np.asarray(r_vals, dtype=float), np.asarray(z_vals, dtype=float)


def compare_grid_to_hdg(
    solution: Any,
    grid: PretabulatedGrid,
    *,
    n_random: int = 1000,
    seed: int = 42,
    fields: Iterable[str] = ("n", "ti", "te", "nn", "S_ion"),
    rel_floor: float = 1e-30,
) -> GridErrorReport:
    """
    Compare bilinear grid lookup to exact HDG evaluation at random plasma points.
    """
    from adjoint_mc.fields.grid_interp import bilinear_sample_array, relative_error

    ensure_hdg_interpolators(solution)
    exact = _exact_evaluators(solution)
    extent = mesh_extent(solution)
    rng = np.random.default_rng(seed)
    r_vals, z_vals = _random_plasma_points(solution, extent, n_random, rng)

    by_field: Dict[str, FieldErrorStats] = {}
    for name in fields:
        if name not in grid.fields:
            continue
        approx = bilinear_sample_array(grid, r_vals, z_vals, name)
        rel_errors = []
        for r, z, approx_val in zip(r_vals, z_vals, approx):
            exact_val = exact[name](float(r), float(z))
            rel = relative_error(exact_val, float(approx_val), floor=rel_floor)
            if np.isfinite(rel):
                rel_errors.append(rel)
        if not rel_errors:
            stats = FieldErrorStats(name, float("nan"), float("nan"), float("nan"), 0)
        else:
            arr = np.asarray(rel_errors, dtype=float)
            stats = FieldErrorStats(
                name=name,
                median_rel=float(np.median(arr)),
                p95_rel=float(np.percentile(arr, 95)),
                max_rel=float(np.max(arr)),
                n_samples=int(arr.size),
            )
        by_field[name] = stats

    return GridErrorReport(
        n_r=grid.n_r,
        n_z=grid.n_z,
        coverage_fraction=grid.coverage_fraction,
        build_seconds=grid.build_seconds,
        by_field=by_field,
    )
