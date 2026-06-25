"""Bilinear interpolation on uniform (R, Z) grids — Python prototype for Cython."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from adjoint_mc.fields.pretabulate import PretabulatedGrid


def _cell_indices(coords: np.ndarray, value: float) -> int | None:
    if value < coords[0] or value > coords[-1]:
        return None
    if np.isclose(value, coords[-1]):
        return len(coords) - 2
    index = int(np.searchsorted(coords, value, side="right") - 1)
    if index >= len(coords) - 1:
        index = len(coords) - 2
    if index < 0:
        return None
    return index


def bilinear_sample(
    grid: PretabulatedGrid,
    r: float,
    z: float,
    field: str,
) -> float:
    """
    Sample ``field`` at (r, z) by bilinear interpolation on the pre-tabulated grid.

    Returns NaN outside the grid box, outside the plasma mask, or if any corner is masked out.
    """
    values = grid.fields[field]

    node_i = np.where(np.isclose(grid.r_coords, r))[0]
    node_j = np.where(np.isclose(grid.z_coords, z))[0]
    if node_i.size and node_j.size:
        j0, i0 = int(node_j[0]), int(node_i[0])
        if grid.mask[j0, i0]:
            return float(values[j0, i0])

    i = _cell_indices(grid.r_coords, r)
    j = _cell_indices(grid.z_coords, z)
    if i is None or j is None:
        return float("nan")

    corners = grid.mask[j : j + 2, i : i + 2]
    if not np.all(corners):
        return float("nan")

    r0, r1 = grid.r_coords[i], grid.r_coords[i + 1]
    z0, z1 = grid.z_coords[j], grid.z_coords[j + 1]
    tx = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)
    ty = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)

    v00 = values[j, i]
    v10 = values[j, i + 1]
    v01 = values[j + 1, i]
    v11 = values[j + 1, i + 1]
    return float(
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def bilinear_sample_array(
    grid: PretabulatedGrid,
    r_values: np.ndarray,
    z_values: np.ndarray,
    field: str,
) -> np.ndarray:
    """Vectorized bilinear sampling (loop over points; sufficient for validation)."""
    r_flat = np.asarray(r_values, dtype=float).reshape(-1)
    z_flat = np.asarray(z_values, dtype=float).reshape(-1)
    out = np.empty(r_flat.size, dtype=float)
    for index, (r, z) in enumerate(zip(r_flat, z_flat)):
        out[index] = bilinear_sample(grid, float(r), float(z), field)
    return out


def relative_error(exact: float, approx: float, *, floor: float = 1e-30) -> float:
    """Relative error |exact - approx| / max(|exact|, floor)."""
    if not np.isfinite(exact) or not np.isfinite(approx):
        return float("nan")
    return float(abs(exact - approx) / max(abs(exact), floor))


def grid_spacing(grid: PretabulatedGrid) -> tuple[float, float]:
    """Uniform node spacing (dr, dz) on the pre-tabulated grid."""
    dr = (grid.r_max - grid.r_min) / (grid.n_r - 1)
    dz = (grid.z_max - grid.z_min) / (grid.n_z - 1)
    return float(dr), float(dz)


def in_plasma(grid: PretabulatedGrid, r: float, z: float) -> bool:
    """True if (R, Z) lies inside the grid plasma mask with finite density."""
    if r < grid.r_min or r > grid.r_max or z < grid.z_min or z > grid.z_max:
        return False
    n_e = bilinear_sample(grid, r, z, "n")
    return bool(np.isfinite(n_e) and n_e > 0.0)


def macroscopic_ionization_rate(
    grid: PretabulatedGrid,
    r: float,
    z: float,
    speed_m_s: float,
) -> float:
    """Sigma_ion [m^-1] = n_e * <sigma v>_ion / |v| from pre-tabulated fields."""
    if speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")
    n_e = bilinear_sample(grid, r, z, "n")
    iz_rate = bilinear_sample(grid, r, z, "iz_rate")
    if not np.isfinite(n_e) or not np.isfinite(iz_rate):
        return 0.0
    return float(max(0.0, n_e * iz_rate / speed_m_s))


def macroscopic_cx_rate(
    grid: PretabulatedGrid,
    r: float,
    z: float,
    speed_m_s: float,
) -> float:
    """Sigma_cx [m^-1] = n_ion * <sigma v>_CX / |v| from pre-tabulated fields.

    A single tracked neutral charge-exchanges with background ions, so the target
    density is the ion density (n_ion ~ n_e in a hydrogenic plasma). The neutral
    density n_n does NOT enter the collision frequency of one neutral.
    """
    if speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")
    n_e = bilinear_sample(grid, r, z, "n")
    cx_rate = bilinear_sample(grid, r, z, "cx_rate")
    if not all(np.isfinite(x) for x in (n_e, cx_rate)):
        return 0.0
    return float(max(0.0, n_e * cx_rate / speed_m_s))


def local_ti_ev(grid: PretabulatedGrid, r: float, z: float, *, default: float = 10.0) -> float:
    """Ion temperature [eV] at (R, Z), with fallback when masked out."""
    ti = bilinear_sample(grid, r, z, "ti")
    if not np.isfinite(ti) or ti <= 0.0:
        return float(default)
    return float(ti)


def sample_neutral_maxwellian_velocity(
    grid: PretabulatedGrid,
    r: float,
    z: float,
    rng: np.random.Generator,
    *,
    fallback_speed_m_s: float = 1.0e4,
) -> np.ndarray:
    """3D Maxwellian neutral velocity at local T_i (T_n ≡ T_i convention)."""
    from adjoint_mc.atomic.cx_rejection import sample_maxwellian_3d

    ti = local_ti_ev(grid, r, z, default=0.0)
    if ti <= 0.0:
        return _isotropic_velocity(rng, fallback_speed_m_s)
    velocity = sample_maxwellian_3d(ti, rng)
    speed = float(np.linalg.norm(velocity))
    if speed < 1.0e2:
        return _isotropic_velocity(rng, fallback_speed_m_s)
    return velocity


def _isotropic_velocity(rng: np.random.Generator, speed_m_s: float) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-15:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        norm = 1.0
    return direction / norm * speed_m_s


def cell_center_relative_error_map(
    grid: PretabulatedGrid,
    field: str,
    exact_at: Callable[[float, float], float],
    *,
    floor: float = 1e-30,
) -> np.ndarray:
    """
    Relative error at cell centers (staggered grid), shape (n_z - 1, n_r - 1).

    ``exact_at(r, z)`` must return the HDG reference value.
    """
    n_r = grid.n_r
    n_z = grid.n_z
    errors = np.full((n_z - 1, n_r - 1), np.nan, dtype=float)
    for j in range(n_z - 1):
        z_c = 0.5 * (grid.z_coords[j] + grid.z_coords[j + 1])
        for i in range(n_r - 1):
            r_c = 0.5 * (grid.r_coords[i] + grid.r_coords[i + 1])
            if not (
                grid.mask[j, i]
                and grid.mask[j, i + 1]
                and grid.mask[j + 1, i]
                and grid.mask[j + 1, i + 1]
            ):
                continue
            exact = float(exact_at(r_c, z_c))
            approx = bilinear_sample(grid, r_c, z_c, field)
            errors[j, i] = relative_error(exact, approx, floor=floor)
    return errors
