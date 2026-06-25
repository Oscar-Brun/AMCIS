"""Seed sampling for adjoint MC histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adjoint_mc.config import DEFAULT_MAX_SEED_REJECTION_ATTEMPTS
from adjoint_mc.fields.grid_interp import grid_spacing, in_plasma
from adjoint_mc.fields.pretabulate import PretabulatedGrid


@dataclass(frozen=True)
class IonizationSeed:
    """One history birth state in 3D Cartesian coordinates."""

    position: tuple[float, float, float]
    r: float
    z: float
    phi: float
    cell_i: int
    cell_j: int
    source_weight: float


def _cell_source_weights(
    grid: PretabulatedGrid,
    *,
    birth_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat birth weights, cell_i indices, cell_j indices."""
    if birth_mask is not None and birth_mask.shape != grid.mask.shape:
        raise ValueError("birth_mask must match grid.mask shape")

    dr, dz = grid_spacing(grid)
    s_ion = grid.fields["S_ion"]
    weights: list[float] = []
    i_idx: list[int] = []
    j_idx: list[int] = []

    for j in range(grid.n_z):
        for i in range(grid.n_r):
            if not grid.mask[j, i]:
                continue
            if birth_mask is not None and not birth_mask[j, i]:
                continue
            value = float(s_ion[j, i])
            if not np.isfinite(value) or value <= 0.0:
                continue
            r_center = float(grid.r_coords[i])
            dV = 2.0 * np.pi * r_center * dr * dz
            weights.append(value * dV)
            i_idx.append(i)
            j_idx.append(j)

    if not weights:
        raise ValueError("No positive S_ion cells in the selected birth region")

    w_arr = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(w_arr))
    if total <= 0.0:
        raise ValueError("Total ionization source weight is zero")
    return w_arr / total, np.asarray(i_idx, dtype=np.int32), np.asarray(j_idx, dtype=np.int32)


def _draw_seed_candidate(
    grid: PretabulatedGrid,
    probs: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    s_ion: np.ndarray,
    dr: float,
    dz: float,
    rng: np.random.Generator,
) -> IonizationSeed:
    choice = int(rng.choice(len(probs), p=probs))
    i = int(i_idx[choice])
    j = int(j_idx[choice])
    r = float(grid.r_coords[i] + rng.uniform(-0.5, 0.5) * dr)
    z = float(grid.z_coords[j] + rng.uniform(-0.5, 0.5) * dz)
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return IonizationSeed(
        position=(x, y, z),
        r=r,
        z=z,
        phi=phi,
        cell_i=i,
        cell_j=j,
        source_weight=float(s_ion[j, i]),
    )


def sample_ionization_seeds(
    grid: PretabulatedGrid,
    n_histories: int,
    rng: np.random.Generator,
    *,
    birth_mask: np.ndarray | None = None,
    max_rejection_attempts: int = DEFAULT_MAX_SEED_REJECTION_ATTEMPTS,
) -> list[IonizationSeed]:
    """
    Draw birth positions with probability proportional to S_ion * dV on the grid.

    Positions are uniform within each chosen cell; phi is uniform in [0, 2 pi).
    Candidates with ``n_e <= 0`` at the drawn point are rejected and resampled.

    When ``birth_mask`` is set, only masked plasma cells are used (e.g. core
    inside the separatrix).
    """
    if n_histories < 1:
        raise ValueError("n_histories must be >= 1")
    if max_rejection_attempts < 1:
        raise ValueError("max_rejection_attempts must be >= 1")

    probs, i_idx, j_idx = _cell_source_weights(grid, birth_mask=birth_mask)
    dr, dz = grid_spacing(grid)
    s_ion = grid.fields["S_ion"]
    seeds: list[IonizationSeed] = []

    for _ in range(n_histories):
        for attempt in range(max_rejection_attempts):
            candidate = _draw_seed_candidate(grid, probs, i_idx, j_idx, s_ion, dr, dz, rng)
            if in_plasma(grid, candidate.r, candidate.z):
                seeds.append(candidate)
                break
        else:
            raise RuntimeError(
                "Failed to sample an in-plasma ionization seed after "
                f"{max_rejection_attempts} attempts; check S_ion / n_e consistency on the grid"
            )

    return seeds
