"""Single (R, Z) target seeding for AMCIS wall→point provenance."""

from __future__ import annotations

import math

import numpy as np

from adjoint_mc.fields.grid_interp import in_plasma
from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.sampling.seeds import IonizationSeed


def sample_point_seeds(
    grid: PretabulatedGrid,
    target_r: float,
    target_z: float,
    n_histories: int,
    rng: np.random.Generator,
) -> list[IonizationSeed]:
    """
    Draw ``n_histories`` birth states at a fixed poloidal target (R, Z).

    Each history gets a uniform toroidal angle phi; the 3D position lies on the
    ring at (target_r, target_z). Used by AMCIS to estimate wall visibility to one
    plasma location.
    """
    if n_histories < 1:
        raise ValueError("n_histories must be >= 1")
    target_r = float(target_r)
    target_z = float(target_z)
    if not in_plasma(grid, target_r, target_z):
        raise ValueError(
            f"Target ({target_r:.4f}, {target_z:.4f}) m is outside the plasma grid mask"
        )

    seeds: list[IonizationSeed] = []
    for _ in range(n_histories):
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        x = target_r * math.cos(phi)
        y = target_r * math.sin(phi)
        seeds.append(
            IonizationSeed(
                position=(x, y, target_z),
                r=target_r,
                z=target_z,
                phi=phi,
                cell_i=-1,
                cell_j=-1,
                source_weight=1.0,
            )
        )
    return seeds
