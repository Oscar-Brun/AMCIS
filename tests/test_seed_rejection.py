"""Birth seed rejection when n_e <= 0 at the drawn point."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from adjoint_mc.fields.grid_interp import in_plasma
from adjoint_mc.sampling.seeds import sample_ionization_seeds

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("hdg_postprocess") is None,
    reason="HDG_postprocess not installed",
)

from adjoint_mc.fields.pretabulate import PretabulatedGrid


def _edge_cell_grid() -> PretabulatedGrid:
    """One S_ion cell with n_e>0 at the node but bilinear n_e=0 on part of the cell."""
    n_r, n_z = 8, 6
    r_coords = np.linspace(1.0, 1.07, n_r)
    z_coords = np.linspace(-0.03, 0.03, n_z)
    mask = np.zeros((n_z, n_r), dtype=bool)
    mask[:, 2:6] = True
    n = np.full((n_z, n_r), np.nan)
    n[:, 3:5] = 1.0e19
    n[:, 2] = 0.0
    n[:, 5] = 0.0
    fields = {
        "n": n,
        "ti": np.full((n_z, n_r), 50.0),
        "te": np.full((n_z, n_r), 50.0),
        "nn": np.full((n_z, n_r), 1.0e16),
        "S_ion": np.zeros((n_z, n_r)),
        "iz_rate": np.full((n_z, n_r), 1.0e-14),
        "cx_rate": np.zeros((n_z, n_r)),
    }
    fields["S_ion"][:, 3] = 2.0
    return PretabulatedGrid(
        r_min=float(r_coords[0]),
        r_max=float(r_coords[-1]),
        z_min=float(z_coords[0]),
        z_max=float(z_coords[-1]),
        n_r=n_r,
        n_z=n_z,
        r_coords=r_coords,
        z_coords=z_coords,
        mask=mask,
        fields=fields,
    )


def test_sampled_seeds_are_in_plasma() -> None:
    grid = _edge_cell_grid()
    rng = np.random.default_rng(0)
    seeds = sample_ionization_seeds(grid, 200, rng)
    assert len(seeds) == 200
    for seed in seeds:
        assert in_plasma(grid, seed.r, seed.z)


def test_rejection_occurs_for_edge_cell_grid() -> None:
    grid = _edge_cell_grid()
    rng = np.random.default_rng(1)
    attempts = 0
    accepted = 0
    dr = (grid.r_max - grid.r_min) / (grid.n_r - 1)
    dz = (grid.z_max - grid.z_min) / (grid.n_z - 1)

    for _ in range(100):
        i, j = 3, 0
        r = float(grid.r_coords[i] + rng.uniform(-0.5, 0.5) * dr)
        z = float(grid.z_coords[j] + rng.uniform(-0.5, 0.5) * dz)
        attempts += 1
        if in_plasma(grid, r, z):
            accepted += 1

    assert accepted < attempts
    assert accepted > 0

    seeds = sample_ionization_seeds(grid, 50, rng, max_rejection_attempts=500)
    assert len(seeds) == 50


def test_rejection_raises_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = _edge_cell_grid()
    monkeypatch.setattr("adjoint_mc.sampling.seeds.in_plasma", lambda *_args, **_kwargs: False)
    rng = np.random.default_rng(2)
    with pytest.raises(RuntimeError, match="Failed to sample an in-plasma ionization seed"):
        sample_ionization_seeds(grid, 1, rng, max_rejection_attempts=5)
