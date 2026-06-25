"""Tests — Python vs Cython equivalence and speed."""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import make_synthetic_wall
from adjoint_mc.tracker.backward_cython import (
    compare_tally_relative_errors,
    cython_available,
    run_backward_full_mc_cython,
)
from adjoint_mc.tracker.backward_full import BackwardFullConfig, run_backward_full_mc

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("hdg_postprocess") is None,
    reason="HDG_postprocess not installed",
)

cython_ext = pytest.mark.skipif(not cython_available(), reason="Cython extension not built")


def _synthetic_grid() -> PretabulatedGrid:
    n_r, n_z = 14, 10
    r_coords = np.linspace(0.5, 0.72, n_r)
    z_coords = np.linspace(-0.1, 0.1, n_z)
    mask = np.ones((n_z, n_r), dtype=bool)
    mask[:, :2] = False
    fields = {
        "n": np.full((n_z, n_r), 1.0e19),
        "ti": np.full((n_z, n_r), 50.0),
        "te": np.full((n_z, n_r), 50.0),
        "nn": np.full((n_z, n_r), 1.0e16),
        "S_ion": np.zeros((n_z, n_r)),
        "iz_rate": np.full((n_z, n_r), 1.0e-14),
        "cx_rate": np.zeros((n_z, n_r)),
    }
    fields["S_ion"][mask] = 1.0
    return PretabulatedGrid(
        r_min=0.5,
        r_max=0.72,
        z_min=-0.1,
        z_max=0.1,
        n_r=n_r,
        n_z=n_z,
        r_coords=r_coords,
        z_coords=z_coords,
        mask=mask,
        fields=fields,
    )


@cython_ext
def test_cython_faster_than_python_synthetic() -> None:
    grid = _synthetic_grid()
    wall = make_synthetic_wall([(0.715, -0.2, 0.715, 0.2, "wall")])
    cfg = BackwardFullConfig(tau_max=0.2, max_path_m=1.0, max_step_m=0.05)
    n = 800

    # Warm up extension and JIT-ish caches before timing.
    run_backward_full_mc(grid, wall, n_histories=20, seed=11, config=cfg)
    run_backward_full_mc_cython(
        grid, wall, n_histories=20, seed=11, config=cfg, enable_cx=False, cx_rejection=False, n_threads=1
    )

    t0 = time.perf_counter()
    run_backward_full_mc(grid, wall, n_histories=n, seed=11, config=cfg)
    py_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_backward_full_mc_cython(
        grid, wall, n_histories=n, seed=11, config=cfg, enable_cx=False, cx_rejection=False, n_threads=1
    )
    cy_s = time.perf_counter() - t0

    assert py_s / max(cy_s, 1e-9) >= 2.0


@cython_ext
def test_cython_tallies_reasonable_vs_python() -> None:
    grid = _synthetic_grid()
    wall = make_synthetic_wall([(0.715, -0.2, 0.715, 0.2, "wall")])
    cfg = BackwardFullConfig(tau_max=0.2, max_path_m=1.0, max_step_m=0.05)
    n = 200

    py = run_backward_full_mc(grid, wall, n_histories=n, seed=3, config=cfg)
    cy = run_backward_full_mc_cython(
        grid, wall, n_histories=n, seed=3, config=cfg, enable_cx=False, cx_rejection=False, n_threads=1
    )

    equiv = compare_tally_relative_errors(py, cy, rtol=0.25)
    assert abs(py.tallies.wall_fraction - cy.tallies.wall_fraction) < 0.15
    assert py.tallies.n_wall > 0
    assert cy.tallies.n_wall > 0


@cython_ext
def test_cython_weights_finite() -> None:
    grid = _synthetic_grid()
    wall = make_synthetic_wall([(0.715, -0.2, 0.715, 0.2, "wall")])
    result = run_backward_full_mc_cython(grid, wall, n_histories=30, seed=1)
    for score in result.scores:
        assert np.isfinite(score.weight)
        assert score.weight >= 0.0
