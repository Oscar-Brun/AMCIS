"""Tests for core fueling provenance (S_ion births inside separatrix)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from matplotlib.path import Path

if importlib.util.find_spec("hdg_postprocess") is None:
    pytest.skip(
        "HDG_postprocess not installed (PretabulatedGrid import)",
        allow_module_level=True,
    )

from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.sampling.seeds import sample_ionization_seeds
from adjoint_mc.scoring.provenance import integrated_ionization_rate_in_mask
from adjoint_mc.viz.separatrix import core_birth_mask_on_grid


def _synthetic_plasma_grid(*, n_r: int = 16, n_z: int = 12) -> PretabulatedGrid:
    r_coords = np.linspace(0.5, 0.72, n_r)
    z_coords = np.linspace(-0.12, 0.12, n_z)
    mask = np.ones((n_z, n_r), dtype=bool)
    mask[:, :2] = False
    rr, zz = np.meshgrid(r_coords, z_coords, indexing="xy")
    fields = {
        "n": np.full((n_z, n_r), 1.0e19),
        "ti": np.full((n_z, n_r), 50.0),
        "te": np.full((n_z, n_r), 20.0),
        "nn": np.full((n_z, n_r), 1.0e17),
        "S_ion": np.where((rr - 0.61) ** 2 + zz**2 < 0.015**2, 5.0e20, 1.0e20),
        "iz_rate": np.full((n_z, n_r), 1.0),
        "cx_rate": np.full((n_z, n_r), 0.1),
    }
    return PretabulatedGrid(
        r_coords=r_coords,
        z_coords=z_coords,
        mask=mask,
        fields=fields,
        build_seconds=0.0,
    )


class _FakeSolution:
    """Minimal stub so separatrix_path_vertices can be monkeypatched."""

    pass


def test_core_birth_mask_selects_interior(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = _synthetic_plasma_grid()
    r_center = float(np.mean(grid.r_coords))
    z_center = 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radius = 0.04
    vertices = np.column_stack(
        [r_center + radius * np.cos(theta), z_center + radius * np.sin(theta)]
    )

    monkeypatch.setattr(
        "adjoint_mc.viz.separatrix.separatrix_path_vertices",
        lambda solution, psi_level=None: (vertices,),
    )

    birth_mask = core_birth_mask_on_grid(grid, _FakeSolution())
    assert birth_mask.shape == grid.mask.shape
    assert np.any(birth_mask)
    assert np.all(birth_mask <= grid.mask)

    rr, zz = np.meshgrid(grid.r_coords, grid.z_coords, indexing="xy")
    points = np.column_stack([rr.ravel(), zz.ravel()])
    inside = Path(vertices).contains_points(points).reshape(grid.n_z, grid.n_r)
    assert np.array_equal(birth_mask, inside & grid.mask)


def test_sample_ionization_seeds_respects_birth_mask() -> None:
    grid = _synthetic_plasma_grid()
    birth_mask = np.zeros_like(grid.mask)
    birth_mask[grid.n_z // 2, grid.n_r // 2] = True
    rng = np.random.default_rng(0)
    seeds = sample_ionization_seeds(grid, 50, rng, birth_mask=birth_mask)
    assert len(seeds) == 50
    i_mid = grid.n_r // 2
    j_mid = grid.n_z // 2
    dr = grid.r_coords[1] - grid.r_coords[0]
    dz = grid.z_coords[1] - grid.z_coords[0]
    r0, r1 = grid.r_coords[i_mid], grid.r_coords[i_mid] + dr
    z0, z1 = grid.z_coords[j_mid], grid.z_coords[j_mid] + dz
    for seed in seeds:
        assert r0 <= seed.r <= r1
        assert z0 <= seed.z <= z1


def test_integrated_ionization_rate_in_mask() -> None:
    grid = _synthetic_plasma_grid()
    birth_mask = np.zeros_like(grid.mask)
    birth_mask[5:8, 6:10] = True
    rate = integrated_ionization_rate_in_mask(grid, birth_mask)
    assert rate > 0.0
    full_rate = integrated_ionization_rate_in_mask(grid, grid.mask)
    assert rate < full_rate


def test_compute_provenance_accepts_fueling_rate_override() -> None:
    from adjoint_mc.geometry.wall import make_synthetic_wall
    from adjoint_mc.scoring.provenance import compute_provenance
    from adjoint_mc.scoring.tallies import HistoryScore, accumulate_wall_scores
    from adjoint_mc.tracker.backward_full import BackwardFullConfig, BackwardFullResult

    grid = _synthetic_plasma_grid()
    wall = make_synthetic_wall()
    scores = [
        HistoryScore(
            weight=1.0,
            termination="wall",
            region_name=wall.segments[0].region_name,
            segment_index=0,
            seed_r=0.61,
            seed_z=0.0,
            hit_r=wall.segments[0].r0,
            hit_z=wall.segments[0].z0,
        )
    ]
    tallies = accumulate_wall_scores(wall, len(scores), scores)
    mc_result = BackwardFullResult(
        config=BackwardFullConfig(),
        n_histories=1,
        seed=0,
        tallies=tallies,
        scores=scores,
    )
    core_mask = np.zeros_like(grid.mask)
    core_mask[: grid.n_z // 2] = grid.mask[: grid.n_z // 2]
    core_rate = integrated_ionization_rate_in_mask(grid, core_mask)
    prov = compute_provenance(
        wall,
        mc_result,
        grid,
        fueling_rate_s=core_rate,
    )
    assert prov.fueling_rate_total_s == pytest.approx(core_rate)
