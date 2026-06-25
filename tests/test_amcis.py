"""Tests for AMCIS — point seeding, survival weights, provenance."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("hdg_postprocess") is None,
    reason="HDG_postprocess not installed (PretabulatedGrid import)",
)

from adjoint_mc.fields.pretabulate import PretabulatedGrid
from adjoint_mc.geometry.wall import make_synthetic_wall
from adjoint_mc.sampling.point_seed import sample_point_seeds
from adjoint_mc.scoring.amcis_provenance import compute_amcis_provenance
from adjoint_mc.scoring.tallies import HistoryScore, accumulate_wall_scores
from adjoint_mc.io.wall_flux import WallNeutralFluxResult
from adjoint_mc.tracker.amcis_backward import (
    AmcisConfig,
    AmcisMcResult,
    _run_amcis_mc_python,
    run_amcis_mc,
    track_amcis_backward,
)
from adjoint_mc.tracker.amcis_cython import run_amcis_mc_cython
from adjoint_mc.tracker.backward_cython import cython_available


def _synthetic_plasma_grid() -> PretabulatedGrid:
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
        "iz_rate": np.full((n_z, n_r), 1.0e-12),
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


def test_point_seeds_fixed_location() -> None:
    grid = _synthetic_plasma_grid()
    rng = np.random.default_rng(0)
    seeds = sample_point_seeds(grid, 0.65, 0.02, 20, rng)
    assert len(seeds) == 20
    for seed in seeds:
        assert seed.r == pytest.approx(0.65)
        assert seed.z == pytest.approx(0.02)
        assert seed.source_weight == pytest.approx(1.0)


def test_point_seed_outside_plasma_raises() -> None:
    grid = _synthetic_plasma_grid()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="outside"):
        sample_point_seeds(grid, 0.52, 0.0, 1, rng)


def test_survival_weight_bounded() -> None:
    grid = _synthetic_plasma_grid()
    wall = make_synthetic_wall([(0.715, -0.2, 0.715, 0.2, "wall")])
    rng = np.random.default_rng(1)
    seeds = sample_point_seeds(grid, 0.65, 0.0, 1, rng)
    score = track_amcis_backward(
        grid,
        wall,
        seeds[0],
        rng,
        config=AmcisConfig(
            enable_cx=False,
            tau_max=0.2,
            max_path_m=2.0,
            max_step_m=0.05,
            fallback_speed_m_s=1.0e4,
        ),
    )
    assert 0.0 <= score.weight <= 1.0 + 1e-9


def test_amcis_provenance_normalization() -> None:
    wall = make_synthetic_wall(
        [
            (0.715, -0.2, 0.715, -0.05, "puff"),
            (0.715, -0.05, 0.715, 0.2, "wall"),
        ]
    )
    scores = [
        HistoryScore(
            weight=0.8,
            termination="wall",
            region_name="puff",
            segment_index=0,
            seed_r=0.65,
            seed_z=0.0,
        ),
        HistoryScore(
            weight=0.2,
            termination="wall",
            region_name="wall",
            segment_index=1,
            seed_r=0.65,
            seed_z=0.0,
        ),
    ]
    tallies = accumulate_wall_scores(wall, 2, scores)
    mc = AmcisMcResult(
        config=AmcisConfig(),
        target_r=0.65,
        target_z=0.0,
        n_histories=2,
        seed=1,
        tallies=tallies,
        scores=scores,
    )
    prov = compute_amcis_provenance(wall, mc)
    assert prov.segment_probability.sum() == pytest.approx(1.0)
    assert prov.region_probability["puff"] == pytest.approx(0.8)
    assert prov.region_probability["wall"] == pytest.approx(0.2)


def test_amcis_emission_weighted_provenance() -> None:
    wall = make_synthetic_wall(
        [
            (0.715, -0.2, 0.715, -0.05, "puff"),
            (0.715, -0.05, 0.715, 0.2, "wall"),
        ]
    )
    scores = [
        HistoryScore(
            weight=0.8,
            termination="wall",
            region_name="puff",
            segment_index=0,
            seed_r=0.65,
            seed_z=0.0,
        ),
        HistoryScore(
            weight=0.2,
            termination="wall",
            region_name="wall",
            segment_index=1,
            seed_r=0.65,
            seed_z=0.0,
        ),
    ]
    tallies = accumulate_wall_scores(wall, 2, scores)
    mc = AmcisMcResult(
        config=AmcisConfig(),
        target_r=0.65,
        target_z=0.0,
        n_histories=2,
        seed=1,
        tallies=tallies,
        scores=scores,
    )
    wall_flux = WallNeutralFluxResult(
        segment_flux=np.array([10.0, 1.0]),
        segment_parallel_flux=np.zeros(2),
        region_flux={"puff": 10.0, "wall": 1.0},
    )
    prov = compute_amcis_provenance(wall, mc, wall_flux=wall_flux)

    assert prov.has_emission_weighting
    assert prov.segment_emission_weight is not None
    assert prov.segment_emission_probability is not None
    assert prov.segment_attributed_flux is not None

    # D_k = C_k * Gamma_k  ->  [8, 0.2];  f_k^flux = [8/8.2, 0.2/8.2]
    assert prov.segment_emission_weight[0] == pytest.approx(8.0)
    assert prov.segment_emission_weight[1] == pytest.approx(0.2)
    assert prov.segment_emission_probability.sum() == pytest.approx(1.0)
    assert prov.segment_emission_probability[0] == pytest.approx(8.0 / 8.2)
    assert prov.segment_emission_probability[1] == pytest.approx(0.2 / 8.2)

    # Phi_k = f_k(visibility) * Gamma_k  ->  [0.8*10, 0.2*1]
    assert prov.segment_attributed_flux[0] == pytest.approx(8.0)
    assert prov.segment_attributed_flux[1] == pytest.approx(0.2)
    assert prov.total_attributed_flux == pytest.approx(8.2)

    assert prov.region_emission_probability is not None
    assert prov.region_emission_probability["puff"] == pytest.approx(8.0 / 8.2)
    assert prov.region_emission_probability["wall"] == pytest.approx(0.2 / 8.2)


def test_amcis_mc_plot_tabs_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from adjoint_mc.viz.amcis_mc_plots import iter_amcis_mc_plot_tabs

    wall = make_synthetic_wall(
        [
            (0.715, -0.2, 0.715, -0.05, "puff"),
            (0.715, -0.05, 0.715, 0.2, "wall"),
        ]
    )
    scores = [
        HistoryScore(
            weight=0.8,
            termination="wall",
            region_name="puff",
            segment_index=0,
            seed_r=0.65,
            seed_z=0.0,
            hit_r=0.715,
            hit_z=-0.1,
        ),
        HistoryScore(
            weight=0.2,
            termination="wall",
            region_name="wall",
            segment_index=1,
            seed_r=0.65,
            seed_z=0.0,
            hit_r=0.715,
            hit_z=0.1,
        ),
    ]
    tallies = accumulate_wall_scores(wall, 2, scores)
    mc = AmcisMcResult(
        config=AmcisConfig(enable_cx=False),
        target_r=0.65,
        target_z=0.0,
        n_histories=2,
        seed=1,
        tallies=tallies,
        scores=scores,
    )
    tabs = list(iter_amcis_mc_plot_tabs(wall, mc))
    assert len(tabs) == 1
    assert tabs[0][0] == "Wall hits"
    for _title, fig in tabs:
        assert len(fig.axes) >= 1


def test_run_amcis_mc_smoke() -> None:
    grid = _synthetic_plasma_grid()
    wall = make_synthetic_wall([(0.715, -0.15, 0.715, 0.15, "wall")])
    result = run_amcis_mc(
        grid,
        wall,
        target_r=0.65,
        target_z=0.0,
        n_histories=30,
        seed=2,
        config=AmcisConfig(enable_cx=False, max_path_m=1.5, max_step_m=0.05),
        use_cython=False,
    )
    assert result.n_histories == 30
    assert len(result.scores) == 30
    prov = compute_amcis_provenance(wall, result)
    assert prov.target_r == pytest.approx(0.65)


cython_ext = pytest.mark.skipif(not cython_available(), reason="Cython extension not built")


@cython_ext
def test_amcis_cython_matches_python() -> None:
    grid = _synthetic_plasma_grid()
    wall = make_synthetic_wall([(0.715, -0.2, 0.715, 0.2, "wall")])
    cfg = AmcisConfig(enable_cx=False, tau_max=0.2, max_path_m=1.0, max_step_m=0.05)
    n = 120

    py = _run_amcis_mc_python(
        grid,
        wall,
        target_r=0.65,
        target_z=0.0,
        n_histories=n,
        seed=11,
        config=cfg,
    )
    cy = run_amcis_mc_cython(
        grid,
        wall,
        target_r=0.65,
        target_z=0.0,
        n_histories=n,
        seed=11,
        config=cfg,
        sync_numpy_rng=True,
        n_threads=1,
        cx_rejection=False,
    )

    assert abs(py.tallies.wall_fraction - cy.tallies.wall_fraction) < 0.05
    assert py.tallies.n_wall == cy.tallies.n_wall
    for ps, cs in zip(py.scores, cy.scores, strict=True):
        assert ps.termination == cs.termination
        assert ps.weight == pytest.approx(cs.weight, rel=1e-9, abs=1e-12)
        assert ps.segment_index == cs.segment_index


@cython_ext
def test_amcis_cython_survival_weights_bounded() -> None:
    grid = _synthetic_plasma_grid()
    wall = make_synthetic_wall([(0.715, -0.2, 0.715, 0.2, "wall")])
    result = run_amcis_mc(
        grid,
        wall,
        target_r=0.65,
        target_z=0.0,
        n_histories=40,
        seed=3,
        config=AmcisConfig(enable_cx=False, max_path_m=2.0, max_step_m=0.05),
        use_cython=True,
    )
    for score in result.scores:
        assert 0.0 <= score.weight <= 1.0 + 1e-9
