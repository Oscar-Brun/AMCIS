"""Tests for SOLEDGE wall flux mapping onto MC wall segments."""

from __future__ import annotations

import numpy as np
import pytest

from adjoint_mc.geometry.wall import make_synthetic_wall
from adjoint_mc.io.wall_flux import (
    _map_boundary_variable_to_segments,
    effective_boundary_emission_flux,
)


def test_effective_boundary_emission_flux_adds_puff() -> None:
    neutral = np.array([0.0, 1.0, -2.0])
    puff = np.array([5.0, 0.0, 0.0])
    out = effective_boundary_emission_flux(neutral, puff)
    assert out[0] == 5.0
    assert out[1] == 1.0
    assert out[2] == 2.0


def test_region_aware_mapping_keeps_puff_flux_on_puff_segment() -> None:
    # Two disjoint wall chains: puff (region 2) vs main wall (region 1).
    wall = make_synthetic_wall(
        [
            (2.95, -0.05, 2.95, 0.05, "puff"),
            (2.50, -0.80, 2.50, 0.80, "main_wall"),
        ],
        region_id=1,
        region_code=50,
    )
    wall.segments[0] = wall.segments[0].__class__(
        r0=2.95,
        z0=-0.05,
        r1=2.95,
        z1=0.05,
        region_id=2,
        region_code=56,
        region_name="puff",
        segment_index=0,
    )

    r_pts = np.array([2.95, 2.50])
    z_pts = np.array([0.0, 0.0])
    ds_pts = np.array([0.01, 0.01])
    values = np.array([100.0, 1.0])
    flags = np.array([2, 1])

    mapped = _map_boundary_variable_to_segments(
        wall,
        r_pts,
        z_pts,
        values,
        ds_pts,
        boundary_flags_pts=flags,
        abs_value=False,
    )
    ring = 2.0 * np.pi * 2.95 * 0.01
    assert mapped[0] == pytest.approx(100.0 * ring)
    assert mapped[1] == pytest.approx(1.0 * 2.0 * np.pi * 2.50 * 0.01)


def test_global_nearest_neighbor_would_mis_assign_puff_without_region_filter() -> None:
    wall = make_synthetic_wall(
        [
            (2.95, -0.05, 2.95, 0.05, "puff"),
            (2.96, -0.80, 2.96, 0.80, "main_wall"),
        ],
    )
    wall.segments[0] = wall.segments[0].__class__(
        r0=2.95,
        z0=-0.05,
        r1=2.95,
        z1=0.05,
        region_id=2,
        region_code=56,
        region_name="puff",
        segment_index=0,
    )

    r_pts = np.array([2.95])
    z_pts = np.array([0.0])
    ds_pts = np.array([0.01])
    values = np.array([50.0])

    without_regions = _map_boundary_variable_to_segments(
        wall, r_pts, z_pts, values, ds_pts, boundary_flags_pts=None, abs_value=False
    )
    with_regions = _map_boundary_variable_to_segments(
        wall, r_pts, z_pts, values, ds_pts, boundary_flags_pts=np.array([2]), abs_value=False
    )
    assert without_regions[0] > 0.0 or without_regions[1] > 0.0
    assert with_regions[0] == pytest.approx(50.0 * 2.0 * np.pi * 2.95 * 0.01)
    assert with_regions[1] == 0.0
