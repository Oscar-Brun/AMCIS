"""Tests for robust separatrix contour selection."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

from adjoint_mc.viz.separatrix import (
    DEFAULT_SEPARATRIX_PSI_LEVEL,
    _estimate_axis,
    _extract_contour_path_components,
    _select_separatrix_paths,
)


def test_estimate_axis_accepts_maximum_psi_sign_convention() -> None:
    # Center node is the magnetic axis. With this sign convention psi is maximal
    # there, while the minimum lies on the mesh boundary.
    r = np.array([-1.0, 1.0, 1.0, -1.0, 0.0])
    z = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    psi = np.array([-2.0, -2.0, -2.0, -2.0, 0.0])
    r_boundary = r[:4]
    z_boundary = z[:4]

    axis_i, method = _estimate_axis(r, z, psi, r_boundary, z_boundary)

    assert axis_i == 4
    assert method == "axis_max_psi"


def test_select_separatrix_paths_prefers_longest_branch_through_axis() -> None:
    short_through_axis = Path(np.array([[0.0, -0.5], [0.0, 0.0], [0.0, 0.5]]))
    long_through_axis = Path(
        np.array(
            [
                [-0.9, -0.9],
                [0.9, -0.9],
                [0.9, 0.9],
                [-0.9, 0.9],
                [-0.9, -0.9],
            ]
        )
    )
    spurious = Path(np.array([[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]]))

    selected = _select_separatrix_paths(
        [short_through_axis, long_through_axis, spurious],
        0.0,
        0.0,
    )

    assert selected == [long_through_axis]


def test_extract_contour_path_components_supports_get_paths() -> None:
    closed = Path(
        np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
    )

    class NewContourSet:
        def get_paths(self):
            return [closed]

    assert _extract_contour_path_components(NewContourSet()) == [closed]


def test_extract_contour_path_components_supports_legacy_collections() -> None:
    closed = Path(
        np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
    )

    class LegacyCollection:
        def get_paths(self):
            return [closed]

    class LegacyContourSet:
        collections = [LegacyCollection()]

    assert _extract_contour_path_components(LegacyContourSet()) == [closed]


def test_default_separatrix_psi_level() -> None:
    assert DEFAULT_SEPARATRIX_PSI_LEVEL == 1.0
