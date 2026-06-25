"""Plasma field grids for fast MC lookup."""

from adjoint_mc.fields.grid_interp import bilinear_sample, relative_error
from adjoint_mc.fields.pretabulate import (
    PRETAB_FIELD_NAMES,
    GridErrorReport,
    PretabulatedGrid,
    build_pretabulated_grid,
    compare_grid_to_hdg,
    ensure_hdg_interpolators,
    mesh_extent,
)

__all__ = [
    "PRETAB_FIELD_NAMES",
    "GridErrorReport",
    "PretabulatedGrid",
    "build_pretabulated_grid",
    "compare_grid_to_hdg",
    "ensure_hdg_interpolators",
    "mesh_extent",
    "bilinear_sample",
    "relative_error",
]
