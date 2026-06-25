"""Tokamak geometry — wall contour and 3D ray intersection."""

from adjoint_mc.geometry.wall import (
    RayHit,
    WallGeometry,
    WallSegment,
    default_region_name,
    extract_wall_geometry,
    infer_machine,
    intersect_ray,
    make_synthetic_wall,
)

__all__ = [
    "RayHit",
    "WallGeometry",
    "WallSegment",
    "default_region_name",
    "extract_wall_geometry",
    "infer_machine",
    "intersect_ray",
    "make_synthetic_wall",
]
