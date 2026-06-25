"""Separatrix overlay from SOLEDGE-HDG poloidal flux (HDG_postprocess)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path
from matplotlib.tri import Triangulation

DEFAULT_SEPARATRIX_PSI_LEVEL = 1.0

_CONTOUR_CACHE: dict[tuple[int, float], SeparatrixContour | None] = {}
_PATH_CACHE: dict[tuple[int, float], tuple[np.ndarray, ...]] = {}


@dataclass(frozen=True)
class SeparatrixContour:
    """Poloidal ψ contour used as separatrix guide."""

    r: np.ndarray
    z: np.ndarray
    psi: np.ndarray
    triangles: np.ndarray
    psi_level: float
    axis_r: float
    axis_z: float


def clear_separatrix_cache() -> None:
    """Drop cached separatrix geometry (e.g. after loading a new solution)."""
    _CONTOUR_CACHE.clear()
    _PATH_CACHE.clear()


def _mesh_triangles(solution: Any) -> np.ndarray:
    from hdg_postprocess.core.solution import preparation as prep_ops

    prep_ops.ensure_connectivity_big(solution)
    connectivity = solution.mesh.derived_geometry.connectivity_big
    if connectivity.ndim != 2 or connectivity.shape[1] < 3:
        raise ValueError("Invalid connectivity_big for separatrix contour")
    return np.asarray(connectivity[:, :3], dtype=np.int32)


def _nodal_poloidal_flux(solution: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Nodal poloidal flux ψ aligned with ``mesh.global_state.vertices``.

    WEST / high-order meshes often expose ``views.simple.equilibrium.poloidal_flux``
    on a reduced node set; in that case fall back to the raw nodal ``magnetic_psi``.
    """
    raw_eq = solution.raw.equilibriums[0]
    vertices = np.asarray(solution.mesh.global_state.vertices, dtype=float)
    n_nodes = vertices.shape[0]
    r = vertices[:, 0]
    z = vertices[:, 1]

    if not solution.metadata.flags.combined_simple_solution:
        solution.assembly.simple()

    simple_psi = getattr(solution.views.simple.equilibrium, "poloidal_flux", None)
    if simple_psi is not None:
        simple_arr = np.asarray(simple_psi, dtype=float).reshape(-1)
        if simple_arr.shape[0] == n_nodes:
            return r, z, simple_arr

    if isinstance(raw_eq, dict) and "poloidal_flux" in raw_eq:
        raw_psi = np.asarray(raw_eq["poloidal_flux"], dtype=float).reshape(-1)
        if raw_psi.shape[0] == n_nodes:
            return r, z, raw_psi

    return None


def _boundary_nodes(triangles: np.ndarray) -> np.ndarray:
    """Mesh vertices on the exterior boundary, detected from single-use edges."""
    edge_count: dict[tuple[int, int], int] = {}
    for i0, i1, i2 in triangles:
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    nodes: set[int] = set()
    for (a, b), count in edge_count.items():
        if count == 1:
            nodes.add(a)
            nodes.add(b)
    return np.fromiter(nodes, dtype=np.int32)


def _min_distance_to_boundary(
    r_query: np.ndarray,
    z_query: np.ndarray,
    r_boundary: np.ndarray,
    z_boundary: np.ndarray,
) -> np.ndarray:
    if r_boundary.size == 0:
        return np.full(r_query.shape, np.inf, dtype=float)
    dr = r_query[:, None] - r_boundary[None, :]
    dz = z_query[:, None] - z_boundary[None, :]
    return np.sqrt(np.min(dr * dr + dz * dz, axis=1))


def _estimate_axis(
    r: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    r_boundary: np.ndarray,
    z_boundary: np.ndarray,
) -> tuple[int, str]:
    """Magnetic axis = ψ extremum best separated from the mesh boundary."""
    finite = np.flatnonzero(np.isfinite(psi))
    if finite.size == 0:
        raise ValueError("No finite poloidal flux values")

    candidates = [
        (int(finite[np.argmin(psi[finite])]), "axis_min_psi"),
        (int(finite[np.argmax(psi[finite])]), "axis_max_psi"),
    ]
    z_mid = float(np.nanmedian(z[finite]))
    candidate_indices = np.asarray([idx for idx, _ in candidates], dtype=np.int32)
    dists = _min_distance_to_boundary(
        r[candidate_indices],
        z[candidate_indices],
        r_boundary,
        z_boundary,
    )

    def score(item: tuple[int, str]) -> float:
        idx, _method = item
        pos = candidate_indices.tolist().index(idx)
        dist = float(dists[pos])
        if not np.isfinite(dist):
            dist = 0.0
        return dist - 0.05 * abs(float(z[idx]) - z_mid)

    return max(candidates, key=score)


def _prepare_separatrix_contour_impl(
    solution: Any,
    *,
    psi_level: float,
) -> SeparatrixContour | None:
    nodal = _nodal_poloidal_flux(solution)
    if nodal is None:
        return None
    r, z, psi = nodal
    triangles = _mesh_triangles(solution)

    boundary = _boundary_nodes(triangles)
    axis_i, _axis_method = _estimate_axis(r, z, psi, r[boundary], z[boundary])

    return SeparatrixContour(
        r=r,
        z=z,
        psi=psi,
        triangles=triangles,
        psi_level=float(psi_level),
        axis_r=float(r[axis_i]),
        axis_z=float(z[axis_i]),
    )


def prepare_separatrix_contour(
    solution: Any,
    *,
    psi_level: float | None = None,
) -> SeparatrixContour | None:
    """
    Build the separatrix as the ψ contour at ``psi_level`` (default 1.0).

    The magnetic axis is estimated only to pick the ψ contour branch that
    passes through the plasma core.
    """
    level = float(DEFAULT_SEPARATRIX_PSI_LEVEL if psi_level is None else psi_level)
    key = (id(solution), level)
    if key not in _CONTOUR_CACHE:
        _CONTOUR_CACHE[key] = _prepare_separatrix_contour_impl(solution, psi_level=level)
    return _CONTOUR_CACHE[key]


def _split_path_components(path: Path) -> list[Path]:
    codes = path.codes
    if codes is None or codes.size <= 1:
        return [path]

    starts = [0]
    for idx, code in enumerate(codes):
        if code == Path.MOVETO and idx > 0:
            starts.append(idx)
    starts.append(codes.size)

    components: list[Path] = []
    for start, stop in zip(starts, starts[1:]):
        if stop - start < 2:
            continue
        components.append(Path(path.vertices[start:stop], path.codes[start:stop]))
    return components or [path]


def _extract_contour_path_components(cs: Any) -> list[Path]:
    raw_paths: list[Path] = []
    get_paths = getattr(cs, "get_paths", None)
    if callable(get_paths):
        raw_paths.extend(get_paths())
    else:
        collections = getattr(cs, "collections", None)
        if collections:
            for collection in collections:
                raw_paths.extend(collection.get_paths())

    components: list[Path] = []
    for path in raw_paths:
        components.extend(_split_path_components(path))
    return components


def _path_arc_length(path: Path) -> float:
    vertices = np.asarray(path.vertices, dtype=float)
    if vertices.shape[0] < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(vertices[:, 0]), np.diff(vertices[:, 1]))))


def _select_separatrix_paths(
    paths: list[Path],
    axis_r: float,
    axis_z: float,
    *,
    min_vertices: int = 4,
) -> list[Path]:
    """
    Keep the main ψ contour branch through the magnetic axis.

    On refined HDG meshes Matplotlib returns many short open segments at one
    level; the separatrix is the longest branch that contains the axis.
    """
    axis = (float(axis_r), float(axis_z))
    candidates: list[Path] = []
    for path in paths:
        if np.asarray(path.vertices, dtype=float).shape[0] < min_vertices:
            continue
        if path.contains_point(axis):
            candidates.append(path)

    if not candidates:
        return []

    return [max(candidates, key=_path_arc_length)]


def _compute_separatrix_path_vertices(contour: SeparatrixContour) -> tuple[np.ndarray, ...]:
    tri = Triangulation(contour.r, contour.z, contour.triangles)
    fig, ax = plt.subplots()
    try:
        cs = ax.tricontour(
            tri,
            contour.psi,
            levels=[contour.psi_level],
            colors=["black"],
            linewidths=[1.0],
        )
        paths = _extract_contour_path_components(cs)
        selected = _select_separatrix_paths(paths, contour.axis_r, contour.axis_z)
    finally:
        plt.close(fig)

    return tuple(np.asarray(path.vertices, dtype=float).copy() for path in selected)


def _separatrix_path_vertices(
    solution: Any,
    contour: SeparatrixContour,
    *,
    psi_level: float,
) -> tuple[np.ndarray, ...]:
    key = (id(solution), psi_level)
    if key not in _PATH_CACHE:
        _PATH_CACHE[key] = _compute_separatrix_path_vertices(contour)
    return _PATH_CACHE[key]


def overlay_separatrix(
    ax: Axes,
    solution: Any,
    *,
    psi_level: float | None = None,
    color: str = "white",
    linewidth: float = 1.6,
    linestyle: str = "-",
    alpha: float = 0.95,
    label: str = "separatrix (ψ=1)",
) -> SeparatrixContour | None:
    """Draw the separatrix as the ψ contour at level 1 (or ``psi_level``)."""
    level = float(DEFAULT_SEPARATRIX_PSI_LEVEL if psi_level is None else psi_level)
    contour = prepare_separatrix_contour(solution, psi_level=level)
    if contour is None:
        return None

    path_vertices = _separatrix_path_vertices(solution, contour, psi_level=level)
    if not path_vertices:
        return None

    for vertices in path_vertices:
        ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=8,
        )
    ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    return contour


def separatrix_path_vertices(
    solution: Any,
    *,
    psi_level: float | None = None,
) -> tuple[np.ndarray, ...]:
    """Poloidal (R, Z) vertices of the separatrix contour at ψ = 1 (or ``psi_level``)."""
    level = float(DEFAULT_SEPARATRIX_PSI_LEVEL if psi_level is None else psi_level)
    contour = prepare_separatrix_contour(solution, psi_level=level)
    if contour is None:
        return ()
    return _separatrix_path_vertices(solution, contour, psi_level=level)


def core_birth_mask_on_grid(
    grid: PretabulatedGrid,
    solution: Any,
    *,
    psi_level: float | None = None,
) -> np.ndarray:
    """
    Boolean mask of plasma grid cells whose centers lie inside the separatrix.

    Used to restrict S_ion-weighted birth positions to the plasma core.
    """
    from adjoint_mc.fields.pretabulate import PretabulatedGrid as _Grid

    if not isinstance(grid, _Grid):
        raise TypeError("grid must be a PretabulatedGrid")

    paths = separatrix_path_vertices(solution, psi_level=psi_level)
    if not paths:
        raise ValueError("Could not build separatrix path for core birth mask")

    rr, zz = np.meshgrid(grid.r_coords, grid.z_coords, indexing="xy")
    points = np.column_stack([rr.ravel(), zz.ravel()])
    inside = np.zeros(points.shape[0], dtype=bool)
    for vertices in paths:
        inside |= Path(np.asarray(vertices, dtype=float)).contains_points(points)

    mask = inside.reshape(grid.n_z, grid.n_r) & grid.mask
    if not np.any(mask):
        raise ValueError("Core birth mask is empty — check separatrix and plasma grid overlap")
    return mask
