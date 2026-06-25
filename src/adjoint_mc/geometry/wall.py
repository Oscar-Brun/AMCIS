"""
Axisymmetric wall geometry and 3D ray intersection.

The poloidal wall contour is a polyline in (R, Z). The physical wall is the
surface of revolution obtained by rotating that contour around the vertical axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

# Generic SOLEDGE boundary-condition codes (same on all machines).
_GENERIC_BOUNDARY_CODE_NAMES: dict[int, str] = {
    0: "none",
    1: "dirichlet",
    2: "neumann",
    3: "symmetry",
}

# Machine-specific labels for scoring regions (same physics code, different geometry).
_MACHINE_REGION_CODE_NAMES: dict[str, dict[int, str]] = {
    # TCABR: cylindrical limiter + gas puff + cryo pump (no divertor).
    "tcabr": {
        50: "wall",
        55: "pump",
        56: "puff",
    },
    # WEST: SOLEDGE BC 56 is the gas puff (HDG gamma_puff_wall); divertor tiles use 50.
    "west": {
        50: "main_wall",
        55: "inner_divertor",
        56: "puff",
    },
}


@dataclass(frozen=True)
class WallSegment:
    """One straight poloidal edge of the wall polyline."""

    r0: float
    z0: float
    r1: float
    z1: float
    region_id: int
    region_code: int
    region_name: str
    segment_index: int

    @property
    def length(self) -> float:
        return float(math.hypot(self.r1 - self.r0, self.z1 - self.z0))


@dataclass
class WallGeometry:
    """Collection of poloidal wall segments (SI metres, R–Z plane)."""

    segments: List[WallSegment] = field(default_factory=list)
    length_scale_m: float = 1.0
    machine: str = "tcabr"

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def region_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for seg in self.segments:
            counts[seg.region_name] = counts.get(seg.region_name, 0) + 1
        return counts


@dataclass(frozen=True)
class RayHit:
    """Ray intersection with the axisymmetric wall."""

    t: float
    position: Tuple[float, float, float]
    r: float
    z: float
    segment_index: int
    region_id: int
    region_code: int
    region_name: str
    u: float


def infer_machine(solution: Any = None, *, solution_path: str | None = None) -> str:
    """Guess machine from the solution path (TCABR vs WEST). Defaults to TCABR."""
    candidates: list[str] = []
    if solution_path:
        candidates.append(str(solution_path))
    if solution is not None:
        for attr in ("path", "filepath", "filename"):
            value = getattr(solution, attr, None)
            if value:
                candidates.append(str(value))
        metadata = getattr(solution, "metadata", None)
        if metadata is not None:
            for attr in ("path", "filepath", "filename"):
                value = getattr(metadata, attr, None)
                if value:
                    candidates.append(str(value))
    blob = " ".join(candidates).upper()
    if "WEST" in blob:
        return "west"
    if "TCABR" in blob:
        return "tcabr"
    return "tcabr"


def default_region_name(region_id: int, region_code: int, *, machine: str = "tcabr") -> str:
    """Human-readable label from mesh boundary type and physics code."""
    machine_key = machine.lower()
    machine_map = _MACHINE_REGION_CODE_NAMES.get(machine_key, {})
    if region_code in machine_map:
        return machine_map[region_code]
    if region_code in _GENERIC_BOUNDARY_CODE_NAMES:
        return _GENERIC_BOUNDARY_CODE_NAMES[region_code]
    return f"boundary_{region_id}"


def _boundary_code_map(solution: Any) -> dict[int, int]:
    flags = np.asarray(solution.parameters["physics"].get("boundary_flags", []), dtype=int).reshape(-1)
    mapping: dict[int, int] = {}
    for index, code in enumerate(flags, start=1):
        mapping[index] = int(code)
    return mapping


def _ensure_mesh_boundary(solution: Any) -> None:
    mesh = solution.mesh
    if not mesh.metadata.flags.combined_to_full:
        mesh.assembly.full()
    if not mesh.metadata.flags.boundary_combined:
        mesh.assembly.boundary(solution.raw.boundary_infos)


def _chain_face_rows(rows: Iterable[Tuple[int, np.ndarray]]) -> List[Tuple[int, List[int]]]:
    """Concatenate boundary face node rows into ordered node chains per region."""
    chains: List[Tuple[int, List[int]]] = []
    current_region: int | None = None
    nodes: List[int] = []

    def flush() -> None:
        nonlocal nodes, current_region
        if nodes:
            chains.append((int(current_region), nodes))
        nodes = []

    for region_id, row in rows:
        row_nodes = [int(v) for v in np.asarray(row).reshape(-1).tolist()]
        if current_region != region_id:
            flush()
            current_region = region_id
            nodes = row_nodes
            continue
        if row_nodes[0] == nodes[-1]:
            nodes.extend(row_nodes[1:])
        elif row_nodes[-1] == nodes[0]:
            nodes = row_nodes[:-1] + nodes
        else:
            flush()
            current_region = region_id
            nodes = row_nodes
    flush()
    return chains


def extract_wall_geometry(
    solution: Any,
    *,
    machine: str | None = None,
    solution_path: str | None = None,
) -> WallGeometry:
    """
    Build labelled poloidal wall segments from an HDG solution mesh.

    Vertex coordinates are already in SI metres (R, Z) in the TCABR layout.
    Region labels depend on the machine (TCABR: wall / puff / pump).
    """
    machine_key = (machine or infer_machine(solution, solution_path=solution_path)).lower()
    _ensure_mesh_boundary(solution)
    mesh = solution.mesh
    code_map = _boundary_code_map(solution)
    length_scale = float(solution.parameters["adimensionalization"]["length_scale"])
    vertices = np.asarray(mesh.global_state.vertices, dtype=float)

    boundaries = tuple(sorted(int(k) for k in mesh.boundary_state.flags.keys() if int(k) > 0))
    if not boundaries:
        return WallGeometry(length_scale_m=length_scale, machine=machine_key)

    from hdg_postprocess.core.mesh.boundary import boundary_ordering

    ordering, _conn_ordered, _iel = boundary_ordering(mesh, solution.raw.boundary_infos, boundaries)
    ordered_rows: List[Tuple[int, np.ndarray]] = []
    for bound_index, component_index in ordering:
        region_id = int(boundaries[bound_index])
        ordered_rows.append((region_id, mesh.boundary_state.connectivity[region_id][component_index]))

    segments: List[WallSegment] = []
    seg_index = 0
    for region_id, chain_nodes in _chain_face_rows(ordered_rows):
        region_code = int(code_map.get(region_id, -1))
        region_name = default_region_name(region_id, region_code, machine=machine_key)
        pts = vertices[np.asarray(chain_nodes, dtype=int)]
        for i in range(len(pts) - 1):
            r0, z0 = float(pts[i, 0]), float(pts[i, 1])
            r1, z1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
            segments.append(
                WallSegment(
                    r0=r0,
                    z0=z0,
                    r1=r1,
                    z1=z1,
                    region_id=region_id,
                    region_code=region_code,
                    region_name=region_name,
                    segment_index=seg_index,
                )
            )
            seg_index += 1

    return WallGeometry(segments=segments, length_scale_m=length_scale, machine=machine_key)


def make_synthetic_wall(
    segments_spec: Sequence[tuple[float, float, float, float, str]],
    *,
    region_id: int = 1,
    region_code: int = 50,
) -> WallGeometry:
    """Build a simple wall from (r0, z0, r1, z1, name) tuples — for unit tests."""
    segments: List[WallSegment] = []
    for index, (r0, z0, r1, z1, name) in enumerate(segments_spec):
        segments.append(
            WallSegment(
                r0=r0,
                z0=z0,
                r1=r1,
                z1=z1,
                region_id=region_id,
                region_code=region_code,
                region_name=name,
                segment_index=index,
            )
        )
    return WallGeometry(segments=segments, length_scale_m=1.0)


def _ray_rz(origin: np.ndarray, direction: np.ndarray, t: float) -> Tuple[float, float]:
    x = origin[0] + t * direction[0]
    y = origin[1] + t * direction[1]
    z = origin[2] + t * direction[2]
    return float(math.hypot(x, y)), float(z)


def _solve_ray_radius(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    target_r: float,
    *,
    t_min: float,
    tol: float = 1e-15,
) -> List[float]:
    """Positive ray parameters where sqrt((ox+t*dx)^2 + (oy+t*dy)^2) = target_r."""
    if target_r < 0.0:
        return []
    a = dx * dx + dy * dy
    if a < tol:
        if abs(math.hypot(ox, oy) - target_r) <= 1e-9:
            return [t_min]
        return []
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - target_r * target_r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    sqrt_disc = math.sqrt(max(disc, 0.0))
    hits: List[float] = []
    for sign in (-1.0, 1.0):
        t_val = (-b + sign * sqrt_disc) / (2.0 * a)
        if t_val >= t_min:
            hits.append(t_val)
    return sorted(set(hits))


def _register_hit(
    best: RayHit | None,
    origin: np.ndarray,
    direction: np.ndarray,
    segment: WallSegment,
    t_hit: float,
    u_hit: float,
) -> RayHit | None:
    rr, zz = _ray_rz(origin, direction, t_hit)
    zt = segment.z0 + u_hit * (segment.z1 - segment.z0)
    rt = segment.r0 + u_hit * (segment.r1 - segment.r0)
    if abs(zz - zt) > 1e-6 or abs(rr - rt) > 1e-6:
        return best
    hit = RayHit(
        t=t_hit,
        position=(
            float(origin[0] + t_hit * direction[0]),
            float(origin[1] + t_hit * direction[1]),
            float(origin[2] + t_hit * direction[2]),
        ),
        r=rr,
        z=zz,
        segment_index=segment.segment_index,
        region_id=segment.region_id,
        region_code=segment.region_code,
        region_name=segment.region_name,
        u=float(u_hit),
    )
    if best is None or hit.t < best.t:
        return hit
    return best


def _intersect_ray_segment(
    origin: np.ndarray,
    direction: np.ndarray,
    segment: WallSegment,
    *,
    t_min: float = 1e-9,
    tol: float = 1e-9,
) -> RayHit | None:
    """
    Intersect a 3D ray with the surface of revolution of one poloidal segment.

    Unknowns: ray parameter t >= t_min and segment parameter u in [0, 1].
    """
    r0, z0, r1, z1 = segment.r0, segment.z0, segment.r1, segment.z1
    dr = r1 - r0
    dz = z1 - z0
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    dx, dy, dz_ray = float(direction[0]), float(direction[1]), float(direction[2])
    best: RayHit | None = None

    if abs(dz_ray) <= tol:
        if abs(dz) <= tol:
            if abs(oz - z0) > 1e-6:
                return None
            for i in range(33):
                u_hit = i / 32.0
                rt = r0 + u_hit * dr
                for t_hit in _solve_ray_radius(ox, oy, dx, dy, rt, t_min=t_min, tol=tol):
                    best = _register_hit(best, origin, direction, segment, t_hit, u_hit)
            return best

        u_hit = (oz - z0) / dz
        if u_hit < -1e-9 or u_hit > 1.0 + 1e-9:
            return None
        u_hit = min(max(u_hit, 0.0), 1.0)
        rt = r0 + u_hit * dr
        for t_hit in _solve_ray_radius(ox, oy, dx, dy, rt, t_min=t_min, tol=tol):
            best = _register_hit(best, origin, direction, segment, t_hit, u_hit)
        return best

    def residual(u: float) -> Tuple[float | None, float]:
        zt = z0 + u * dz
        rt = r0 + u * dr
        t_val = (zt - oz) / dz_ray
        if t_val < t_min:
            return None, 0.0
        rr = math.hypot(ox + t_val * dx, oy + t_val * dy)
        return t_val, rr - rt

    n_scan = 32
    for i in range(n_scan):
        u_a = i / n_scan
        u_b = (i + 1) / n_scan
        t_a, res_a = residual(u_a)
        t_b, res_b = residual(u_b)
        if t_a is None and t_b is None:
            continue
        if t_a is None or t_b is None:
            continue
        if res_a == 0.0 or res_b == 0.0 or res_a * res_b < 0.0:
            lo, hi = u_a, u_b
            u_hit = 0.5 * (lo + hi)
            t_hit: float | None = None
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                t_mid, res_mid = residual(mid)
                if t_mid is None:
                    break
                if abs(res_mid) < 1e-10:
                    t_hit = t_mid
                    u_hit = mid
                    break
                _, res_lo = residual(lo)
                if res_lo is None:
                    break
                if res_lo * res_mid <= 0.0:
                    hi = mid
                else:
                    lo = mid
            else:
                u_hit = 0.5 * (lo + hi)
                t_hit, _ = residual(u_hit)
            if t_hit is None:
                continue
            best = _register_hit(best, origin, direction, segment, t_hit, u_hit)
    return best


def _segment_bbox_overlaps_ray(
    segment: WallSegment,
    r_lo: float,
    r_hi: float,
    z_lo: float,
    z_hi: float,
    *,
    margin: float = 0.02,
) -> bool:
    seg_r_lo = min(segment.r0, segment.r1) - margin
    seg_r_hi = max(segment.r0, segment.r1) + margin
    seg_z_lo = min(segment.z0, segment.z1) - margin
    seg_z_hi = max(segment.z0, segment.z1) + margin
    return not (seg_r_hi < r_lo or seg_r_lo > r_hi or seg_z_hi < z_lo or seg_z_lo > z_hi)


def _ray_rz_bounds(
    origin: np.ndarray,
    direction: np.ndarray,
    t_min: float,
    t_max: float,
) -> Tuple[float, float, float, float]:
    rs: List[float] = []
    zs: List[float] = []
    for t in (t_min, t_max):
        r, z = _ray_rz(origin, direction, t)
        rs.append(r)
        zs.append(z)
    return min(rs), max(rs), min(zs), max(zs)


def intersect_ray(
    origin: Sequence[float],
    direction: Sequence[float],
    wall: WallGeometry,
    *,
    t_min: float = 1e-9,
    t_max: float | None = None,
) -> RayHit | None:
    """Return the closest forward intersection along a normalised 3D direction."""
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm < 1e-15:
        raise ValueError("direction must be non-zero")
    d = d / norm

    t_upper = t_max if t_max is not None else None
    if t_upper is not None:
        r_lo, r_hi, z_lo, z_hi = _ray_rz_bounds(o, d, t_min, t_upper)
    else:
        r_lo = z_lo = float("-inf")
        r_hi = z_hi = float("inf")

    best: RayHit | None = None
    for segment in wall.segments:
        if t_upper is not None and not _segment_bbox_overlaps_ray(segment, r_lo, r_hi, z_lo, z_hi):
            continue
        hit = _intersect_ray_segment(o, d, segment, t_min=t_min)
        if hit is None:
            continue
        if t_upper is not None and hit.t > t_upper:
            continue
        if best is None or hit.t < best.t:
            best = hit
    return best
