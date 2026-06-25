"""Plasma field visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from hdg_postprocess.core.solution import preparation as prep_ops

_SIGNED_CMAP = "RdBu_r"


@dataclass(frozen=True)
class PhysicalFieldSpec:
    """Physical field description for one GUI tab."""

    key: str
    tab_title: str
    label: str
    log_scale: bool
    signed: bool


def _conservative_name(solution: Any, index: int) -> bytes:
    return solution.parameters["physics"]["conservative_variable_names"][index]


def _spec_for_conservative_index(solution: Any, index: int) -> PhysicalFieldSpec:
    """One GUI/plot spec per conservative index (matches HDG_postprocess layout)."""
    if index == 0:
        return PhysicalFieldSpec("n", "n", r"n [m$^{-3}$]", log_scale=True, signed=False)
    if index == 1:
        return PhysicalFieldSpec("n_n", r"n_n", r"$n_n$ [m$^{-3}$]", log_scale=True, signed=False)
    if index == 2:
        return PhysicalFieldSpec("T_i", r"T_i", r"$T_i$ [eV]", log_scale=True, signed=False)
    if index == 3:
        return PhysicalFieldSpec("T_e", r"T_e", r"$T_e$ [eV]", log_scale=True, signed=False)
    if index == 4:
        return PhysicalFieldSpec("M", "M", "M", log_scale=False, signed=True)

    var_name = _conservative_name(solution, index)
    if var_name == b"k":
        return PhysicalFieldSpec("k", "k", r"$k$ [m$^2$/s$^2$]", log_scale=False, signed=False)
    if var_name == b"Gamman":
        return PhysicalFieldSpec(
            "Gamma_n",
            r"$\Gamma_n$",
            r"$\Gamma_n$ [m$^{-2}$ s$^{-1}$]",
            log_scale=False,
            signed=False,
        )
    if var_name == b"Gammanx":
        return PhysicalFieldSpec(
            "Gamma_nx",
            r"$\Gamma_{nx}$",
            r"$\Gamma_{nx}$ [m$^{-2}$ s$^{-1}$]",
            log_scale=False,
            signed=False,
        )
    if var_name == b"Gammany":
        return PhysicalFieldSpec(
            "Gamma_ny",
            r"$\Gamma_{ny}$",
            r"$\Gamma_{ny}$ [m$^{-2}$ s$^{-1}$]",
            log_scale=False,
            signed=False,
        )
    name = var_name.decode("utf-8", errors="replace")
    return PhysicalFieldSpec(name, name, name, log_scale=False, signed=False)


def physical_field_specs(solution: Any) -> List[PhysicalFieldSpec]:
    """
    Fields shown by physical_overview (same order as HDG_postprocess).

    One entry per conservative variable (``solution.neq``), not always six.
    """
    return [_spec_for_conservative_index(solution, i) for i in range(int(solution.neq))]


def _build_solutions_plot(solution: Any) -> np.ndarray:
    """Nodal fields per conservative index — mirrors ``plot_overview_physical``."""
    prep_ops.ensure_simple_physical(solution)

    simple_phys = solution.views.simple.solution.physical
    simple_cons = solution.views.simple.solution.conservative
    solutions_plot = np.zeros_like(simple_cons)
    indices = solution.metadata.indices.physical
    adim = solution.parameters["adimensionalization"]

    rho_idx = indices.get(b"rho", 0)
    solutions_plot[:, 0] = simple_phys[:, rho_idx]

    rhon_idx = indices.get(b"rhon", -1)
    if rhon_idx >= 0:
        solutions_plot[:, 1] = simple_phys[:, rhon_idx]
    else:
        solutions_plot[:, 1] = simple_phys[:, -1]

    if solution.neq > 2:
        ti_idx = indices.get(b"Ti", -1)
        if ti_idx >= 0:
            solutions_plot[:, 2] = simple_phys[:, ti_idx]
        te_idx = indices.get(b"Te", -1)
        if te_idx >= 0:
            solutions_plot[:, 3] = simple_phys[:, te_idx]

    if solution.neq > 4:
        m_idx = indices.get(b"M", -1)
        if m_idx >= 0:
            solutions_plot[:, 4] = simple_phys[:, m_idx]

    for idx in range(5, solution.neq):
        var_name = _conservative_name(solution, idx)
        if var_name == b"k":
            k_phys_idx = indices.get(b"k", 11)
            solutions_plot[:, idx] = simple_phys[:, k_phys_idx]
        elif var_name in (b"Gamman", b"Gammanx", b"Gammany"):
            solutions_plot[:, idx] = (
                simple_cons[:, idx] * adim["density_scale"] * adim["speed_scale"]
            )
        else:
            solutions_plot[:, idx] = simple_cons[:, idx]

    return solutions_plot


def _physical_field_arrays(solution: Any) -> Dict[str, np.ndarray]:
    """Extract nodal arrays like plot_overview_physical (HDG)."""
    cache = getattr(solution, "_adjoint_mc_physical_field_cache", None)
    if cache is not None:
        return cache

    prep_ops.ensure_connectivity_big(solution)
    solutions_plot = _build_solutions_plot(solution)
    specs = physical_field_specs(solution)
    out: Dict[str, np.ndarray] = {}
    for index, spec in enumerate(specs):
        data = solutions_plot[:, index].copy()
        if spec.signed:
            data[np.isnan(data)] = 0.0
        elif spec.key in ("n", "n_n"):
            data[data < 0] = 1e8
        else:
            data[data < 0] = 1e-3
        out[spec.key] = data
    solution._adjoint_mc_physical_field_cache = out
    return out


def physical_field_array(solution: Any, key: str) -> np.ndarray:
    """Return one nodal physical field array (keys match :func:`physical_field_specs`)."""
    arrays = _physical_field_arrays(solution)
    if key not in arrays:
        raise KeyError(f"Unknown physical field {key!r}")
    return arrays[key]


def neutral_perpendicular_flux_spec(solution: Any) -> PhysicalFieldSpec | None:
    """Perpendicular neutral flux Γ_n (or kinetic energy k if no Gamman)."""
    for spec in physical_field_specs(solution):
        if spec.key == "Gamma_n":
            return spec
    for spec in physical_field_specs(solution):
        if spec.key == "k":
            return spec
    return None


def plot_physical_field_with_target(
    solution: Any,
    spec: PhysicalFieldSpec,
    data: np.ndarray,
    *,
    target_r: float,
    target_z: float,
    n_levels: int = 50,
    figsize: Tuple[float, float] = (9.0, 8.0),
    suptitle: str | None = None,
    separatrix_psi_level: float | None = None,
    show_separatrix: bool = True,
) -> Figure:
    """HDG mesh map for one field with an AMCIS target marker."""
    fig = plot_physical_field(solution, spec, data, n_levels=n_levels, figsize=figsize)
    ax = fig.axes[0]
    if show_separatrix:
        from adjoint_mc.viz.separatrix import overlay_separatrix

        overlay_separatrix(ax, solution, psi_level=separatrix_psi_level, color="white")
    ax.scatter(
        [target_r],
        [target_z],
        s=120,
        c="red",
        marker="*",
        zorder=10,
        edgecolors="white",
        linewidths=0.6,
        label=r"target $\Omega$",
    )
    ax.legend(loc="upper right")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout()
    return fig


def plot_physical_field(
    solution: Any,
    spec: PhysicalFieldSpec,
    data: np.ndarray,
    *,
    n_levels: int = 50,
    figsize: Tuple[float, float] = (9.0, 8.0),
) -> Figure:
    """Full-screen map for one physical field."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_kwargs: Dict[str, Any] = {
        "ax": ax,
        "label": spec.label,
        "connectivity": solution.mesh.derived_geometry.connectivity_big,
        "n_levels": n_levels,
        "log": spec.log_scale,
    }
    if spec.signed:
        plot_kwargs["cmap"] = _SIGNED_CMAP
    solution.mesh.plot.full(data, **plot_kwargs)
    ax.set_title(spec.label, fontsize=12)
    fig.tight_layout()
    return fig


def iter_plasma_field_plot_tabs(
    solution: Any,
    *,
    n_levels: int = 50,
) -> Iterator[Tuple[str, str, Figure]]:
    """
    Yield (key, tab_title, figure) for each plasma field map.

    One tab per physical field plus one tab for S_ion.
    """
    arrays = _physical_field_arrays(solution)
    for spec in physical_field_specs(solution):
        fig = plot_physical_field(solution, spec, arrays[spec.key], n_levels=n_levels)
        yield spec.key, spec.tab_title, fig

    fig_ion, _ax = plot_ionization_source(solution, n_levels=n_levels)
    fig_ion.set_size_inches(9.0, 8.0)
    fig_ion.tight_layout()
    yield "S_ion", r"S$_{\mathrm{ion}}$", fig_ion


def compute_ionization_source_field(solution: Any) -> np.ndarray:
    """
    Volume ionization source S_ion on the simple mesh.

    Computed by HDG_postprocess: S_ion = n_e * n_n * <sigma v>_ion(T_e, n_e)
    with AMJUEL atomic coefficients configured at load time.
    """
    solution.sources.ionization(view="simple")
    return np.asarray(solution.views.simple.sources.ionization_source, dtype=float)


def ionization_source_stats(solution: Any) -> Dict[str, float]:
    """Quick statistics on S_ion (simple domain)."""
    field = compute_ionization_source_field(solution)
    positive = field[field > 0]
    if positive.size == 0:
        return {"max": 0.0, "mean": 0.0, "fraction_positive": 0.0}
    return {
        "max": float(np.max(positive)),
        "mean": float(np.mean(positive)),
        "fraction_positive": float(positive.size / field.size),
    }


def plot_physical_overview(
    solution: Any,
    *,
    n_levels: int = 40,
) -> Tuple[Figure, Any, Any]:
    """
    Overview of physical fields on the simple mesh.

    Shows n_e, n_n, T_i, T_e, M, Gamma_n — not S_ion (see plot_ionization_source).
    """
    fig, axes, overview = solution.plot.physical_overview(n_levels=n_levels)
    fig.suptitle("Physical fields — simple mesh", fontsize=11)
    fig.tight_layout()
    return fig, axes, overview


def plot_ionization_source(
    solution: Any,
    *,
    n_levels: int = 50,
) -> Tuple[Figure, Any]:
    """
    Poloidal map of the volume ionization source S_ion(R, Z).

    Log scale; units: code source term (n_e * n_n * <sv>_ion),
    consistent with SOLEDGE-HDG / HDG_postprocess.
    """
    prep_ops.ensure_connectivity_big(solution)
    data = compute_ionization_source_field(solution).copy()
    data[data <= 0] = np.nan

    positive = data[np.isfinite(data) & (data > 0)]
    limits = None
    if positive.size:
        limits = (
            float(np.log10(np.percentile(positive, 5))),
            float(np.log10(np.percentile(positive, 99))),
        )

    fig, ax = plt.subplots(figsize=(9, 7))
    solution.mesh.plot.full(
        data,
        ax=ax,
        log=True,
        label=r"$S_{\mathrm{ion}}$ [code units m$^{-3}$ s$^{-1}$] (5–99 %)",
        connectivity=solution.mesh.derived_geometry.connectivity_big,
        n_levels=n_levels,
        limits=limits,
    )
    stats = ionization_source_stats(solution)
    if limits is not None:
        scale_note = f"log scale {10 ** limits[0]:.2e} – {10 ** limits[1]:.2e}"
    else:
        scale_note = "log scale"
    fig.suptitle(
        "Volume ionization source  "
        f"(max = {stats['max']:.2e}, mean = {stats['mean']:.2e}; {scale_note})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, ax


def format_summary_text(summary, ion_stats: Dict[str, float] | None = None) -> str:
    """Informative text for the GUI summary panel."""
    text = (
        f"File   : {summary.path.name}\n"
        f"Model  : {summary.model}\n"
        f"Neq    : {summary.neq}  |  partitions: {summary.n_partitions}\n"
        f"Mesh   : {summary.n_elements} elements, {summary.n_nodes} nodes\n"
        f"\n"
        f"Non-dimensionalization (SI):\n"
        f"  length        L0 = {summary.length_scale_m:.6g} m\n"
        f"  density       n0 = {summary.density_scale_m3:.6g} m^-3\n"
        f"  temperature   T0 = {summary.temperature_scale_ev:.6g} eV\n"
        f"  time          t0 = {summary.time_scale_s:.6g} s\n"
    )
    if ion_stats is not None:
        text += (
            f"\n"
            f"Ionization source S_ion (simple mesh):\n"
            f"  max     = {ion_stats['max']:.4e}\n"
            f"  mean    = {ion_stats['mean']:.4e}\n"
            f"  fraction nodes > 0: {100 * ion_stats['fraction_positive']:.1f} %\n"
            f"\n"
            f"Formula: S_ion = n_e * n_n * <sigma v>_ion(T_e, n_e)\n"
            f"(AMJUEL coefficients via HDG_postprocess)\n"
        )
    return text
