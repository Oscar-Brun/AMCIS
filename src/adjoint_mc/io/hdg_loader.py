"""Load SOLEDGE-HDG solutions via HDG_postprocess."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from adjoint_mc.config import (
    ATOMIC_DATA_DIR,
    HDG_POSTPROCESS_PATH,
    REFERENCE_ELEMENT,
)


@dataclass(frozen=True)
class SolutionSummary:
    """Metadata for the GUI and reporting."""

    path: Path
    model: str
    neq: int
    n_partitions: int
    length_scale_m: float
    density_scale_m3: float
    temperature_scale_ev: float
    time_scale_s: float
    n_nodes: int
    n_elements: int


@dataclass
class LoadedSolution:
    solution: Any
    summary: SolutionSummary


def parse_h5_path(h5_path: Path | str) -> tuple[str, str]:
    """
    Convert an .h5 path to (solpath, solname_base) for load_solution.

    Assumption: mesh embedded in the same file (recent SOLEDGE-HDG layout).
    """
    path = Path(h5_path).expanduser().resolve()
    if path.suffix.lower() != ".h5":
        raise ValueError(f"Expected .h5 file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Solution not found: {path}")
    return str(path.with_suffix("")), ""


def _extract_summary(solution: Any, path: Path) -> SolutionSummary:
    params = solution.parameters
    adim = params["adimensionalization"]
    physics = params["physics"]
    model = params.get("model", ["?"])
    if isinstance(model, (list, tuple)):
        model = model[0]
    if isinstance(model, bytes):
        model = model.decode()
    mesh_numbers = solution.mesh.raw.mesh_numbers
    if isinstance(mesh_numbers, list) and mesh_numbers:
        mn = mesh_numbers[0]
    elif isinstance(mesh_numbers, dict):
        mn = mesh_numbers
    else:
        mn = {}
    n_nodes = int(mn.get("Nnodes", 0))
    n_elements = int(mn.get("Nelems", 0))
    return SolutionSummary(
        path=path,
        model=str(model),
        neq=int(solution.neq),
        n_partitions=int(solution.n_partitions),
        length_scale_m=float(adim["length_scale"]),
        density_scale_m3=float(adim["density_scale"]),
        temperature_scale_ev=float(adim["temperature_scale"]),
        time_scale_s=float(adim["time_scale"]),
        n_nodes=n_nodes,
        n_elements=n_elements,
    )


def configure_solution(
    solution: Any,
    *,
    hdg_root: Path | None = None,
    reference_element: Path | None = None,
    atomic_data_dir: Path | None = None,
) -> Any:
    """P8 reference element + AMJUEL/OpenADAS atomic data (for S_ion)."""
    from hdg_postprocess.api import configure_solution_setup

    root = hdg_root or HDG_POSTPROCESS_PATH
    ref = Path(reference_element or REFERENCE_ELEMENT)
    atomic = Path(atomic_data_dir or ATOMIC_DATA_DIR)
    if not ref.is_file():
        raise FileNotFoundError(
            f"HDG reference element not found: {ref}\n"
            f"HDG_postprocess root: {root}\n"
            "Set HDG_POSTPROCESS_PATH to your HDG_postprocess clone."
        )
    if not atomic.is_dir():
        raise FileNotFoundError(
            f"HDG atomic data directory not found: {atomic}\n"
            f"HDG_postprocess root: {root}"
        )
    configure_solution_setup(
        solution,
        reference_element=str(ref),
        radiation_model="none",
        atomic_data_dir=str(atomic),
    )
    return solution


def load_hdg_solution(
    h5_path: Path | str,
    *,
    hdg_root: Path | None = None,
    configure: bool = True,
) -> LoadedSolution:
    """Load a TCABR / SOLEDGE-HDG .h5 solution."""
    from hdg_postprocess.api import load_solution

    path = Path(h5_path).expanduser().resolve()
    solpath, solbase = parse_h5_path(path)
    solution = load_solution(solpath, solbase, n_partitions=1)
    if configure:
        configure_solution(solution, hdg_root=hdg_root)
    summary = _extract_summary(solution, path)
    return LoadedSolution(solution=solution, summary=summary)


@lru_cache(maxsize=2)
def load_hdg_solution_cached(h5_path: str) -> LoadedSolution:
    """Light cache to avoid reloading on every GUI click."""
    return load_hdg_solution(h5_path)
