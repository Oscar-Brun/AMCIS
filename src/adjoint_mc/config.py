"""Default parameters and project paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOLUTION_PATH = Path("/path/to/your/solution.h5")


def resolve_hdg_postprocess_path() -> Path:
    """
    Locate the HDG_postprocess source tree (reference element + atomic data).

    Order: ``HDG_POSTPROCESS_PATH`` env var, then the editable install root of
    ``hdg_postprocess``.
    """
    env = os.environ.get("HDG_POSTPROCESS_PATH", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if root.is_dir():
            return root
        raise FileNotFoundError(
            f"HDG_POSTPROCESS_PATH is not a directory: {root}"
        )

    try:
        import hdg_postprocess

        root = Path(hdg_postprocess.__file__).resolve().parent.parent
        ref = root / "demos/data/reference_elements/reference_triangle_P8.mat"
        if ref.is_file():
            return root
    except ImportError:
        pass

    raise RuntimeError(
        "HDG_postprocess not configured.\n"
        "Install it with: pip install -e /path/to/HDG_postprocess\n"
        "Or set: export HDG_POSTPROCESS_PATH=/path/to/HDG_postprocess"
    )


HDG_POSTPROCESS_PATH = resolve_hdg_postprocess_path()
REFERENCE_ELEMENT = (
    HDG_POSTPROCESS_PATH / "demos/data/reference_elements/reference_triangle_P8.mat"
)
ATOMIC_DATA_DIR = HDG_POSTPROCESS_PATH / "demos/data/atomic"

FIGURES_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = PROJECT_ROOT / "cache"

DEFAULT_N_HISTORIES = 1000
DEFAULT_SEED = 42
DEFAULT_GRID_N_R = 120
DEFAULT_GRID_N_Z = 180
DEFAULT_TAU_MAX = 0.1
DEFAULT_NEUTRAL_SPEED_M_S = 1.0e4
DEFAULT_MAX_STEP_M = 0.02
DEFAULT_VACUUM_WALL_SEARCH_M = 0.15
DEFAULT_MAX_PATH_M = 5.0
DEFAULT_MAX_SEED_REJECTION_ATTEMPTS = 1000
