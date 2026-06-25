"""Cython performance core."""

from __future__ import annotations

try:
    from adjoint_mc.core._tracker import CYTHON_AVAILABLE, run_backward_batch_cython
except ImportError:
    CYTHON_AVAILABLE = False
    run_backward_batch_cython = None  # type: ignore[misc, assignment]

__all__ = ["CYTHON_AVAILABLE", "run_backward_batch_cython"]
