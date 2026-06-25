"""Shared pipeline timing dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunTiming:
    grid_build_s: float
    mc_s: float
    provenance_s: float
    kernel_s: float
    pack_s: float
    total_s: float
