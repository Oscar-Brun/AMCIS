"""Step-by-step backward trajectory recording for GUI animation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackwardTraceFrame:
    """One frame of a backward history in the poloidal (R, Z) plane."""

    r: float
    z: float
    log_weight: float
    path_m: float
    event: str  # birth, step, cx, wall, lost, max_path, max_steps
    n_cx: int = 0
    region_name: str | None = None


def append_trace_frame(
    trace: list[BackwardTraceFrame] | None,
    *,
    r: float,
    z: float,
    log_weight: float,
    path_m: float,
    event: str,
    n_cx: int = 0,
    region_name: str | None = None,
) -> None:
    if trace is None:
        return
    trace.append(
        BackwardTraceFrame(
            r=float(r),
            z=float(z),
            log_weight=float(log_weight),
            path_m=float(path_m),
            event=str(event),
            n_cx=int(n_cx),
            region_name=region_name,
        )
    )
