"""Inverse charge-exchange rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adjoint_mc.atomic.cx_cross_section import majorant_sv, maxwellian_sv, maxwellian_speed_std_m_s, sigma_v


@dataclass(frozen=True)
class CxRejectionConfig:
    """Settings for isolated CX rejection tests."""

    m_ref_amu: float = 1.0
    calibrate_hdg: bool = True
    max_trials_per_sample: int = 10_000


@dataclass
class CxRejectionSample:
    """One accepted CX velocity exchange."""

    v_n_before: np.ndarray
    v_i_proposed: np.ndarray
    v_n_after: np.ndarray
    v_rel_m_s: float
    n_trials: int
    ti_ev: float


@dataclass
class CxRejectionRateEstimate:
    """Rate and acceptance statistics at one (T_i, v_n) point."""

    ti_ev: float
    v_n: np.ndarray
    n_trials: int
    n_accepted: int
    acceptance_fraction: float
    majorant_sv_m3s: float
    rate_mc_m3s: float
    rate_quadrature_m3s: float


@dataclass
class CxRejectionBatchResult:
    """Batch validation over a temperature grid (v_n = 0)."""

    temperature_ev: np.ndarray
    estimates: list[CxRejectionRateEstimate] = field(default_factory=list)
    cx_params: Any | None = None
    seed: int = 0
    n_trials_per_temperature: int = 0

    @property
    def reference_rates(self) -> np.ndarray:
        if self.cx_params is None:
            return np.array([e.rate_quadrature_m3s for e in self.estimates], dtype=float)
        from adjoint_mc.atomic.cx_cross_section import hdg_openadas_sv

        return np.array(
            [float(hdg_openadas_sv(e.ti_ev, cx_params=self.cx_params)) for e in self.estimates],
            dtype=float,
        )

    @property
    def mc_rates(self) -> np.ndarray:
        return np.array([e.rate_mc_m3s for e in self.estimates], dtype=float)

    @property
    def acceptance_fractions(self) -> np.ndarray:
        return np.array([e.acceptance_fraction for e in self.estimates], dtype=float)

    def max_relative_error_vs_reference(self) -> float:
        ref = self.reference_rates
        mc = self.mc_rates
        mask = ref > 0.0
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(mc[mask] - ref[mask]) / ref[mask]))

    def min_acceptance_fraction(self) -> float:
        if not self.estimates:
            return 0.0
        return float(min(e.acceptance_fraction for e in self.estimates))


def sample_maxwellian_3d(ti_ev: float, rng: np.random.Generator, *, m_ref_amu: float = 1.0) -> np.ndarray:
    """Draw one ion velocity from a 3D Maxwellian at T_i [eV]."""
    std = maxwellian_speed_std_m_s(ti_ev, m_ref_amu=m_ref_amu)
    return rng.normal(0.0, std, size=3)


def sample_cx_velocity(
    v_n: np.ndarray,
    ti_ev: float,
    rng: np.random.Generator,
    *,
    config: CxRejectionConfig | None = None,
    cx_params: Any | None = None,
) -> CxRejectionSample | None:
    """
    Rejection sample of post-CX neutral velocity (backward: v_n <- v_i).

    Returns None if no acceptance within ``max_trials_per_sample``.
    """
    if config is None:
        config = CxRejectionConfig()
    v_n = np.asarray(v_n, dtype=float).reshape(3)
    m = majorant_sv(
        ti_ev,
        float(v_n[0]),
        float(v_n[1]),
        float(v_n[2]),
        m_ref_amu=config.m_ref_amu,
        calibrate_hdg=config.calibrate_hdg,
    )
    if m <= 0.0:
        return None

    for trial in range(1, config.max_trials_per_sample + 1):
        v_i = sample_maxwellian_3d(ti_ev, rng, m_ref_amu=config.m_ref_amu)
        v_rel = float(np.linalg.norm(v_n - v_i))
        sv = float(
            sigma_v(
                v_rel,
                ti_ev=ti_ev,
                m_ref_amu=config.m_ref_amu,
                calibrate_hdg=config.calibrate_hdg,
                cx_params=cx_params,
            )
        )
        if rng.random() < sv / m:
            return CxRejectionSample(
                v_n_before=v_n.copy(),
                v_i_proposed=v_i.copy(),
                v_n_after=v_i.copy(),
                v_rel_m_s=v_rel,
                n_trials=trial,
                ti_ev=float(ti_ev),
            )
    return None


def estimate_cx_rate_rejection(
    ti_ev: float,
    v_n: np.ndarray | None = None,
    *,
    n_trials: int = 50_000,
    seed: int = 0,
    config: CxRejectionConfig | None = None,
    cx_params: Any | None = None,
) -> CxRejectionRateEstimate:
    """
    Estimate <σv> via rejection acceptance: rate ≈ (N_accept / N) * (σv)_max.

    Also returns the deterministic quadrature ``maxwellian_sv`` for comparison.
    """
    if config is None:
        config = CxRejectionConfig()
    if v_n is None:
        v_n = np.zeros(3, dtype=float)
    v_n = np.asarray(v_n, dtype=float).reshape(3)
    ti_ev = float(ti_ev)

    m = majorant_sv(
        ti_ev,
        float(v_n[0]),
        float(v_n[1]),
        float(v_n[2]),
        m_ref_amu=config.m_ref_amu,
        calibrate_hdg=config.calibrate_hdg,
    )
    rng = np.random.default_rng(seed)
    n_accepted = 0
    for _ in range(n_trials):
        v_i = sample_maxwellian_3d(ti_ev, rng, m_ref_amu=config.m_ref_amu)
        v_rel = float(np.linalg.norm(v_n - v_i))
        sv = float(
            sigma_v(
                v_rel,
                ti_ev=ti_ev,
                m_ref_amu=config.m_ref_amu,
                calibrate_hdg=config.calibrate_hdg,
                cx_params=cx_params,
            )
        )
        if m > 0.0 and rng.random() < sv / m:
            n_accepted += 1

    acceptance = n_accepted / max(n_trials, 1)
    rate_mc = acceptance * m
    rate_quad = maxwellian_sv(
        ti_ev,
        v_n,
        m_ref_amu=config.m_ref_amu,
        calibrate_hdg=config.calibrate_hdg,
        cx_params=cx_params,
        seed=seed + 1,
    )
    return CxRejectionRateEstimate(
        ti_ev=ti_ev,
        v_n=v_n.copy(),
        n_trials=n_trials,
        n_accepted=n_accepted,
        acceptance_fraction=acceptance,
        majorant_sv_m3s=m,
        rate_mc_m3s=rate_mc,
        rate_quadrature_m3s=rate_quad,
    )


def run_cx_rejection_batch(
    temperatures_ev: np.ndarray | list[float],
    *,
    n_trials_per_temperature: int = 40_000,
    seed: int = 0,
    config: CxRejectionConfig | None = None,
    cx_params: Any | None = None,
) -> CxRejectionBatchResult:
    """Run rate + acceptance validation on a temperature grid with v_n = 0."""
    if config is None:
        config = CxRejectionConfig()
    temps = np.asarray(temperatures_ev, dtype=float).reshape(-1)
    estimates: list[CxRejectionRateEstimate] = []
    for idx, ti in enumerate(temps):
        estimates.append(
            estimate_cx_rate_rejection(
                float(ti),
                np.zeros(3),
                n_trials=n_trials_per_temperature,
                seed=seed + idx * 17,
                config=config,
                cx_params=cx_params,
            )
        )
    return CxRejectionBatchResult(
        temperature_ev=temps,
        estimates=estimates,
        cx_params=cx_params,
        seed=seed,
        n_trials_per_temperature=n_trials_per_temperature,
    )
