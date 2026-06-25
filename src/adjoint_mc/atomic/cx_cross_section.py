"""Janev-Smith H.2 charge-exchange cross section and Maxwellian rate helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

ELECTRON_CHARGE = 1.602176634e-19  # C
PROTON_MASS = 1.67262192369e-27  # kg
AMU_TO_KG = 1.66053906660e-27

# Janev-Smith Table 9, n = 1 (FIDASIM / AMJUEL H.2).
_JANEV_A = np.array([3.2345, 2.3588e2, 2.3713, 3.8371e-2, 3.8068e-6, 1.1832e-10])
_JANEV_N = 1.0

# Temperature grid used for OpenADAS calibration and majorant tables [eV].
_T_CALIB_GRID = np.array([5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0])


def sigma_cx_janev_cm2(energy_keV_amu: np.ndarray | float) -> np.ndarray | float:
    """
    Total H⁺ + H(1s) CX cross section (Janev-Smith, reaction H.2).

    Parameters
    ----------
    energy_keV_amu
        Relative collision energy in keV/amu (laboratory frame).
    """
    scalar = np.isscalar(energy_keV_amu)
    ehat = np.maximum(np.asarray(energy_keV_amu, dtype=float), 1.0e-12) * _JANEV_N**2
    sigma = (
        1.0e-16
        * _JANEV_A[0]
        * (_JANEV_N**4)
        * np.log(_JANEV_A[1] / ehat + _JANEV_A[2])
        / (1.0 + _JANEV_A[3] * ehat + _JANEV_A[4] * ehat**3.5 + _JANEV_A[5] * ehat**5.4)
    )
    return float(sigma) if scalar else sigma


def sigma_cx_janev_m2(energy_keV_amu: np.ndarray | float) -> np.ndarray | float:
    """Janev H.2 cross section in m²."""
    return np.asarray(sigma_cx_janev_cm2(energy_keV_amu), dtype=float) * 1.0e-4


def reduced_mass_kg(m_ref_amu: float = 1.0) -> float:
    """Reduced mass for H⁺ on H(1s)."""
    m = m_ref_amu * AMU_TO_KG
    return 0.5 * m


def relative_energy_keV(v_rel_m_s: np.ndarray | float, *, m_ref_amu: float = 1.0) -> np.ndarray | float:
    """Convert relative speed to collision energy in keV/amu."""
    mu = reduced_mass_kg(m_ref_amu)
    scalar = np.isscalar(v_rel_m_s)
    v = np.asarray(v_rel_m_s, dtype=float)
    energy_j = 0.5 * mu * v * v
    energy_keV = energy_j / (ELECTRON_CHARGE * 1000.0)
    return float(energy_keV) if scalar else energy_keV


def sigma_cx(
    v_rel_m_s: np.ndarray | float,
    *,
    ti_ev: float,
    m_ref_amu: float = 1.0,
    calibrate_hdg: bool = True,
    cx_params: Any | None = None,
) -> np.ndarray | float:
    """
    CX cross section σ(v_rel) in m².

    When ``calibrate_hdg`` is True, a temperature-dependent factor aligns the
    Maxwellian average with OpenADAS / HDG ``calculate_cx_rate(T_i)``.
    """
    energy_keV = relative_energy_keV(v_rel_m_s, m_ref_amu=m_ref_amu)
    sigma = sigma_cx_janev_m2(energy_keV)
    if calibrate_hdg:
        sigma = np.asarray(sigma, dtype=float) * calibration_factor(ti_ev, cx_params=cx_params)
    return sigma


def maxwellian_speed_std_m_s(ti_ev: float, *, m_ref_amu: float = 1.0) -> float:
    """Per-component standard deviation of a 3D Maxwellian at T_i [eV]."""
    mass = m_ref_amu * AMU_TO_KG
    return float(np.sqrt(ti_ev * ELECTRON_CHARGE / mass))


def _maxwellian_sv_janev_raw(
    ti_ev: float,
    v_n: np.ndarray,
    *,
    m_ref_amu: float = 1.0,
    n_samples: int = 80_000,
    seed: int = 0,
) -> float:
    """Monte Carlo estimate of <σ v> for raw Janev σ, ions Maxwellian at T_i."""
    ti_ev = float(ti_ev)
    if ti_ev <= 0.0:
        return 0.0
    std = maxwellian_speed_std_m_s(ti_ev, m_ref_amu=m_ref_amu)
    rng = np.random.default_rng(seed)
    v_i = rng.normal(0.0, std, size=(n_samples, 3))
    v_rel = np.linalg.norm(v_n.reshape(1, 3) - v_i, axis=1)
    energy_keV = relative_energy_keV(v_rel, m_ref_amu=m_ref_amu)
    sigma = sigma_cx_janev_m2(energy_keV)
    return float(np.mean(sigma * v_rel))


def hdg_openadas_sv(ti_ev: float | np.ndarray, *, cx_params: Any) -> np.ndarray | float:
    """OpenADAS rate coefficient <σv>(T_i) [m³/s] at ion temperature (T_n ≡ T_i convention)."""
    from hdg_postprocess.routines.atomic import calculate_cx_rate

    te = np.atleast_1d(np.asarray(ti_ev, dtype=float))
    rates = np.asarray(calculate_cx_rate(te, cx_params), dtype=float)
    return float(rates[0]) if np.isscalar(ti_ev) else rates


@lru_cache(maxsize=1)
def _default_cx_params() -> Any | None:
    """Load TCABR cx fit parameters when the reference .h5 is available."""
    try:
        from adjoint_mc.config import DEFAULT_SOLUTION_PATH
        from adjoint_mc.io.hdg_loader import load_hdg_solution
    except Exception:
        return None
    if not DEFAULT_SOLUTION_PATH.is_file():
        return None
    loaded = load_hdg_solution(str(DEFAULT_SOLUTION_PATH))
    atomic = loaded.solution.additional_parameters.atomic
    if atomic is None or "cx" not in atomic:
        return None
    return atomic["cx"]


@lru_cache(maxsize=128)
def _calibration_factor_cached(ti_ev: float, cx_key: int) -> float:
    cx_params = _default_cx_params() if cx_key == 0 else None
    if cx_params is None:
        return 1.0
    janev = _maxwellian_sv_janev_raw(ti_ev, np.zeros(3), seed=int(ti_ev * 100) % 10_000)
    if janev <= 0.0:
        return 1.0
    ref = float(hdg_openadas_sv(ti_ev, cx_params=cx_params))
    return ref / janev


def calibration_factor(ti_ev: float, *, cx_params: Any | None = None) -> float:
    """
    Scale Janev σ so that <σv>_Maxwellian matches OpenADAS at T_i.

    Uses log-log interpolation on a fixed grid when ``cx_params`` is omitted.
    """
    ti_ev = float(ti_ev)
    if cx_params is not None:
        janev = _maxwellian_sv_janev_raw(ti_ev, np.zeros(3), seed=int(ti_ev * 100) % 10_000)
        if janev <= 0.0:
            return 1.0
        return float(hdg_openadas_sv(ti_ev, cx_params=cx_params)) / janev

    cx_key = 0 if _default_cx_params() is not None else -1
    if cx_key < 0:
        return 1.0

    factors = np.array(
        [_calibration_factor_cached(float(t), cx_key) for t in _T_CALIB_GRID],
        dtype=float,
    )
    log_t = np.log(_T_CALIB_GRID)
    log_f = np.log(np.maximum(factors, 1.0e-12))
    return float(np.exp(np.interp(np.log(max(ti_ev, _T_CALIB_GRID[0])), log_t, log_f)))


def maxwellian_sv(
    ti_ev: float,
    v_n: np.ndarray | None = None,
    *,
    m_ref_amu: float = 1.0,
    calibrate_hdg: bool = True,
    cx_params: Any | None = None,
    n_samples: int = 80_000,
    seed: int = 0,
) -> float:
    """Maxwellian average <σ v> [m³/s] at ion temperature T_i [eV]."""
    ti_ev = float(ti_ev)
    if ti_ev <= 0.0:
        return 0.0
    if v_n is None:
        v_n = np.zeros(3, dtype=float)
    v_n = np.asarray(v_n, dtype=float).reshape(3)
    std = maxwellian_speed_std_m_s(ti_ev, m_ref_amu=m_ref_amu)
    rng = np.random.default_rng(seed)
    v_i = rng.normal(0.0, std, size=(n_samples, 3))
    v_rel = np.linalg.norm(v_n.reshape(1, 3) - v_i, axis=1)
    sigma = sigma_cx(
        v_rel,
        ti_ev=ti_ev,
        m_ref_amu=m_ref_amu,
        calibrate_hdg=calibrate_hdg,
        cx_params=cx_params,
    )
    return float(np.mean(np.asarray(sigma, dtype=float) * v_rel))


@lru_cache(maxsize=256)
def majorant_sv(
    ti_ev: float,
    v_n_x: float = 0.0,
    v_n_y: float = 0.0,
    v_n_z: float = 0.0,
    *,
    m_ref_amu: float = 1.0,
    calibrate_hdg: bool = True,
) -> float:
    """
    Majorant (σ v)_max(T_i, v_n) for inverse-CX rejection sampling.

    Computed on a 1-D speed grid (v_n antiparallel to v_i maximises v_rel).
    """
    ti_ev = float(ti_ev)
    if ti_ev <= 0.0:
        return 0.0
    v_n = np.array([v_n_x, v_n_y, v_n_z], dtype=float)
    v_n_norm = float(np.linalg.norm(v_n))
    std = maxwellian_speed_std_m_s(ti_ev, m_ref_amu=m_ref_amu)
    # |v_n - v_i| is maximised for v_i antiparallel to v_n; σv can peak at
    # v_rel well above the thermal scale, so scan a wide speed range.
    v_max = max(v_n_norm + 50.0 * std, 2.0e6)
    speeds = np.linspace(0.0, v_max, 8000)
    v_rel = v_n_norm + speeds
    sigma = sigma_cx(
        v_rel,
        ti_ev=ti_ev,
        m_ref_amu=m_ref_amu,
        calibrate_hdg=calibrate_hdg,
    )
    sv = np.asarray(sigma, dtype=float) * v_rel
    return float(np.max(sv)) if sv.size else 0.0


def sigma_v(
    v_rel_m_s: np.ndarray | float,
    *,
    ti_ev: float,
    m_ref_amu: float = 1.0,
    calibrate_hdg: bool = True,
    cx_params: Any | None = None,
) -> np.ndarray | float:
    """Product σ(v_rel) v_rel [m²/s] used in the rejection kernel."""
    sigma = sigma_cx(
        v_rel_m_s,
        ti_ev=ti_ev,
        m_ref_amu=m_ref_amu,
        calibrate_hdg=calibrate_hdg,
        cx_params=cx_params,
    )
    return np.asarray(sigma, dtype=float) * np.asarray(v_rel_m_s, dtype=float)
