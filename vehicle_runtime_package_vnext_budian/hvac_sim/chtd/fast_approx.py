"""FAST_APPROX first-order thermal model for non-TRACE CHTD zones (T7-0).

Mirrors generate_golden_cases_chtd.m compute_generic / one_step_chtd category
loop (§8 generic template, placeholder coef = 1.0).
"""

from __future__ import annotations

import numpy as np

from .bus_index import U_INDEX
from .heat_terms import q_solar_gain, q_zone_convection, q_zone_radiation
from .helpers import DefectMode, cp_m_ex
from .params import CHTDParams
from .zone_config import (
    CHTD_ZONE_CONFIG,
    FAST_APPROX_ZONE_INDICES,
    ZoneImplementationMode,
)

_PLACEHOLDER_COEF = 1.0
_MAX_FAST_APPROX_DELTA_C = 2.0

# mass / cp / area per Python state index (HeadTempSp [6] unused — TRACE only)
_ZONE_MASS_ATTR: tuple[str, ...] = (
    "CHTD_HoodMassAtb_P",
    "CHTD_CabinFrntMassAtb_P",
    "CHTD_ConsoleMassAtb_P",
    "HeadFdMassAtb",
    "HeadFpMassAtb",
    "HeadSdMassAtb",
    "HeadTdMassAtb",
    "HeadTdMassAtb",
    "HeadTpMassAtb",
    "FeetFdMassAtb",
    "FeetFpMassAtb",
    "FeetSdMassAtb",
    "FeetSpMassAtb",
    "FeetTdMassAtb",
    "FeetTpMassAtb",
    "CHTD_CabinFdMassAtb_P",
    "CHTD_CabinFpMassAtb_P",
    "CHTD_CabinSdMassAtb_P",
    "CHTD_CabinSpMassAtb_P",
    "CHTD_CabinTdMassAtb_P",
    "CHTD_CabinTpMassAtb_P",
    "CHTD_WinFdMassAtb_P",
    "CHTD_WinFpMassAtb_P",
    "CHTD_WinSdMassAtb_P",
    "CHTD_WinSpMassAtb_P",
    "CHTD_WinTdMassAtb_P",
    "CHTD_WinTpMassAtb_P",
    "CHTD_RoofMassAtb_P",
)

_ZONE_CP_ATTR: tuple[str | None, ...] = (
    "CHTD_HoodCpAtb_P",
    "CHTD_CabinFrntCpAtb_P",
    "CHTD_ConsoleCpAtb_P",
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    "CHTD_CabinFdCpAtb_P",
    "CHTD_CabinFpCpAtb_P",
    "CHTD_CabinSdCpAtb_P",
    "CHTD_CabinSpCpAtb_P",
    "CHTD_CabinTdCpAtb_P",
    "CHTD_CabinTpCpAtb_P",
    "CHTD_WinFdCpAtb_P",
    "CHTD_WinFpCpAtb_P",
    "CHTD_WinSdCpAtb_P",
    "CHTD_WinSpCpAtb_P",
    "CHTD_WinTdCpAtb_P",
    "CHTD_WinTpCpAtb_P",
    "CHTD_RoofCpAtb_P",
)

_ZONE_AREA_ATTR: tuple[str, ...] = (
    "CHTD_HoodAreaAtb_P",
    "CHTD_CabinFrntAreaAtb_P",
    "CHTD_ConsoleAreaAtb_P",
    "CHTD_HeadAreaAtb_P",
    "CHTD_HeadAreaAtb_P",
    "CHTD_HeadAreaAtb_P",
    "CHTD_HeadAreaAtb_P",
    "CHTD_HeadAreaAtb_P",
    "CHTD_HeadAreaAtb_P",
    "CHTD_FeetAreaAtb_P",
    "CHTD_FeetAreaAtb_P",
    "CHTD_FeetAreaAtb_P",
    "CHTD_FeetAreaAtb_P",
    "CHTD_FeetAreaAtb_P",
    "CHTD_FeetAreaAtb_P",
    "CHTD_CabinFdAreaAtb_P",
    "CHTD_CabinFpAreaAtb_P",
    "CHTD_CabinSdAreaAtb_P",
    "CHTD_CabinSpAreaAtb_P",
    "CHTD_CabinTdAreaAtb_P",
    "CHTD_CabinTpAreaAtb_P",
    "CHTD_WinFdAreaAtb_P",
    "CHTD_WinFpAreaAtb_P",
    "CHTD_WinSdAreaAtb_P",
    "CHTD_WinSpAreaAtb_P",
    "CHTD_WinTdAreaAtb_P",
    "CHTD_WinTpAreaAtb_P",
    "CHTD_RoofAreaAtb_P",
)


def _zone_scalar(params: CHTDParams, attr: str) -> float:
    return float(getattr(params, attr, 1.0))


def _zone_capacitance(params: CHTDParams, idx: int) -> float:
    mass = _zone_scalar(params, _ZONE_MASS_ATTR[idx])
    cp_attr = _ZONE_CP_ATTR[idx]
    cp = 1.0 if cp_attr is None else _zone_scalar(params, cp_attr)
    return cp_m_ex(mass, cp)


def _zone_area(params: CHTDParams, idx: int) -> float:
    return _zone_scalar(params, _ZONE_AREA_ATTR[idx])


def _category_for_zone(idx: int) -> tuple[int, ...]:
    """Neighbor category (MATLAB golden generator categories, Python indices)."""
    if 3 <= idx <= 8:
        return tuple(range(3, 9))
    if 9 <= idx <= 14:
        return tuple(range(9, 15))
    if 15 <= idx <= 20:
        return tuple(range(15, 21))
    if 21 <= idx <= 26:
        return tuple(range(21, 27))
    return (0, 1, 2, 27)


def _solar_intensity(idx: int, u: np.ndarray) -> float:
    if 3 <= idx <= 8:
        return float(u[U_INDEX["SolarFp"]])
    if 15 <= idx <= 20 or 21 <= idx <= 26:
        return float(u[U_INDEX["SolarFd"]])
    return 0.0


def compute_fast_approx_zone_delta(
    idx: int,
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    mode: DefectMode,
) -> float:
    """Single-zone §8 generic Forward-Euler delta [°C/step]."""
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    t_zone = float(x[idx])
    area = _zone_area(params, idx)
    flow_amb = veh_spd * area
    solar_i = _solar_intensity(idx, u)

    q_net = 0.0
    for nbr in _category_for_zone(idx):
        if nbr == idx:
            continue
        t_nbr = float(x[nbr])
        q_net += q_zone_convection(t_nbr, t_zone, 1.0, _PLACEHOLDER_COEF)
        q_net += q_zone_radiation(t_nbr, t_zone, area, _PLACEHOLDER_COEF, mode)

    q_amb_rad = q_zone_radiation(t_zone, amb_t, area, _PLACEHOLDER_COEF, mode)
    q_amb_conv = q_zone_convection(t_zone, amb_t, flow_amb, _PLACEHOLDER_COEF)
    q_net += params.CHTD_QlossCo_P * (q_amb_rad + q_amb_conv)
    q_net += q_solar_gain(solar_i, area, _PLACEHOLDER_COEF) * params.CHTD_QgainCo_P

    c_zone = _zone_capacitance(params, idx)
    raw_delta = q_net * params.CHTD_Dt_P / c_zone
    return float(np.clip(raw_delta, -_MAX_FAST_APPROX_DELTA_C, _MAX_FAST_APPROX_DELTA_C))


def apply_fast_approx_deltas(
    delta: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    mode: DefectMode,
) -> None:
    """Fill FAST_APPROX / GAP zone entries in delta in-place."""
    for entry in CHTD_ZONE_CONFIG:
        if entry.mode is ZoneImplementationMode.TRACE:
            continue
        idx = entry.index
        delta[idx] = compute_fast_approx_zone_delta(idx, x, u, params, mode)


def fast_approx_zone_indices() -> frozenset[int]:
    return FAST_APPROX_ZONE_INDICES
