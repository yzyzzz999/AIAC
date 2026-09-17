"""CHTD helper functions — T3 implementation.

Pure thermal helper functions for the 28-state CHTD model.
All functions are stateless and do NOT depend on thermal.py.

Defect policy: all functions that involve RadCoBaseTempEx accept a
`mode` argument (DefectMode) to switch between the as-found Simulink
defect and the physically correct formula.  Golden case replay uses
DefectMode.AS_FOUND.  See formulas/CHTD_defect_policy.md.
"""

from __future__ import annotations

import enum
from typing import Union

import numpy as np

from .bus_index import X_INDEX, U_INDEX


# ── Defect mode ──────────────────────────────────────────────────────────────

class DefectMode(enum.Enum):
    """Controls as-found vs corrected behaviour for confirmed Simulink defects."""
    AS_FOUND  = "as_found"
    CORRECTED = "corrected"


# Module-level aliases for ergonomic call sites
AS_FOUND  = DefectMode.AS_FOUND
CORRECTED = DefectMode.CORRECTED


# ── RadCoBaseTempEx breakpoints and LUT tables ────────────────────────────────
# Simulink block: PMV/RadCoBaseTempEx/1-D Lookup Table
# BP: [-40, -30, ..., 90] °C (14 points, step 10)
# AS_FOUND table: power(BP - 273.15, 4) * 5.67e-8  ← CONFIRMED DEFECT (sign error)
# CORRECTED table: power(BP + 273.15, 4) * 5.67e-8  ← physically correct
# See formulas/RadCoBaseTempEx_model_defect.md for full analysis.

_RAD_BP: np.ndarray = np.arange(-40.0, 91.0, 10.0)   # shape (14,)
_RAD_TBL_AS_FOUND:  np.ndarray = (_RAD_BP - 273.15) ** 4 * 5.67e-8
_RAD_TBL_CORRECTED: np.ndarray = (_RAD_BP + 273.15) ** 4 * 5.67e-8


# ── ConvCoBaseFlowEx breakpoints and LUT table ───────────────────────────────
# Simulink block: PMV/ConvCoBaseFlowEx/1-D Lookup Table
# BP: [1, 5, 9, ..., 97, 99, 100, 200, 300, 400, 500, 600]
# Table: power(BP, 0.8)  (Dittus-Boelter Nu ∝ Re^0.8 correlation)
# Extrapolation: clip (Simulink 'Clip')

_CONV_BP: np.ndarray = np.concatenate([
    np.arange(1.0, 98.0, 4.0),                        # [1, 5, 9, ..., 97]  25 pts
    np.array([99.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
])
_CONV_TBL: np.ndarray = np.power(_CONV_BP, 0.8)


# ── LUT lookup ───────────────────────────────────────────────────────────────

def lookup_ambt(
    table: np.ndarray,
    amb_t: float,
    params,
) -> float:
    """Evaluate a CHTD_*_M LUT at the current ambient temperature.

    Linear interpolation, clip extrapolation (matches Simulink 1-D Lookup
    Table with 'Clip' extrapolation setting).

    Parameters
    ----------
    table  : 1-D LUT values, shape (7,), indexed over CHTD_AmbT_X
    amb_t  : ambient temperature [°C]
    params : CHTDParams instance (provides CHTD_AmbT_X breakpoints)

    Returns
    -------
    Interpolated scalar float.
    """
    if not isinstance(table, np.ndarray) or table.shape != (7,):
        raise ValueError(
            f"lookup_ambt: table must be ndarray shape (7,), "
            f"got {type(table).__name__} shape {getattr(table, 'shape', '?')}"
        )
    return float(np.interp(float(amb_t), params.CHTD_AmbT_X, table))


# ── RadCoBaseTempEx ──────────────────────────────────────────────────────────

def rad_co_base_temp_ex(
    temp_c: float,
    mode: DefectMode = AS_FOUND,
) -> float:
    """Radiant emissive power from 1-D LUT (shared Simulink helper).

    AS_FOUND  (Simulink-faithful, Policy P1):
        table = (BP - 273.15)^4 * 5.67e-8   ← CONFIRMED DEFECT: sign error.
        Function is MONOTONICALLY DECREASING.  Radiation direction is physically
        reversed for all non-equilibrium conditions.  Replicated as-found for
        golden-case fidelity.

    CORRECTED (physically correct Stefan-Boltzmann):
        table = (BP + 273.15)^4 * 5.67e-8
        Function is MONOTONICALLY INCREASING.

    Both modes agree exactly at T = 0°C (coincidental equal point).

    Parameters
    ----------
    temp_c : temperature [°C], clipped to [-40, 90] before lookup
    mode   : DefectMode.AS_FOUND or DefectMode.CORRECTED

    Returns
    -------
    Radiant emissive power proxy [W/m²] (float).
    """
    t = float(np.clip(float(temp_c), _RAD_BP[0], _RAD_BP[-1]))
    if mode == DefectMode.AS_FOUND:
        return float(np.interp(t, _RAD_BP, _RAD_TBL_AS_FOUND))
    return float(np.interp(t, _RAD_BP, _RAD_TBL_CORRECTED))


# ── ConvCoBaseFlowEx ─────────────────────────────────────────────────────────

def conv_co_base_flow_ex(flow: float) -> float:
    """Flow-dependent convection multiplier (Simulink ConvCoBaseFlowEx block).

    LUT with bp = [1, 5, 9, ..., 97, 99, 100, 200, 300, 400, 500, 600],
    table = bp^0.8 (Dittus-Boelter turbulent correlation: Nu ∝ Re^0.8).
    Extrapolation: clip to boundary values (Simulink 'Clip').

    flow below first breakpoint (< 1.0) clips to 1.0^0.8 = 1.0.
    flow above last breakpoint (> 600.0) clips to 600.0^0.8.

    Parameters
    ----------
    flow : duct or exterior flow value [m³/h or normalised]

    Returns
    -------
    Convection multiplier (float, ≥ 1.0^0.8).
    """
    return float(np.interp(float(flow), _CONV_BP, _CONV_TBL))


# ── RhoAirEx ─────────────────────────────────────────────────────────────────

def rho_air_ex(temp_c: float) -> float:
    """Dry air density at 1 atm via ideal gas law.

    Formula confirmed from HeadTempSp_topology_spec.md §3 (CpVEx capacitance):
        ρ = P₀ / (R_air × T_K)
        P₀ = 101325 Pa, R_air = 287 J/(kg·K)

    Parameters
    ----------
    temp_c : air temperature [°C]

    Returns
    -------
    Air density [kg/m³].
    """
    return 101325.0 / (287.0 * (float(temp_c) + 273.15))


# ── CpVEx / CpMEx ────────────────────────────────────────────────────────────

def cp_v_ex(temp_c: float, volume: float, air_cp: float) -> float:
    """Thermal capacitance for air zones (CpVEx Simulink block).

    C = max(ρ_air(T) × V, 0.1) × Cp_air

    The 0.1 floor guard is applied to the (ρ × V) product before
    multiplying by Cp_air, matching the Simulink CpMEx clamp applied
    to (rho*V) before the Cp multiplication.

    Parameters
    ----------
    temp_c  : zone air temperature [°C]
    volume  : air zone volume [m³]  (CHTD_HeadAirVAtb_P or similar)
    air_cp  : air specific heat [J/(kg·K)]  (CHTD_AirCpAtb_P = 1006.0)

    Returns
    -------
    Thermal capacitance [J/K].
    """
    rho_v = rho_air_ex(float(temp_c)) * float(volume)
    return max(rho_v, 0.1) * float(air_cp)


def cp_m_ex(mass_or_rho_volume: float, cp: float) -> float:
    """Thermal capacitance for body zones (CpMEx Simulink block).

    C = max(M × Cp, 0.1)

    Used for Hood, CabinFrnt, Console, Cabin*, Win*, Roof zones where
    zone capacitance is given directly as mass × specific-heat.

    Parameters
    ----------
    mass_or_rho_volume : zone mass [kg] or pre-computed (ρ×V) product
    cp                 : zone specific heat [J/(kg·K)]

    Returns
    -------
    Thermal capacitance [J/K], minimum 0.1.
    """
    return max(float(mass_or_rho_volume) * float(cp), 0.1)


# ── Zone-level heat exchange helpers ─────────────────────────────────────────

def radiation_heat(
    t_ref: float,
    t_src: float,
    area: float,
    rad_coef: float,
    mode: DefectMode = AS_FOUND,
) -> float:
    """Radiation heat exchange from t_ref to t_src.

    Q = (RadCoBase(t_ref) − RadCoBase(t_src)) × area × rad_coef

    Sign convention (from CHTD formula spec §4):
    - Positive Q means t_ref zone emits more radiation than t_src zone
      (heat flows from t_ref toward t_src if both are coupled).
    - In AS_FOUND mode the sign is physically reversed for T > 0°C
      (confirmed Simulink defect P1 — see CHTD_defect_policy.md §1).

    Parameters
    ----------
    t_ref    : reference zone temperature [°C]
    t_src    : source zone temperature [°C]
    area     : zone area parameter [m²]
    rad_coef : evaluated CHTD_*RadCo_M LUT value [dimensionless]
    mode     : DefectMode.AS_FOUND (Simulink-faithful) or CORRECTED

    Returns
    -------
    Radiation heat flux [W] (scalar float).
    """
    return (
        rad_co_base_temp_ex(t_ref, mode) - rad_co_base_temp_ex(t_src, mode)
    ) * float(area) * float(rad_coef)


def convection_heat(
    t_ref: float,
    t_src: float,
    flow_or_area: float,
    conv_coef: float,
) -> float:
    """Convection heat exchange from t_ref to t_src.

    Q = (t_ref − t_src) × ConvCoBaseFlowEx(flow_or_area) × conv_coef

    Parameters
    ----------
    t_ref        : reference temperature [°C]
    t_src        : source temperature [°C]
    flow_or_area : flow input to ConvCoBaseFlowEx (duct flow or VehSpd × area)
    conv_coef    : evaluated CHTD_*ConvCo_M LUT value [dimensionless]

    Returns
    -------
    Convection heat flux [W] (scalar float).
    """
    return (
        (float(t_ref) - float(t_src))
        * conv_co_base_flow_ex(float(flow_or_area))
        * float(conv_coef)
    )


def solar_heat(solar: float, area: float, solar_coef: float) -> float:
    """Solar radiation heat gain.

    Q = solar × area × solar_coef

    Parameters
    ----------
    solar      : solar irradiance proxy [W/m²] (SolarFd, SolarFp, or alias)
    area       : zone solar-absorbing area [m²]
    solar_coef : evaluated CHTD_*SolarRadCo_M LUT value [dimensionless]

    Returns
    -------
    Solar heat gain [W] (scalar float).
    """
    return float(solar) * float(area) * float(solar_coef)


def hvac_convection_heat(
    tma: float,
    zone_temp: float,
    flow: float,
    coef: float,
) -> float:
    """HVAC duct convection heat (face/floor vents, defrost ducts).

    Q = (tma − zone_temp) × ConvCoBaseFlowEx(flow) × coef

    Parameters
    ----------
    tma       : HVAC supply air temperature (Tma signal) [°C]
    zone_temp : current zone temperature [°C]
    flow      : duct volumetric flow (e.g. FrntFdvFlow) [m³/h or normalised]
    coef      : evaluated duct convection LUT value [dimensionless]

    Returns
    -------
    HVAC heat gain [W] (scalar float). Positive if tma > zone_temp.
    """
    return (
        (float(tma) - float(zone_temp))
        * conv_co_base_flow_ex(float(flow))
        * float(coef)
    )


# ── Solar alias helpers (Policy P2 / P3) ─────────────────────────────────────

def get_solar_fd_horiz(u: np.ndarray) -> float:
    """Return SolarFdHoriz proxy from Bus_CHTD_u.

    TEMPORARY ALIAS STRATEGY — Phase 4 manual resolution.
    BusSelector62 in PMV/OneStepCHTD/HeadTempFd is configured to extract
    'SolarFdHoriz' from Bus_CHTD_u, but this field DOES NOT EXIST in the bus
    definition (confirmed via SLDD).

    Proxy per technical team direction: use SolarFd = u[3].
    This uses a single interior-mirror sensor value.  It is NOT a solar
    projection model (no vehicle attitude, sun azimuth, or window normal).

    Policy P2 — CHTD_defect_policy.md §2.
    THIS IS NOT THE FINAL SOLAR PROJECTION MODEL.
    """
    return float(u[U_INDEX["SolarFd"]])   # u[3]


def get_solar_fp_horiz(u: np.ndarray) -> float:
    """Return SolarFpHoriz proxy from Bus_CHTD_u.

    TEMPORARY ALIAS STRATEGY — Phase 4 manual resolution.
    BusSelector44 in PMV/OneStepCHTD/HeadTempSp extracts 'SolarFpHoriz'
    which does NOT exist in Bus_CHTD_u.

    Proxy per technical team direction: use SolarFp = u[4].

    Policy P3 — CHTD_defect_policy.md §3.
    THIS IS NOT THE FINAL SOLAR PROJECTION MODEL.
    """
    return float(u[U_INDEX["SolarFp"]])   # u[4]


# ── HeadTempSp RearSpf T_src helper (Policy P4) ──────────────────────────────

def select_head_sp_rear_spf_t_src(
    x: np.ndarray,
    mode: DefectMode = AS_FOUND,
) -> float:
    """Return T_src for HeadTempSp RearSpf duct convection term.

    AS_FOUND  (Policy P4 — Simulink-faithful):
        T_src = x[HeadTempSd]  ← confirmed copy-paste defect in SubRef4.
        BusSelector in PMV/OneStepCHTD/HeadTempSp/SubRef4 reads HeadTempSd
        (x[5]) instead of HeadTempSp (x[6]).

    CORRECTED (physically correct):
        T_src = x[HeadTempSp]  ← the zone whose duct heat we are computing.

    Uses X_INDEX from bus_index.py — no hardcoded integers.

    Parameters
    ----------
    x    : 28-element CHTD state vector
    mode : DefectMode.AS_FOUND or CORRECTED

    Returns
    -------
    Zone temperature used as T_src for the RearSpf duct convection [°C].
    """
    if mode == DefectMode.AS_FOUND:
        return float(x[X_INDEX["HeadTempSd"]])   # x[5] — DEFECT: wrong zone
    return float(x[X_INDEX["HeadTempSp"]])        # x[6] — physically correct


def select_defect_lut(
    mode: DefectMode,
    as_found_lut: str,
    corrected_lut: str,
) -> str:
    """Pick LUT workspace field for AS_FOUND (Simulink copy-paste) vs CORRECTED."""
    if mode == DefectMode.CORRECTED:
        return corrected_lut
    return as_found_lut
