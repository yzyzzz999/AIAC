"""CHTD zone-level heat term functions — T5 implementation.

Thin wrappers over helpers.py that encode per-term physics without applying
QgainCo/QlossCo sign gains (caller handles sign convention per
CHTD_formula_spec.md §4).

Sign convention (caller's responsibility):
    CHTD_QgainCo_P = +1  — heat GAIN: solar, HVAC, inter-zone inflow
    CHTD_QlossCo_P = -1  — heat LOSS: ambient rad/conv, inter-zone outflow

Exceptions — functions that do apply a gain internally:
    q_interzone_from_bus  — caller supplies the gain explicitly
    apply_gain_loss       — explicit multiplier helper
    head_sp_rear_spf_hvac_term — applies QgainCo per Simulink (HVAC gain term)

All functions are stateless and do NOT depend on thermal.py.
"""

from __future__ import annotations

from .bus_index import U_INDEX, X_INDEX
from .helpers import (
    AS_FOUND,
    DefectMode,
    convection_heat,
    hvac_convection_heat,
    radiation_heat,
    solar_heat,
)


# ── 1. HVAC duct convection ───────────────────────────────────────────────────

def q_hvac_duct(
    tma: float,
    zone_temp: float,
    flow: float,
    coef: float,
) -> float:
    """HVAC duct convection: heat delivered to zone from supply air.

    Q = (tma - zone_temp) × ConvCoBaseFlowEx(flow) × coef

    Positive when tma > zone_temp (heating), negative when cooling.
    Caller multiplies by CHTD_QgainCo_P (+1).

    Parameters
    ----------
    tma       : HVAC supply air temperature [°C]
    zone_temp : zone temperature [°C]
    flow      : duct flow input to ConvCoBaseFlowEx [m³/h or normalized]
    coef      : evaluated *ConvCo_M LUT value at current AmbT [dimensionless]

    Returns
    -------
    HVAC duct heat flux [W] (float). Positive = heats zone.
    """
    return hvac_convection_heat(tma, zone_temp, flow, coef)


# ── 2. Zone radiation ─────────────────────────────────────────────────────────

def q_zone_radiation(
    t_ref: float,
    t_src: float,
    area: float,
    coef: float,
    mode: DefectMode = AS_FOUND,
) -> float:
    """Radiation heat exchange between two zones.

    Q = (RadCoBase(t_ref) - RadCoBase(t_src)) × area × coef

    AS_FOUND: radiation direction is physically reversed for T > 0°C
    (confirmed Simulink defect P1 — see CHTD_defect_policy.md §1).
    CORRECTED: physically correct Stefan-Boltzmann direction.

    QgainCo / QlossCo NOT applied — caller decides sign.

    Parameters
    ----------
    t_ref : reference zone temperature [°C]
    t_src : source zone temperature [°C]
    area  : zone area parameter [m²]
    coef  : evaluated *RadCo_M LUT value at current AmbT [dimensionless]
    mode  : DefectMode.AS_FOUND or CORRECTED

    Returns
    -------
    Radiation heat flux [W] (float).
    """
    return radiation_heat(t_ref, t_src, area, coef, mode)


# ── 3. Zone convection ────────────────────────────────────────────────────────

def q_zone_convection(
    t_ref: float,
    t_src: float,
    flow: float,
    coef: float,
) -> float:
    """Convection heat exchange between two zones.

    Q = (t_ref - t_src) × ConvCoBaseFlowEx(flow) × coef

    QgainCo / QlossCo NOT applied — caller decides sign.

    Parameters
    ----------
    t_ref : reference zone temperature [°C]
    t_src : source zone temperature [°C]
    flow  : flow input to ConvCoBaseFlowEx [m³/h or normalized]
    coef  : evaluated *ConvCo_M LUT value at current AmbT [dimensionless]

    Returns
    -------
    Convection heat flux [W] (float). Positive when t_ref > t_src.
    """
    return convection_heat(t_ref, t_src, flow, coef)


# ── 4. Solar gain ─────────────────────────────────────────────────────────────

def q_solar_gain(
    solar: float,
    area: float,
    coef: float,
) -> float:
    """Solar radiation heat gain.

    Q = solar × area × coef

    Caller multiplies by CHTD_QgainCo_P (+1).

    Parameters
    ----------
    solar : solar irradiance proxy [W/m²] (SolarFd, SolarFp, or alias)
    area  : zone solar-absorbing area [m²]
    coef  : evaluated *SolarRadCo_M LUT value at current AmbT [dimensionless]

    Returns
    -------
    Solar heat gain [W] (float). Non-negative when solar >= 0.
    """
    return solar_heat(solar, area, coef)


# ── 5. Leakage / air infiltration ────────────────────────────────────────────

def q_leakage(
    zone_temp: float,
    amb_t: float,
    veh_spd: float,
    area: float,
    coef: float,
) -> float:
    """Air infiltration/leakage heat exchange between zone and exterior ambient.

    flow_proxy = veh_spd × area  (Simulink convention for exterior leakage)
    Q = (zone_temp - amb_t) × ConvCoBaseFlowEx(flow_proxy) × coef

    Positive base value when zone_temp > amb_t (zone is warmer).
    Caller multiplies by CHTD_QlossCo_P (-1) to convert to heat loss.

    Parameters
    ----------
    zone_temp : zone air temperature [°C]
    amb_t     : ambient temperature [°C]
    veh_spd   : vehicle speed [m/s or km/h, consistent with area scaling]
    area      : window/panel area as flow scaling parameter [m²]
    coef      : evaluated *LeakageCo_M LUT value at current AmbT [dimensionless]

    Returns
    -------
    Leakage base heat flux [W] (float). Positive = zone warmer than ambient.
    """
    flow_proxy = float(veh_spd) * float(area)
    return convection_heat(float(zone_temp), float(amb_t), flow_proxy, coef)


# ── 6. Q-bus inter-zone term ──────────────────────────────────────────────────

def q_interzone_from_bus(
    q_bus_value: float,
    gain: float,
) -> float:
    """Scale a pre-computed q-bus heat exchange term by a gain factor.

    Q = q_bus_value × gain

    Used when a neighboring zone exports its computed heat exchange on the
    q-bus and this zone receives it with a gain (typically CHTD_QgainCo_P = +1
    for inflow, or CHTD_QlossCo_P = -1 for outflow terms re-exported to
    another zone).  No additional physical logic is applied.

    Parameters
    ----------
    q_bus_value : pre-computed heat exchange value from q-bus [W]
    gain        : scaling factor (e.g. params.CHTD_QgainCo_P = +1.0)

    Returns
    -------
    Scaled heat term [W] (float).
    """
    return float(q_bus_value) * float(gain)


# ── 7. Gain / loss multiplier ─────────────────────────────────────────────────

def apply_gain_loss(q: float, gain: float) -> float:
    """Apply a QgainCo / QlossCo sign gain to a heat term.

    Q_scaled = q × gain

    Typical usage:
        apply_gain_loss(q, params.CHTD_QgainCo_P)   → +1 × q
        apply_gain_loss(q, params.CHTD_QlossCo_P)   → -1 × q

    Parameters
    ----------
    q    : heat term [W]
    gain : multiplicative gain (CHTD_QgainCo_P = +1 or CHTD_QlossCo_P = -1)

    Returns
    -------
    Scaled heat term [W] (float).
    """
    return float(q) * float(gain)


# ── 8. Net heat combiner ──────────────────────────────────────────────────────

def combine_terms(*terms: float) -> float:
    """Sum all zone heat terms into Q_net.

    Q_net = sum(terms)

    NaN and inf propagate per Python/IEEE default — not silently suppressed.

    Parameters
    ----------
    *terms : individual heat term values [W], already signed

    Returns
    -------
    Total net heat flux [W] (float).
    """
    return sum(float(t) for t in terms)


# ── 9. HeadTempSp RearSpf duct term (defect policy P4) ───────────────────────

def head_sp_rear_spf_hvac_term(
    x,
    u,
    params,
    lut_values: dict,
    mode: DefectMode = AS_FOUND,
) -> float:
    """HeadTempSp RearSpf duct convection heat term (defect P4 integration test).

    Computes only the RearSpf duct contribution to HeadTempSp Q_net.
    Full HeadTempSp heat balance is NOT computed here (that is T6 scope).

    AS_FOUND (Policy P4 — Simulink-faithful, default):
        T_src = x[HeadTempSd]  ← confirmed copy-paste defect D1.
        SubRef4 BusSelector reads HeadTempSd (x[5]) instead of HeadTempSp (x[6]).

    CORRECTED (physically correct):
        T_src = x[HeadTempSp]  ← the zone whose duct heat we are computing.

    Formula (both modes):
        Q = hvac_convection_heat(RearSpfTma, T_src, RearSpfFlow, coef) × QgainCo_P

    Signal indices (0-based):
        u[19]  RearSpfTma  — rear side-passenger floor supply temperature
        u[37]  RearSpfFlow — rear side-passenger floor volumetric flow
        x[5]   HeadTempSd  — AS_FOUND T_src (wrong zone)
        x[6]   HeadTempSp  — CORRECTED T_src (correct zone)

    Parameters
    ----------
    x          : 28-element CHTD state vector (array-like)
    u          : 54-element CHTD input vector (array-like)
    params     : CHTDParams instance (provides CHTD_QgainCo_P)
    lut_values : dict from eval_lut_map (must contain 'CHTD_RearSpfSpConvCo_M')
    mode       : DefectMode.AS_FOUND or CORRECTED

    Returns
    -------
    RearSpf duct heat contribution to HeadTempSp [W] (float).
    Positive = duct supply is warmer than T_src (heating the zone).
    """
    tma  = float(u[U_INDEX["RearSpfTma"]])    # u[19]
    flow = float(u[U_INDEX["RearSpfFlow"]])   # u[37]
    coef = float(lut_values["CHTD_RearSpfSpConvCo_M"])

    if mode == DefectMode.AS_FOUND:
        t_src = float(x[X_INDEX["HeadTempSd"]])   # x[5] — DEFECT: wrong zone
    else:
        t_src = float(x[X_INDEX["HeadTempSp"]])   # x[6] — physically correct

    q = hvac_convection_heat(tma, t_src, flow, coef)
    return q * float(params.CHTD_QgainCo_P)
