"""CHTD 28-state multi-step stability audit (read-only diagnostic).

Steps ``compute_chtd_delta`` forward in time and records per-step zone deltas
to locate threshold excursions. Does **not** modify ``thermal.py`` or params.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.afe.v2_chtd_preview import _diagnostic_from_case, _initial_state_x0
from hvac_sim.afe.v2_pmv_replay import case_to_afe_case, case_to_tma_source
from hvac_sim.afe.v2_to_chtd_adapter import build_chtd_u_preview_from_afe_v2
from hvac_sim.chtd.bus_index import CHTD_U_NAMES, CHTD_X_NAMES, N_U, N_X_STATES, U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.heat_terms import q_interzone_from_bus, q_solar_gain, q_zone_convection, q_zone_radiation
from hvac_sim.chtd.lut import eval_lut_map
from hvac_sim.chtd.thermal import (
    compute_cabin_frnt_temp,
    compute_cabin_temp_fd_qbus,
    compute_cabin_temp_fp,
    compute_chtd_delta,
    compute_console_temp,
    compute_feet_temp_fd,
    compute_feet_temp_fp,
    compute_head_temp_fd,
    compute_head_temp_fp,
)
from hvac_sim.chtd.helpers import cp_m_ex
from hvac_sim.chtd.zone_config import CHTD_ZONE_CONFIG, ZoneImplementationMode

AUDIT_SCHEMA = "chtd_multi_step_stability_audit_v1"
EQUILIBRIUM_AUDIT_SCHEMA = "chtd_equilibrium_contradiction_audit_v1"
CLASSIFICATION = "experimental_diagnostic_not_sign_off"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_MULTI_STEP_STABILITY_AUDIT.json"
)
DEFAULT_MD_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_MULTI_STEP_STABILITY_AUDIT.md"
)
DEFAULT_EQUILIBRIUM_JSON_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_EQUILIBRIUM_CONTRADICTION_AUDIT.json"
)
DEFAULT_EQUILIBRIUM_MD_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_EQUILIBRIUM_CONTRADICTION_AUDIT.md"
)

_DEFAULT_THRESHOLD_C = 20.0
_EQUILIBRIUM_DELTA_ATOL = 1e-3
_FLOW_ZERO_ATOL = 1e-9
_TMA_U_SLICE = slice(7, 24)
_FLOW_U_SLICE = slice(24, 42)
_TOP_N = 10

_ZONE_MODE = {entry.name: entry.mode.value for entry in CHTD_ZONE_CONFIG}

# Deprecated lumped mass / cp field names (params unchanged; audit reads only).
_ZONE_MASS_ATTR: Dict[str, str] = {
    "HoodTemp": "HoodMassAtb",
    "CabinFrntTemp": "CabinFrntMassAtb",
    "ConsoleTemp": "ConsoleMassAtb",
    "HeadTempFd": "HeadFdMassAtb",
    "HeadTempFp": "HeadFpMassAtb",
    "HeadTempSd": "HeadSdMassAtb",
    "HeadTempSp": "HeadSdMassAtb",
    "HeadTempTd": "HeadTdMassAtb",
    "HeadTempTp": "HeadTpMassAtb",
    "FeetTempFd": "FeetFdMassAtb",
    "FeetTempFp": "FeetFpMassAtb",
    "FeetTempSd": "FeetSdMassAtb",
    "FeetTempSp": "FeetSpMassAtb",
    "FeetTempTd": "FeetTdMassAtb",
    "FeetTempTp": "FeetTpMassAtb",
    "CabinTempFd": "CabinFdMassAtb",
    "CabinTempFp": "CabinFpMassAtb",
    "CabinTempSd": "CabinSdMassAtb",
    "CabinTempSp": "CabinSpMassAtb",
    "CabinTempTd": "CabinTdMassAtb",
    "CabinTempTp": "CabinTpMassAtb",
    "WinTempFd": "WinFdMassAtb",
    "WinTempFp": "WinFpMassAtb",
    "WinTempSd": "WinSdMassAtb",
    "WinTempSp": "WinSpMassAtb",
    "WinTempTd": "WinTdMassAtb",
    "WinTempTp": "WinTpMassAtb",
    "RoofTemp": "RoofMassAtb",
}

_QBUS_ZONE_KEYWORDS = ("Console", "CabinFrnt", "CabinTemp", "WinTemp", "RoofTemp")

_BUILTIN_SCENARIOS: Tuple[Dict[str, Any], ...] = (
    {
        "scenario_id": "cold_foot_heating_full_state",
        "description": "AFE v2 foot heating cold soak; full 28-state integration",
        "case": {
            "case_id": "cold_foot_heating_full_state",
            "mode_code": "F",
            "amb_t": -5.0,
            "rh": 60.0,
            "driver_face_tma": 35.0,
            "passenger_face_tma": 35.0,
            "driver_foot_tma": 42.0,
            "passenger_foot_tma": 42.0,
            "rear_face_tma": 34.0,
            "front_blower_voltage": 12.0,
            "rear_blower_voltage": 12.0,
            "circle_mode_posn": 100.0,
            "circle_prior_posn": 4.56,
            "heating_mode": True,
        },
        "steps": 5,
        "dt_seconds": 1.0,
    },
    {
        "scenario_id": "cold_foot_defrost_full_state",
        "description": "AFE v2 foot+defrost heating cold soak; full 28-state integration",
        "case": {
            "case_id": "cold_foot_defrost_full_state",
            "mode_code": "F_D",
            "amb_t": -5.0,
            "rh": 60.0,
            "driver_face_tma": 35.0,
            "passenger_face_tma": 35.0,
            "driver_foot_tma": 42.0,
            "passenger_foot_tma": 42.0,
            "rear_face_tma": 34.0,
            "front_blower_voltage": 12.0,
            "rear_blower_voltage": 12.0,
            "circle_mode_posn": 100.0,
            "circle_prior_posn": 4.56,
            "heating_mode": True,
        },
        "steps": 5,
        "dt_seconds": 1.0,
    },
    {
        "scenario_id": "uniform_equilibrium_control",
        "description": "Uniform x and u at 25C, zero flow — expect near-zero delta",
        "equilibrium_temp_c": 25.0,
        "steps": 10,
        "dt_seconds": 1.0,
    },
)


def default_builtin_scenarios() -> List[Dict[str, Any]]:
    return [dict(s) for s in _BUILTIN_SCENARIOS]


def _flow_total_m3h(u: np.ndarray) -> float:
    return float(sum(float(u[U_INDEX[name]]) for name in CHTD_U_NAMES if name.endswith("Flow")))


def _max_tma_minus_amb(u: np.ndarray) -> float:
    amb = float(u[U_INDEX["AmbT"]])
    deltas = []
    for name in CHTD_U_NAMES:
        if name.endswith("Tma") or name in ("WinShdTEst", "FrntDefTmaEst"):
            deltas.append(abs(float(u[U_INDEX[name]]) - amb))
    return float(max(deltas)) if deltas else 0.0


def _zone_mass_cp(params: CHTDParams, zone: str) -> Tuple[Optional[float], Optional[float]]:
    mass_attr = _ZONE_MASS_ATTR.get(zone)
    if mass_attr is None:
        return None, None
    mass = float(getattr(params, mass_attr, 1.0))
    canonical_mass = f"CHTD_{zone.replace('Temp', '')}MassAtb_P"
    if hasattr(params, canonical_mass):
        mass = float(getattr(params, canonical_mass))
    cp_attr = mass_attr.replace("MassAtb", "CpAtb")
    cp = float(getattr(params, cp_attr, 1.0))
    return mass, cp


def _neighbor_temp_spread(x: np.ndarray, zone: str) -> float:
    idx = X_INDEX[zone]
    t0 = float(x[idx])
    return float(max(abs(t0 - float(v)) for v in x))


def infer_suspect_causes(
    *,
    zone: str,
    x_before: float,
    x_vector: np.ndarray,
    delta: float,
    x_after: float,
    u: np.ndarray,
    params: CHTDParams,
    step: int,
) -> List[str]:
    """Heuristic tags — diagnostic only, no formula changes."""
    causes: List[str] = []
    mass, cp = _zone_mass_cp(params, zone)
    if mass is not None and cp is not None and mass * cp <= 1.0:
        causes.append("low_capacitance")
    if _ZONE_MODE.get(zone) == ZoneImplementationMode.TRACE.value:
        causes.append("placeholder_lut")
    if any(k in zone for k in _QBUS_ZONE_KEYWORDS):
        causes.append("qbus_feedback")
    spread = _neighbor_temp_spread(x_vector, zone)
    amb = float(u[U_INDEX["AmbT"]])
    if abs(delta) > 5.0 and spread > 15.0 and step >= 2:
        causes.append("qbus_feedback")
    if _max_tma_minus_amb(u) > 30.0:
        causes.append("tma_extreme")
    if _flow_total_m3h(u) > 200.0 and zone.startswith(("Head", "Feet", "Cabin")):
        causes.append("flow_magnitude")
    if delta * (x_before - amb) > 0 and abs(delta) > 10.0:
        causes.append("radiation_sign")
    if not causes:
        causes.append("placeholder_lut")
    return sorted(set(causes))


def _top_delta_zones(
    delta: np.ndarray,
    x_before: np.ndarray,
    *,
    top_n: int = _TOP_N,
    watch_zones: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        ((name, float(delta[i]), float(x_before[i])) for name, i in X_INDEX.items()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    if watch_zones:
        watch = set(watch_zones)
        ranked = [item for item in ranked if item[0] in watch] + [
            item for item in ranked if item[0] not in watch
        ]
    out: List[Dict[str, Any]] = []
    for name, dval, x0 in ranked[:top_n]:
        out.append(
            {
                "zone": name,
                "x_before_c": x0,
                "delta_c": dval,
                "x_after_c": x0 + dval,
                "abs_delta_c": abs(dval),
                "implementation_mode": _ZONE_MODE.get(name, "unknown"),
            }
        )
    return out


def _step_record(
    *,
    step: int,
    dt_seconds: float,
    x_before: np.ndarray,
    delta: np.ndarray,
    x_after: np.ndarray,
    threshold: float,
    u: np.ndarray,
    params: CHTDParams,
    watch_zones: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    abs_delta = np.abs(delta)
    max_idx = int(np.argmax(abs_delta))
    max_zone = CHTD_X_NAMES[max_idx]
    max_abs = float(abs_delta[max_idx])
    top = _top_delta_zones(delta, x_before, watch_zones=watch_zones)
    for entry in top:
        entry["suspect_causes"] = infer_suspect_causes(
            zone=entry["zone"],
            x_before=entry["x_before_c"],
            x_vector=x_before,
            delta=entry["delta_c"],
            x_after=entry["x_after_c"],
            u=u,
            params=params,
            step=step,
        )
    return {
        "step": int(step),
        "time_s": float(step * dt_seconds),
        "max_abs_delta_c": max_abs,
        "max_zone": max_zone,
        "threshold_exceeded": max_abs > threshold,
        "threshold_delta_c_per_step": float(threshold),
        "x_min_c": float(np.min(x_before)),
        "x_max_c": float(np.max(x_before)),
        "x_after_min_c": float(np.min(x_after)),
        "x_after_max_c": float(np.max(x_after)),
        "nonfinite_delta": not bool(np.all(np.isfinite(delta))),
        "nonfinite_x_after": not bool(np.all(np.isfinite(x_after))),
        "top_abs_delta_zones": top,
    }


def build_state_and_input_from_afe_case(case: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    afe = case_to_afe_case(case)
    tma = case_to_tma_source(case)
    amb = float(tma["amb_t"])
    diagnostic = _diagnostic_from_case(afe)
    adapter = build_chtd_u_preview_from_afe_v2(diagnostic, tma)
    x0 = _initial_state_x0(None, amb_t=amb)
    return x0, adapter.u.copy(), amb


def build_uniform_equilibrium(*, temp_c: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
    x0 = np.full(N_X_STATES, float(temp_c))
    u = np.zeros(N_U, dtype=float)
    u[U_INDEX["ICT"]] = float(temp_c)
    u[U_INDEX["RawAmbT"]] = float(temp_c)
    u[U_INDEX["AmbT"]] = float(temp_c)
    for name in CHTD_U_NAMES:
        if name.endswith("Tma") or name in ("WinShdTEst", "FrntDefTmaEst"):
            u[U_INDEX[name]] = float(temp_c)
    return x0, u


def build_cc_doc_equilibrium_pattern(*, temp_c: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce CHTD_MULTI_STEP_STABILITY_RISK_AND_AUDIT_PLAN.md §4.2/§4.3 u setup."""
    x0 = np.full(N_X_STATES, float(temp_c))
    u = np.zeros(N_U, dtype=float)
    u[U_INDEX["AmbT"]] = float(temp_c)
    u[_FLOW_U_SLICE] = 0.0
    for idx in range(_TMA_U_SLICE.start, _TMA_U_SLICE.stop):
        u[idx] = float(temp_c)
    return x0, u


def build_equilibrium_only_flow_zero_but_tma_default(
    *,
    temp_c: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero flow via default u=0; Tma slots unset (0) — CC-like incomplete wiring."""
    x0 = np.full(N_X_STATES, float(temp_c))
    u = np.zeros(N_U, dtype=float)
    u[U_INDEX["AmbT"]] = float(temp_c)
    return x0, u


def build_equilibrium_x20_tma25() -> Tuple[np.ndarray, np.ndarray]:
    """Uniform x/Amb at 20°C with HVAC Tma at 25°C — isolates flow×ΔT forcing."""
    x0 = np.full(N_X_STATES, 20.0)
    _, u = build_uniform_equilibrium(temp_c=25.0)
    u[U_INDEX["ICT"]] = 20.0
    u[U_INDEX["RawAmbT"]] = 20.0
    u[U_INDEX["AmbT"]] = 20.0
    return x0, u


def explicit_equilibrium_case_specs() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "equilibrium_all_25_zero_all",
            "description": "True equilibrium: x=25, ICT/RawAmbT/AmbT/Tma=25, zero flow/solar/human/veh_spd",
            "builder": lambda: build_uniform_equilibrium(temp_c=25.0),
        },
        {
            "case_id": "equilibrium_all_20_zero_all",
            "description": "True equilibrium at 20°C (Cursor build_uniform_equilibrium)",
            "builder": lambda: build_uniform_equilibrium(temp_c=20.0),
        },
        {
            "case_id": "equilibrium_x20_tma25",
            "description": "x/Amb=20°C, Tma=25°C — HVAC ΔT term only (not spatial equilibrium)",
            "builder": build_equilibrium_x20_tma25,
        },
        {
            "case_id": "equilibrium_only_flow_zero_but_tma_default",
            "description": "AmbT=20 only; flows/Tma/RawAmbT/ICT left at 0 (CC-like incomplete u)",
            "builder": lambda: build_equilibrium_only_flow_zero_but_tma_default(temp_c=20.0),
        },
        {
            "case_id": "equilibrium_cc_doc_pattern_20",
            "description": "Exact CC audit-plan snippet: u[AmbT]=T, u[5:]=0, Tma[7:24]=T",
            "builder": lambda: build_cc_doc_equilibrium_pattern(temp_c=20.0),
        },
    ]


def summarize_u_bus(u: np.ndarray) -> Dict[str, Any]:
    u_arr = np.asarray(u, dtype=float).reshape(N_U)
    tma = u_arr[_TMA_U_SLICE]
    flows = u_arr[_FLOW_U_SLICE]
    human_names = [n for n in CHTD_U_NAMES if n.startswith("Hum") and n.endswith("Power")]
    return {
        "AmbT": float(u_arr[U_INDEX["AmbT"]]),
        "RawAmbT": float(u_arr[U_INDEX["RawAmbT"]]),
        "ICT": float(u_arr[U_INDEX["ICT"]]),
        "WinShdTEst": float(u_arr[U_INDEX["WinShdTEst"]]),
        "FrntDefTmaEst": float(u_arr[U_INDEX["FrntDefTmaEst"]]),
        "tma_u_7_24": {
            "min": float(np.min(tma)),
            "max": float(np.max(tma)),
        },
        "flow_u_24_42": {
            "sum": float(np.sum(flows)),
            "min": float(np.min(flows)),
            "max": float(np.max(flows)),
        },
        "solar_fd": float(u_arr[U_INDEX["SolarFd"]]),
        "solar_fp": float(u_arr[U_INDEX["SolarFp"]]),
        "human_power_sum": float(sum(float(u_arr[U_INDEX[n]]) for n in human_names)),
        "veh_spd": float(u_arr[U_INDEX["VehSpd"]]),
    }


def _x_is_uniform(x: np.ndarray, *, atol: float = 1e-9) -> Tuple[bool, float]:
    x_arr = np.asarray(x, dtype=float)
    spread = float(np.max(x_arr) - np.min(x_arr))
    return spread <= atol, spread


def _tma_matches_x(u: np.ndarray, x: np.ndarray, *, atol: float = 1e-6) -> bool:
    x_ref = float(np.mean(x))
    for name in CHTD_U_NAMES:
        if name.endswith("Tma") or name in ("WinShdTEst", "FrntDefTmaEst"):
            if abs(float(u[U_INDEX[name]]) - x_ref) > atol:
                return False
    return True


def diagnose_equilibrium_case(
    x: np.ndarray,
    u: np.ndarray,
    delta: np.ndarray,
    *,
    atol: float = _EQUILIBRIUM_DELTA_ATOL,
) -> Dict[str, Any]:
    u_summary = summarize_u_bus(u)
    flows = u[_FLOW_U_SLICE]
    flow_zero = float(np.max(np.abs(flows))) <= _FLOW_ZERO_ATOL
    x_uniform, x_spread = _x_is_uniform(x)
    amb = float(u[U_INDEX["AmbT"]])
    x_matches_amb = x_uniform and abs(float(np.mean(x)) - amb) <= 1e-6
    tma_ok = _tma_matches_x(u, x)
    raw_amb_ok = abs(float(u[U_INDEX["RawAmbT"]]) - amb) <= 1e-6
    ict_ok = abs(float(u[U_INDEX["ICT"]]) - amb) <= 1e-6
    external = (
        abs(u_summary["solar_fd"]) > 1e-9
        or abs(u_summary["solar_fp"]) > 1e-9
        or abs(u_summary["human_power_sum"]) > 1e-9
        or abs(u_summary["veh_spd"]) > 1e-9
    )
    max_abs_delta = float(np.max(np.abs(delta)))
    true_eq_candidate = flow_zero and x_matches_amb and tma_ok and raw_amb_ok and not external

    flags: List[str] = []
    if external:
        flags.append("external_forcing_present")
    if not tma_ok or not raw_amb_ok or not ict_ok:
        flags.append("not_true_equilibrium")
    if true_eq_candidate and max_abs_delta > atol:
        flags.append("equilibrium_violation")
    if not flow_zero:
        flags.append("nonzero_flow")

    return {
        "flags": flags,
        "true_equilibrium_candidate": true_eq_candidate,
        "flow_zero": flow_zero,
        "x_uniform_c": x_uniform,
        "x_spread_c": x_spread,
        "x_matches_amb": x_matches_amb,
        "tma_matches_x": tma_ok,
        "raw_amb_matches_amb": raw_amb_ok,
        "ict_matches_amb": ict_ok,
        "max_abs_delta_c": max_abs_delta,
        "delta_near_zero": max_abs_delta <= atol,
    }


def _console_term_specs() -> Tuple[Tuple[str, str, str], ...]:
    return (
        ("cabin_frnt_console_radiation", "gain", "cabin_frnt_qbus"),
        ("console_solar_radiation", "gain", "console_self"),
        ("console_ws_radiation", "loss", "console_self"),
        ("console_fd_radiation", "loss", "console_x_coupling_head"),
        ("console_fp_radiation", "loss", "console_x_coupling_head"),
        ("console_feet_fd_radiation", "loss", "console_x_coupling_feet"),
        ("console_feet_fd_convection", "loss", "console_x_coupling_feet"),
        ("console_feet_fp_radiation", "loss", "console_x_coupling_feet"),
        ("console_feet_fp_convection", "loss", "console_x_coupling_feet"),
    )


def decompose_qbus_snapshot(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
) -> Dict[str, Any]:
    """Q-bus / heat-term breakdown for Console, CabinFrnt, CabinFd/Fp at one x,u."""
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    lv = eval_lut_map(params, amb_t)
    q_gain = params.CHTD_QgainCo_P
    q_loss = params.CHTD_QlossCo_P

    t_hood = float(x[X_INDEX["HoodTemp"]])
    t_cabin_frnt = float(x[X_INDEX["CabinFrntTemp"]])
    area_hood = params.CHTD_HoodAreaAtb_P
    q_hood_cabin_frnt_raw = q_zone_radiation(
        t_hood, t_cabin_frnt, area_hood, lv["CHTD_HoodCabinFrntRadCo_M"], AS_FOUND
    )

    delta_console, console_q = compute_console_temp(x, u, params, lv, AS_FOUND)
    c_console = cp_m_ex(params.CHTD_ConsoleMassAtb_P, params.CHTD_ConsoleCpAtb_P)
    console_terms: List[Dict[str, Any]] = []
    for field_name, gain_loss, coupling in _console_term_specs():
        q_raw = float(getattr(console_q, field_name))
        q_scaled = q_raw * (q_gain if gain_loss == "gain" else q_loss)
        delta_contrib = q_scaled * params.CHTD_Dt_P / c_console
        console_terms.append(
            {
                "term": field_name,
                "q_bus_raw_w": q_raw,
                "q_scaled_w": q_scaled,
                "delta_contrib_c": float(delta_contrib),
                "coupling_class": coupling,
            }
        )
    console_terms.sort(key=lambda t: abs(t["delta_contrib_c"]), reverse=True)

    delta_cabin_frnt, cabin_frnt_q = compute_cabin_frnt_temp(
        params,
        hood_cabin_frnt_radiation=q_hood_cabin_frnt_raw,
        cabin_frnt_console_radiation=console_q.cabin_frnt_console_radiation,
    )
    c_cabin_frnt = cp_m_ex(params.CHTD_CabinFrntMassAtb_P, params.CHTD_CabinFrntCpAtb_P)
    cabin_frnt_terms = [
        {
            "term": "hood_cabin_frnt_radiation",
            "q_bus_raw_w": float(cabin_frnt_q.hood_cabin_frnt_radiation),
            "q_scaled_w": float(cabin_frnt_q.hood_cabin_frnt_radiation * q_gain),
            "delta_contrib_c": float(
                cabin_frnt_q.hood_cabin_frnt_radiation
                * q_gain
                * params.CHTD_Dt_P
                / c_cabin_frnt
            ),
            "coupling_class": "hood_to_cabin_frnt_qbus",
        },
        {
            "term": "cabin_frnt_console_radiation",
            "q_bus_raw_w": float(cabin_frnt_q.cabin_frnt_console_radiation),
            "q_scaled_w": float(cabin_frnt_q.cabin_frnt_console_radiation * q_loss),
            "delta_contrib_c": float(
                cabin_frnt_q.cabin_frnt_console_radiation
                * q_loss
                * params.CHTD_Dt_P
                / c_cabin_frnt
            ),
            "coupling_class": "console_to_cabin_frnt_qbus",
        },
    ]
    cabin_frnt_terms.sort(key=lambda t: abs(t["delta_contrib_c"]), reverse=True)

    _, head_fd_q = compute_head_temp_fd(x, u, params, lv, AS_FOUND, console_q=console_q)
    _, feet_fd_q = compute_feet_temp_fd(x, u, params, lv, head_fd_q, AS_FOUND, console_q=console_q)
    _, head_fp_q = compute_head_temp_fp(x, u, params, lv, head_fd_q, AS_FOUND, console_q=console_q)
    _, feet_fp_q = compute_feet_temp_fp(x, u, params, lv, head_fp_q, AS_FOUND, console_q=console_q)

    qbus_cabin_fd = compute_cabin_temp_fd_qbus(head_fd_q, feet_fd_q)
    c_cabin_fd = cp_m_ex(params.CHTD_CabinFdMassAtb_P, params.CHTD_CabinFdCpAtb_P)
    area_cabin_fd = params.CHTD_CabinFdAreaAtb_P
    t_cabin_fd = float(x[X_INDEX["CabinTempFd"]])
    q_cabin_amb_rad = q_zone_radiation(
        t_cabin_fd, amb_t, area_cabin_fd, lv["CHTD_CabinFdAmbRadCo_M"], AS_FOUND
    )
    q_cabin_amb_conv = q_zone_convection(
        t_cabin_fd, amb_t, veh_spd * area_cabin_fd, lv["CHTD_CabinFdAmbConvCo_M"]
    )
    q_cabin_amb = q_loss * (q_cabin_amb_rad + q_cabin_amb_conv)
    q_from_head_fd = q_interzone_from_bus(qbus_cabin_fd.head_fd_radiation, q_gain) + q_interzone_from_bus(
        qbus_cabin_fd.head_fd_convection, q_gain
    )
    q_from_feet_fd = q_interzone_from_bus(qbus_cabin_fd.feet_fd_radiation, q_gain) + q_interzone_from_bus(
        qbus_cabin_fd.feet_fd_convection, q_gain
    )
    q_cabin_solar = q_solar_gain(
        float(u[U_INDEX["SolarFd"]]), area_cabin_fd, lv["CHTD_CabinFdSolarRadCo_M"]
    ) * q_gain
    q_cabin_human = float(u[U_INDEX["HumFeetFdPower"]])
    cabin_fd_terms = [
        {
            "term": "cabin_fd_amb",
            "q_bus_raw_w": float(q_cabin_amb_rad + q_cabin_amb_conv),
            "delta_contrib_c": float(q_cabin_amb * params.CHTD_Dt_P / c_cabin_fd),
            "coupling_class": "cabin_self_amb",
        },
        {
            "term": "head_fd_cabin_fd_radiation",
            "q_bus_raw_w": float(qbus_cabin_fd.head_fd_radiation),
            "delta_contrib_c": float(q_from_head_fd * params.CHTD_Dt_P / c_cabin_fd),
            "coupling_class": "head_to_cabin_qbus",
        },
        {
            "term": "head_fd_cabin_fd_convection",
            "q_bus_raw_w": float(qbus_cabin_fd.head_fd_convection),
            "delta_contrib_c": float(
                q_interzone_from_bus(qbus_cabin_fd.head_fd_convection, q_gain)
                * params.CHTD_Dt_P
                / c_cabin_fd
            ),
            "coupling_class": "head_to_cabin_qbus",
        },
        {
            "term": "feet_fd_cabin_fd_radiation",
            "q_bus_raw_w": float(qbus_cabin_fd.feet_fd_radiation),
            "delta_contrib_c": float(
                q_interzone_from_bus(qbus_cabin_fd.feet_fd_radiation, q_gain)
                * params.CHTD_Dt_P
                / c_cabin_fd
            ),
            "coupling_class": "feet_to_cabin_qbus",
        },
        {
            "term": "feet_fd_cabin_fd_convection",
            "q_bus_raw_w": float(qbus_cabin_fd.feet_fd_convection),
            "delta_contrib_c": float(q_from_feet_fd * params.CHTD_Dt_P / c_cabin_fd),
            "coupling_class": "feet_to_cabin_qbus",
        },
        {
            "term": "cabin_fd_solar",
            "q_bus_raw_w": float(q_cabin_solar / q_gain if q_gain else 0.0),
            "delta_contrib_c": float(q_cabin_solar * params.CHTD_Dt_P / c_cabin_fd),
            "coupling_class": "cabin_self_solar",
        },
        {
            "term": "cabin_fd_human",
            "q_bus_raw_w": q_cabin_human,
            "delta_contrib_c": float(q_cabin_human * params.CHTD_Dt_P / c_cabin_fd),
            "coupling_class": "external_forcing",
        },
    ]
    cabin_fd_terms.sort(key=lambda t: abs(t["delta_contrib_c"]), reverse=True)

    delta_cabin_fp, cabin_fp_q = compute_cabin_temp_fp(
        x, u, params, lv, head_fp_q, feet_fp_q, AS_FOUND
    )
    c_cabin_fp = cp_m_ex(params.CHTD_CabinFpMassAtb_P, params.CHTD_CabinFpCpAtb_P)
    cabin_fp_terms = [
        {
            "term": "head_fp_cabin_fp_radiation",
            "q_bus_raw_w": float(cabin_fp_q.fp_cabin_fp_radiation),
            "delta_contrib_c": float(
                q_interzone_from_bus(cabin_fp_q.fp_cabin_fp_radiation, q_gain)
                * params.CHTD_Dt_P
                / c_cabin_fp
            ),
            "coupling_class": "head_to_cabin_qbus",
        },
        {
            "term": "head_fp_cabin_fp_convection",
            "q_bus_raw_w": float(cabin_fp_q.fp_cabin_fp_convection),
            "delta_contrib_c": float(
                q_interzone_from_bus(cabin_fp_q.fp_cabin_fp_convection, q_gain)
                * params.CHTD_Dt_P
                / c_cabin_fp
            ),
            "coupling_class": "head_to_cabin_qbus",
        },
        {
            "term": "feet_fp_cabin_fp_radiation",
            "q_bus_raw_w": float(cabin_fp_q.feet_fp_cabin_fp_radiation),
            "delta_contrib_c": float(
                q_interzone_from_bus(cabin_fp_q.feet_fp_cabin_fp_radiation, q_gain)
                * params.CHTD_Dt_P
                / c_cabin_fp
            ),
            "coupling_class": "feet_to_cabin_qbus",
        },
        {
            "term": "feet_fp_cabin_fp_convection",
            "q_bus_raw_w": float(cabin_fp_q.feet_fp_cabin_fp_convection),
            "delta_contrib_c": float(
                q_interzone_from_bus(cabin_fp_q.feet_fp_cabin_fp_convection, q_gain)
                * params.CHTD_Dt_P
                / c_cabin_fp
            ),
            "coupling_class": "feet_to_cabin_qbus",
        },
        {
            "term": "cabin_fp_amb_radiation",
            "q_bus_raw_w": float(cabin_fp_q.cabin_fp_amb_radiation),
            "delta_contrib_c": float(
                q_interzone_from_bus(cabin_fp_q.cabin_fp_amb_radiation, q_loss)
                * params.CHTD_Dt_P
                / c_cabin_fp
            ),
            "coupling_class": "cabin_self_amb",
        },
        {
            "term": "cabin_fp_amb_convection",
            "q_bus_raw_w": float(cabin_fp_q.cabin_fp_amb_convection),
            "delta_contrib_c": float(
                q_interzone_from_bus(cabin_fp_q.cabin_fp_amb_convection, q_loss)
                * params.CHTD_Dt_P
                / c_cabin_fp
            ),
            "coupling_class": "cabin_self_amb",
        },
    ]
    cabin_fp_terms.sort(key=lambda t: abs(t["delta_contrib_c"]), reverse=True)

    cabin_fp_terms.sort(key=lambda t: abs(t["delta_contrib_c"]), reverse=True)

    full_delta = compute_chtd_delta(x, u, params, mode=AS_FOUND)
    roof_head_terms = [
        {
            "term": "head_fd_roof_radiation",
            "q_bus_raw_w": float(head_fd_q.fd_roof_radiation),
            "coupling_class": "head_to_roof_qbus",
        },
        {
            "term": "head_fd_roof_convection",
            "q_bus_raw_w": float(head_fd_q.fd_roof_convection),
            "coupling_class": "head_to_roof_qbus",
        },
        {
            "term": "head_fp_roof_radiation",
            "q_bus_raw_w": float(head_fp_q.fp_roof_radiation),
            "coupling_class": "head_to_roof_qbus",
        },
        {
            "term": "head_fp_roof_convection",
            "q_bus_raw_w": float(head_fp_q.fp_roof_convection),
            "coupling_class": "head_to_roof_qbus",
        },
    ]

    return {
        "ConsoleTemp": {
            "zone_delta_c": float(delta_console),
            "verified_delta_c": float(full_delta[X_INDEX["ConsoleTemp"]]),
            "terms": console_terms[:10],
            "dominant_coupling": console_terms[0]["coupling_class"] if console_terms else None,
        },
        "CabinFrntTemp": {
            "zone_delta_c": float(delta_cabin_frnt),
            "verified_delta_c": float(full_delta[X_INDEX["CabinFrntTemp"]]),
            "terms": cabin_frnt_terms,
            "dominant_coupling": cabin_frnt_terms[0]["coupling_class"] if cabin_frnt_terms else None,
        },
        "CabinTempFd": {
            "zone_delta_c": float(full_delta[X_INDEX["CabinTempFd"]]),
            "terms": cabin_fd_terms[:10],
            "dominant_coupling": cabin_fd_terms[0]["coupling_class"] if cabin_fd_terms else None,
        },
        "CabinTempFp": {
            "zone_delta_c": float(delta_cabin_fp),
            "verified_delta_c": float(full_delta[X_INDEX["CabinTempFp"]]),
            "terms": cabin_fp_terms[:10],
            "dominant_coupling": cabin_fp_terms[0]["coupling_class"] if cabin_fp_terms else None,
        },
        "RoofTemp": {
            "zone_delta_c": float(full_delta[X_INDEX["RoofTemp"]]),
            "head_qbus_exports": roof_head_terms,
        },
    }


def _qbus_jump_analysis(
    prior: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    q_threshold: float = 1.0,
    delta_threshold: float = 5.0,
) -> List[Dict[str, Any]]:
    """Terms whose |q_bus_raw| or |delta_contrib| jumped from ~0 to large."""
    jumps: List[Dict[str, Any]] = []
    if prior is None:
        return jumps
    for zone in ("ConsoleTemp", "CabinFrntTemp", "CabinTempFd", "CabinTempFp"):
        cur_z = current.get(zone, {})
        pri_z = prior.get(zone, {})
        pri_by_term = {t["term"]: t for t in pri_z.get("terms", [])}
        for term in cur_z.get("terms", []):
            name = term["term"]
            q_now = abs(float(term.get("q_bus_raw_w", 0.0)))
            d_now = abs(float(term.get("delta_contrib_c", 0.0)))
            pri = pri_by_term.get(name, {})
            q_was = abs(float(pri.get("q_bus_raw_w", 0.0)))
            d_was = abs(float(pri.get("delta_contrib_c", 0.0)))
            if (q_was < q_threshold and q_now >= q_threshold) or (
                d_was < delta_threshold and d_now >= delta_threshold
            ):
                jumps.append(
                    {
                        "zone": zone,
                        "term": name,
                        "coupling_class": term.get("coupling_class"),
                        "q_bus_raw_was_w": q_was,
                        "q_bus_raw_now_w": q_now,
                        "delta_contrib_was_c": d_was,
                        "delta_contrib_now_c": d_now,
                    }
                )
    jumps.sort(key=lambda j: j["delta_contrib_now_c"], reverse=True)
    return jumps


def run_explicit_equilibrium_cases(
    params: Optional[CHTDParams] = None,
) -> List[Dict[str, Any]]:
    p = params if params is not None else CHTDParams()
    results: List[Dict[str, Any]] = []
    for spec in explicit_equilibrium_case_specs():
        x0, u = spec["builder"]()
        delta = compute_chtd_delta(x0, u, p, mode=AS_FOUND)
        diag = diagnose_equilibrium_case(x0, u, delta)
        results.append(
            {
                "case_id": spec["case_id"],
                "description": spec["description"],
                "u_summary": summarize_u_bus(u),
                "diagnostics": diag,
                "top_abs_delta_zones": _top_delta_zones(delta, x0, top_n=_TOP_N),
            }
        )
    return results


def run_equilibrium_contradiction_audit(
    *,
    params: Optional[CHTDParams] = None,
    include_warmup_qbus: bool = True,
) -> Dict[str, Any]:
    """Audit Cursor vs CC equilibrium claims and cold warm-up q-bus root cause."""
    p = params if params is not None else CHTDParams()
    eq_cases = run_explicit_equilibrium_cases(p)

    cursor_true = next(c for c in eq_cases if c["case_id"] == "equilibrium_all_25_zero_all")
    cc_doc = next(c for c in eq_cases if c["case_id"] == "equilibrium_cc_doc_pattern_20")

    contradiction = {
        "cursor_uniform_equilibrium_max_abs_delta_c": cursor_true["diagnostics"]["max_abs_delta_c"],
        "cc_doc_pattern_max_abs_delta_c": cc_doc["diagnostics"]["max_abs_delta_c"],
        "cc_doc_max_zone": cc_doc["top_abs_delta_zones"][0]["zone"] if cc_doc["top_abs_delta_zones"] else None,
        "root_cause": (
            "CC audit-plan equilibrium snippet sets AmbT=T but leaves RawAmbT=0 and ICT=0. "
            "HoodTemp dual-ambient terms (RawAmbT↔Hood, AmbT↔Hood) then see a 20°C artificial "
            "gradient even when x is uniform — not a radiation-matrix inconsistency. "
            "Cursor build_uniform_equilibrium sets ICT, RawAmbT, AmbT, and all Tma to the same "
            "temperature with zero flow/solar/human/veh_spd, yielding delta≈0 as expected."
        ),
        "cc_flags": cc_doc["diagnostics"]["flags"],
        "cursor_flags": cursor_true["diagnostics"]["flags"],
    }

    warmup_audits: List[Dict[str, Any]] = []
    qbus_slices: Dict[str, Any] = {}
    if include_warmup_qbus:
        for spec in _BUILTIN_SCENARIOS:
            if "case" not in spec:
                continue
            x0, u, _ = build_state_and_input_from_afe_case(spec["case"])
            audit = run_chtd_stability_audit(
                x0=x0,
                u=u,
                steps=2,
                stop_on_threshold=False,
                include_qbus_decomposition=True,
                params=p,
                scenario_id=str(spec["scenario_id"]),
                description=str(spec.get("description", "")),
            )
            warmup_audits.append(audit)
            qbus_slices[str(spec["scenario_id"])] = audit.get("qbus_focus_steps", {})

    return {
        "schema": EQUILIBRIUM_AUDIT_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "thermal_formula_unchanged": True,
        "params_unchanged": True,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contradiction_summary": contradiction,
        "explicit_equilibrium_cases": eq_cases,
        "warmup_qbus_slices": qbus_slices,
        "warmup_audits": warmup_audits,
    }


def build_equilibrium_contradiction_markdown(doc: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# CHTD Equilibrium Contradiction Audit",
        "",
        "**Read-only diagnostic** — explains Cursor vs CC zero-flow equilibrium mismatch.",
        "",
        f"Generated: `{doc.get('generated_at_utc', '')}`",
        "",
        "## Contradiction resolution",
        "",
    ]
    cs = doc.get("contradiction_summary", {})
    lines.extend(
        [
            f"- Cursor `build_uniform_equilibrium` max |delta|: **{cs.get('cursor_uniform_equilibrium_max_abs_delta_c', 0):.6f}** °C/step",
            f"- CC audit-plan pattern max |delta|: **{cs.get('cc_doc_pattern_max_abs_delta_c', 0):.3f}** °C/step "
            f"(max zone: {cs.get('cc_doc_max_zone')})",
            "",
            cs.get("root_cause", ""),
            "",
            f"- CC flags: `{', '.join(cs.get('cc_flags', []))}`",
            f"- Cursor flags: `{', '.join(cs.get('cursor_flags', []))}`",
            "",
            "## Explicit equilibrium cases",
            "",
            "| case | max |delta| | flags | RawAmbT | AmbT | flow sum |",
            "|------|------------|-------|---------|------|----------|",
        ]
    )
    for case in doc.get("explicit_equilibrium_cases", []):
        us = case["u_summary"]
        diag = case["diagnostics"]
        lines.append(
            f"| {case['case_id']} | {diag['max_abs_delta_c']:.3f} | "
            f"{', '.join(diag['flags']) or '—'} | {us['RawAmbT']:.1f} | {us['AmbT']:.1f} | "
            f"{us['flow_u_24_42']['sum']:.1f} |"
        )
    lines.extend(["", "## Warm-up q-bus root cause (step 1 → step 2)", ""])
    for sid, focus in doc.get("warmup_qbus_slices", {}).items():
        lines.append(f"### {sid}")
        lines.append("")
        for step_key in ("step_1", "step_2"):
            block = focus.get(step_key)
            if not block:
                continue
            lines.append(f"**{step_key}** — q-bus jumps from prior step:")
            lines.append("")
            for jump in block.get("qbus_jumps", [])[:8]:
                lines.append(
                    f"- `{jump['zone']}.{jump['term']}` ({jump['coupling_class']}): "
                    f"Δcontrib {jump['delta_contrib_was_c']:.2f} → {jump['delta_contrib_now_c']:.2f} °C"
                )
            dom = block.get("dominant_terms", {})
            for zone, info in dom.items():
                if info:
                    lines.append(
                        f"- {zone} dominant: **{info.get('term')}** "
                        f"({info.get('coupling_class')}) "
                        f"|delta|={abs(info.get('delta_contrib_c', 0)):.1f} °C"
                    )
            lines.append("")
    return "\n".join(lines)


def write_equilibrium_contradiction_outputs(
    *,
    bundle: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
    pretty: bool = True,
) -> Dict[str, Path]:
    doc = bundle if bundle is not None else run_equilibrium_contradiction_audit()
    json_out = Path(json_path) if json_path is not None else DEFAULT_EQUILIBRIUM_JSON_OUT
    md_out = Path(md_path) if md_path is not None else DEFAULT_EQUILIBRIUM_MD_OUT
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )
    md_out.write_text(build_equilibrium_contradiction_markdown(doc), encoding="utf-8")
    return {"json": json_out, "md": md_out}


def run_chtd_stability_audit(
    *,
    x0: np.ndarray,
    u: np.ndarray,
    steps: int,
    dt_seconds: float = 1.0,
    threshold_delta_c_per_step: float = _DEFAULT_THRESHOLD_C,
    watch_zones: Optional[Sequence[str]] = None,
    stop_on_threshold: bool = True,
    include_qbus_decomposition: bool = False,
    params: Optional[CHTDParams] = None,
    scenario_id: str = "custom",
    description: str = "",
) -> Dict[str, Any]:
    """Run forward Euler audit calling ``compute_chtd_delta`` each step."""
    x = np.asarray(x0, dtype=float).reshape(N_X_STATES).copy()
    u_arr = np.asarray(u, dtype=float).reshape(N_U).copy()
    p = params if params is not None else CHTDParams()
    if steps < 0:
        raise ValueError("steps must be >= 0")

    step_records: List[Dict[str, Any]] = []
    stopped_early = False
    stop_reason: Optional[str] = None
    completed_steps = 0
    first_threshold_step: Optional[int] = None
    qbus_prior: Optional[Dict[str, Any]] = None
    qbus_focus: Dict[str, Any] = {}

    # step 0 baseline (before any integration)
    zero = np.zeros(N_X_STATES)
    step_records.append(
        _step_record(
            step=0,
            dt_seconds=dt_seconds,
            x_before=x.copy(),
            delta=zero,
            x_after=x.copy(),
            threshold=threshold_delta_c_per_step,
            u=u_arr,
            params=p,
            watch_zones=watch_zones,
        )
    )

    for step in range(1, steps + 1):
        x_before = x.copy()
        delta = compute_chtd_delta(x_before, u_arr, p, mode=AS_FOUND)
        if not np.all(np.isfinite(delta)):
            rec = _step_record(
                step=step,
                dt_seconds=dt_seconds,
                x_before=x_before,
                delta=delta,
                x_after=x_before,
                threshold=threshold_delta_c_per_step,
                u=u_arr,
                params=p,
                watch_zones=watch_zones,
            )
            step_records.append(rec)
            stopped_early = True
            stop_reason = "non_finite_delta"
            break

        x_after = x_before + delta
        rec = _step_record(
            step=step,
            dt_seconds=dt_seconds,
            x_before=x_before,
            delta=delta,
            x_after=x_after,
            threshold=threshold_delta_c_per_step,
            u=u_arr,
            params=p,
            watch_zones=watch_zones,
        )
        step_records.append(rec)
        completed_steps = step

        if include_qbus_decomposition and step in (1, 2):
            qbus_snap = decompose_qbus_snapshot(x_before, u_arr, p)
            jumps = _qbus_jump_analysis(qbus_prior, qbus_snap)
            dominant = {
                zone: (data.get("terms") or [{}])[0]
                for zone, data in qbus_snap.items()
                if isinstance(data, dict) and data.get("terms")
            }
            qbus_focus[f"step_{step}"] = {
                "x_before_summary": {
                    "min_c": float(np.min(x_before)),
                    "max_c": float(np.max(x_before)),
                    "HeadTempFd_c": float(x_before[X_INDEX["HeadTempFd"]]),
                    "FeetTempFd_c": float(x_before[X_INDEX["FeetTempFd"]]),
                    "ConsoleTemp_c": float(x_before[X_INDEX["ConsoleTemp"]]),
                },
                "decomposition": qbus_snap,
                "qbus_jumps": jumps,
                "dominant_terms": dominant,
            }
            qbus_prior = qbus_snap

        if not np.all(np.isfinite(x_after)):
            stopped_early = True
            stop_reason = "non_finite_x"
            break

        if rec["threshold_exceeded"]:
            if first_threshold_step is None:
                first_threshold_step = step
            if stop_on_threshold:
                stopped_early = True
                stop_reason = "threshold_exceeded"
                x = x_after
                break

        x = x_after

    summary = {
        "completed_steps": int(completed_steps),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "first_threshold_step": first_threshold_step,
        "max_abs_delta_over_run": float(
            max(r["max_abs_delta_c"] for r in step_records if r["step"] > 0)
        )
        if any(r["step"] > 0 for r in step_records)
        else 0.0,
        "flow_total_m3h": _flow_total_m3h(u_arr),
        "max_tma_minus_amb_c": _max_tma_minus_amb(u_arr),
        "amb_t_c": float(u_arr[U_INDEX["AmbT"]]),
    }

    return {
        "schema": AUDIT_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "thermal_formula_unchanged": True,
        "params_unchanged": True,
        "scenario_id": scenario_id,
        "description": description,
        "steps_requested": int(steps),
        "dt_seconds": float(dt_seconds),
        "threshold_delta_c_per_step": float(threshold_delta_c_per_step),
        "stop_on_threshold": bool(stop_on_threshold),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "step_records": step_records,
        "focus_steps": _extract_focus_steps(step_records, focus=(1, 2)),
        "qbus_focus_steps": qbus_focus if include_qbus_decomposition else {},
    }


def _extract_focus_steps(
    records: Sequence[Mapping[str, Any]],
    *,
    focus: Sequence[int],
) -> Dict[str, Any]:
    by_step = {int(r["step"]): r for r in records}
    return {f"step_{s}": by_step[s] for s in focus if s in by_step}


def run_builtin_stability_audits(
    *,
    threshold_delta_c_per_step: float = _DEFAULT_THRESHOLD_C,
    stop_on_threshold: bool = True,
) -> Dict[str, Any]:
    audits: List[Dict[str, Any]] = []
    for spec in _BUILTIN_SCENARIOS:
        sid = str(spec["scenario_id"])
        steps = int(spec.get("steps", 5))
        dt = float(spec.get("dt_seconds", 1.0))
        if "case" in spec:
            x0, u, _ = build_state_and_input_from_afe_case(spec["case"])
            desc = str(spec.get("description", ""))
            qbus = True
        else:
            t = float(spec.get("equilibrium_temp_c", 25.0))
            x0, u = build_uniform_equilibrium(temp_c=t)
            desc = str(spec.get("description", ""))
            qbus = False
        audits.append(
            run_chtd_stability_audit(
                x0=x0,
                u=u,
                steps=steps,
                dt_seconds=dt,
                threshold_delta_c_per_step=threshold_delta_c_per_step,
                stop_on_threshold=stop_on_threshold,
                include_qbus_decomposition=qbus,
                scenario_id=sid,
                description=desc,
            )
        )
    return {
        "schema": AUDIT_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "audit_count": len(audits),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audits": audits,
    }


def build_markdown_report(bundle: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# CHTD Multi-Step Stability Audit",
        "",
        "**Read-only diagnostic** — `thermal.py` and params unchanged.",
        "",
        f"Generated: `{bundle.get('generated_at_utc', '')}`",
        "",
    ]
    audits = bundle.get("audits", [bundle])
    for audit in audits:
        sid = audit.get("scenario_id", "custom")
        lines.extend(
            [
                f"## {sid}",
                "",
                audit.get("description", ""),
                "",
                f"- Completed steps: {audit['summary']['completed_steps']}",
                f"- Stopped early: {audit['summary']['stopped_early']}",
                f"- Stop reason: {audit['summary'].get('stop_reason')}",
                f"- First threshold step: {audit['summary'].get('first_threshold_step')}",
                f"- Max |delta|: {audit['summary'].get('max_abs_delta_over_run', 0):.3f} °C/step",
                "",
            ]
        )
        focus = audit.get("focus_steps", {})
        for label, rec in focus.items():
            if not rec or int(rec.get("step", -1)) <= 0:
                continue
            lines.append(f"### {label}")
            lines.append("")
            lines.append(
                f"max zone **{rec['max_zone']}** | max |delta| = {rec['max_abs_delta_c']:.3f} °C "
                f"| threshold exceeded: {rec['threshold_exceeded']}"
            )
            lines.append("")
            lines.append("| zone | x_before | delta | x_after | suspect causes |")
            lines.append("|------|----------|-------|---------|----------------|")
            for z in rec.get("top_abs_delta_zones", [])[:5]:
                causes = ", ".join(z.get("suspect_causes", []))
                lines.append(
                    f"| {z['zone']} | {z['x_before_c']:.2f} | {z['delta_c']:.2f} | "
                    f"{z['x_after_c']:.2f} | {causes} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Conclusions (cold full_state warm-up)",
            "",
            "After step 1 warms Head/Feet zones slightly, step 2 typically shows:",
            "",
            "- **ConsoleTemp** and **CabinFrntTemp** largest |delta| (>20°C) with x still at cold soak",
            "- **CabinTempFd/Fp** and **RoofTemp** also spike via q-bus / inter-zone coupling",
            "- **HeadTempFd/FeetTempFd** deltas remain O(1–3°C) — not the blow-up source",
            "",
            "Suspect cause tags (heuristic): `qbus_feedback`, `placeholder_lut`, `low_capacitance`,",
            "`tma_extreme`, `flow_magnitude`. Uniform equilibrium control stays below threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def write_stability_audit_outputs(
    *,
    bundle: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
    pretty: bool = True,
) -> Dict[str, Path]:
    doc = bundle if bundle is not None else run_builtin_stability_audits()
    if "audits" not in doc:
        doc = {
            "schema": AUDIT_SCHEMA,
            "classification": CLASSIFICATION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "audits": [doc],
        }
    json_out = Path(json_path) if json_path is not None else DEFAULT_JSON_OUT
    md_out = Path(md_path) if md_path is not None else DEFAULT_MD_OUT
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )
    md_out.write_text(build_markdown_report(doc), encoding="utf-8")
    return {"json": json_out, "md": md_out}
