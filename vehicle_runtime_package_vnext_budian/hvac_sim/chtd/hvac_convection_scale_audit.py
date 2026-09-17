"""HVAC duct convection unit and scale audit (read-only diagnostic).

Audits ConvCoBaseFlowEx / flow units / LUT placeholder semantics for all direct
HVAC duct convection terms. Never mutates ``thermal.py`` or default params.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.chtd.bus_index import U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND, conv_co_base_flow_ex, cp_v_ex
from hvac_sim.chtd.heat_terms import q_hvac_duct
from hvac_sim.chtd.lut import eval_lut_map, lookup_chtd_amb
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.physical_params import build_physical_capacity_params
from hvac_sim.chtd.stability_audit import build_state_and_input_from_afe_case
from hvac_sim.chtd.thermal import compute_chtd_delta

AUDIT_SCHEMA = "chtd_hvac_convection_scale_audit_v1"
CLASSIFICATION = "experimental_diagnostic_not_sign_off"
DISCLAIMER = (
    "Read-only unit/scale audit on copied interpretation paths. Not calibration "
    "sign-off. Default CHTDParams, thermal.py, and golden unchanged."
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_HVAC_CONVECTION_SCALE_AUDIT.json"
)
DEFAULT_MD_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_HVAC_CONVECTION_SCALE_AUDIT.md"
)

_M3H_TO_M3S = 1.0 / 3600.0
_CONV_CURVE_FLOWS = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 300.0)

_AUDIT_CASE = {
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
}


@dataclass(frozen=True)
class _HvacDuctSpec:
    zone: str
    zone_group: str
    duct_name: str
    tma_u_key: str
    flow_u_key: str
    lut_name: str
    air_volume_attr: str
    zone_temp_x_key: Optional[str] = None
    notes: str = ""


def _hvac_duct_registry() -> Tuple[_HvacDuctSpec, ...]:
    head_v = "CHTD_HeadAirVAtb_P"
    feet_v = "CHTD_FeetAirVAtb_P"
    return (
        # HeadTempFd
        _HvacDuctSpec("HeadTempFd", "Head", "FrntFdDef", "FrntDefTmaEst", "FrntFdDefFlow", "CHTD_FdDefFdConvCo_M", head_v),
        _HvacDuctSpec("HeadTempFd", "Head", "FrntFdv", "FrntFdvTma", "FrntFdvFlow", "CHTD_FdvFdConvCo_M", head_v),
        _HvacDuctSpec("HeadTempFd", "Head", "FrntFdfHead", "FrntFdfTma", "FrntFdfFlow", "CHTD_FdfFdConvCo_M", head_v,
                      notes="Same FrntFdf duct as FeetTempFd; head-side LUT"),
        # HeadTempFp
        _HvacDuctSpec("HeadTempFp", "Head", "FrntFpDef", "FrntDefTmaEst", "FrntFpDefFlow", "CHTD_FpDefFpConvCo_M", head_v),
        _HvacDuctSpec("HeadTempFp", "Head", "FrntFpv", "FrntFpvTma", "FrntFpvFlow", "CHTD_FpvFpConvCo_M", head_v),
        _HvacDuctSpec("HeadTempFp", "Head", "FrntFpfHead", "FrntFpfTma", "FrntFpfFlow", "CHTD_FpfFpConvCo_M", head_v),
        # HeadTempSd
        _HvacDuctSpec("HeadTempSd", "Head", "FrntSdv", "FrntSdvTma", "FrntSdvFlow", "CHTD_SdvSdConvCo_M", head_v),
        _HvacDuctSpec("HeadTempSd", "Head", "FrntSdfHead", "FrntSdfTma", "FrntSdfFlow", "CHTD_SdfSdConvCo_M", head_v),
        _HvacDuctSpec("HeadTempSd", "Head", "RearSdv", "RearSdvTma", "RearSdvFlow", "CHTD_RearSdvSdConvCo_M", head_v),
        _HvacDuctSpec("HeadTempSd", "Head", "RearSdfHead", "RearSdfTma", "RearSdfFlow", "CHTD_RearSdfSdConvCo_M", head_v),
        # HeadTempSp (inline in compute_chtd_delta)
        _HvacDuctSpec("HeadTempSp", "Head", "FrntSpv", "FrntSpvTma", "FrntSpvFlow", "CHTD_SpvSpConvCo_M", head_v),
        _HvacDuctSpec("HeadTempSp", "Head", "FrntSpfHead", "FrntSpfTma", "FrntSpfFlow", "CHTD_SpfSpConvCo_M", head_v),
        _HvacDuctSpec("HeadTempSp", "Head", "RearSpv", "RearSpvTma", "RearSpvFlow", "CHTD_RearSpvSpConvCo_M", head_v),
        _HvacDuctSpec(
            "HeadTempSp", "Head", "RearSpfHead", "RearSpfTma", "RearSpfFlow", "CHTD_RearSpfSpConvCo_M", head_v,
            zone_temp_x_key="HeadTempSd",
            notes="AS_FOUND defect P4: T_src=HeadTempSd not HeadTempSp",
        ),
        # HeadTempTd
        _HvacDuctSpec("HeadTempTd", "Head", "RearTdv", "RearTdvTma", "RearTdvFlow", "CHTD_TdvTdConvCo_M", head_v),
        _HvacDuctSpec("HeadTempTd", "Head", "RearTdfHead", "RearTdfTma", "RearTdfFlow", "CHTD_TdfTdConvCo_M", head_v),
        # HeadTempTp
        _HvacDuctSpec("HeadTempTp", "Head", "RearTpv", "RearTpvTma", "RearTpvFlow", "CHTD_TpvTpConvCo_M", head_v),
        _HvacDuctSpec("HeadTempTp", "Head", "RearTpfHead", "RearTpfTma", "RearTpfFlow", "CHTD_TpfTpConvCo_M", head_v),
        # FeetTemp*
        _HvacDuctSpec("FeetTempFd", "Feet", "FrntFdf", "FrntFdfTma", "FrntFdfFlow", "CHTD_FdfFeetFdConvCo_M", feet_v),
        _HvacDuctSpec("FeetTempFp", "Feet", "FrntFpf", "FrntFpfTma", "FrntFpfFlow", "CHTD_FpfFeetFpConvCo_M", feet_v),
        _HvacDuctSpec("FeetTempSd", "Feet", "FrntSdf", "FrntSdfTma", "FrntSdfFlow", "CHTD_SdfFeetSdConvCo_M", feet_v),
        _HvacDuctSpec("FeetTempSd", "Feet", "RearSdf", "RearSdfTma", "RearSdfFlow", "CHTD_RearSdfFeetSdConvCo_M", feet_v),
        _HvacDuctSpec("FeetTempSp", "Feet", "FrntSpf", "FrntSpfTma", "FrntSpfFlow", "CHTD_SpfFeetSpConvCo_M", feet_v),
        _HvacDuctSpec("FeetTempSp", "Feet", "RearSpf", "RearSpfTma", "RearSpfFlow", "CHTD_RearSpfFeetSpConvCo_M", feet_v),
        _HvacDuctSpec("FeetTempTd", "Feet", "RearTdf", "RearTdfTma", "RearTdfFlow", "CHTD_TdfFeetTdConvCo_M", feet_v),
        _HvacDuctSpec("FeetTempTp", "Feet", "RearTpf", "RearTpfTma", "RearTpfFlow", "CHTD_TpfFeetTpConvCo_M", feet_v),
    )


def _upstream_tma(u: np.ndarray, key: str, amb_t: float) -> float:
    val = float(u[U_INDEX[key]])
    return amb_t if val == 0.0 else val


def _lut_coef(params: CHTDParams, lv: Mapping[str, float], lut_name: str, amb_t: float) -> float:
    if lut_name in lv:
        return float(lv[lut_name])
    if hasattr(params, lut_name):
        return lookup_chtd_amb(getattr(params, lut_name), amb_t, params)
    return 1.0


def _flow_for_conv_base(raw_flow: float, unit_path: str) -> Tuple[float, str]:
    if unit_path == "current_as_m3h":
        return float(raw_flow), "m3/h (code path: raw u → ConvCoBase)"
    if unit_path == "flow_m3h_to_m3s_before_conv":
        return float(raw_flow) * _M3H_TO_M3S, "m3/s (hypothesis: divide m3/h by 3600 before ConvCoBase)"
    if unit_path == "flow_lps":
        return float(raw_flow) / 3.6, "m3/h equivalent (hypothesis: u is L/s, ÷3.6 → m3/h for LUT)"
    raise ValueError(unit_path)


def _q_and_dt_from_duct(
    *,
    tma: float,
    t_zone: float,
    raw_flow: float,
    lut: float,
    C: float,
    dt: float,
    q_gain: float,
    unit_path: str = "current_as_m3h",
) -> Dict[str, float]:
    flow_conv, _ = _flow_for_conv_base(raw_flow, unit_path)
    conv_base = conv_co_base_flow_ex(flow_conv)
    q_raw = (tma - t_zone) * conv_base * lut
    q_scaled = q_raw * q_gain
    d_contrib = q_scaled * dt / C if C else float("inf")
    return {
        "conv_co_base": conv_base,
        "q_hvac_raw_w": q_raw,
        "q_hvac_scaled_w": q_scaled,
        "delta_contrib_c_per_step": d_contrib,
        "flow_into_conv_co_base": flow_conv,
    }


def audit_single_hvac_duct(
    spec: _HvacDuctSpec,
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: Mapping[str, float],
) -> Dict[str, Any]:
    amb_t = float(u[U_INDEX["AmbT"]])
    x_zone = spec.zone_temp_x_key or spec.zone
    t_zone = float(x[X_INDEX[x_zone]])
    tma = _upstream_tma(u, spec.tma_u_key, amb_t)
    raw_flow = float(u[U_INDEX[spec.flow_u_key]])
    lut = _lut_coef(params, lv, spec.lut_name, amb_t)
    air_v = float(getattr(params, spec.air_volume_attr))
    C = cp_v_ex(t_zone, air_v, params.CHTD_AirCpAtb_P)
    dt = float(params.CHTD_Dt_P)
    q_gain = float(params.CHTD_QgainCo_P)
    delta_t = tma - t_zone

    calc = _q_and_dt_from_duct(
        tma=tma,
        t_zone=t_zone,
        raw_flow=raw_flow,
        lut=lut,
        C=C,
        dt=dt,
        q_gain=q_gain,
        unit_path="current_as_m3h",
    )

    return {
        "zone": spec.zone,
        "zone_group": spec.zone_group,
        "duct_name": spec.duct_name,
        "zone_temp_x_key": x_zone,
        "tma_u_key": spec.tma_u_key,
        "flow_u_key": spec.flow_u_key,
        "lut_name": spec.lut_name,
        "notes": spec.notes,
        "tma_c": tma,
        "t_zone_c": t_zone,
        "delta_t_c": delta_t,
        "flow_raw_u": raw_flow,
        "assumed_unit_current_code": "m3/h (Simulink ConvCoBase BP documented as m3/h or normalized)",
        "flow_m3h_if_current": raw_flow,
        "flow_m3s_if_current": raw_flow * _M3H_TO_M3S,
        "conv_co_base_flow_ex": calc["conv_co_base"],
        "lut_coefficient": lut,
        "capacitance_j_per_k": C,
        "air_volume_m3": air_v,
        "qgain_co": q_gain,
        "dt_s": dt,
        "q_hvac_raw_w": calc["q_hvac_raw_w"],
        "q_hvac_scaled_w": calc["q_hvac_scaled_w"],
        "effective_ua_w_per_k": calc["q_hvac_raw_w"] / delta_t if delta_t else None,
        "delta_contrib_c_per_step": calc["delta_contrib_c_per_step"],
    }


def audit_all_hvac_ducts(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
) -> List[Dict[str, Any]]:
    amb_t = float(u[U_INDEX["AmbT"]])
    lv = eval_lut_map(params, amb_t)
    return [audit_single_hvac_duct(spec, x, u, params, lv) for spec in _hvac_duct_registry()]


def compare_flow_unit_assumptions_feet_fd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
) -> Dict[str, Any]:
    """Compare FeetTempFd FrntFdf dT under three flow unit interpretations."""
    spec = next(s for s in _hvac_duct_registry() if s.zone == "FeetTempFd" and s.duct_name == "FrntFdf")
    amb_t = float(u[U_INDEX["AmbT"]])
    lv = eval_lut_map(params, amb_t)
    t_zone = float(x[X_INDEX["FeetTempFd"]])
    tma = _upstream_tma(u, spec.tma_u_key, amb_t)
    raw_flow = float(u[U_INDEX[spec.flow_u_key]])
    lut = _lut_coef(params, lv, spec.lut_name, amb_t)
    C = cp_v_ex(t_zone, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)
    dt = float(params.CHTD_Dt_P)
    q_gain = float(params.CHTD_QgainCo_P)

    paths: List[Dict[str, Any]] = []
    for path_id, label in (
        ("current_as_m3h", "current path (u as m3/h)"),
        ("flow_m3h_to_m3s_before_conv", "if flow_m3h_to_m3s_before_conv"),
        ("flow_lps", "if flow_lps (u L/s → u/3.6 m3/h for ConvCoBase)"),
    ):
        flow_conv, unit_desc = _flow_for_conv_base(raw_flow, path_id)
        calc = _q_and_dt_from_duct(
            tma=tma, t_zone=t_zone, raw_flow=raw_flow, lut=lut, C=C, dt=dt, q_gain=q_gain, unit_path=path_id
        )
        paths.append(
            {
                "path_id": path_id,
                "label": label,
                "flow_into_conv_co_base": flow_conv,
                "unit_description": unit_desc,
                **calc,
            }
        )

    verified = float(compute_chtd_delta(x, u, params, AS_FOUND)[X_INDEX["FeetTempFd"]])
    current = paths[0]["delta_contrib_c_per_step"]

    return {
        "zone": "FeetTempFd",
        "duct": "FrntFdf",
        "verified_zone_delta_c": verified,
        "current_path_matches_verified": abs(current - verified) < 0.05,
        "paths": paths,
        "interpretation": _flow_unit_interpretation(paths),
    }


def _flow_unit_interpretation(paths: Sequence[Mapping[str, Any]]) -> str:
    current = float(paths[0]["delta_contrib_c_per_step"])
    m3s = float(paths[1]["delta_contrib_c_per_step"])
    lps = float(paths[2]["delta_contrib_c_per_step"])
    if abs(m3s - current) < 0.5:
        return "m3/s-before-conv hypothesis does NOT explain overshoot (ConvCoBase clips to ~1 at tiny flow)."
    if abs(lps - current) < 1.0:
        return "L/s hypothesis (÷3.6) partially matches — verify AFE flow unit export."
    if m3s < 5.0 and current > 20.0:
        return (
            "Overshoot persists on current m3/h path; m3/s misread would collapse dT — "
            "likely NOT a simple m3/s-vs-m3/h swap alone."
        )
    return (
        "Unit swap alone insufficient; combine with C (FeetAirV) and/or LUT calibration."
    )


def conv_co_base_curve_table(
    flows: Sequence[float] = _CONV_CURVE_FLOWS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in flows:
        cb = conv_co_base_flow_ex(f)
        rows.append(
            {
                "flow_m3h": f,
                "conv_co_base_flow_ex": cb,
                "flow_power_0p8": float(f ** 0.8),
                "interpretation_effective_ua_w_per_k_at_lut_1": cb,
            }
        )
    return rows


def _conv_base_interpretation() -> Dict[str, Any]:
    return {
        "formula": "ConvCoBaseFlowEx(flow) = clip_interp(flow, BP, BP^0.8)",
        "bp_unit_documented": "m3/h or normalized (CHTD_formula_spec.md §3.1)",
        "physical_basis_note": "Nu ∝ Re^0.8 proxy; NOT ρ·cp·V̇ alone",
        "dimensions_if_lut_is_1": (
            "Q = ΔT × ConvCoBase × LUT → ConvCoBase acts as effective UA [W/K] "
            "when LUT is dimensionless"
        ),
        "typical_automotive_ua_w_per_k": "50–300 W/K for modest cabin air control volumes",
        "at_flow_150_m3h": conv_co_base_flow_ex(150.0),
        "verdict": (
            "Curve magnitude at 100–200 m3/h (~40–56) resembles hA [W/K], not a "
            "0–1 normalization factor; placeholder LUT=1.0 leaves full UA on the table."
        ),
    }


def _invert_conv_base_target(target_ua: float, *, lo: float = 0.0, hi: float = 600.0) -> float:
    """Find flow [m3/h] such that ConvCoBaseFlowEx(flow) ≈ target_ua."""
    target = max(float(target_ua), 1.0)
    if conv_co_base_flow_ex(hi) < target:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if conv_co_base_flow_ex(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_feet_fd_calibration_targets(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
) -> Dict[str, Any]:
    """LUT / FeetAirV / flow needed for FeetTempFd dT targets (FrntFdf duct, LUT=1 paths)."""
    spec = next(s for s in _hvac_duct_registry() if s.zone == "FeetTempFd")
    row = audit_single_hvac_duct(spec, x, u, params, eval_lut_map(params, float(u[U_INDEX["AmbT"]])))
    baseline_dt = float(row["delta_contrib_c_per_step"])
    verified = float(compute_chtd_delta(x, u, params, AS_FOUND)[X_INDEX["FeetTempFd"]])
    delta_t = float(row["delta_t_c"])
    lut = float(row["lut_coefficient"])
    C = float(row["capacitance_j_per_k"])
    dt = float(row["dt_s"])
    q_gain = float(row["qgain_co"])
    q_scaled = float(row["q_hvac_scaled_w"])
    raw_flow = float(row["flow_raw_u"])
    conv_base = float(row["conv_co_base_flow_ex"])
    feet_v = float(row["air_volume_m3"])

    targets: Dict[str, Any] = {}
    for label, dT_target in (("2c", 2.0), ("5c", 5.0)):
        lut_scale = dT_target / baseline_dt if baseline_dt else None
        C_needed = q_scaled * dt / dT_target
        feet_v_needed = feet_v * (baseline_dt / dT_target)
        ua_needed = (dT_target * C) / (delta_t * lut * dt * q_gain) if delta_t else None
        conv_base_needed = ua_needed / lut if ua_needed and lut else None
        flow_needed = (
            _invert_conv_base_target(conv_base_needed) if conv_base_needed else None
        )
        targets[f"required_lut_scale_for_{label}"] = lut_scale
        targets[f"required_feet_airv_m3_for_{label}_lut_1"] = feet_v_needed
        targets[f"required_flow_m3h_for_{label}_lut_1"] = flow_needed
        targets[f"required_conv_co_base_for_{label}_lut_1"] = conv_base_needed

    return {
        "baseline_feet_temp_fd_duct_delta_c": baseline_dt,
        "baseline_verified_zone_delta_c": verified,
        "baseline_q_hvac_scaled_w": q_scaled,
        "baseline_conv_co_base": conv_base,
        "baseline_flow_m3h": raw_flow,
        "baseline_feet_airv_m3": feet_v,
        "baseline_capacitance_j_per_k": C,
        "baseline_lut": lut,
        **targets,
        "diagnosis": _calibration_diagnosis(baseline_dt, targets),
    }


def _calibration_diagnosis(baseline_dt: float, targets: Mapping[str, Any]) -> str:
    lut2 = float(targets["required_lut_scale_for_2c"])
    lut5 = float(targets["required_lut_scale_for_5c"])
    v2 = float(targets["required_feet_airv_m3_for_2c_lut_1"])
    if lut2 < 0.15 and v2 > 0.25:
        return (
            "Primary mismatch is small feet control volume (FeetAirV) plus full-scale "
            "ConvCoBase×LUT=1 UA; unit-only fix unlikely without capacity or LUT scale."
        )
    return "See sensitivity table; multiple knobs can achieve target dT."


def top_duct_deltas(rows: Sequence[Mapping[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: abs(float(r["delta_contrib_c_per_step"])), reverse=True)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(ranked[:n], start=1):
        out.append({"rank": i, **dict(row)})
    return out


def run_hvac_convection_scale_audit(
    *,
    params: Optional[CHTDParams] = None,
    vehicle_class: str = "m8_estimated",
) -> Dict[str, Any]:
    physical_params, provenance = (
        build_physical_capacity_params(vehicle_class=vehicle_class)
        if params is None
        else (copy.deepcopy(params), {})
    )
    x0, u, amb = build_state_and_input_from_afe_case(_AUDIT_CASE)

    all_ducts = audit_all_hvac_ducts(x0, u, physical_params)
    feet_fd = next(r for r in all_ducts if r["zone"] == "FeetTempFd" and r["duct_name"] == "FrntFdf")
    unit_compare = compare_flow_unit_assumptions_feet_fd(x0, u, physical_params)
    curve = conv_co_base_curve_table()
    cal_targets = compute_feet_fd_calibration_targets(x0, u, physical_params)

    cabin_win_note = (
        "CabinTemp* and WinTemp* have no direct HVAC duct q_hvac_duct terms in thermal.py; "
        "they receive inter-zone q-bus heat only."
    )

    return {
        "schema": AUDIT_SCHEMA,
        "classification": CLASSIFICATION,
        "disclaimer": DISCLAIMER,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vehicle_class": vehicle_class,
        "params_bundle": "physical_capacity_params",
        "case_id": _AUDIT_CASE["case_id"],
        "amb_t_c": amb,
        "provenance_note": provenance.get("_meta", {}),
        "cabin_win_direct_hvac": cabin_win_note,
        "hvac_duct_terms": all_ducts,
        "top_10_duct_delta_contrib": top_duct_deltas(all_ducts, 10),
        "feet_temp_fd_primary": feet_fd,
        "flow_unit_comparison_feet_fd": unit_compare,
        "conv_co_base_curve": curve,
        "conv_co_base_interpretation": _conv_base_interpretation(),
        "feet_fd_calibration_targets": cal_targets,
        "overall_verdict": _overall_verdict(feet_fd, unit_compare, cal_targets),
    }


def _overall_verdict(
    feet_fd: Mapping[str, Any],
    unit_compare: Mapping[str, Any],
    cal_targets: Mapping[str, Any],
) -> Dict[str, str]:
    return {
        "feet_overshoot_driver": (
            "Coefficient + control-volume definition: C=FeetAirV·ρ·cp ≈ "
            f"{feet_fd['capacitance_j_per_k']:.0f} J/K with effective UA ≈ "
            f"{feet_fd.get('effective_ua_w_per_k', 0):.0f} W/K at LUT=1 — NOT dt alone."
        ),
        "flow_unit": unit_compare.get("interpretation", ""),
        "lut_placeholder": (
            f"LUT=1.0 leaves ConvCoBase as full UA; need scale ≈ "
            f"{cal_targets['required_lut_scale_for_2c']:.3f} for 2°C/step or "
            f"{cal_targets['required_lut_scale_for_5c']:.3f} for 5°C/step."
        ),
        "conv_co_base": _conv_base_interpretation()["verdict"],
    }


def build_hvac_convection_markdown(doc: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# CHTD HVAC Convection Unit and Scale Audit",
        "",
        f"**{doc.get('disclaimer', DISCLAIMER)}**",
        "",
        f"Generated: `{doc.get('generated_at_utc', '')}`",
        f"Case: `{doc.get('case_id', '')}` | bundle: `{doc.get('params_bundle', '')}`",
        "",
        "## FeetTempFd primary duct (FrntFdf)",
        "",
    ]
    fd = doc["feet_temp_fd_primary"]
    lines.extend(
        [
            f"- Verified zone delta: **{doc['feet_fd_calibration_targets']['baseline_verified_zone_delta_c']:.3f} °C/step**",
            f"- Duct isolated dT: **{fd['delta_contrib_c_per_step']:.3f} °C/step**",
            f"- ΔT={fd['delta_t_c']:.1f}°C, flow={fd['flow_raw_u']:.1f} m³/h, ConvCoBase={fd['conv_co_base_flow_ex']:.2f}, LUT={fd['lut_coefficient']:.3g}",
            f"- C={fd['capacitance_j_per_k']:.1f} J/K (FeetAirV={fd['air_volume_m3']:.4f} m³), Q={fd['q_hvac_scaled_w']:.1f} W",
            "",
            "## Flow unit comparison (FeetTempFd)",
            "",
            "| path | flow into ConvCoBase | ConvCoBase | dT °C/step |",
            "|------|---------------------|------------|------------|",
        ]
    )
    for p in doc["flow_unit_comparison_feet_fd"]["paths"]:
        lines.append(
            f"| {p['path_id']} | {p['flow_into_conv_co_base']:.4g} | {p['conv_co_base']:.3f} | {p['delta_contrib_c_per_step']:.3f} |"
        )
    lines.extend(["", f"*{doc['flow_unit_comparison_feet_fd']['interpretation']}*", ""])

    cal = doc["feet_fd_calibration_targets"]
    lines.extend(
        [
            "## Calibration targets (FrntFdf → FeetTempFd, isolated duct)",
            "",
            f"- `required_lut_scale_for_2c`: **{cal['required_lut_scale_for_2c']:.4f}**",
            f"- `required_lut_scale_for_5c`: **{cal['required_lut_scale_for_5c']:.4f}**",
            f"- FeetAirV for 2°C/step (LUT=1): **{cal['required_feet_airv_m3_for_2c_lut_1']:.4f} m³**",
            f"- FeetAirV for 5°C/step (LUT=1): **{cal['required_feet_airv_m3_for_5c_lut_1']:.4f} m³**",
            f"- flow for 2°C/step (LUT=1): **{cal['required_flow_m3h_for_2c_lut_1']:.2f} m³/h**",
            f"- flow for 5°C/step (LUT=1): **{cal['required_flow_m3h_for_5c_lut_1']:.2f} m³/h**",
            "",
            "## ConvCoBase curve",
            "",
            "| flow m³/h | ConvCoBase | ~flow^0.8 |",
            "|-----------|------------|-----------|",
        ]
    )
    for row in doc["conv_co_base_curve"]:
        lines.append(
            f"| {row['flow_m3h']:.0f} | {row['conv_co_base_flow_ex']:.3f} | {row['flow_power_0p8']:.3f} |"
        )
    lines.extend(["", f"*{doc['conv_co_base_interpretation']['verdict']}*", ""])

    lines.extend(["## Top 10 HVAC duct dT contributions", ""])
    lines.append("| rank | zone | duct | dT °C/step | flow | ConvCoBase | LUT |")
    lines.append("|------|------|------|------------|------|------------|-----|")
    for r in doc["top_10_duct_delta_contrib"]:
        lines.append(
            f"| {r['rank']} | {r['zone']} | {r['duct_name']} | {r['delta_contrib_c_per_step']:.3f} | "
            f"{r['flow_raw_u']:.1f} | {r['conv_co_base_flow_ex']:.2f} | {r['lut_coefficient']:.3g} |"
        )
    lines.extend(["", f"*{doc['cabin_win_direct_hvac']}*", ""])
    verdict = doc.get("overall_verdict", {})
    lines.extend(["## Overall verdict", ""])
    for k, v in verdict.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines)


def write_hvac_convection_scale_outputs(
    *,
    bundle: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    doc = bundle if bundle is not None else run_hvac_convection_scale_audit()
    json_out = Path(json_path) if json_path is not None else DEFAULT_JSON_OUT
    md_out = Path(md_path) if md_path is not None else DEFAULT_MD_OUT
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(build_hvac_convection_markdown(doc), encoding="utf-8")
    return {"json": json_out, "md": md_out}
