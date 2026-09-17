"""Feet-zone HVAC heating magnitude audit (read-only diagnostic).

Decomposes FeetTemp* HVAC duct Q/C chain and runs sensitivity sweeps on
copied ``CHTDParams`` / ``u`` — never mutates defaults or ``thermal.py``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.chtd.bus_index import U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND, conv_co_base_flow_ex, cp_v_ex
from hvac_sim.chtd.heat_terms import q_hvac_duct
from hvac_sim.chtd.lut import eval_lut_map, lookup_chtd_amb
from hvac_sim.chtd.params import CHTDParams, as_lut
from hvac_sim.chtd.physical_params import build_physical_capacity_params
from hvac_sim.chtd.stability_audit import build_state_and_input_from_afe_case
from hvac_sim.chtd.thermal import (
    HeadTempSpQBusOutputs,
    compute_chtd_delta,
    compute_console_temp,
    compute_feet_temp_fd,
    compute_feet_temp_fp,
    compute_feet_temp_sd,
    compute_feet_temp_sp,
    compute_feet_temp_td,
    compute_feet_temp_tp,
    compute_head_temp_fd,
    compute_head_temp_fp,
    compute_head_temp_sd,
    compute_head_temp_td,
    compute_head_temp_tp,
)

AUDIT_SCHEMA = "chtd_feet_hvac_magnitude_audit_v1"
CLASSIFICATION = "experimental_diagnostic_not_sign_off"
DISCLAIMER = (
    "Read-only Q/C decomposition on copied params/u. Not calibration sign-off. "
    "Default CHTDParams, thermal.py, and golden unchanged."
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_FEET_HVAC_MAGNITUDE_AUDIT.json"
)
DEFAULT_MD_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_FEET_HVAC_MAGNITUDE_AUDIT.md"
)

_M3H_TO_M3S = 1.0 / 3600.0

_CASES: Tuple[Dict[str, Any], ...] = (
    {
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
    {
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
)

_FEET_AIRV_SCALES = (1.0, 3.0, 10.0, 30.0)
_FLOW_SCALES = (1.0, 0.5, 0.25)
_CONV_LUT_SCALES = (1.0, 0.3, 0.1)
_DT_VALUES = (1.0, 0.5, 0.1)

# Feet duct conv LUT names (scale together in conv_lut sensitivity).
_FEET_DUCT_CONV_LUTS = (
    "CHTD_FdfFeetFdConvCo_M",
    "CHTD_FpfFeetFpConvCo_M",
    "CHTD_SdfFeetSdConvCo_M",
    "CHTD_RearSdfFeetSdConvCo_M",
    "CHTD_SpfFeetSpConvCo_M",
    "CHTD_RearSpfFeetSpConvCo_M",
    "CHTD_TdfFeetTdConvCo_M",
    "CHTD_TpfFeetTpConvCo_M",
)

_FOOT_FLOW_U_KEYS = (
    "FrntFdfFlow",
    "FrntFpfFlow",
    "FrntSdfFlow",
    "FrntSpfFlow",
    "RearSdfFlow",
    "RearSpfFlow",
    "RearTdfFlow",
    "RearTpfFlow",
)


@dataclass(frozen=True)
class _HvacDuctSpec:
    duct_id: str
    tma_u_key: str
    flow_u_key: str
    lut_name: str


@dataclass(frozen=True)
class _FeetZoneSpec:
    zone: str
    x_key: str
    ducts: Tuple[_HvacDuctSpec, ...]
    compute_fn: Any
    head_chain: str  # "fd" | "fp" | "sd" | "sp" | "td" | "tp"


def _lut_coef(params: CHTDParams, lv: Mapping[str, float], lut_name: str, amb_t: float) -> float:
    if lut_name in lv:
        return float(lv[lut_name])
    if hasattr(params, lut_name):
        return lookup_chtd_amb(getattr(params, lut_name), amb_t, params)
    return 1.0


def _upstream_tma(u: np.ndarray, key: str, amb_t: float) -> float:
    val = float(u[U_INDEX[key]])
    return amb_t if val == 0.0 else val


def _feet_zone_specs() -> Tuple[_FeetZoneSpec, ...]:
    return (
        _FeetZoneSpec(
            "FeetTempFd",
            "FeetTempFd",
            (
                _HvacDuctSpec(
                    "FrntFdf",
                    "FrntFdfTma",
                    "FrntFdfFlow",
                    "CHTD_FdfFeetFdConvCo_M",
                ),
            ),
            compute_feet_temp_fd,
            "fd",
        ),
        _FeetZoneSpec(
            "FeetTempFp",
            "FeetTempFp",
            (
                _HvacDuctSpec(
                    "FrntFpf",
                    "FrntFpfTma",
                    "FrntFpfFlow",
                    "CHTD_FpfFeetFpConvCo_M",
                ),
            ),
            compute_feet_temp_fp,
            "fp",
        ),
        _FeetZoneSpec(
            "FeetTempSd",
            "FeetTempSd",
            (
                _HvacDuctSpec("FrntSdf", "FrntSdfTma", "FrntSdfFlow", "CHTD_SdfFeetSdConvCo_M"),
                _HvacDuctSpec("RearSdf", "RearSdfTma", "RearSdfFlow", "CHTD_RearSdfFeetSdConvCo_M"),
            ),
            compute_feet_temp_sd,
            "sd",
        ),
        _FeetZoneSpec(
            "FeetTempSp",
            "FeetTempSp",
            (
                _HvacDuctSpec("FrntSpf", "FrntSpfTma", "FrntSpfFlow", "CHTD_SpfFeetSpConvCo_M"),
                _HvacDuctSpec("RearSpf", "RearSpfTma", "RearSpfFlow", "CHTD_RearSpfFeetSpConvCo_M"),
            ),
            compute_feet_temp_sp,
            "sp",
        ),
        _FeetZoneSpec(
            "FeetTempTd",
            "FeetTempTd",
            (
                _HvacDuctSpec("RearTdf", "RearTdfTma", "RearTdfFlow", "CHTD_TdfFeetTdConvCo_M"),
            ),
            compute_feet_temp_td,
            "td",
        ),
        _FeetZoneSpec(
            "FeetTempTp",
            "FeetTempTp",
            (
                _HvacDuctSpec("RearTpf", "RearTpfTma", "RearTpfFlow", "CHTD_TpfFeetTpConvCo_M"),
            ),
            compute_feet_temp_tp,
            "tp",
        ),
    )


def _build_head_q(
    chain: str,
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
):
    _, console_q = compute_console_temp(x, u, params, lv, AS_FOUND)
    if chain == "fd":
        _, head_q = compute_head_temp_fd(x, u, params, lv, AS_FOUND, console_q=console_q)
        return head_q, console_q
    if chain == "fp":
        _, head_fd_q = compute_head_temp_fd(x, u, params, lv, AS_FOUND, console_q=console_q)
        _, head_q = compute_head_temp_fp(x, u, params, lv, head_fd_q, AS_FOUND, console_q=console_q)
        return head_q, console_q
    if chain == "sd":
        _, head_fd_q = compute_head_temp_fd(x, u, params, lv, AS_FOUND, console_q=console_q)
        _, head_q = compute_head_temp_sd(x, u, params, lv, head_fd_q, AS_FOUND)
        return head_q, None
    if chain == "sp":
        # HeadTempSp is inline in compute_chtd_delta; at cold-soak step 1 interzone
        # head→feet coupling is ~0 (uniform T), so zero q-bus is sufficient for audit.
        head_q = HeadTempSpQBusOutputs(
            sp_feet_temp_radiation=0.0,
            sp_feet_temp_convection=0.0,
            sp_cabin_sp_radiation=0.0,
            sp_cabin_sp_convection=0.0,
            sp_win_sp_radiation=0.0,
            sp_win_sp_convection=0.0,
            sp_roof_radiation=0.0,
            sp_roof_convection=0.0,
            sp_tp_radiation=0.0,
            sp_tp_convection=0.0,
        )
        return head_q, None
    if chain == "td":
        _, head_fd_q = compute_head_temp_fd(x, u, params, lv, AS_FOUND, console_q=console_q)
        _, head_sd_q = compute_head_temp_sd(x, u, params, lv, head_fd_q, AS_FOUND)
        _, head_q = compute_head_temp_td(x, u, params, lv, head_sd_q, AS_FOUND)
        return head_q, None
    if chain == "tp":
        _, head_fd_q = compute_head_temp_fd(x, u, params, lv, AS_FOUND, console_q=console_q)
        _, head_sd_q = compute_head_temp_sd(x, u, params, lv, head_fd_q, AS_FOUND)
        _, head_td_q = compute_head_temp_td(x, u, params, lv, head_sd_q, AS_FOUND)
        head_sp_q = HeadTempSpQBusOutputs(
            sp_feet_temp_radiation=0.0,
            sp_feet_temp_convection=0.0,
            sp_cabin_sp_radiation=0.0,
            sp_cabin_sp_convection=0.0,
            sp_win_sp_radiation=0.0,
            sp_win_sp_convection=0.0,
            sp_roof_radiation=0.0,
            sp_roof_convection=0.0,
            sp_tp_radiation=0.0,
            sp_tp_convection=0.0,
        )
        _, head_q = compute_head_temp_tp(
            x, u, params, lv, head_sp_q, AS_FOUND, head_td_q
        )
        return head_q, None
    raise ValueError(chain)


def decompose_feet_hvac_duct(
    *,
    zone_spec: _FeetZoneSpec,
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    step: int = 1,
) -> Dict[str, Any]:
    """Per-duct HVAC breakdown + verified zone delta at given x,u."""
    amb_t = float(u[U_INDEX["AmbT"]])
    lv = eval_lut_map(params, amb_t)
    t_zone = float(x[X_INDEX[zone_spec.x_key]])
    C = cp_v_ex(t_zone, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)
    q_gain = float(params.CHTD_QgainCo_P)
    dt = float(params.CHTD_Dt_P)

    head_q, console_q = _build_head_q(zone_spec.head_chain, x, u, params, lv)
    if console_q is not None:
        delta, _qbus = zone_spec.compute_fn(
            x, u, params, lv, head_q, AS_FOUND, console_q=console_q
        )
    else:
        delta, _qbus = zone_spec.compute_fn(x, u, params, lv, head_q, AS_FOUND)

    verified = float(compute_chtd_delta(x, u, params, AS_FOUND)[X_INDEX[zone_spec.x_key]])

    ducts: List[Dict[str, Any]] = []
    hvac_q_scaled_sum = 0.0
    for duct in zone_spec.ducts:
        tma = _upstream_tma(u, duct.tma_u_key, amb_t)
        flow_m3h = float(u[U_INDEX[duct.flow_u_key]])
        flow_m3s = flow_m3h * _M3H_TO_M3S
        delta_t = tma - t_zone
        conv_base = conv_co_base_flow_ex(flow_m3h)
        lut = _lut_coef(params, lv, duct.lut_name, amb_t)
        q_raw = q_hvac_duct(tma, t_zone, flow_m3h, lut)
        q_scaled = q_raw * q_gain
        d_contrib = q_scaled * dt / C
        hvac_q_scaled_sum += q_scaled
        ducts.append(
            {
                "duct_id": duct.duct_id,
                "tma_u_key": duct.tma_u_key,
                "flow_u_key": duct.flow_u_key,
                "lut_name": duct.lut_name,
                "zone_temp_c": t_zone,
                "tma_c": tma,
                "delta_t_c": float(delta_t),
                "flow_m3h": flow_m3h,
                "flow_m3s": flow_m3s,
                "conv_co_base_flow_ex": conv_base,
                "lut_coefficient": lut,
                "qgain_co": q_gain,
                "q_hvac_raw_w": float(q_raw),
                "q_hvac_scaled_w": float(q_scaled),
                "capacitance_j_per_k": float(C),
                "feet_air_volume_m3": float(params.CHTD_FeetAirVAtb_P),
                "dt_s": dt,
                "delta_contrib_c_per_step": float(d_contrib),
            }
        )

    return {
        "zone": zone_spec.zone,
        "step": int(step),
        "zone_delta_c_verified": verified,
        "zone_delta_c_from_compute": float(delta),
        "hvac_ducts": ducts,
        "hvac_duct_delta_sum_c": float(sum(d["delta_contrib_c_per_step"] for d in ducts)),
        "non_hvac_residual_delta_c": float(delta - sum(d["delta_contrib_c_per_step"] for d in ducts)),
        "dominant_hvac_duct": max(ducts, key=lambda d: abs(d["delta_contrib_c_per_step"]))["duct_id"]
        if ducts
        else None,
    }


def decompose_all_feet_zones(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    *,
    step: int = 1,
) -> List[Dict[str, Any]]:
    return [
        decompose_feet_hvac_duct(zone_spec=zs, x=x, u=u, params=params, step=step)
        for zs in _feet_zone_specs()
    ]


def _feet_fd_delta_step1(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
) -> float:
    return float(compute_chtd_delta(x, u, params, AS_FOUND)[X_INDEX["FeetTempFd"]])


def _apply_sensitivity(
    base_params: CHTDParams,
    base_u: np.ndarray,
    *,
    feet_airv_scale: float = 1.0,
    foot_flow_scale: float = 1.0,
    foot_conv_lut_scale: float = 1.0,
    dt_seconds: float = 1.0,
) -> Tuple[CHTDParams, np.ndarray]:
    p = copy.deepcopy(base_params)
    u = np.asarray(base_u, dtype=float).copy()
    p.CHTD_FeetAirVAtb_P = float(p.CHTD_FeetAirVAtb_P) * float(feet_airv_scale)
    p.CHTD_Dt_P = float(dt_seconds)
    if foot_conv_lut_scale != 1.0:
        factor = float(foot_conv_lut_scale)
        for name in _FEET_DUCT_CONV_LUTS:
            if hasattr(p, name):
                arr = np.asarray(getattr(p, name), dtype=float)
                setattr(p, name, arr * factor)
            else:
                setattr(p, name, as_lut(factor))
    if foot_flow_scale != 1.0:
        factor = float(foot_flow_scale)
        for key in _FOOT_FLOW_U_KEYS:
            u[U_INDEX[key]] = float(u[U_INDEX[key]]) * factor
    return p, u


def run_sensitivity_matrix(
    x: np.ndarray,
    u: np.ndarray,
    base_params: CHTDParams,
) -> Dict[str, Any]:
    baseline = _feet_fd_delta_step1(x, u, base_params)
    rows: List[Dict[str, Any]] = []

    def add(group: str, label: str, **kwargs: float) -> None:
        p, u_adj = _apply_sensitivity(base_params, u, **kwargs)
        d = _feet_fd_delta_step1(x, u_adj, p)
        rows.append(
            {
                "group": group,
                "label": label,
                "feet_temp_fd_delta_step1_c": d,
                "ratio_vs_baseline": d / baseline if baseline else None,
                **kwargs,
            }
        )

    for s in _FEET_AIRV_SCALES:
        add("feet_airv_scale", f"{s:g}x", feet_airv_scale=s)
    for s in _FLOW_SCALES:
        add("foot_flow_scale", f"{s:g}x", foot_flow_scale=s)
    for s in _CONV_LUT_SCALES:
        add("foot_conv_lut_scale", f"{s:g}x", foot_conv_lut_scale=s)
    for dt in _DT_VALUES:
        add("dt_sensitivity", f"{dt:g}s", dt_seconds=dt)

    return {
        "baseline_feet_temp_fd_delta_step1_c": baseline,
        "rows": rows,
    }


def diagnose_feet_magnitude(
    decomposition: Sequence[Mapping[str, Any]],
    sensitivity: Mapping[str, Any],
    *,
    baseline_delta: float,
) -> Dict[str, Any]:
    flags: List[str] = []
    fd = next(d for d in decomposition if d["zone"] == "FeetTempFd")
    primary = fd["hvac_ducts"][0] if fd["hvac_ducts"] else {}

    C = float(primary.get("capacitance_j_per_k", 1.0))
    flow = float(primary.get("flow_m3h", 0.0))
    conv_base = float(primary.get("conv_co_base_flow_ex", 1.0))
    lut = float(primary.get("lut_coefficient", 1.0))
    feet_v = float(primary.get("feet_air_volume_m3", 1.0))

    if C < 500.0:
        flags.append("low_feet_air_capacitance")
    if flow > 100.0:
        flags.append("afe_flow_scale_suspect")
    if conv_base > 20.0:
        flags.append("conv_base_suspect")
    if abs(lut - 1.0) < 1e-9:
        flags.append("foot_conv_lut_needs_calibration")

    rows = list(sensitivity.get("rows", []))
    dt_only = _best_reduction(rows, "dt_sensitivity")
    cap_only = _best_reduction(rows, "feet_airv_scale")
    flow_only = _best_reduction(rows, "foot_flow_scale")
    lut_only = _best_reduction(rows, "foot_conv_lut_scale")

    if dt_only and dt_only["ratio_vs_baseline"] < 0.2:
        if all(
            _best_reduction(rows, g)["ratio_vs_baseline"] >= 0.2
            for g in ("feet_airv_scale", "foot_flow_scale", "foot_conv_lut_scale")
            if _best_reduction(rows, g)
        ):
            flags.append("integration_step_suspect")

    factor_ranking = _rank_factors(rows, baseline_delta)
    repair_suggestions = _repair_suggestions(factor_ranking, feet_v, flow, lut)

    return {
        "flags": sorted(set(flags)),
        "primary_hvac_chain": {
            "formula": "dT = (Tma-Tzone)*ConvCoBaseFlowEx(flow)*LUT * QgainCo * dt / C",
            "measured_step1": primary,
            "baseline_feet_temp_fd_delta_c": baseline_delta,
            "hvac_explains_pct": (
                100.0 * fd["hvac_duct_delta_sum_c"] / baseline_delta if baseline_delta else None
            ),
        },
        "factor_ranking": factor_ranking,
        "repair_suggestions": repair_suggestions,
    }


def _best_reduction(rows: Sequence[Mapping[str, Any]], group: str) -> Optional[Dict[str, Any]]:
    group_rows = [r for r in rows if r["group"] == group and r.get("ratio_vs_baseline", 1.0) < 1.0]
    if not group_rows:
        return None
    return min(group_rows, key=lambda r: r["ratio_vs_baseline"])


def _rank_factors(rows: Sequence[Mapping[str, Any]], baseline: float) -> List[Dict[str, Any]]:
    """Rank sensitivity knobs by |delta reduction| at step 1 (excluding baseline 1x)."""
    scored: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("ratio_vs_baseline") is None:
            continue
        if r["label"] in ("1x", "1.0x", "1p0"):
            continue
        d = float(r["feet_temp_fd_delta_step1_c"])
        scored.append(
            {
                "group": r["group"],
                "label": r["label"],
                "feet_temp_fd_delta_step1_c": d,
                "reduction_c": baseline - d,
                "ratio_vs_baseline": r["ratio_vs_baseline"],
            }
        )
    scored.sort(key=lambda x: x["reduction_c"], reverse=True)
    for i, row in enumerate(scored, start=1):
        row["rank"] = i
    return scored


def _repair_suggestions(
    ranking: Sequence[Mapping[str, Any]],
    feet_v: float,
    flow: float,
    lut: float,
) -> Dict[str, Any]:
    return {
        "A_feet_airv": {
            "knob": "CHTD_FeetAirVAtb_P",
            "current_m3": feet_v,
            "experiment_range_m3": [max(feet_v * 3, 0.05), feet_v * 30],
            "rationale": "Directly scales C=ρV·cp; excel 0.029 m³ is far below placeholder 1.0 m³",
        },
        "B_foot_conv_lut_scale": {
            "knob": "CHTD_*Feet*ConvCo_M LUT maps",
            "current_placeholder": lut,
            "experiment_range_scale": [0.3, 0.1],
            "rationale": "Placeholder LUT=1.0 with ConvCoBase(flow) yields large Q at ~150 m³/h",
        },
        "C_flow_scale": {
            "knob": "AFE→CHTD foot flow u[24:42] (FrntFdfFlow etc.)",
            "current_m3h_example": flow,
            "experiment_range_scale": [0.5, 0.25],
            "rationale": "ConvCoBaseFlowEx ~ flow^0.8; verify AFE flow unit/scaling vs Simulink",
        },
        "D_dt": {
            "knob": "CHTD_Dt_P",
            "experiment_range_s": [0.5, 0.1],
            "rationale": "Euler dt scales dT linearly; use only to test integration vs parameter root cause",
        },
        "priority_order": [r["group"] for r in ranking[:4]],
    }


def run_feet_hvac_magnitude_audit(
    *,
    params: Optional[CHTDParams] = None,
    vehicle_class: str = "m8_estimated",
) -> Dict[str, Any]:
    physical_params, provenance = (
        build_physical_capacity_params(vehicle_class=vehicle_class)
        if params is None
        else (copy.deepcopy(params), {})
    )

    case_results: List[Dict[str, Any]] = []
    for case in _CASES:
        x0, u, amb = build_state_and_input_from_afe_case(case)
        decomp = decompose_all_feet_zones(x0, u, physical_params, step=1)
        sens = run_sensitivity_matrix(x0, u, physical_params)
        fd_delta = float(decomp[0]["zone_delta_c_verified"])
        diag = diagnose_feet_magnitude(decomp, sens, baseline_delta=fd_delta)
        case_results.append(
            {
                "case_id": case["case_id"],
                "mode_code": case["mode_code"],
                "amb_t_c": amb,
                "step1_feet_decomposition": decomp,
                "sensitivity_matrix": sens,
                "diagnosis": diag,
            }
        )

    return {
        "schema": AUDIT_SCHEMA,
        "classification": CLASSIFICATION,
        "disclaimer": DISCLAIMER,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vehicle_class": vehicle_class,
        "params_bundle": "physical_capacity_params",
        "provenance_note": provenance.get("_meta", {}),
        "cases": case_results,
    }


def build_feet_hvac_markdown(doc: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# CHTD Feet HVAC Magnitude Audit",
        "",
        f"**{doc.get('disclaimer', DISCLAIMER)}**",
        "",
        f"Generated: `{doc.get('generated_at_utc', '')}`",
        "",
    ]
    for case in doc.get("cases", []):
        lines.extend([f"## {case['case_id']}", ""])
        fd = next(z for z in case["step1_feet_decomposition"] if z["zone"] == "FeetTempFd")
        p = fd["hvac_ducts"][0]
        lines.extend(
            [
                f"**FeetTempFd step-1 verified delta: {fd['zone_delta_c_verified']:.3f} °C/step**",
                "",
                "### Primary HVAC duct (FrntFdf)",
                "",
                f"- T_zone = {p['zone_temp_c']:.1f}°C, Tma = {p['tma_c']:.1f}°C, ΔT = {p['delta_t_c']:.1f}°C",
                f"- flow = {p['flow_m3h']:.1f} m³/h ({p['flow_m3s']:.4f} m³/s)",
                f"- ConvCoBaseFlowEx(flow) = {p['conv_co_base_flow_ex']:.2f}",
                f"- LUT = {p['lut_coefficient']:.3g}, QgainCo = {p['qgain_co']:.1f}",
                f"- C = {p['capacitance_j_per_k']:.1f} J/K (FeetAirV = {p['feet_air_volume_m3']:.4f} m³)",
                f"- Q_hvac = {p['q_hvac_scaled_w']:.1f} W → dT = {p['delta_contrib_c_per_step']:.3f} °C",
                "",
                f"HVAC explains **{case['diagnosis']['primary_hvac_chain']['hvac_explains_pct']:.1f}%** of FeetTempFd delta.",
                "",
                "### Flags",
                "",
            ]
        )
        for f in case["diagnosis"]["flags"]:
            lines.append(f"- `{f}`")
        lines.extend(["", "### Factor ranking (FeetTempFd step-1)", ""])
        lines.append("| rank | group | label | delta °C | reduction |")
        lines.append("|------|-------|-------|----------|-----------|")
        for r in case["diagnosis"]["factor_ranking"][:8]:
            lines.append(
                f"| {r['rank']} | {r['group']} | {r['label']} | {r['feet_temp_fd_delta_step1_c']:.3f} | "
                f"{r['reduction_c']:.3f} |"
            )
        lines.extend(["", "### Repair experiment ranges (not calibrated values)", ""])
        for key in ("A_feet_airv", "B_foot_conv_lut_scale", "C_flow_scale", "D_dt"):
            sug = case["diagnosis"]["repair_suggestions"][key]
            lines.append(f"- **{key}** `{sug['knob']}`: {sug['rationale']}")
        lines.append("")

    return "\n".join(lines)


def write_feet_hvac_magnitude_outputs(
    *,
    bundle: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    doc = bundle if bundle is not None else run_feet_hvac_magnitude_audit()
    json_out = Path(json_path) if json_path is not None else DEFAULT_JSON_OUT
    md_out = Path(md_path) if md_path is not None else DEFAULT_MD_OUT
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(build_feet_hvac_markdown(doc), encoding="utf-8")
    return {"json": json_out, "md": md_out}
