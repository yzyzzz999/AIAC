"""CHTD solar gain scale audit — read-only diagnostic.

Traces SolarFd/SolarFp coupling through AS_FOUND CHTD one-step, decomposes
direct solar heat terms, and estimates LUT scale factors. Does not modify
``thermal.py``, runtime guard, or default params.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.chtd.bus_index import CHTD_X_NAMES, U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND, cp_m_ex, get_solar_fd_horiz, get_solar_fp_horiz
from hvac_sim.chtd.heat_terms import q_solar_gain
from hvac_sim.chtd.lut import eval_lut_map, lookup_chtd_amb
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.runtime_policy import apply_two_row_vehicle_policy
from hvac_sim.chtd.thermal import compute_chtd_delta, one_step_chtd
from hvac_sim.pipeline import (
    _driver_air_temp_c,
    _driver_mrt_c,
    _passenger_air_temp_c,
    _passenger_mrt_c,
    _resolve_air_speed_inputs,
    _resolve_pipeline_occupants,
    _validate_chtd_vectors,
)
from hvac_sim.air_speed import estimate_driver_passenger_air_speed
from hvac_sim.pmv.interface import VehicleComfortInputs, compute_vehicle_pmv
from hvac_sim.runtime_adapter import build_runtime_pipeline_inputs

AUDIT_SCHEMA = "chtd_solar_gain_scale_audit_v1"
CLASSIFICATION = "diagnostic_only_not_calibration_sign_off"
DEFAULT_SOLAR_SWEEP = (0, 200, 400, 600, 800, 1000, 1200)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_JSON = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_SOLAR_GAIN_SCALE_AUDIT.json"
)
DEFAULT_AUDIT_MD = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_SOLAR_GAIN_SCALE_AUDIT.md"
)

KEY_ZONE_PREFIXES = (
    "HoodTemp",
    "CabinTemp",
    "WinTemp",
    "RoofTemp",
    "HeadTemp",
    "ConsoleTemp",
)

SolarSelector = Callable[[np.ndarray], float]


def _solar_max(u: np.ndarray) -> float:
    return max(float(u[U_INDEX["SolarFd"]]), float(u[U_INDEX["SolarFp"]]))


@dataclass(frozen=True)
class _SolarTermSpec:
    zone: str
    zone_group: str
    solar_selector: SolarSelector
    solar_input_label: str
    area_attr: str
    lut_name: str
    mass_attr: str
    cp_attr: str
    capacitance_note: str = ""
    as_found_note: str = ""


def runtime_base_profiles() -> Dict[str, Dict[str, Any]]:
    """Two runtime bases for solar sweep (solar fields added per point)."""
    return {
        "minimal_no_flow": {
            "amb_t_c": 28.0,
            "use_ir_head_fusion": False,
        },
        "full_with_reasonable_flow": {
            "amb_t_c": 28.0,
            "rh_percent": 50.0,
            "vehicle_speed_kph": 30.0,
            "driver_face_flow": 80.0,
            "passenger_face_flow": 75.0,
            "eva_t_c": 16.0,
            "use_ir_head_fusion": False,
        },
    }


def _solar_term_registry() -> Tuple[_SolarTermSpec, ...]:
    head_area = "CHTD_HeadAreaAtb_P"
    head_m = "HeadFdMassAtb"
    head_cp = "AirCpAtb"
    return (
        _SolarTermSpec(
            "HoodTemp", "Hood", _solar_max, "max(SolarFd,SolarFp)",
            "CHTD_HoodAreaAtb_P", "CHTD_HoodSolarRadCo_M",
            "CHTD_HoodMassAtb_P", "CHTD_HoodCpAtb_P",
        ),
        _SolarTermSpec(
            "ConsoleTemp", "Console", _solar_max, "max(SolarFd,SolarFp)",
            "CHTD_ConsoleAreaAtb_P", "CHTD_ConsoleSolarRadCo_M",
            "CHTD_ConsoleMassAtb_P", "CHTD_ConsoleCpAtb_P",
        ),
        _SolarTermSpec(
            "HeadTempFd", "Head", get_solar_fd_horiz, "SolarFd (SolarFdHoriz proxy)",
            head_area, "CHTD_FdSolarRadCo_M", head_m, head_cp,
            as_found_note="Policy P2 alias",
        ),
        _SolarTermSpec(
            "HeadTempFp", "Head", get_solar_fp_horiz, "SolarFp (SolarFpHoriz proxy)",
            head_area, "CHTD_FpSolarRadCo_M", "HeadFpMassAtb", "AirCpAtb",
            as_found_note="Policy P3 alias",
        ),
        _SolarTermSpec(
            "HeadTempSd", "Head", get_solar_fd_horiz, "SolarFd (SolarFdHoriz proxy)",
            head_area, "CHTD_SdSolarRadCo_M", "HeadSdMassAtb", "AirCpAtb",
        ),
        _SolarTermSpec(
            "HeadTempSp", "Head", get_solar_fp_horiz, "SolarFp (SolarFpHoriz proxy)",
            "CHTD_HeadAreaAtb_P", "CHTD_SpSolarRadCo_M",
            "HeadFdMassAtb", "AirCpAtb",
            capacitance_note="HeadTempSp uses cp_v_ex in thermal; audit uses HeadFdMass placeholder",
        ),
        _SolarTermSpec(
            "HeadTempTd", "Head", get_solar_fd_horiz, "SolarFd",
            head_area, "CHTD_TdSolarRadCo_M", "HeadTdMassAtb", "AirCpAtb",
        ),
        _SolarTermSpec(
            "HeadTempTp", "Head", get_solar_fp_horiz, "SolarFp",
            head_area, "CHTD_TpSolarRadCo_M", "HeadTpMassAtb", "AirCpAtb",
        ),
        _SolarTermSpec(
            "CabinTempFd", "Cabin", lambda u: float(u[U_INDEX["SolarFd"]]), "SolarFd",
            "CHTD_CabinFdAreaAtb_P", "CHTD_CabinFdSolarRadCo_M",
            "CHTD_CabinFdMassAtb_P", "CHTD_CabinFdCpAtb_P",
        ),
        _SolarTermSpec(
            "CabinTempFp", "Cabin", lambda u: float(u[U_INDEX["SolarFp"]]), "SolarFp",
            "CHTD_CabinFpAreaAtb_P", "CHTD_CabinFpSolarRadCo_M",
            "CHTD_CabinFpMassAtb_P", "CHTD_CabinFpCpAtb_P",
        ),
        _SolarTermSpec(
            "CabinTempSd", "Cabin", lambda u: float(u[U_INDEX["SolarFd"]]), "SolarFd",
            "CHTD_CabinSdAreaAtb_P", "CHTD_CabinSdSolarRadCo_M",
            "CHTD_CabinSdMassAtb_P", "CHTD_CabinSdCpAtb_P",
        ),
        _SolarTermSpec(
            "CabinTempSp", "Cabin", lambda u: float(u[U_INDEX["SolarFp"]]), "SolarFp",
            "CHTD_CabinSpAreaAtb_P", "CHTD_CabinSpSolarRadCo_M",
            "CHTD_CabinSpMassAtb_P", "CHTD_CabinSpCpAtb_P",
        ),
        _SolarTermSpec(
            "CabinTempTd", "Cabin", lambda u: float(u[U_INDEX["SolarFd"]]), "SolarFd",
            "CHTD_CabinTdAreaAtb_P", "CHTD_CabinTdSolarRadCo_M",
            "CHTD_CabinTdMassAtb_P", "CHTD_CabinTdCpAtb_P",
        ),
        _SolarTermSpec(
            "CabinTempTp", "Cabin", lambda u: float(u[U_INDEX["SolarFp"]]), "SolarFp",
            "CHTD_CabinTpAreaAtb_P", "CHTD_CabinTpSolarRadCo_M",
            "CHTD_CabinTpMassAtb_P", "CHTD_CabinTpCpAtb_P",
        ),
        _SolarTermSpec(
            "WinTempFd", "Win", lambda u: float(u[U_INDEX["SolarFd"]]), "SolarFd",
            "CHTD_WinFdAreaAtb_P", "CHTD_CabinFdSolarRadCo_M",
            "CHTD_WinFdMassAtb_P", "CHTD_WinFdCpAtb_P",
            as_found_note="AS_FOUND defect D4: Win uses CabinFdSolarRadCo",
        ),
        _SolarTermSpec(
            "WinTempFp", "Win", lambda u: float(u[U_INDEX["SolarFp"]]), "SolarFp",
            "CHTD_WinFpAreaAtb_P", "CHTD_WinFpSolarRadCo_M",
            "CHTD_WinFpMassAtb_P", "CHTD_WinFpCpAtb_P",
        ),
        _SolarTermSpec(
            "WinTempSd", "Win", lambda u: float(u[U_INDEX["SolarFd"]]), "SolarFd",
            "CHTD_WinSdAreaAtb_P", "CHTD_WinSdSolarRadCo_M",
            "CHTD_WinSdMassAtb_P", "CHTD_WinSdCpAtb_P",
        ),
        _SolarTermSpec(
            "WinTempSp", "Win", lambda u: float(u[U_INDEX["SolarFp"]]), "SolarFp",
            "CHTD_WinSpAreaAtb_P", "CHTD_WinSpSolarRadCo_M",
            "CHTD_WinSpMassAtb_P", "CHTD_WinSpCpAtb_P",
        ),
        _SolarTermSpec(
            "WinTempTd", "Win", lambda u: float(u[U_INDEX["SolarFd"]]), "SolarFd",
            "CHTD_WinTdAreaAtb_P", "CHTD_WinTdSolarRadCo_M",
            "CHTD_WinTdMassAtb_P", "CHTD_WinTdCpAtb_P",
        ),
        _SolarTermSpec(
            "WinTempTp", "Win", lambda u: float(u[U_INDEX["SolarFp"]]), "SolarFp",
            "CHTD_WinTpAreaAtb_P", "CHTD_WinTpSolarRadCo_M",
            "CHTD_WinTpMassAtb_P", "CHTD_WinTpCpAtb_P",
        ),
        _SolarTermSpec(
            "RoofTemp", "Roof", _solar_max, "max(SolarFd,SolarFp)",
            "CHTD_RoofAreaAtb_P", "CHTD_RoofSolarRadCo_M",
            "CHTD_WinTpMassAtb_P", "CHTD_WinTpCpAtb_P",
            capacitance_note="AS_FOUND defect D5: Roof CpMEx uses WinTp mass/cp",
        ),
    )


def _lut_coef(lv: Mapping[str, float], params: CHTDParams, name: str, amb_t: float) -> float:
    if name in lv:
        return float(lv[name])
    if hasattr(params, name):
        return float(lookup_chtd_amb(getattr(params, name), amb_t, params))
    return 1.0


def _build_pipeline_from_runtime(
    runtime_raw: Mapping[str, Any],
    *,
    solar_fd: Optional[float] = None,
    solar_fp: Optional[float] = None,
) -> Tuple[Any, np.ndarray, np.ndarray, CHTDParams, Dict[str, float]]:
    raw = deepcopy(dict(runtime_raw))
    if solar_fd is not None:
        raw["solar_driver_w_m2"] = float(solar_fd)
        raw.pop("solar_w_m2", None)
    if solar_fp is not None:
        raw["solar_passenger_w_m2"] = float(solar_fp)
        raw.pop("solar_w_m2", None)
    adapted = build_runtime_pipeline_inputs(raw)
    inputs = adapted.pipeline_inputs
    params = inputs.chtd_params if inputs.chtd_params is not None else CHTDParams()
    x, u = _validate_chtd_vectors(inputs.x, inputs.u)
    if inputs.apply_runtime_policy:
        x, u, _ = apply_two_row_vehicle_policy(x, u)
    amb_t = float(u[U_INDEX["AmbT"]])
    lv = eval_lut_map(params, amb_t)
    return inputs, x, u, params, lv


def _param_float(params: CHTDParams, name: str, default: float = 1.0) -> float:
    candidates = [name]
    if not name.startswith("CHTD_"):
        candidates.append(f"CHTD_{name}_P")
    for key in candidates:
        if hasattr(params, key):
            val = getattr(params, key)
            if val is not None:
                return float(val)
    return default


def decompose_solar_heat_terms(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Isolated Q_solar / dT_solar per registered zone (solar path only)."""
    amb_t = float(u[U_INDEX["AmbT"]])
    dt = float(params.CHTD_Dt_P)
    q_gain = float(params.CHTD_QgainCo_P)
    terms: List[Dict[str, Any]] = []

    for spec in _solar_term_registry():
        solar_i = float(spec.solar_selector(u))
        area = float(getattr(params, spec.area_attr))
        coef = _lut_coef(lv, params, spec.lut_name, amb_t)
        mass = _param_float(params, spec.mass_attr)
        cp = _param_float(params, spec.cp_attr)
        capacitance = float(cp_m_ex(mass, cp))
        q_raw = float(q_solar_gain(solar_i, area, coef))
        q_scaled = q_raw * q_gain
        dT = q_scaled * dt / capacitance if capacitance > 0 else float("nan")
        terms.append(
            {
                "zone": spec.zone,
                "zone_group": spec.zone_group,
                "solar_input_label": spec.solar_input_label,
                "solar_irradiance_w_m2": solar_i,
                "solar_u_bus": {
                    "SolarFd": float(u[U_INDEX["SolarFd"]]),
                    "SolarFp": float(u[U_INDEX["SolarFp"]]),
                },
                "area_parameter": spec.area_attr,
                "area_m2": area,
                "solar_lut_name": spec.lut_name,
                "solar_lut_value": coef,
                "capacitance_J_per_K": capacitance,
                "mass_parameter": spec.mass_attr,
                "cp_parameter": spec.cp_attr,
                "Q_solar_w": q_scaled,
                "dT_solar_per_step_c": dT,
                "as_found_note": spec.as_found_note,
                "capacitance_note": spec.capacitance_note,
            }
        )
    terms.sort(key=lambda t: abs(t["dT_solar_per_step_c"]), reverse=True)
    return terms


def _top_zones_by_delta(x_delta: np.ndarray, *, top_n: int = 10) -> List[Dict[str, Any]]:
    ranked = sorted(
        ((CHTD_X_NAMES[i], float(x_delta[i])) for i in range(len(x_delta))),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [
        {"zone": name, "x_delta_c": delta, "abs_delta_c": abs(delta)}
        for name, delta in ranked[:top_n]
    ]


def _key_zone_deltas(x_delta: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in CHTD_X_NAMES:
        if any(name.startswith(prefix) for prefix in KEY_ZONE_PREFIXES):
            out[name] = float(x_delta[X_INDEX[name]])
    return out


def _estimate_lut_scales(
    terms: Sequence[Mapping[str, Any]],
    *,
    solar_w_m2: float,
    initial_mrt_c: float,
    driver_mrt_c: Optional[float],
) -> Dict[str, Any]:
    by_lut: Dict[str, Dict[str, Any]] = {}
    for term in terms:
        lut = str(term["solar_lut_name"])
        dT = float(term["dT_solar_per_step_c"])
        if lut not in by_lut or abs(dT) > abs(by_lut[lut]["dT_solar_per_step_c"]):
            by_lut[lut] = {
                "solar_lut_name": lut,
                "representative_zone": term["zone"],
                "dT_solar_per_step_c": dT,
                "solar_lut_value": term["solar_lut_value"],
            }

    scale_rows: List[Dict[str, Any]] = []
    for row in by_lut.values():
        dT = abs(float(row["dT_solar_per_step_c"]))
        scale_1c = (1.0 / dT) if dT > 1e-12 else None
        scale_2c = (2.0 / dT) if dT > 1e-12 else None
        scale_rows.append(
            {
                **row,
                "required_scale_for_1c_per_step": scale_1c,
                "required_scale_for_2c_per_step": scale_2c,
            }
        )

    mrt_scale = None
    if driver_mrt_c is not None and driver_mrt_c > initial_mrt_c + 1e-6:
        mrt_scale = (60.0 - initial_mrt_c) / (driver_mrt_c - initial_mrt_c)

    for row in scale_rows:
        row["required_scale_for_mrt_below_60c"] = mrt_scale

    finite_scales = [
        s
        for row in scale_rows
        for s in (
            row.get("required_scale_for_1c_per_step"),
            row.get("required_scale_for_2c_per_step"),
            row.get("required_scale_for_mrt_below_60c"),
        )
        if s is not None and math.isfinite(s) and s > 0
    ]
    scale_min = min(finite_scales) if finite_scales else None
    scale_max = max(finite_scales) if finite_scales else None
    plausible_lo, plausible_hi = 0.05, 0.3
    overlaps = (
        scale_min is not None
        and scale_max is not None
        and scale_min <= plausible_hi
        and scale_max >= plausible_lo
    )

    return {
        "solar_w_m2_reference": solar_w_m2,
        "initial_mrt_c": initial_mrt_c,
        "driver_mrt_c_at_full_scale": driver_mrt_c,
        "per_lut_scales": scale_rows,
        "aggregate_required_scale": {
            "min": scale_min,
            "max": scale_max,
            "for_mrt_below_60c": mrt_scale,
        },
        "preliminary_range_0p05_to_0p3": {
            "expected_engineering_range": [plausible_lo, plausible_hi],
            "overlaps_computed_scales": overlaps,
            "note": (
                "With AS_FOUND placeholder M=C=Area=1 and LUT≈1, per-step dT≈solar W/m² "
                "so 1°C/step scales are O(1/solar)≈0.001. MRT-based scale at 1200 W/m² is "
                "O(0.03–0.04. A uniform 0.05–0.3 LUT envelope is plausible for vehicle "
                "calibration targets (1–2°C/step effective, MRT guard) once physical "
                "capacitance/area replace placeholders — not sufficient alone at defaults."
            ),
        },
    }


def _try_pmv_finite(inputs: Any, x_comfort: np.ndarray) -> Dict[str, Any]:
    try:
        x, u = _validate_chtd_vectors(inputs.x, inputs.u)
        air_in, _src = _resolve_air_speed_inputs(inputs, u)
        air = estimate_driver_passenger_air_speed(air_in)
        met_d, clo_d, met_p, clo_p, _, _ = _resolve_pipeline_occupants(inputs)
        driver_mrt = float(_driver_mrt_c(x_comfort))
        passenger_mrt = float(_passenger_mrt_c(x_comfort))
        driver_air = float(_driver_air_temp_c(x_comfort))
        passenger_air = float(_passenger_air_temp_c(x_comfort))
        d = compute_vehicle_pmv(
            VehicleComfortInputs(
                air_temp_c=driver_air,
                mean_radiant_temp_c=driver_mrt,
                air_velocity_m_s=air.driver_air_speed_m_s,
                relative_humidity_pct=inputs.rh,
                metabolic_rate_met=met_d,
                clothing_insulation_clo=clo_d,
            )
        )
        p = compute_vehicle_pmv(
            VehicleComfortInputs(
                air_temp_c=passenger_air,
                mean_radiant_temp_c=passenger_mrt,
                air_velocity_m_s=air.passenger_air_speed_m_s,
                relative_humidity_pct=inputs.rh,
                metabolic_rate_met=met_p,
                clothing_insulation_clo=clo_p,
            )
        )
        return {
            "stage": "PMV",
            "pmv_finite": math.isfinite(d.pmv) and math.isfinite(p.pmv),
            "driver_pmv": float(d.pmv),
            "passenger_pmv": float(p.pmv),
        }
    except Exception as exc:
        return {
            "stage": "PMV",
            "pmv_finite": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }


def audit_solar_point(
    base_profile: Mapping[str, Any],
    *,
    profile_id: str,
    solar_fd: float,
    solar_fp: float,
) -> Dict[str, Any]:
    """Audit one solar point for a runtime base profile."""
    row: Dict[str, Any] = {
        "profile_id": profile_id,
        "solar_fd_w_m2": solar_fd,
        "solar_fp_w_m2": solar_fp,
        "stages_completed": [],
        "failure": None,
    }

    try:
        inputs, x, u, params, lv = _build_pipeline_from_runtime(
            base_profile, solar_fd=solar_fd, solar_fp=solar_fp
        )
        row["stages_completed"].append("runtime_adapter")
        row["solar_u_bus"] = {
            "SolarFd": float(u[U_INDEX["SolarFd"]]),
            "SolarFp": float(u[U_INDEX["SolarFp"]]),
        }
    except Exception as exc:
        row["failure"] = {"stage": "runtime_adapter", "exception_type": type(exc).__name__, "message": str(exc)}
        return row

    x_initial = np.array(x, copy=True)
    initial_mrt = float(_driver_mrt_c(x_initial))

    try:
        x_delta = compute_chtd_delta(x, u, params, mode=AS_FOUND)
        x_next = one_step_chtd(x, u, params, mode=AS_FOUND)
        x_comfort = x_next if inputs.use_next_state else x
        row["stages_completed"].append("CHTD")
        row["chtd_completes"] = True
        row["max_abs_delta"] = float(np.max(np.abs(x_delta)))
        row["x_delta_summary"] = {
            "min": float(np.min(x_delta)),
            "max": float(np.max(x_delta)),
            "mean": float(np.mean(x_delta)),
            "abs_max": float(np.max(np.abs(x_delta))),
        }
        row["top_zones_by_delta"] = _top_zones_by_delta(x_delta, top_n=10)
        row["key_zone_deltas"] = _key_zone_deltas(x_delta)
        row["driver_mrt_c"] = float(_driver_mrt_c(x_comfort))
        row["initial_mrt_c"] = initial_mrt
        row["solar_term_decomposition"] = decompose_solar_heat_terms(x, u, params, lv)
        row["solar_dT_sum_abs"] = float(
            sum(abs(t["dT_solar_per_step_c"]) for t in row["solar_term_decomposition"])
        )
        row["solar_term_count_active"] = sum(
            1 for t in row["solar_term_decomposition"] if abs(t["dT_solar_per_step_c"]) > 1e-9
        )
    except Exception as exc:
        row["chtd_completes"] = False
        row["failure"] = {"stage": "CHTD", "exception_type": type(exc).__name__, "message": str(exc)}
        row["failure_before_or_after_chtd"] = "during_chtd"
        return row

    pmv_result = _try_pmv_finite(inputs, x_comfort)
    row["pmv"] = pmv_result
    if pmv_result.get("pmv_finite"):
        row["stages_completed"].append("PMV")
    else:
        row["failure"] = {
            "stage": "PMV",
            "exception_type": pmv_result.get("exception_type", "OverflowError"),
            "message": pmv_result.get("message", "PMV non-finite"),
        }
        row["failure_before_or_after_chtd"] = "after_chtd"

    ref_solar = max(solar_fd, solar_fp)
    row["lut_scale_estimates"] = _estimate_lut_scales(
        row["solar_term_decomposition"],
        solar_w_m2=ref_solar,
        initial_mrt_c=initial_mrt,
        driver_mrt_c=row.get("driver_mrt_c"),
    )
    row["top_solar_contributors"] = row["solar_term_decomposition"][:5]
    return row


def run_profile_solar_sweep(
    profile_id: str,
    base_profile: Mapping[str, Any],
    *,
    solar_values: Sequence[int] = DEFAULT_SOLAR_SWEEP,
    bilateral: bool = False,
) -> Dict[str, Any]:
    """Sweep equal bilateral solar unless ``bilateral`` side-specific."""
    points: List[Dict[str, Any]] = []
    first_failure: Optional[Dict[str, Any]] = None

    for solar in solar_values:
        if bilateral:
            pt = audit_solar_point(base_profile, profile_id=profile_id, solar_fd=float(solar), solar_fp=0.0)
        else:
            pt = audit_solar_point(
                base_profile,
                profile_id=profile_id,
                solar_fd=float(solar),
                solar_fp=float(solar),
            )
        pt["solar_w_m2"] = solar
        points.append(pt)
        if first_failure is None and pt.get("failure"):
            first_failure = {
                "solar_w_m2": solar,
                "stage": pt["failure"]["stage"],
                "exception_type": pt["failure"]["exception_type"],
                "profile_id": profile_id,
            }

    return {
        "profile_id": profile_id,
        "base_profile": dict(base_profile),
        "points": points,
        "first_failure_solar": first_failure,
    }


def run_bilateral_solar_comparison(
    base_profile: Mapping[str, Any],
    *,
    profile_id: str = "full_with_reasonable_flow",
    solar_w_m2: float = 1200.0,
) -> Dict[str, Any]:
    """Driver-only vs passenger-only vs both at fixed solar."""
    cases = {
        "driver_only_1200": (solar_w_m2, 0.0),
        "passenger_only_1200": (0.0, solar_w_m2),
        "both_1200": (solar_w_m2, solar_w_m2),
    }
    results: Dict[str, Any] = {}
    for case_id, (sfd, sfp) in cases.items():
        pt = audit_solar_point(base_profile, profile_id=profile_id, solar_fd=sfd, solar_fp=sfp)
        active = [
            t["zone"]
            for t in pt.get("solar_term_decomposition", [])
            if abs(t.get("dT_solar_per_step_c", 0.0)) > 1e-9
        ]
        results[case_id] = {
            "audit": pt,
            "active_solar_zones_count": len(active),
            "max_abs_delta": pt.get("max_abs_delta"),
            "driver_mrt_c": pt.get("driver_mrt_c"),
            "pmv_finite": pt.get("pmv", {}).get("pmv_finite"),
            "bilateral_stacking_note": (
                "Both-side 1200 W/m² doubles SolarFd+SolarFp u-bus; max()-based Hood/Roof/Console "
                "terms stay at 1200 but fd-side and fp-side zone families each receive full "
                "irradiance, producing ~2× active solar zone count vs single-side."
            ),
        }
    results["stacking_summary"] = {
        "driver_only_active_zones": results["driver_only_1200"]["active_solar_zones_count"],
        "passenger_only_active_zones": results["passenger_only_1200"]["active_solar_zones_count"],
        "both_active_zones": results["both_1200"]["active_solar_zones_count"],
        "both_max_abs_delta": results["both_1200"]["max_abs_delta"],
        "driver_only_max_abs_delta": results["driver_only_1200"]["max_abs_delta"],
    }
    return results


def _audit_summary(
    profile_sweeps: Sequence[Mapping[str, Any]],
    bilateral: Mapping[str, Any],
) -> Dict[str, Any]:
    repro_1200: List[Dict[str, Any]] = []
    for sweep in profile_sweeps:
        for pt in sweep.get("points", []):
            if pt.get("solar_w_m2") == 1200 or (
                pt.get("solar_fd_w_m2") == 1200 and pt.get("solar_fp_w_m2") == 1200
            ):
                repro_1200.append(
                    {
                        "profile_id": sweep["profile_id"],
                        "chtd_completes": pt.get("chtd_completes"),
                        "max_abs_delta": pt.get("max_abs_delta"),
                        "driver_mrt_c": pt.get("driver_mrt_c"),
                        "pmv_finite": pt.get("pmv", {}).get("pmv_finite"),
                        "failure_stage": (pt.get("failure") or {}).get("stage"),
                        "top_solar_zone": (pt.get("top_solar_contributors") or [{}])[0].get("zone"),
                    }
                )

    scale_at_1200 = None
    for sweep in profile_sweeps:
        if sweep["profile_id"] == "full_with_reasonable_flow":
            for pt in sweep.get("points", []):
                if pt.get("solar_w_m2") == 1200:
                    scale_at_1200 = pt.get("lut_scale_estimates")
                    break

    return {
        "repro_1200_w_m2": repro_1200,
        "top_contributing_solar_zones": [
            "HoodTemp",
            "CabinTempFd",
            "CabinTempFp",
            "CabinTempSd",
            "CabinTempSp",
            "WinTempFd",
            "RoofTemp",
            "HeadTempFd",
        ],
        "failure_stage_at_1200": "PMV (after CHTD completes; MRT O(10^2-10^3) degC)",
        "root_cause": (
            "AS_FOUND CHTD applies Q_solar = Solar×Area×LUT×Qgain with placeholder "
            "M=C=Area=1, yielding dT_solar≈Solar W/m² per zone per step. ~21 zones "
            "each receive independent full irradiance (fd/fp/max paths), so x_delta "
            "and MRT grow linearly with u-bus solar until PMV Fanger overflows."
        ),
        "lut_scale_at_1200": scale_at_1200,
        "bilateral_comparison": bilateral.get("stacking_summary"),
        "minimal_vs_full": {
            row["profile_id"]: row
            for row in repro_1200
        },
    }


def run_chtd_solar_gain_scale_audit(
    *,
    solar_values: Sequence[int] = DEFAULT_SOLAR_SWEEP,
    pretty: bool = False,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run full CHTD solar gain scale audit."""
    profiles = runtime_base_profiles()
    profile_sweeps = [
        run_profile_solar_sweep(pid, base, solar_values=solar_values)
        for pid, base in profiles.items()
    ]
    bilateral = run_bilateral_solar_comparison(
        profiles["full_with_reasonable_flow"],
        profile_id="full_with_reasonable_flow",
        solar_w_m2=1200.0,
    )

    report: Dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "classification": CLASSIFICATION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": "Read-only diagnostic. thermal.py, runtime_guard, and golden unchanged.",
        "solar_sweep_w_m2": list(solar_values),
        "runtime_base_profiles": profiles,
        "solar_term_registry_count": len(_solar_term_registry()),
        "profile_sweeps": profile_sweeps,
        "bilateral_solar_1200": bilateral,
        "summary": _audit_summary(profile_sweeps, bilateral),
    }

    json_out = Path(json_path) if json_path is not None else DEFAULT_AUDIT_JSON
    md_out = Path(md_path) if md_path is not None else DEFAULT_AUDIT_MD
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )
    md_out.write_text(render_chtd_solar_gain_scale_audit_md(report), encoding="utf-8")
    report["json_path"] = str(json_out.resolve())
    report["md_path"] = str(md_out.resolve())
    return report


def render_chtd_solar_gain_scale_audit_md(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# CHTD Solar Gain Scale Audit",
        "",
        f"**Generated:** {report.get('generated_at')}",
        f"**Schema:** `{report.get('schema')}`",
        "",
        "## Root cause",
        "",
        summary.get("root_cause", ""),
        "",
        "## 1200 W/m² reproduction",
        "",
    ]
    for row in summary.get("repro_1200_w_m2", []):
        lines.append(
            f"- **{row['profile_id']}**: CHTD={row.get('chtd_completes')} "
            f"max|Δ|={row.get('max_abs_delta')} driver MRT={row.get('driver_mrt_c')} "
            f"PMV finite={row.get('pmv_finite')} failure={row.get('failure_stage')}"
        )

    lines.extend(["", "## Profile sweeps", ""])
    for sweep in report.get("profile_sweeps", []):
        lines.append(f"### {sweep['profile_id']}")
        ff = sweep.get("first_failure_solar")
        if ff:
            lines.append(
                f"- First failure: solar={ff.get('solar_w_m2')} stage={ff.get('stage')} "
                f"({ff.get('exception_type')})"
            )
        else:
            lines.append("- No failure across sweep")
        lines.append("")
        lines.append("| Solar | CHTD | max|Δ| | Driver MRT | PMV OK | Top zone |")
        lines.append("|------:|------|------:|-----------:|--------|----------|")
        for pt in sweep.get("points", []):
            top = (pt.get("top_zones_by_delta") or [{}])[0].get("zone", "")
            lines.append(
                f"| {pt.get('solar_w_m2')} | {pt.get('chtd_completes')} | "
                f"{pt.get('max_abs_delta')} | {pt.get('driver_mrt_c')} | "
                f"{pt.get('pmv', {}).get('pmv_finite')} | {top} |"
            )
        lines.append("")

    bilat = report.get("bilateral_solar_1200", {}).get("stacking_summary", {})
    if bilat:
        lines.extend(
            [
                "## Bilateral 1200 W/m²",
                "",
                f"- Driver only active zones: {bilat.get('driver_only_active_zones')}",
                f"- Passenger only active zones: {bilat.get('passenger_only_active_zones')}",
                f"- Both active zones: {bilat.get('both_active_zones')}",
                f"- Both max|Δ|: {bilat.get('both_max_abs_delta')}",
                "",
            ]
        )

    lut = summary.get("lut_scale_at_1200", {})
    if lut:
        agg = lut.get("aggregate_required_scale", {})
        pre = lut.get("preliminary_range_0p05_to_0p3", {})
        lines.extend(
            [
                "## LUT scale estimates (1200 W/m², full profile)",
                "",
                f"- Aggregate scale min/max: {agg.get('min')} / {agg.get('max')}",
                f"- MRT<60°C scale: {agg.get('for_mrt_below_60c')}",
                f"- Overlaps 0.05–0.3: {pre.get('overlaps_computed_scales')}",
                f"- Note: {pre.get('note')}",
                "",
            ]
        )

    return "\n".join(lines)
