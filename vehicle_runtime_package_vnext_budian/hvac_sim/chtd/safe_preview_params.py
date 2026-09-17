"""Opt-in CHTD safe_preview parameter bundle for demo / guard preview.

Builds on ``physical_capacity_params`` + HVAC conv scales + solar LUT scales.
Does not modify default ``CHTDParams``, AS_FOUND golden, or ``thermal.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.calibration.chtd_hvac_scale_adapter import (
    DEMO_COLD_FOOT_HEATING_CASE,
    apply_chtd_hvac_scale_params,
)
from hvac_sim.chtd.bus_index import U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND
from hvac_sim.chtd.params import CHTDParams, as_lut
from hvac_sim.chtd.physical_params import build_physical_capacity_params
from hvac_sim.chtd.stability_audit import (
    build_state_and_input_from_afe_case,
    build_uniform_equilibrium,
)
from hvac_sim.chtd.thermal import compute_chtd_delta, one_step_chtd
from hvac_sim.pipeline import (
    _driver_air_temp_c,
    _driver_mrt_c,
    _passenger_air_temp_c,
    _passenger_mrt_c,
    run_comfort_pipeline,
)
from hvac_sim.pmv.interface import VehicleComfortInputs, compute_vehicle_pmv
from hvac_sim.runtime_adapter import build_runtime_pipeline_inputs
from hvac_sim.runtime_guard import _strip_sanitize_meta, sanitize_runtime_input

SAFE_PREVIEW_SCHEMA = "chtd_safe_preview_params_v1"
CLASSIFICATION = "safe_preview_not_vehicle_calibration"
NOT_VEHICLE_CALIBRATION = True

DEFAULT_HVAC_SCALES: Dict[str, float] = {
    "front_foot_feet_conv_scale": 0.1,
    "front_face_head_conv_scale": 0.2,
    "second_row_face_head_conv_scale": 0.2,
    "second_row_foot_feet_conv_scale": 0.1,
}
DEFAULT_SOLAR_GLOBAL_SCALE = 0.05
DEFAULT_SOLAR_HEAD_PROXY_SCALE = 0.03

HEAD_SOLAR_LUT_NAMES = frozenset(
    {
        "CHTD_FdSolarRadCo_M",
        "CHTD_FpSolarRadCo_M",
        "CHTD_SdSolarRadCo_M",
        "CHTD_SpSolarRadCo_M",
        "CHTD_TdSolarRadCo_M",
        "CHTD_TpSolarRadCo_M",
    }
)

DEFAULT_SOLAR_SWEEP = (0, 200, 400, 600, 800, 1000, 1200)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECK_JSON = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_SAFE_PREVIEW_CHECK.json"
)
DEFAULT_CHECK_MD = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_SAFE_PREVIEW_CHECK.md"
)


def _all_solar_lut_field_names(params: CHTDParams) -> List[str]:
    return [f.name for f in fields(params) if f.name.endswith("SolarRadCo_M")]


def _scale_lut_field(params: CHTDParams, lut_name: str, factor: float) -> None:
    fac = float(factor)
    if hasattr(params, lut_name):
        arr = np.asarray(getattr(params, lut_name), dtype=float)
        setattr(params, lut_name, arr * fac)
    else:
        setattr(params, lut_name, as_lut(fac))


def apply_solar_lut_scales(
    params: CHTDParams,
    *,
    global_scale: float = DEFAULT_SOLAR_GLOBAL_SCALE,
    head_proxy_scale: Optional[float] = DEFAULT_SOLAR_HEAD_PROXY_SCALE,
) -> Dict[str, float]:
    """Scale all ``*SolarRadCo_M`` LUTs on copied params; returns per-LUT factors applied."""
    applied: Dict[str, float] = {}
    for lut_name in _all_solar_lut_field_names(params):
        if head_proxy_scale is not None and lut_name in HEAD_SOLAR_LUT_NAMES:
            factor = float(head_proxy_scale)
        else:
            factor = float(global_scale)
        _scale_lut_field(params, lut_name, factor)
        applied[lut_name] = factor
    return applied


def build_safe_preview_chtd_params(
    base: Optional[CHTDParams] = None,
    *,
    vehicle_class: str = "m8_estimated",
    hvac_scales: Optional[Mapping[str, float]] = None,
    solar_global_scale: float = DEFAULT_SOLAR_GLOBAL_SCALE,
    solar_head_proxy_scale: Optional[float] = DEFAULT_SOLAR_HEAD_PROXY_SCALE,
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Build opt-in safe_preview ``CHTDParams`` (physical capacity + conv + solar scales)."""
    physical, physical_prov = build_physical_capacity_params(
        base=base,
        vehicle_class=vehicle_class,
    )
    scales = dict(DEFAULT_HVAC_SCALES)
    if hvac_scales:
        scales.update(dict(hvac_scales))

    params = apply_chtd_hvac_scale_params(physical, scales, include_optional=False)
    solar_applied = apply_solar_lut_scales(
        params,
        global_scale=solar_global_scale,
        head_proxy_scale=solar_head_proxy_scale,
    )

    provenance: Dict[str, Any] = {
        "schema": SAFE_PREVIEW_SCHEMA,
        "classification": CLASSIFICATION,
        "not_vehicle_calibration": NOT_VEHICLE_CALIBRATION,
        "calibration_status": "safe_preview_not_vehicle_calibration",
        "vehicle_class": vehicle_class,
        "base_chain": [
            "CHTDParams() defaults (unchanged on disk)",
            "build_physical_capacity_params (cabin/roof/win/console physical bundle)",
            "apply_chtd_hvac_scale_params (HVAC conv scales)",
            "apply_solar_lut_scales (solar_global_scale + optional head proxy)",
        ],
        "hvac_scales_applied": scales,
        "solar_global_scale": float(solar_global_scale),
        "solar_head_proxy_scale": solar_head_proxy_scale,
        "solar_lut_scales_applied": solar_applied,
        "physical_capacity_provenance_meta": physical_prov.get("_meta", {}),
        "opt_in_only": True,
        "as_found_golden_unchanged": True,
        "thermal_py_unchanged": True,
    }
    return params, provenance


def _summer_solar_state(
    *,
    amb_t_c: float = 28.0,
    solar_w_m2: float = 1200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x, u = build_uniform_equilibrium(temp_c=amb_t_c)
    u[U_INDEX["SolarFd"]] = float(solar_w_m2)
    u[U_INDEX["SolarFp"]] = float(solar_w_m2)
    u[U_INDEX["VehSpd"]] = 30.0
    return x, u


def _pmv_from_state(
    x_comfort: np.ndarray,
    *,
    rh: float = 50.0,
) -> Dict[str, Any]:
    driver_mrt = float(_driver_mrt_c(x_comfort))
    passenger_mrt = float(_passenger_mrt_c(x_comfort))
    driver_air = float(_driver_air_temp_c(x_comfort))
    passenger_air = float(_passenger_air_temp_c(x_comfort))
    try:
        d = compute_vehicle_pmv(
            VehicleComfortInputs(
                air_temp_c=driver_air,
                mean_radiant_temp_c=driver_mrt,
                air_velocity_m_s=0.1,
                relative_humidity_pct=rh,
                metabolic_rate_met=1.0,
                clothing_insulation_clo=0.5,
            )
        )
        p = compute_vehicle_pmv(
            VehicleComfortInputs(
                air_temp_c=passenger_air,
                mean_radiant_temp_c=passenger_mrt,
                air_velocity_m_s=0.1,
                relative_humidity_pct=rh,
                metabolic_rate_met=1.0,
                clothing_insulation_clo=0.5,
            )
        )
        return {
            "pmv_finite": math.isfinite(d.pmv) and math.isfinite(p.pmv),
            "driver_pmv": float(d.pmv),
            "passenger_pmv": float(p.pmv),
            "driver_mrt_c": driver_mrt,
            "passenger_mrt_c": passenger_mrt,
        }
    except Exception as exc:
        return {
            "pmv_finite": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "driver_mrt_c": driver_mrt,
            "passenger_mrt_c": passenger_mrt,
        }


def _check_true_equilibrium(params: CHTDParams) -> Dict[str, Any]:
    x, u = build_uniform_equilibrium(temp_c=25.0)
    delta = compute_chtd_delta(x, u, params, AS_FOUND)
    max_abs = float(np.max(np.abs(delta)))
    return {
        "case_id": "true_equilibrium",
        "max_abs_delta_c": max_abs,
        "passed": max_abs <= 1e-6,
        "all_finite": bool(np.all(np.isfinite(delta))),
    }


def _check_cold_foot_heating(params: CHTDParams, *, steps: int = 10) -> Dict[str, Any]:
    x, u, _ = build_state_and_input_from_afe_case(DEMO_COLD_FOOT_HEATING_CASE)
    step_rows: List[Dict[str, Any]] = []
    max_feet_dt = 0.0
    for step in range(steps):
        delta = compute_chtd_delta(x, u, params, AS_FOUND)
        feet_dt = float(delta[X_INDEX["FeetTempFd"]])
        max_feet_dt = max(max_feet_dt, abs(feet_dt))
        step_rows.append({"step": step + 1, "feet_temp_fd_delta_c": feet_dt})
        if not np.all(np.isfinite(delta)):
            break
        x = one_step_chtd(x, u, params, AS_FOUND)
    return {
        "case_id": "cold_foot_heating",
        "steps": steps,
        "step1_feet_temp_fd_delta_c": step_rows[0]["feet_temp_fd_delta_c"] if step_rows else None,
        "max_abs_feet_delta_c": max_feet_dt,
        "passed": max_feet_dt < 5.0 and all(np.isfinite(r["feet_temp_fd_delta_c"]) for r in step_rows),
        "step_trace": step_rows,
    }


def _check_summer_solar_one_step(
    params: CHTDParams,
    *,
    solar_w_m2: float = 1200.0,
) -> Dict[str, Any]:
    x, u = _summer_solar_state(solar_w_m2=solar_w_m2)
    delta = compute_chtd_delta(x, u, params, AS_FOUND)
    x_next = one_step_chtd(x, u, params, AS_FOUND)
    pmv = _pmv_from_state(x_next)
    return {
        "case_id": "summer_solar_one_step",
        "solar_w_m2": solar_w_m2,
        "max_abs_delta_c": float(np.max(np.abs(delta))),
        "driver_mrt_c": pmv.get("driver_mrt_c"),
        "pmv_finite": pmv.get("pmv_finite"),
        "driver_pmv": pmv.get("driver_pmv"),
        "passed": (
            pmv.get("pmv_finite") is True
            and pmv.get("driver_mrt_c", 999.0) < 80.0
            and bool(np.all(np.isfinite(delta)))
        ),
    }


def _check_summer_solar_sweep(
    params: CHTDParams,
    *,
    solar_values: Sequence[int] = DEFAULT_SOLAR_SWEEP,
) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    for solar in solar_values:
        row = _check_summer_solar_one_step(params, solar_w_m2=float(solar))
        points.append(row)
    sweep_ok = all(p.get("passed") for p in points)
    return {
        "case_id": "summer_solar_sweep",
        "solar_values": list(solar_values),
        "points": points,
        "all_passed": sweep_ok,
        "passed": sweep_ok,
    }


def _check_runtime_guard_overrange_preview(params: CHTDParams) -> Dict[str, Any]:
    raw = {
        "amb_t_c": 28.0,
        "solar_w_m2": 5000.0,
        "solar_driver_w_m2": 2000.0,
        "rh_percent": 50.0,
        "vehicle_speed_kph": 30.0,
        "driver_face_flow": 80.0,
        "eva_t_c": 16.0,
    }
    runtime = _strip_sanitize_meta(sanitize_runtime_input(raw))
    adapted = build_runtime_pipeline_inputs(runtime)
    inputs = replace(adapted.pipeline_inputs, chtd_params=params)
    try:
        result = run_comfort_pipeline(inputs)
        pmv_ok = math.isfinite(result.driver.pmv) and math.isfinite(result.passenger.pmv)
        return {
            "case_id": "runtime_guard_overrange_solar_preview",
            "passed": pmv_ok,
            "driver_pmv": float(result.driver.pmv),
            "passenger_pmv": float(result.passenger.pmv),
            "driver_mrt_c": float(_driver_mrt_c(result.x_next)),
            "note": "Uses safe_preview chtd_params on guarded runtime input (clamp still applied)",
        }
    except Exception as exc:
        return {
            "case_id": "runtime_guard_overrange_solar_preview",
            "passed": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }


def run_safe_preview_checks(
    params: Optional[CHTDParams] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    *,
    include_runtime_guard_preview: bool = True,
    solar_sweep: Sequence[int] = DEFAULT_SOLAR_SWEEP,
) -> Dict[str, Any]:
    """Run acceptance check cases against safe_preview params."""
    if params is None or provenance is None:
        params, provenance = build_safe_preview_chtd_params()

    cases = [
        _check_true_equilibrium(params),
        _check_cold_foot_heating(params, steps=10),
        _check_summer_solar_one_step(params, solar_w_m2=1200.0),
        _check_summer_solar_sweep(params, solar_values=solar_sweep),
    ]
    if include_runtime_guard_preview:
        cases.append(_check_runtime_guard_overrange_preview(params))

    all_passed = all(c.get("passed") for c in cases)
    return {
        "schema": "chtd_safe_preview_check_v1",
        "classification": CLASSIFICATION,
        "not_vehicle_calibration": NOT_VEHICLE_CALIBRATION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params_provenance": dict(provenance),
        "cases": cases,
        "all_passed": all_passed,
    }


def render_safe_preview_check_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# CHTD Safe Preview Check",
        "",
        f"**Generated:** {report.get('generated_at')}",
        f"**Classification:** `{report.get('classification')}`",
        f"**not_vehicle_calibration:** `{report.get('not_vehicle_calibration')}`",
        "",
        "Opt-in demo/guard preview bundle — **not** formal vehicle calibration.",
        "",
        f"**All passed:** {report.get('all_passed')}",
        "",
        "## Cases",
        "",
    ]
    for case in report.get("cases", []):
        lines.append(f"### {case['case_id']} — {'PASS' if case.get('passed') else 'FAIL'}")
        for key, val in case.items():
            if key in ("case_id", "passed", "step_trace", "points"):
                continue
            lines.append(f"- {key}: {val}")
        lines.append("")
    prov = report.get("params_provenance", {})
    lines.extend(
        [
            "## Params provenance",
            "",
            f"- HVAC scales: {prov.get('hvac_scales_applied')}",
            f"- Solar global scale: {prov.get('solar_global_scale')}",
            f"- Solar head proxy scale: {prov.get('solar_head_proxy_scale')}",
            "",
        ]
    )
    return "\n".join(lines)


def run_chtd_safe_preview_check(
    *,
    pretty: bool = False,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
    include_runtime_guard_preview: bool = True,
) -> Dict[str, Any]:
    """Build safe_preview params, run checks, write JSON/Markdown report."""
    params, provenance = build_safe_preview_chtd_params()
    report = run_safe_preview_checks(
        params,
        provenance,
        include_runtime_guard_preview=include_runtime_guard_preview,
    )
    report["params_snapshot"] = {
        "CHTD_FeetAirVAtb_P": params.CHTD_FeetAirVAtb_P,
        "CHTD_HeadAirVAtb_P": params.CHTD_HeadAirVAtb_P,
        "CHTD_CabinFdMassAtb_P": params.CHTD_CabinFdMassAtb_P,
        "CHTD_HoodSolarRadCo_M_first": float(params.CHTD_HoodSolarRadCo_M[0]),
        "CHTD_FdSolarRadCo_M_first": float(params.CHTD_FdSolarRadCo_M[0]),
    }

    json_out = Path(json_path) if json_path is not None else DEFAULT_CHECK_JSON
    md_out = Path(md_path) if md_path is not None else DEFAULT_CHECK_MD
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )
    md_out.write_text(render_safe_preview_check_md(report), encoding="utf-8")
    report["json_path"] = str(json_out.resolve())
    report["md_path"] = str(md_out.resolve())
    return report
