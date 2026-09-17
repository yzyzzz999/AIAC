"""Deployment guard for runtime PMV — input sanitization and guard diagnostics.

Does not alter Fanger PMV, CHTD thermal core, or runtime schema requirements.
Wraps ``run_runtime_pmv`` with bounded/clamped upstream fields and traceable
degradation reporting aligned with ``build_degradation_report``.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

GUARD_REPORT_SCHEMA = "runtime_guard_report_v1"
SANITIZE_META_SCHEMA = "runtime_guard_sanitize_v1"

_TEMP_MIN_C = -40.0
_TEMP_MAX_C = 90.0
_RH_MIN = 0.0
_RH_MAX = 100.0
_FLOW_MIN_M3H = 0.0
_FLOW_MAX_M3H = 600.0
_VEHSPD_MIN_KPH = 0.0
_VEHSPD_MAX_KPH = 250.0
_SOLAR_MIN_W_M2 = 0.0
_SOLAR_MAX_W_M2 = 1200.0

_DEFAULT_AMB_T_C = 24.0
_DEFAULT_RH_PERCENT = 50.0
_DEFAULT_VEH_SPD_KPH = 0.0
_DEFAULT_SOLAR_W_M2 = 0.0
_DEFAULT_FLOW_M3H = 0.0

_TEMP_KEYS = (
    "amb_t_c",
    "raw_amb_t_c",
    "raw_ambient_temp_c",
    "eva_t_c",
    "ac_evap_temp_c",
    "ac_fevap_current_temp_c",
    "hct_c",
    "heater_core_outlet_temp_c",
    "posn_fdh",
    "defrost_mix_position",
    "blend_request",
    "defrost_blend_request",
    "frnt_def_tma_est_c",
    "tma_def_c",
    "windshield_glass_temp_c",
    "glass_temp_c",
    "windshield_temp_state_c",
    "win_shd_t_est_c",
    "windshield_temp_c",
    "ict_c",
)

_FLOW_KEYS = (
    "driver_face_flow",
    "passenger_face_flow",
    "driver_floor_flow",
    "passenger_floor_flow",
    "driver_defrost_flow",
    "passenger_defrost_flow",
    "frnt_fd_def_flow",
    "frnt_fdv_flow",
    "frnt_fdf_flow",
    "frnt_fp_def_flow",
    "frnt_fpv_flow",
    "frnt_fpf_flow",
    "frnt_sdv_flow",
    "frnt_sdf_flow",
    "frnt_spv_flow",
    "frnt_spf_flow",
)

_RH_KEYS = ("rh_percent", "rh", "relative_humidity_percent")
_VEHSPD_KEYS = ("vehicle_speed_kph", "veh_spd_kph", "veh_spd")
_SOLAR_KEYS = (
    "solar_driver_w_m2",
    "solar_passenger_w_m2",
    "solar_fd_w_m2",
    "solar_fd",
    "solar_fp_w_m2",
    "solar_fp",
    "solar_w_m2",
    "solar_total_w_m2",
    "sig_solar_w_m2",
    "solar_irradiance_w_m2",
)

_IR_FLAG_KEYS = ("use_ir_head_fusion", "enable_ir_fusion")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMOKE_REPORT_JSON = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "RUNTIME_GUARD_SMOKE_REPORT.json"
)
DEFAULT_SMOKE_REPORT_MD = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "RUNTIME_GUARD_SMOKE_REPORT.md"
)


def _is_missing(value: Any) -> bool:
    return value is None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _finite_or(value: Any, fallback: float) -> Tuple[float, bool]:
    parsed = _to_float(value)
    if parsed is None or not math.isfinite(parsed):
        return float(fallback), True
    return parsed, False


def _clamp_value(
    value: Any,
    *,
    lo: float,
    hi: float,
    fallback: float,
    field: str,
    clamped_fields: List[Dict[str, Any]],
    fallback_fields: List[Dict[str, Any]],
    invalid_fields: List[Dict[str, Any]],
    warnings: List[str],
) -> float:
    parsed = _to_float(value)
    if parsed is None:
        invalid_fields.append({"field": field, "reason": "non_numeric", "raw": value})
        fallback_fields.append({"field": field, "fallback": fallback, "reason": "non_numeric"})
        warnings.append(f"{field}: non-numeric → fallback {fallback}")
        return float(fallback)
    if not math.isfinite(parsed):
        invalid_fields.append({"field": field, "reason": "non_finite", "raw": value})
        fallback_fields.append({"field": field, "fallback": fallback, "reason": "non_finite"})
        warnings.append(f"{field}: non-finite → fallback {fallback}")
        return float(fallback)
    if parsed < lo or parsed > hi:
        clamped = max(lo, min(hi, parsed))
        clamped_fields.append(
            {"field": field, "raw": parsed, "clamped": clamped, "lo": lo, "hi": hi}
        )
        warnings.append(f"{field}: clamped {parsed} → {clamped}")
        return clamped
    return parsed


def _sanitize_image_inputs(
    raw: Mapping[str, Any],
    out: MutableMapping[str, Any],
    *,
    fallback_fields: List[Dict[str, Any]],
    disabled_features: List[str],
    warnings: List[str],
    missing_fields: List[str],
) -> None:
    if "image_inputs" not in raw or raw.get("image_inputs") is None:
        missing_fields.append("image_inputs")
        out["image_inputs"] = {
            "driver": {"occupied": True},
            "passenger": {"occupied": True},
        }
        fallback_fields.append(
            {
                "field": "image_inputs",
                "fallback": "occupied_true_default",
                "reason": "missing_image_module",
            }
        )
        warnings.append("image_inputs missing → occupied=true default, IR disabled")
        for key in _IR_FLAG_KEYS:
            if key in out:
                out[key] = False
            else:
                out[key] = False
        if "ir_fusion" not in disabled_features:
            disabled_features.append("ir_fusion")
        return

    image = raw.get("image_inputs")
    if not isinstance(image, dict):
        fallback_fields.append(
            {"field": "image_inputs", "fallback": "occupied_true_default", "reason": "invalid_type"}
        )
        warnings.append("image_inputs invalid → occupied=true default, IR disabled")
        out["image_inputs"] = {
            "driver": {"occupied": True},
            "passenger": {"occupied": True},
        }
        for key in _IR_FLAG_KEYS:
            out[key] = False
        if "ir_fusion" not in disabled_features:
            disabled_features.append("ir_fusion")
        return

    sanitized_image: Dict[str, Any] = {}
    for seat in ("driver", "passenger"):
        seat_raw = image.get(seat)
        if seat_raw is None:
            sanitized_image[seat] = {"occupied": True}
            fallback_fields.append(
                {
                    "field": f"image_inputs.{seat}",
                    "fallback": "occupied_true",
                    "reason": "missing_seat",
                }
            )
            continue
        if not isinstance(seat_raw, dict):
            sanitized_image[seat] = {"occupied": True}
            fallback_fields.append(
                {
                    "field": f"image_inputs.{seat}",
                    "fallback": "occupied_true",
                    "reason": "invalid_seat_type",
                }
            )
            continue
        seat_out = dict(seat_raw)
        if seat_out.get("occupied") is None:
            seat_out["occupied"] = True
            fallback_fields.append(
                {
                    "field": f"image_inputs.{seat}.occupied",
                    "fallback": True,
                    "reason": "missing_occupied",
                }
            )
        ir_temp = seat_out.get("ir_head_surface_temp_c")
        ir_parsed = _to_float(ir_temp)
        if ir_temp is not None and (ir_parsed is None or not math.isfinite(ir_parsed)):
            seat_out.pop("ir_head_surface_temp_c", None)
            fallback_fields.append(
                {
                    "field": f"image_inputs.{seat}.ir_head_surface_temp_c",
                    "fallback": "removed",
                    "reason": "non_finite_ir",
                }
            )
            warnings.append(f"image_inputs.{seat}.ir_head_surface_temp_c non-finite → removed")
        elif ir_parsed is not None:
            clamped_ir = _clamp_value(
                ir_parsed,
                lo=_TEMP_MIN_C,
                hi=_TEMP_MAX_C,
                fallback=_DEFAULT_AMB_T_C + 10.0,
                field=f"image_inputs.{seat}.ir_head_surface_temp_c",
                clamped_fields=[],
                fallback_fields=fallback_fields,
                invalid_fields=[],
                warnings=warnings,
            )
            seat_out["ir_head_surface_temp_c"] = clamped_ir
        sanitized_image[seat] = seat_out
    out["image_inputs"] = sanitized_image


def _sanitize_tma_map(
    tma: Any,
    *,
    clamped_fields: List[Dict[str, Any]],
    fallback_fields: List[Dict[str, Any]],
    invalid_fields: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, float]:
    if not isinstance(tma, dict):
        return {}
    out: Dict[str, float] = {}
    for name, value in tma.items():
        field = f"tma.{name}"
        out[str(name)] = _clamp_value(
            value,
            lo=_TEMP_MIN_C,
            hi=_TEMP_MAX_C,
            fallback=_DEFAULT_AMB_T_C,
            field=field,
            clamped_fields=clamped_fields,
            fallback_fields=fallback_fields,
            invalid_fields=invalid_fields,
            warnings=warnings,
        )
    return out


def _sanitize_flows_map(
    flows: Any,
    *,
    clamped_fields: List[Dict[str, Any]],
    fallback_fields: List[Dict[str, Any]],
    invalid_fields: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, float]:
    if not isinstance(flows, dict):
        return {}
    out: Dict[str, float] = {}
    for name, value in flows.items():
        field = f"flows.{name}"
        out[str(name)] = _clamp_value(
            value,
            lo=_FLOW_MIN_M3H,
            hi=_FLOW_MAX_M3H,
            fallback=_DEFAULT_FLOW_M3H,
            field=field,
            clamped_fields=clamped_fields,
            fallback_fields=fallback_fields,
            invalid_fields=invalid_fields,
            warnings=warnings,
        )
    return out


def _sanitize_previous_state(
    state: Any,
    *,
    clamped_fields: List[Dict[str, Any]],
    fallback_fields: List[Dict[str, Any]],
    invalid_fields: List[Dict[str, Any]],
    warnings: List[str],
) -> Optional[List[float]]:
    if state is None:
        return None
    if not isinstance(state, list):
        invalid_fields.append({"field": "previous_state", "reason": "invalid_type"})
        fallback_fields.append({"field": "previous_state", "fallback": "removed", "reason": "invalid_type"})
        warnings.append("previous_state invalid type → removed")
        return None
    out: List[float] = []
    for idx, value in enumerate(state):
        out.append(
            _clamp_value(
                value,
                lo=_TEMP_MIN_C,
                hi=_TEMP_MAX_C,
                fallback=_DEFAULT_AMB_T_C,
                field=f"previous_state[{idx}]",
                clamped_fields=clamped_fields,
                fallback_fields=fallback_fields,
                invalid_fields=invalid_fields,
                warnings=warnings,
            )
        )
    return out


def sanitize_runtime_input(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Sanitize runtime upstream JSON before ``run_runtime_pmv``.

    Returns a deep copy with clamped/bounded fields, finite fallbacks, and
    default image occupancy when the image module is absent. Original ``data`` is
    not modified.
    """
    raw = dict(data)
    if "runtime" in raw and isinstance(raw["runtime"], dict):
        inner = sanitize_runtime_input(raw["runtime"])
        wrapper = copy.deepcopy(raw)
        wrapper["runtime"] = inner
        return wrapper

    out: Dict[str, Any] = copy.deepcopy(raw)
    clamped_fields: List[Dict[str, Any]] = []
    missing_fields: List[str] = []
    fallback_fields: List[Dict[str, Any]] = []
    invalid_fields: List[Dict[str, Any]] = []
    warnings: List[str] = []
    disabled_features: List[str] = []

    if "amb_t_c" not in out or out.get("amb_t_c") is None:
        missing_fields.append("amb_t_c")
        out["amb_t_c"] = _DEFAULT_AMB_T_C
        fallback_fields.append(
            {"field": "amb_t_c", "fallback": _DEFAULT_AMB_T_C, "reason": "missing_required"}
        )
        warnings.append(f"amb_t_c missing → fallback {_DEFAULT_AMB_T_C}")
    else:
        out["amb_t_c"] = _clamp_value(
            out["amb_t_c"],
            lo=_TEMP_MIN_C,
            hi=_TEMP_MAX_C,
            fallback=_DEFAULT_AMB_T_C,
            field="amb_t_c",
            clamped_fields=clamped_fields,
            fallback_fields=fallback_fields,
            invalid_fields=invalid_fields,
            warnings=warnings,
        )

    for key in _TEMP_KEYS:
        if key == "amb_t_c" or key not in out:
            continue
        if out[key] is None:
            missing_fields.append(key)
            continue
        out[key] = _clamp_value(
            out[key],
            lo=_TEMP_MIN_C,
            hi=_TEMP_MAX_C,
            fallback=float(out["amb_t_c"]),
            field=key,
            clamped_fields=clamped_fields,
            fallback_fields=fallback_fields,
            invalid_fields=invalid_fields,
            warnings=warnings,
        )

    rh_present = False
    for key in _RH_KEYS:
        if key in out and out[key] is not None:
            rh_present = True
            out[key] = _clamp_value(
                out[key],
                lo=_RH_MIN,
                hi=_RH_MAX,
                fallback=_DEFAULT_RH_PERCENT,
                field=key,
                clamped_fields=clamped_fields,
                fallback_fields=fallback_fields,
                invalid_fields=invalid_fields,
                warnings=warnings,
            )
    if not rh_present:
        missing_fields.append("rh_percent")

    veh_present = False
    for key in _VEHSPD_KEYS:
        if key in out and out[key] is not None:
            veh_present = True
            out[key] = _clamp_value(
                out[key],
                lo=_VEHSPD_MIN_KPH,
                hi=_VEHSPD_MAX_KPH,
                fallback=_DEFAULT_VEH_SPD_KPH,
                field=key,
                clamped_fields=clamped_fields,
                fallback_fields=fallback_fields,
                invalid_fields=invalid_fields,
                warnings=warnings,
            )
    if not veh_present:
        missing_fields.append("vehicle_speed_kph")

    solar_present = False
    for key in _SOLAR_KEYS:
        if key in out and out[key] is not None:
            solar_present = True
            out[key] = _clamp_value(
                out[key],
                lo=_SOLAR_MIN_W_M2,
                hi=_SOLAR_MAX_W_M2,
                fallback=_DEFAULT_SOLAR_W_M2,
                field=key,
                clamped_fields=clamped_fields,
                fallback_fields=fallback_fields,
                invalid_fields=invalid_fields,
                warnings=warnings,
            )
    if not solar_present:
        missing_fields.append("solar")

    flow_present = False
    for key in _FLOW_KEYS:
        if key in out and out[key] is not None:
            flow_present = True
            out[key] = _clamp_value(
                out[key],
                lo=_FLOW_MIN_M3H,
                hi=_FLOW_MAX_M3H,
                fallback=_DEFAULT_FLOW_M3H,
                field=key,
                clamped_fields=clamped_fields,
                fallback_fields=fallback_fields,
                invalid_fields=invalid_fields,
                warnings=warnings,
            )
    if not flow_present and not isinstance(out.get("flows"), dict):
        missing_fields.append("vent_flows")

    if "flows" in out:
        out["flows"] = _sanitize_flows_map(
            out.get("flows"),
            clamped_fields=clamped_fields,
            fallback_fields=fallback_fields,
            invalid_fields=invalid_fields,
            warnings=warnings,
        )

    if "tma" in out:
        out["tma"] = _sanitize_tma_map(
            out.get("tma"),
            clamped_fields=clamped_fields,
            fallback_fields=fallback_fields,
            invalid_fields=invalid_fields,
            warnings=warnings,
        )

    if "previous_state" in out:
        sanitized_state = _sanitize_previous_state(
            out.get("previous_state"),
            clamped_fields=clamped_fields,
            fallback_fields=fallback_fields,
            invalid_fields=invalid_fields,
            warnings=warnings,
        )
        if sanitized_state is None:
            out.pop("previous_state", None)
        else:
            out["previous_state"] = sanitized_state

    _sanitize_image_inputs(
        raw,
        out,
        fallback_fields=fallback_fields,
        disabled_features=disabled_features,
        warnings=warnings,
        missing_fields=missing_fields,
    )

    out["_runtime_guard_sanitize"] = {
        "schema": SANITIZE_META_SCHEMA,
        "clamped_fields": clamped_fields,
        "missing_fields": sorted(set(missing_fields)),
        "fallback_fields": fallback_fields,
        "invalid_fields": invalid_fields,
        "warnings": warnings,
        "disabled_features": disabled_features,
    }
    return out


def _strip_sanitize_meta(sanitized: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(sanitized))
    out.pop("_runtime_guard_sanitize", None)
    return out


def _pmv_finite(pipeline_result: Mapping[str, Any]) -> bool:
    pipe = pipeline_result.get("pipeline", {})
    if not isinstance(pipe, dict):
        return False
    driver = pipe.get("driver", {})
    passenger = pipe.get("passenger", {})
    if not isinstance(driver, dict) or not isinstance(passenger, dict):
        return False
    try:
        d_pmv = float(driver.get("pmv", float("nan")))
        p_pmv = float(passenger.get("pmv", float("nan")))
    except (TypeError, ValueError):
        return False
    return math.isfinite(d_pmv) and math.isfinite(p_pmv)


def _valid_for_comfort_interpretation(pipeline_result: Mapping[str, Any]) -> bool:
    if not _pmv_finite(pipeline_result):
        return False
    pipe = pipeline_result["pipeline"]
    driver_valid = bool(pipe.get("driver", {}).get("valid", True))
    passenger_valid = bool(pipe.get("passenger", {}).get("valid", True))
    deg = pipeline_result.get("degradation_report", {})
    quality = deg.get("pmv_quality_level", "baseline_only")
    return driver_valid and passenger_valid and quality != "baseline_only"


def build_runtime_guard_report(
    raw_input: Mapping[str, Any],
    sanitized_input: Mapping[str, Any],
    pipeline_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build deployment guard report aligned with ``degradation_report``."""
    sanitize_meta = dict(sanitized_input.get("_runtime_guard_sanitize", {}))
    deg = dict(pipeline_result.get("degradation_report", {}))

    clamped_fields = list(sanitize_meta.get("clamped_fields", []))
    missing_fields = sorted(
        set(sanitize_meta.get("missing_fields", [])) | set(deg.get("missing_inputs", []))
    )
    fallback_fields = list(sanitize_meta.get("fallback_fields", []))
    for key, item in deg.get("fallback_used", {}).items():
        fallback_fields.append({"field": key, "fallback": item, "source": "degradation_report"})
    invalid_fields = list(sanitize_meta.get("invalid_fields", []))

    disabled_features = sorted(
        set(sanitize_meta.get("disabled_features", []))
        | set(deg.get("disabled_features", []))
        | set(pipeline_result.get("disabled_features", []))
    )

    quality_level = deg.get("pmv_quality_level", "baseline_only")
    guard_warnings = list(sanitize_meta.get("warnings", []))
    if deg.get("reasons"):
        guard_warnings.extend(str(r) for r in deg["reasons"])

    return {
        "schema": GUARD_REPORT_SCHEMA,
        "clamped_fields": clamped_fields,
        "missing_fields": missing_fields,
        "fallback_fields": fallback_fields,
        "invalid_fields": invalid_fields,
        "disabled_features": disabled_features,
        "pmv_finite": _pmv_finite(pipeline_result),
        "valid_for_comfort_interpretation": _valid_for_comfort_interpretation(pipeline_result),
        "quality_level": quality_level,
        "degradation_report_aligned": {
            "pmv_quality_level": quality_level,
            "missing_inputs": deg.get("missing_inputs", []),
            "fallback_used": deg.get("fallback_used", {}),
            "disabled_features": deg.get("disabled_features", []),
            "ir_status": deg.get("ir_status"),
            "image_status": deg.get("image_status"),
        },
        "warnings": guard_warnings,
        "driver_pmv": pipeline_result.get("pipeline", {}).get("driver", {}).get("pmv"),
        "passenger_pmv": pipeline_result.get("pipeline", {}).get("passenger", {}).get("pmv"),
        "driver_valid": pipeline_result.get("pipeline", {}).get("driver", {}).get("valid"),
        "passenger_valid": pipeline_result.get("pipeline", {}).get("passenger", {}).get("valid"),
    }


def run_runtime_pmv_guarded(
    raw_input: Mapping[str, Any],
    *,
    previous_state: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Sanitize input, run runtime PMV, attach guard report."""
    from examples.run_runtime_pmv import run_runtime_pmv

    sanitized = sanitize_runtime_input(raw_input)
    runtime_payload = _strip_sanitize_meta(sanitized)
    pipeline_error: Optional[str] = None
    try:
        pipeline_result = run_runtime_pmv(runtime_payload, previous_state=previous_state)
    except Exception as exc:
        pipeline_error = f"{type(exc).__name__}: {exc}"
        pipeline_result = {
            "schema": "runtime_pmv_output_v1",
            "purpose": "runtime_upstream_pmv_step_guard_failure",
            "trace_status": "guard_pipeline_error",
            "pipeline": {
                "driver": {"pmv": float("nan"), "valid": False},
                "passenger": {"pmv": float("nan"), "valid": False},
            },
            "degradation_report": {
                "schema": "runtime_degradation_report_v1",
                "pmv_quality_level": "baseline_only",
                "missing_inputs": [],
                "fallback_used": {},
                "disabled_features": list(
                    sanitized.get("_runtime_guard_sanitize", {}).get("disabled_features", [])
                ),
                "active_features": [],
                "reasons": [pipeline_error],
                "ir_status": "disabled",
                "image_status": "fallback_default",
                "signal_status": {},
                "feature_flags": {},
                "warnings_count": 0,
                "params_loaded": False,
            },
            "pmv_quality_level": "baseline_only",
            "fallback_used": {},
            "disabled_features": [],
            "runtime_adapter_provenance": {},
            "runtime_adapter_warnings": [],
        }

    guard_report = build_runtime_guard_report(raw_input, sanitized, pipeline_result)
    if pipeline_error:
        guard_report["pipeline_error"] = pipeline_error
        guard_report["pmv_finite"] = False
        guard_report["valid_for_comfort_interpretation"] = False
        guard_report["quality_level"] = "baseline_only"

    return {
        "raw_input": dict(raw_input),
        "sanitized_input": runtime_payload,
        "pipeline_result": pipeline_result,
        "guard_report": guard_report,
        "pipeline_error": pipeline_error,
    }


def builtin_guard_smoke_cases() -> Dict[str, Dict[str, Any]]:
    """Built-in extreme/missing-input scenarios for deployment smoke."""
    return {
        "minimal_only_amb": {"amb_t_c": 26.0},
        "no_image_no_ir": {
            "amb_t_c": 24.0,
            "driver_face_flow": 80.0,
            "eva_t_c": 16.0,
        },
        "nan_inputs": {
            "amb_t_c": 24.0,
            "rh_percent": float("nan"),
            "vehicle_speed_kph": float("inf"),
            "driver_face_flow": float("nan"),
            "solar_w_m2": float("nan"),
            "eva_t_c": 15.0,
        },
        "negative_flow": {
            "amb_t_c": 22.0,
            "driver_face_flow": -50.0,
            "passenger_face_flow": -10.0,
        },
        "overrange_temperature": {
            "amb_t_c": 150.0,
            "raw_amb_t_c": -55.0,
            "eva_t_c": 200.0,
        },
        "overrange_solar": {
            "amb_t_c": 28.0,
            "solar_w_m2": 5000.0,
            "solar_driver_w_m2": 2000.0,
        },
        "overrange_solar_with_safe_preview": {
            "amb_t_c": 28.0,
            "solar_w_m2": 5000.0,
            "solar_driver_w_m2": 2000.0,
            "chtd_param_mode": "safe_preview",
        },
        "occupied_false": {
            "amb_t_c": 24.0,
            "driver_face_flow": 80.0,
            "image_inputs": {
                "driver": {"occupied": False},
                "passenger": {"occupied": True},
            },
        },
        "missing_rh_vehspd_solar": {"amb_t_c": 23.5},
    }


def run_runtime_guard_smoke(
    *,
    pretty: bool = False,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run all built-in guard smoke cases; write JSON/Markdown report."""
    cases = builtin_guard_smoke_cases()
    results: List[Dict[str, Any]] = []
    for case_id, raw in cases.items():
        guarded = run_runtime_pmv_guarded(raw)
        pipeline_error = guarded.get("pipeline_error")
        status = "warning" if pipeline_error else "passed"

        guard = guarded["guard_report"]
        row: Dict[str, Any] = {
            "case_id": case_id,
            "status": status,
            "driver_pmv": guard.get("driver_pmv"),
            "passenger_pmv": guard.get("passenger_pmv"),
            "driver_valid": guard.get("driver_valid"),
            "passenger_valid": guard.get("passenger_valid"),
            "pmv_finite": guard.get("pmv_finite"),
            "valid_for_comfort_interpretation": guard.get("valid_for_comfort_interpretation"),
            "quality_level": guard.get("quality_level"),
            "clamped_fields": guard.get("clamped_fields", []),
            "fallback_fields": guard.get("fallback_fields", []),
            "missing_fields": guard.get("missing_fields", []),
            "disabled_features": guard.get("disabled_features", []),
            "guard_report": guard,
            "degradation_report": guarded["pipeline_result"].get("degradation_report"),
        }
        if pipeline_error:
            row["pipeline_error"] = pipeline_error
        results.append(row)

    report: Dict[str, Any] = {
        "schema": "runtime_guard_smoke_report_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case_count": len(results),
        "cases": results,
        "summary": {
            "all_finite_or_structured": all(
                r.get("status") in ("passed", "warning")
                and (
                    r.get("pmv_finite") is True
                    or r.get("driver_valid") is False
                    or r.get("passenger_valid") is False
                    or r.get("pipeline_error")
                )
                for r in results
            ),
            "failed_cases": [r["case_id"] for r in results if r["status"] == "failed"],
            "warning_cases": [r["case_id"] for r in results if r["status"] == "warning"],
        },
    }

    json_out = Path(json_path) if json_path is not None else DEFAULT_SMOKE_REPORT_JSON
    md_out = Path(md_path) if md_path is not None else DEFAULT_SMOKE_REPORT_MD
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )
    md_out.write_text(render_guard_smoke_report_md(report), encoding="utf-8")
    report["json_path"] = str(json_out.resolve())
    report["md_path"] = str(md_out.resolve())
    return report


def render_guard_smoke_report_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Runtime PMV Guard Smoke Report",
        "",
        f"**Generated:** {report.get('generated_at')}",
        f"**Cases:** {report.get('case_count')}",
        "",
        "## Summary",
        "",
        f"- All finite or structured: {report.get('summary', {}).get('all_finite_or_structured')}",
        "",
        "## Cases",
        "",
    ]
    for row in report.get("cases", []):
        lines.append(f"### {row['case_id']} — `{row['status']}`")
        if row.get("pipeline_error"):
            lines.append(f"- Pipeline error (caught): {row['pipeline_error']}")
        elif row.get("error"):
            lines.append(f"- Error: {row['error']}")
        else:
            lines.append(f"- Driver PMV: {row.get('driver_pmv')}")
            lines.append(f"- Passenger PMV: {row.get('passenger_pmv')}")
            lines.append(f"- PMV finite: {row.get('pmv_finite')}")
            lines.append(f"- Valid for comfort: {row.get('valid_for_comfort_interpretation')}")
            lines.append(f"- Quality: {row.get('quality_level')}")
            if row.get("clamped_fields"):
                lines.append(f"- Clamped: {len(row['clamped_fields'])} field(s)")
            if row.get("fallback_fields"):
                lines.append(f"- Fallbacks: {len(row['fallback_fields'])} field(s)")
            if row.get("missing_fields"):
                lines.append(f"- Missing: {', '.join(row['missing_fields'])}")
            if row.get("disabled_features"):
                lines.append(f"- Disabled: {', '.join(row['disabled_features'])}")
        lines.append("")
    return "\n".join(lines)
