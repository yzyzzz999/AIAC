"""Input sanitization helpers used by engineering preview parameter audits."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

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
