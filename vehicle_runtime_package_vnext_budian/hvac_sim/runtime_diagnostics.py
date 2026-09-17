"""Runtime PMV degradation report (diagnostics only; does not change PMV values)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

_FALLBACK_STATUSES = frozenset(
    {"missing", "default", "fallback", "precision_fallback"}
)


def _signal_status_from_provenance(
    provenance: Mapping[str, Any],
    *,
    key: str,
    missing_label: str,
) -> str:
    item = provenance.get(key, {})
    if not isinstance(item, dict):
        return "unknown"
    status = item.get("status")
    if status in _FALLBACK_STATUSES:
        return str(status)
    if key == "Solar" and status is None and item.get("fallback_w_m2") is not None:
        return "missing"
    if key == "vent_flows":
        return str(item.get("status", "unknown"))
    if key == "RH" and status == "default":
        return "default"
    if key == "VehSpd" and status == "missing":
        return "missing"
    if key == "tma_fallback" and item.get("explicit_count", 0) == 0:
        return "fallback"
    if status == "sig" or item.get("source") == "runtime.frnt_def_tma_est_c":
        return "explicit"
    if item.get("TmaDef_approx"):
        return "precision_fallback"
    return "ok" if status not in _FALLBACK_STATUSES else str(status)


def _collect_missing_inputs(provenance: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    checks = (
        ("raw_amb_t_c", "RawAmbT"),
        ("rh_percent", "RH"),
        ("vehicle_speed_kph", "VehSpd"),
        ("solar", "Solar"),
        ("vent_flows", "vent_flows"),
        ("hct_c", "Hct"),
        ("posn_fdh", "PosnFdh"),
        ("eva_t_c", "TmaDef"),
        ("image_inputs", "image_inputs"),
    )
    for field_name, prov_key in checks:
        item = provenance.get(prov_key, {})
        if prov_key == "image_inputs":
            if provenance.get("image_module") == "absent":
                missing.append(field_name)
            continue
        if prov_key == "Solar":
            if isinstance(item, dict) and item.get("status") == "missing":
                missing.append(field_name)
            continue
        if prov_key == "vent_flows":
            if isinstance(item, dict) and item.get("status") == "missing":
                missing.append("vent_flows")
            continue
        if prov_key == "RH":
            if isinstance(item, dict) and item.get("status") == "default":
                missing.append(field_name)
            continue
        if prov_key == "VehSpd":
            if isinstance(item, dict) and item.get("status") == "missing":
                missing.append(field_name)
            continue
        if prov_key == "RawAmbT":
            if isinstance(item, dict) and item.get("status") == "missing":
                missing.append(field_name)
            continue
        if prov_key == "Hct":
            if isinstance(item, dict) and item.get("status") == "precision_fallback":
                missing.append(field_name)
            continue
        if prov_key == "PosnFdh":
            if isinstance(item, dict) and item.get("status") == "precision_fallback":
                missing.append(field_name)
            continue
    return sorted(set(missing))


def _collect_fallback_used(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    fallbacks: Dict[str, Any] = {}
    for key, item in provenance.items():
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status in _FALLBACK_STATUSES:
            fallbacks[key] = {
                "status": status,
                "fallback": item.get("fallback") or item.get("fallback_kph") or item.get("fallback_w_m2"),
            }
        if key == "TmaDef" and item.get("TmaDef_approx"):
            fallbacks[key] = {
                "status": "precision_fallback",
                "fallback": item.get("fallback", "eva_t_or_amb_t"),
            }
        if key == "WinShdTEst" and item.get("estimated"):
            fallbacks[key] = {"status": "estimated", "source": item.get("source")}
    if isinstance(provenance.get("tma_fallback"), dict):
        tma = provenance["tma_fallback"]
        if int(tma.get("explicit_count", 0)) == 0:
            fallbacks["tma_bus"] = {
                "status": "fallback",
                "default": tma.get("default", "eva_t_or_amb_t"),
            }
    params = provenance.get("params_bundle", {})
    if isinstance(params, dict) and params.get("status") == "default_model_params":
        fallbacks["model_params"] = {"status": "default_model_params"}
    return fallbacks


def _ir_status(
    *,
    feature_flags: Mapping[str, bool],
    inputs_used: Mapping[str, Any],
    image_present: bool,
) -> str:
    if not feature_flags.get("enable_ir_fusion", False):
        return "disabled"
    if not image_present:
        return "missing"
    driver_fusion = inputs_used.get("driver_ir_fusion", {})
    if isinstance(driver_fusion, dict) and driver_fusion.get("source") == "model_ir_fused":
        return "used"
    driver_occ = inputs_used.get("occupant_driver", {})
    ir_temp = None
    if isinstance(driver_occ, dict):
        ir_temp = driver_occ.get("ir_head_surface_temp_c")
    if ir_temp is None:
        return "missing"
    return "model_only"


def _image_status(
    *,
    feature_flags: Mapping[str, bool],
    inputs_used: Mapping[str, Any],
    image_present: bool,
) -> str:
    if not feature_flags.get("enable_image_occupancy", True):
        return "disabled"
    if not image_present:
        return "fallback_default"
    driver_occ = inputs_used.get("occupant_driver", {})
    if isinstance(driver_occ, dict):
        source = driver_occ.get("source", {})
        if isinstance(source, dict) and source.get("occupied") == "default_true":
            return "fallback_default"
    return "used"


def _pmv_quality_level(
    *,
    missing_inputs: Sequence[str],
    fallback_used: Mapping[str, Any],
    feature_flags: Mapping[str, bool],
    params_loaded: bool,
    warnings_count: int,
) -> str:
    fallback_n = len(fallback_used)
    missing_n = len(missing_inputs)
    if not params_loaded and (fallback_n >= 4 or missing_n >= 4):
        return "baseline_only"
    if fallback_n >= 2 or missing_n >= 3 or warnings_count >= 4:
        return "degraded"
    if not feature_flags.get("enable_vehicle_adapted_params") and fallback_n >= 1:
        return "degraded"
    return "full"


def build_degradation_report(
    *,
    pipeline_result: Mapping[str, Any],
    provenance: Mapping[str, Any],
    warnings: Sequence[str],
    feature_flags: Optional[Mapping[str, bool]] = None,
    runtime_raw: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build additive degradation diagnostics; does not alter PMV computation."""
    flags = dict(feature_flags or provenance.get("feature_flags") or {})
    inputs_used = pipeline_result.get("inputs_used", {})
    if not isinstance(inputs_used, dict):
        inputs_used = {}

    image_present = provenance.get("image_module") != "absent"
    if runtime_raw is not None and runtime_raw.get("image_inputs") is not None:
        image_present = True
    if provenance.get("image_module") == "disabled":
        image_present = False

    missing_inputs = _collect_missing_inputs(provenance)
    fallback_used = _collect_fallback_used(provenance)

    params_bundle = provenance.get("params_bundle", {})
    params_loaded = (
        isinstance(params_bundle, dict)
        and params_bundle.get("status") == "initial_calibration_params"
    )

    disabled: List[str] = list(provenance.get("disabled_features") or [])
    active: List[str] = list(provenance.get("active_features") or [])

    ir_status = _ir_status(
        feature_flags=flags,
        inputs_used=inputs_used,
        image_present=image_present,
    )
    image_status = _image_status(
        feature_flags=flags,
        inputs_used=inputs_used,
        image_present=image_present,
    )

    signal_status = {
        "solar": _signal_status_from_provenance(provenance, key="Solar", missing_label="solar"),
        "veh_spd": _signal_status_from_provenance(provenance, key="VehSpd", missing_label="veh_spd"),
        "rh": _signal_status_from_provenance(provenance, key="RH", missing_label="rh"),
        "tma": (
            "precision_fallback"
            if isinstance(provenance.get("TmaDef"), dict)
            and provenance["TmaDef"].get("TmaDef_approx")
            else (
                "fallback"
                if isinstance(provenance.get("tma_fallback"), dict)
                and int(provenance["tma_fallback"].get("explicit_count", 0)) == 0
                else "ok"
            )
        ),
        "flow": _signal_status_from_provenance(
            provenance, key="vent_flows", missing_label="flow"
        ),
    }

    reasons: List[str] = []
    if missing_inputs:
        reasons.append(f"missing upstream fields: {', '.join(missing_inputs)}")
    if fallback_used:
        reasons.append(f"fallback paths active: {', '.join(sorted(fallback_used.keys()))}")
    if disabled:
        reasons.append(f"features disabled by flags: {', '.join(disabled)}")
    if ir_status == "disabled":
        reasons.append("IR head fusion disabled (enable_ir_fusion=false)")
    elif ir_status == "model_only":
        reasons.append("IR present but not fused or below confidence threshold")
    elif ir_status == "missing":
        reasons.append("no IR surface temperature supplied")
    if image_status == "fallback_default":
        reasons.append("image module absent or occupied defaulted to true")
    if not params_loaded:
        reasons.append("using default CHTD/AFE model parameters (not vehicle bundle)")

    pmv_quality_level = _pmv_quality_level(
        missing_inputs=missing_inputs,
        fallback_used=fallback_used,
        feature_flags=flags,
        params_loaded=params_loaded,
        warnings_count=len(warnings),
    )

    return {
        "schema": "runtime_degradation_report_v1",
        "pmv_quality_level": pmv_quality_level,
        "missing_inputs": missing_inputs,
        "fallback_used": fallback_used,
        "disabled_features": disabled,
        "active_features": active,
        "reasons": reasons,
        "ir_status": ir_status,
        "image_status": image_status,
        "signal_status": signal_status,
        "feature_flags": flags,
        "warnings_count": len(warnings),
        "params_loaded": params_loaded,
    }
