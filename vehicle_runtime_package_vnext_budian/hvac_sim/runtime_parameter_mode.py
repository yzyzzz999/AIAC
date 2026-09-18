"""CHTD parameter-mode selection isolated from runtime signal adaptation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from hvac_sim.config.param_loader import load_runtime_params


CHTD_PARAM_MODE_DEFAULT = "default"
CHTD_PARAM_MODE_SAFE_PREVIEW = "safe_preview"
CHTD_PARAM_MODE_PHASE3 = "phase3_engineering_preview"
VALID_CHTD_PARAM_MODES = frozenset(
    {CHTD_PARAM_MODE_DEFAULT, CHTD_PARAM_MODE_SAFE_PREVIEW, CHTD_PARAM_MODE_PHASE3}
)
SAFE_PREVIEW_NOT_VEHICLE_CALIBRATION = "safe_preview_not_vehicle_calibration"


def resolve_chtd_param_mode(raw: Mapping[str, Any]) -> str:
    if raw.get("use_safe_preview_chtd_params") is True:
        return CHTD_PARAM_MODE_SAFE_PREVIEW
    mode = raw.get("chtd_param_mode", CHTD_PARAM_MODE_DEFAULT)
    if mode is None:
        return CHTD_PARAM_MODE_DEFAULT
    normalized = str(mode).strip().lower()
    return normalized if normalized in VALID_CHTD_PARAM_MODES else CHTD_PARAM_MODE_DEFAULT


def apply_chtd_parameter_mode(
    raw: Mapping[str, Any],
    signal: Any,
    pipeline_kw: Dict[str, Any],
    provenance: Dict[str, Any],
    warnings: List[str],
) -> None:
    """Resolve a parameter bundle and annotate pipeline/provenance mappings."""
    mode = resolve_chtd_param_mode(raw)
    is_safe_preview = mode == CHTD_PARAM_MODE_SAFE_PREVIEW
    is_phase3 = mode == CHTD_PARAM_MODE_PHASE3
    provenance.update(
        {
            "chtd_param_mode": mode,
            "safe_preview_active": is_safe_preview,
            "phase3_engineering_preview_active": is_phase3,
        }
    )
    pipeline_kw.update(
        {
            "chtd_param_mode": mode,
            "safe_preview_active": is_safe_preview,
            "phase3_engineering_preview_active": is_phase3,
        }
    )

    if signal.use_initial_calibration_params or signal.params_bundle_path is not None:
        bundle = load_runtime_params(signal.params_bundle_path)
        pipeline_kw["chtd_params"] = bundle.chtd_params
        provenance["params_bundle"] = bundle.provenance_dict()
        provenance["params_bundle"]["loaded_via"] = (
            "params_bundle_path"
            if signal.params_bundle_path is not None
            else "use_initial_calibration_params"
        )
        provenance["params_bundle"]["status"] = "initial_calibration_params"
        if bundle.classification == "initial_guess_not_final_calibration":
            provenance["params_bundle"]["vehicle_scope"] = "other_vehicle_initial_guess"
        if is_safe_preview or is_phase3:
            warnings.append(
                f"chtd_param_mode={mode} ignored because params_bundle_path is set."
            )
        return

    if is_phase3:
        from hvac_sim.chtd.phase3_engineering_preview_params import (
            PHASE3_NOT_VEHICLE_CALIBRATION,
            build_phase3_engineering_preview_chtd_params,
            phase3_diagnostics_flags,
            resolve_phase3_structure_from_runtime,
        )

        parameters, parameter_provenance = build_phase3_engineering_preview_chtd_params()
        structure = resolve_phase3_structure_from_runtime(raw)
        flags = phase3_diagnostics_flags(structure)
        pipeline_kw.update(
            {
                "chtd_params": parameters,
                "params_provenance": {**parameter_provenance, **flags},
                "phase3_capacity_active": flags["phase3_capacity_active"],
                "solar_shell_routing_active": flags["solar_shell_routing_active"],
                "actuator_distribution_active": flags["actuator_distribution_active"],
                "foot_leakage_active": flags["foot_leakage_active"],
                "phase3_classification": flags["classification"],
            }
        )
        provenance["params_provenance"] = pipeline_kw["params_provenance"]
        provenance["phase3_structure"] = structure.provenance()
        provenance["params_bundle"] = {
            "status": "phase3_engineering_preview",
            "classification": PHASE3_NOT_VEHICLE_CALIBRATION,
            "not_vehicle_calibration": True,
            "calibration_status": PHASE3_NOT_VEHICLE_CALIBRATION,
            "opt_in_only": True,
            "note": (
                "Opt-in phase3 engineering preview: capacity bundle + solar shell routing; "
                "actuator/foot leakage off unless phase3_experimental overrides."
            ),
            **flags,
        }
        warnings.append(PHASE3_NOT_VEHICLE_CALIBRATION)
        return

    if is_safe_preview:
        from hvac_sim.chtd.safe_preview_params import build_safe_preview_chtd_params

        parameters, parameter_provenance = build_safe_preview_chtd_params()
        pipeline_kw["chtd_params"] = parameters
        pipeline_kw["params_provenance"] = dict(parameter_provenance)
        provenance["params_provenance"] = dict(parameter_provenance)
        provenance["params_bundle"] = {
            "status": "safe_preview",
            "classification": SAFE_PREVIEW_NOT_VEHICLE_CALIBRATION,
            "not_vehicle_calibration": True,
            "calibration_status": SAFE_PREVIEW_NOT_VEHICLE_CALIBRATION,
            "opt_in_only": True,
            "note": (
                "Opt-in safe_preview CHTDParams for demo/guard preview; "
                "not formal vehicle calibration."
            ),
        }
        warnings.append(SAFE_PREVIEW_NOT_VEHICLE_CALIBRATION)
        return

    provenance["params_bundle"] = {
        "status": "default_model_params",
        "note": (
            "CHTDParams() defaults used; set params_bundle_path or "
            "use_initial_calibration_params to load initial_calibration_params.json"
        ),
    }
