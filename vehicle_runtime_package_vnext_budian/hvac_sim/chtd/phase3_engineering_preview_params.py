"""Phase3 engineering preview CHTD param bundle (TDC-CHTD-PHASE3-INTEGRATION-GUARD).

Opt-in mode ``phase3_engineering_preview`` — does not replace default or safe_preview.
Does not modify ``thermal.py`` or AS_FOUND golden defaults on disk.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Tuple

from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.phase3_structure_adapter import (
    Phase3StructureConfig,
    apply_phase3_structure_to_params,
    build_phase3_capacity_params,
    apply_solar_shell_routing_params,
)
from hvac_sim.chtd.safe_preview_params import build_safe_preview_chtd_params

PHASE3_PARAM_MODE = "phase3_engineering_preview"
PHASE3_CLASSIFICATION = "phase3_engineering_preview_not_vehicle_calibration"
PHASE3_NOT_VEHICLE_CALIBRATION = PHASE3_CLASSIFICATION
PHASE3_SCHEMA = "chtd_phase3_engineering_preview_params_v1"

DEFAULT_CAPACITY_SCALE = 100.0
DEFAULT_HEAD_SOLAR_SCALE = 0.0
DEFAULT_SHELL_SOLAR_SCALE = 2.0


def default_actuator_phase3_structure_config(
    *,
    foot_leakage_ratio: float = 0.10,
) -> Phase3StructureConfig:
    """Phase3 + voltage actuator distribution (no AC_ModeVentilaPosn, no foot gating)."""
    return Phase3StructureConfig(
        variant_id="actuator_voltage_phase3",
        use_actuator_based_distribution=True,
        use_voltage_actuator_scores=True,
        use_foot_leakage_routing=True,
        foot_leakage_ratio=foot_leakage_ratio,
        use_capacity_bundle=True,
        capacity_scale=DEFAULT_CAPACITY_SCALE,
        use_solar_shell_routing=True,
        head_solar_scale=DEFAULT_HEAD_SOLAR_SCALE,
        shell_solar_scale=DEFAULT_SHELL_SOLAR_SCALE,
        ac_mode_ventila_posn_used=False,
    )


def default_phase3_structure_config(
    *,
    actuator_based_distribution: bool = False,
    foot_leakage_routing: bool = False,
    foot_leakage_ratio: float = 0.10,
) -> Phase3StructureConfig:
    """Production Phase3 flags: capacity + solar on; actuator/foot off by default."""
    return Phase3StructureConfig(
        variant_id=PHASE3_PARAM_MODE,
        use_actuator_based_distribution=actuator_based_distribution,
        use_foot_leakage_routing=foot_leakage_routing,
        foot_leakage_ratio=foot_leakage_ratio,
        use_capacity_bundle=True,
        capacity_scale=DEFAULT_CAPACITY_SCALE,
        use_solar_shell_routing=True,
        head_solar_scale=DEFAULT_HEAD_SOLAR_SCALE,
        shell_solar_scale=DEFAULT_SHELL_SOLAR_SCALE,
        ac_mode_ventila_posn_used=False,
    )


def resolve_phase3_structure_from_runtime(raw: Mapping[str, Any]) -> Phase3StructureConfig:
    """Optional experimental overrides under ``phase3_experimental``."""
    exp = raw.get("phase3_experimental") or {}
    if not isinstance(exp, dict):
        exp = {}
    return default_phase3_structure_config(
        actuator_based_distribution=bool(exp.get("actuator_based_distribution", False)),
        foot_leakage_routing=bool(exp.get("foot_leakage_routing", False)),
        foot_leakage_ratio=float(exp.get("foot_leakage_ratio", 0.10)),
    )


def phase3_diagnostics_flags(struct_cfg: Phase3StructureConfig) -> Dict[str, Any]:
    return {
        "chtd_param_mode": PHASE3_PARAM_MODE,
        "phase3_capacity_active": bool(struct_cfg.use_capacity_bundle),
        "solar_shell_routing_active": bool(struct_cfg.use_solar_shell_routing),
        "actuator_distribution_active": bool(struct_cfg.use_actuator_based_distribution),
        "foot_leakage_active": bool(struct_cfg.use_foot_leakage_routing),
        "classification": PHASE3_CLASSIFICATION,
        "foot_leakage_ratio": struct_cfg.foot_leakage_ratio if struct_cfg.use_foot_leakage_routing else None,
        "capacity_scale": struct_cfg.capacity_scale if struct_cfg.use_capacity_bundle else None,
    }


def build_phase3_engineering_preview_chtd_params(
    *,
    capacity_scale: float = DEFAULT_CAPACITY_SCALE,
    head_solar_scale: float = DEFAULT_HEAD_SOLAR_SCALE,
    shell_solar_scale: float = DEFAULT_SHELL_SOLAR_SCALE,
    vehicle_class: str = "m8_estimated",
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """safe_preview chain + Phase3 capacity bundle + solar shell routing."""
    base, safe_prov = build_safe_preview_chtd_params(vehicle_class=vehicle_class)
    struct_cfg = default_phase3_structure_config()
    struct_cfg = replace(
        struct_cfg,
        capacity_scale=capacity_scale,
        head_solar_scale=head_solar_scale,
        shell_solar_scale=shell_solar_scale,
    )
    params, layer_meta = apply_phase3_structure_to_params(base, struct_cfg)
    provenance: Dict[str, Any] = {
        "schema": PHASE3_SCHEMA,
        "classification": PHASE3_CLASSIFICATION,
        "not_vehicle_calibration": True,
        "calibration_status": PHASE3_CLASSIFICATION,
        "opt_in_only": True,
        "as_found_golden_unchanged": True,
        "thermal_py_unchanged": True,
        "base_chain": list(safe_prov.get("base_chain", []))
        + [
            "phase3_capacity_bundle (cabin/shell/win/hood mass x scale)",
            "solar_shell_routing (head solar off, shell solar boost)",
        ],
        "safe_preview_provenance": safe_prov,
        "phase3_structure_layers": layer_meta.get("param_layers", []),
        "phase3_defaults": {
            "capacity_scale": capacity_scale,
            "head_solar_scale": head_solar_scale,
            "shell_solar_scale": shell_solar_scale,
            "actuator_based_distribution_default": False,
            "foot_leakage_routing_default": False,
        },
        **phase3_diagnostics_flags(struct_cfg),
    }
    return params, provenance
