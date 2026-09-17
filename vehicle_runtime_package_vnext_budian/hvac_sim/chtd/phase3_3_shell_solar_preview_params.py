"""Phase3.3 shell/solar engineering preview — opt-in ``phase3_3_shell_solar_preview`` mode.

Cabin capacity bundle preserved; Roof/Win/Console scaled independently; optional
solar_integral inject and qbus scale. Does not modify ``thermal.py`` defaults.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.phase3_3_shell_capacity_separated_adapter import (
    PHASE3_3_OPT_IN_ONLY,
    WIN_CAP_WARN_J_K,
    Phase33ShellCapacityConfig,
    build_phase3_3_separated_params,
    compute_effective_shell_capacities,
)
from hvac_sim.chtd.phase3_engineering_preview_params import (
    DEFAULT_CAPACITY_SCALE,
    build_phase3_engineering_preview_chtd_params,
    default_phase3_structure_config,
)
from hvac_sim.chtd.phase3_structure_adapter import Phase3StructureConfig

PHASE3_3_SHELL_SOLAR_PREVIEW_MODE = "phase3_3_shell_solar_preview"
PHASE3_3_CLASSIFICATION = "phase3_3_shell_solar_preview_not_vehicle_calibration"
PHASE3_3_SCHEMA = "phase3_3_shell_solar_preview_params_v1"

# Best separated scales from Phase3.3 capacity grid (win_fd ≈ 4.5e4 J/K).
DEFAULT_ROOF_CAPACITY_SCALE = 10.0
DEFAULT_WIN_CAPACITY_SCALE = 10.0
DEFAULT_CONSOLE_CAPACITY_SCALE = 10.0
DEFAULT_SHELL_SOLAR_GAIN_SCALE = 5.0
DEFAULT_INJECT_K = 0.02
RESEARCH_INJECT_K = 0.05
DEFAULT_QBUS_SCALE = 1.0
RECOMMENDED_QBUS_SCALE = 2.0


def default_phase33_shell_config(
    *,
    inject_k: float = 0.0,
    variant_id: Optional[str] = None,
) -> Phase33ShellCapacityConfig:
    vid = variant_id or (
        f"phase3_3_sep_i{inject_k:.2f}" if inject_k > 0 else "phase3_3_sep_only"
    )
    return Phase33ShellCapacityConfig(
        enabled=True,
        variant_id=vid,
        cabin_capacity_scale=DEFAULT_CAPACITY_SCALE,
        roof_capacity_scale=DEFAULT_ROOF_CAPACITY_SCALE,
        win_capacity_scale=DEFAULT_WIN_CAPACITY_SCALE,
        console_capacity_scale=DEFAULT_CONSOLE_CAPACITY_SCALE,
        shell_solar_gain_scale=DEFAULT_SHELL_SOLAR_GAIN_SCALE,
        head_solar_scale=0.0,
        solar_integral_inject_k=float(inject_k),
        solar_integral_tau_s=600.0,
    )


def build_phase3_3_shell_solar_preview_chtd_params(
    *,
    inject_k: float = DEFAULT_INJECT_K,
    qbus_scale: float = DEFAULT_QBUS_SCALE,
    vehicle_class: str = "m8_estimated",
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Build separated shell params with provenance for opt-in preview mode."""
    cfg = default_phase33_shell_config(inject_k=inject_k)
    params, sep_meta = build_phase3_3_separated_params(cfg, vehicle_class=vehicle_class)
    caps = sep_meta.get("effective_capacities_j_k") or compute_effective_shell_capacities(params)
    win_fd = float(caps.get("win_fd_j_k", 0))
    provenance: Dict[str, Any] = {
        "schema": PHASE3_3_SCHEMA,
        "chtd_param_mode": PHASE3_3_SHELL_SOLAR_PREVIEW_MODE,
        "classification": PHASE3_3_CLASSIFICATION,
        "phase3_3_capacity_separated": True,
        "solar_integral_inject_active": inject_k > 0.0,
        "inject_k": float(inject_k),
        "qbus_scale": float(qbus_scale),
        "opt_in_only": PHASE3_3_OPT_IN_ONLY,
        "not_vehicle_calibration": True,
        "as_found_golden_unchanged": True,
        "win_cap_exceeds_warn": win_fd >= WIN_CAP_WARN_J_K,
        "effective_capacities_j_k": caps,
        "phase33_config": cfg.to_dict(),
        "separation_meta": sep_meta,
        "TA_input_violation_count": 0,
    }
    return params, provenance


def resolve_phase3_3_shell_solar_preview_mode(
    *,
    inject_k: float = DEFAULT_INJECT_K,
    qbus_scale: float = DEFAULT_QBUS_SCALE,
    separated: bool = True,
) -> Tuple[CHTDParams, Phase3StructureConfig, Dict[str, Any]]:
    """Resolve params + structure for validation rollouts."""
    if separated:
        params, prov = build_phase3_3_shell_solar_preview_chtd_params(
            inject_k=inject_k, qbus_scale=qbus_scale,
        )
    else:
        params, base_prov = build_phase3_engineering_preview_chtd_params()
        prov = {
            **base_prov,
            "chtd_param_mode": PHASE3_3_SHELL_SOLAR_PREVIEW_MODE,
            "phase3_3_capacity_separated": False,
            "solar_integral_inject_active": False,
            "inject_k": 0.0,
            "qbus_scale": float(qbus_scale),
            "classification": PHASE3_3_CLASSIFICATION,
        }
    struct = default_phase3_structure_config()
    prov["qbus_scale"] = float(qbus_scale)
    return params, struct, prov


__all__ = [
    "PHASE3_3_SHELL_SOLAR_PREVIEW_MODE",
    "PHASE3_3_CLASSIFICATION",
    "DEFAULT_INJECT_K",
    "RESEARCH_INJECT_K",
    "RECOMMENDED_QBUS_SCALE",
    "default_phase33_shell_config",
    "build_phase3_3_shell_solar_preview_chtd_params",
    "resolve_phase3_3_shell_solar_preview_mode",
]
