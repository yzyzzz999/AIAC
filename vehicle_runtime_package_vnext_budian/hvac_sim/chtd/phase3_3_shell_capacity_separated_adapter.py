"""Phase3.3 separated shell capacity — opt-in params preview.

Cabin masses keep Phase3 ×100 bundle; Roof/Win/Console scaled independently from
safe_preview base (avoids 4.5e7 J/K Win stagnation). Does not modify ``thermal.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from hvac_sim.chtd.helpers import AS_FOUND, cp_m_ex
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.phase3_engineering_preview_params import DEFAULT_CAPACITY_SCALE
from hvac_sim.chtd.phase3_structure_adapter import (
    apply_solar_shell_routing_params,
)
from hvac_sim.chtd.phase3_shell_solar_fix_adapter import (
    ShellSolarIntegralState,
    apply_shell_solar_post_step,
    init_shell_solar_integral,
)
from hvac_sim.chtd.safe_preview_params import build_safe_preview_chtd_params

PHASE3_3_SCHEMA = "phase3_3_shell_capacity_separated_v1"
PHASE3_3_OPT_IN_ONLY = True
WIN_CAP_WARN_J_K = 1.0e7

CABIN_CAPACITY_KEYS = (
    "CHTD_HoodMassAtb_P",
    "CHTD_CabinFrntMassAtb_P",
    "CHTD_CabinFdMassAtb_P",
    "CHTD_CabinFpMassAtb_P",
    "CHTD_CabinSdMassAtb_P",
    "CHTD_CabinSpMassAtb_P",
    "CHTD_CabinTdMassAtb_P",
    "CHTD_CabinTpMassAtb_P",
)
ROOF_MASS_KEYS = ("CHTD_RoofMassAtb_P",)
WIN_MASS_KEYS = (
    "CHTD_WinFdMassAtb_P",
    "CHTD_WinFpMassAtb_P",
    "CHTD_WinSdMassAtb_P",
    "CHTD_WinSpMassAtb_P",
    "CHTD_WinTdMassAtb_P",
    "CHTD_WinTpMassAtb_P",
)
CONSOLE_MASS_KEYS = ("CHTD_ConsoleMassAtb_P",)


@dataclass
class Phase33ShellCapacityConfig:
    """Opt-in Phase3.3 separated shell capacity; disabled = use caller params unchanged."""

    enabled: bool = False
    variant_id: str = "baseline"
    cabin_capacity_scale: float = DEFAULT_CAPACITY_SCALE
    roof_capacity_scale: float = 3.0
    win_capacity_scale: float = 3.0
    console_capacity_scale: float = 3.0
    shell_solar_gain_scale: float = 2.0
    head_solar_scale: float = 0.0
    solar_integral_inject_k: float = 0.0
    solar_integral_tau_s: float = 600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PHASE3_3_SCHEMA,
            "phase3_3_opt_in_only": PHASE3_3_OPT_IN_ONLY,
            "enabled": self.enabled,
            "variant_id": self.variant_id,
            "cabin_capacity_scale": self.cabin_capacity_scale,
            "roof_capacity_scale": self.roof_capacity_scale,
            "win_capacity_scale": self.win_capacity_scale,
            "console_capacity_scale": self.console_capacity_scale,
            "shell_solar_gain_scale": self.shell_solar_gain_scale,
            "head_solar_scale": self.head_solar_scale,
            "solar_integral_inject_k": self.solar_integral_inject_k,
            "solar_integral_tau_s": self.solar_integral_tau_s,
        }


def _scale_keys_from_base(
    target: CHTDParams,
    base: CHTDParams,
    keys: Tuple[str, ...],
    scale: float,
) -> None:
    for key in keys:
        if hasattr(base, key) and hasattr(target, key):
            setattr(target, key, float(getattr(base, key)) * float(scale))


def build_phase3_3_separated_params(
    cfg: Phase33ShellCapacityConfig,
    *,
    vehicle_class: str = "m8_estimated",
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Rebuild params: cabin ×cabin_scale from safe_preview; shell scales independent."""
    base, safe_prov = build_safe_preview_chtd_params(vehicle_class=vehicle_class)
    p = copy.deepcopy(base)

    _scale_keys_from_base(p, base, CABIN_CAPACITY_KEYS, cfg.cabin_capacity_scale)
    _scale_keys_from_base(p, base, ROOF_MASS_KEYS, cfg.roof_capacity_scale)
    _scale_keys_from_base(p, base, WIN_MASS_KEYS, cfg.win_capacity_scale)
    _scale_keys_from_base(p, base, CONSOLE_MASS_KEYS, cfg.console_capacity_scale)

    p, sol_meta = apply_solar_shell_routing_params(
        p,
        head_solar_scale=cfg.head_solar_scale,
        shell_solar_scale=cfg.shell_solar_gain_scale,
    )
    caps = compute_effective_shell_capacities(p)
    meta: Dict[str, Any] = {
        "schema": PHASE3_3_SCHEMA,
        "phase3_3_opt_in_only": PHASE3_3_OPT_IN_ONLY,
        "classification": "phase3_3_engineering_preview_not_vehicle_calibration",
        "safe_preview_provenance": safe_prov,
        "solar_routing": sol_meta,
        "effective_capacities_j_k": caps,
        "win_cap_exceeds_warn": caps.get("win_fd_j_k", 0) >= WIN_CAP_WARN_J_K,
        **cfg.to_dict(),
    }
    return p, meta


def compute_effective_shell_capacities(params: CHTDParams) -> Dict[str, float]:
    """Effective thermal capacitance [J/K] for shell zones used in diagnostics."""
    return {
        "roof_j_k": float(cp_m_ex(params.CHTD_RoofMassAtb_P, params.CHTD_RoofCpAtb_P)),
        "win_fd_j_k": float(cp_m_ex(params.CHTD_WinFdMassAtb_P, params.CHTD_WinFdCpAtb_P)),
        "win_fp_j_k": float(cp_m_ex(params.CHTD_WinFpMassAtb_P, params.CHTD_WinFpCpAtb_P)),
        "console_j_k": float(cp_m_ex(params.CHTD_ConsoleMassAtb_P, params.CHTD_ConsoleCpAtb_P)),
        "cabin_fd_j_k": float(cp_m_ex(params.CHTD_CabinFdMassAtb_P, params.CHTD_CabinFdCpAtb_P)),
    }


def resolve_params_for_config(
    cfg: Phase33ShellCapacityConfig,
    fallback_params: CHTDParams,
) -> Tuple[CHTDParams, Dict[str, Any]]:
    if not cfg.enabled:
        caps = compute_effective_shell_capacities(fallback_params)
        return fallback_params, {"effective_capacities_j_k": caps, "variant_id": cfg.variant_id}
    return build_phase3_3_separated_params(cfg)


def apply_phase33_inject_post_step(x, u, aux: ShellSolarIntegralState, cfg: Phase33ShellCapacityConfig, dt: float):
    """Solar integral injection using Phase3.3 config (shared post-step with v1 adapter)."""
    from hvac_sim.chtd.phase3_shell_solar_fix_adapter import ShellSolarFixConfig

    if cfg.solar_integral_inject_k <= 0.0:
        return x
    fix = ShellSolarFixConfig(
        enabled=True,
        solar_shell_inject_k=cfg.solar_integral_inject_k,
        solar_integral_tau_s=cfg.solar_integral_tau_s,
    )
    return apply_shell_solar_post_step(x, u, aux, fix, dt)


__all__ = [
    "Phase33ShellCapacityConfig",
    "PHASE3_3_SCHEMA",
    "PHASE3_3_OPT_IN_ONLY",
    "build_phase3_3_separated_params",
    "compute_effective_shell_capacities",
    "resolve_params_for_config",
    "apply_phase33_inject_post_step",
    "init_shell_solar_integral",
    "ShellSolarIntegralState",
    "WIN_CAP_WARN_J_K",
]
