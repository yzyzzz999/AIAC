"""Phase3 shell solar fix adapter — opt-in Roof/Win/Console response tuning.

Does not modify ``thermal.py``. Applies param scaling and optional solar-integral
injection to shell states after ``one_step_chtd``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hvac_sim.chtd.bus_index import U_INDEX, X_INDEX
from hvac_sim.validation.chtd_global_calibration_phase1 import _scale_lut_params

PHASE3_SHELL_SOLAR_FIX_SCHEMA = "phase3_shell_solar_fix_adapter_v1"
OPT_IN_ONLY = True

ROOF_SOLAR_LUTS = ("CHTD_RoofSolarRadCo_M",)
WIN_SOLAR_LUTS = (
    "CHTD_WinFdSolarRadCo_M",
    "CHTD_WinFpSolarRadCo_M",
    "CHTD_CabinFdSolarRadCo_M",
)
CONSOLE_SOLAR_LUTS = ("CHTD_ConsoleSolarRadCo_M",)
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

MAX_SHELL_INJECT_C_PER_STEP = 0.05


@dataclass
class ShellSolarFixConfig:
    """Opt-in shell submodel correction; disabled by default."""

    enabled: bool = False
    variant_id: str = "baseline"
    roof_solar_gain_scale: float = 1.0
    win_solar_gain_scale: float = 1.0
    console_solar_gain_scale: float = 1.0
    roof_capacity_scale: float = 1.0
    win_capacity_scale: float = 1.0
    console_capacity_scale: float = 1.0
    solar_shell_inject_k: float = 0.0
    solar_integral_tau_s: float = 600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PHASE3_SHELL_SOLAR_FIX_SCHEMA,
            "opt_in_only": OPT_IN_ONLY,
            "enabled": self.enabled,
            "variant_id": self.variant_id,
            "roof_solar_gain_scale": self.roof_solar_gain_scale,
            "win_solar_gain_scale": self.win_solar_gain_scale,
            "console_solar_gain_scale": self.console_solar_gain_scale,
            "roof_capacity_scale": self.roof_capacity_scale,
            "win_capacity_scale": self.win_capacity_scale,
            "console_capacity_scale": self.console_capacity_scale,
            "solar_shell_inject_k": self.solar_shell_inject_k,
            "solar_integral_tau_s": self.solar_integral_tau_s,
        }


@dataclass
class ShellSolarIntegralState:
    solar_int_fd: float = 0.0
    solar_int_fp: float = 0.0


def _scale_mass_keys(params, keys: Tuple[str, ...], scale: float):
    if scale == 1.0:
        return params
    p = copy.deepcopy(params)
    for key in keys:
        if hasattr(p, key):
            setattr(p, key, float(getattr(p, key)) * float(scale))
    return p


def apply_shell_solar_fix_params(params, cfg: ShellSolarFixConfig):
    """Apply per-zone solar gain and capacity scales on top of Phase3 baseline params."""
    if not cfg.enabled:
        return params
    p = copy.deepcopy(params)
    if cfg.roof_solar_gain_scale != 1.0:
        p = _scale_lut_params(p, ROOF_SOLAR_LUTS, cfg.roof_solar_gain_scale)
    if cfg.win_solar_gain_scale != 1.0:
        p = _scale_lut_params(p, WIN_SOLAR_LUTS, cfg.win_solar_gain_scale)
    if cfg.console_solar_gain_scale != 1.0:
        p = _scale_lut_params(p, CONSOLE_SOLAR_LUTS, cfg.console_solar_gain_scale)
    p = _scale_mass_keys(p, ROOF_MASS_KEYS, cfg.roof_capacity_scale)
    p = _scale_mass_keys(p, WIN_MASS_KEYS, cfg.win_capacity_scale)
    p = _scale_mass_keys(p, CONSOLE_MASS_KEYS, cfg.console_capacity_scale)
    return p


def _solar_effective(u: np.ndarray, side: str) -> float:
    key = "SolarFd" if side == "fd" else "SolarFp"
    return max(0.0, float(u[U_INDEX[key]]))


def update_shell_solar_integral(aux: ShellSolarIntegralState, u: np.ndarray, dt: float, tau_s: float) -> None:
    tau = max(float(tau_s), 1e-3)
    alpha = min(1.0, dt / tau)
    aux.solar_int_fd += alpha * (_solar_effective(u, "fd") - aux.solar_int_fd)
    aux.solar_int_fp += alpha * (_solar_effective(u, "fp") - aux.solar_int_fp)


def init_shell_solar_integral(u: np.ndarray) -> ShellSolarIntegralState:
    return ShellSolarIntegralState(
        solar_int_fd=_solar_effective(u, "fd"),
        solar_int_fp=_solar_effective(u, "fp"),
    )


def apply_shell_solar_post_step(
    x: np.ndarray,
    u: np.ndarray,
    aux: ShellSolarIntegralState,
    cfg: ShellSolarFixConfig,
    dt: float,
) -> np.ndarray:
    """Optional solar-integral injection into Roof/Win/Console after CHTD step."""
    if not cfg.enabled or cfg.solar_shell_inject_k <= 0.0:
        return x
    update_shell_solar_integral(aux, u, dt, cfg.solar_integral_tau_s)
    solar_max = max(aux.solar_int_fd, aux.solar_int_fp)
    delta = min(MAX_SHELL_INJECT_C_PER_STEP, cfg.solar_shell_inject_k * solar_max / 500.0)
    if delta <= 0.0:
        return x
    out = x.copy()
    for key in ("RoofTemp", "WinTempFd", "WinTempFp", "ConsoleTemp"):
        zi = X_INDEX[key]
        out[zi] = float(out[zi]) + delta
    return out
