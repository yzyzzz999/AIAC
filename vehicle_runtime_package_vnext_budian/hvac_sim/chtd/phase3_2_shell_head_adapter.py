"""Phase3.2 shell→head adapter — extended shell-state proxy types."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hvac_sim.chtd.bus_index import U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import cp_v_ex

PHASE3_2_SCHEMA = "phase3_2_shell_head_adapter_v2"
PHASE3_2_OPT_IN_ONLY = True

LEGACY_PROXY_TYPES = ("cabin", "cabin_roof_win", "cabin_solar_index", "heat_soak_index")
SHELL_STATE_PROXY_TYPES = (
    "proxy_roof",
    "proxy_shell_mean",
    "proxy_weighted_shell",
    "proxy_solar_integral",
    "proxy_amb_solar_shell",
)
PROXY_TYPES = LEGACY_PROXY_TYPES + SHELL_STATE_PROXY_TYPES

SOLAR_NORM_W_M2 = 500.0
MAX_HEAD_DELTA_C_PER_STEP = 0.15
DEFAULT_SOLAR_INTEGRAL_TAU_S = 600.0
DEFAULT_AMB_SOLAR_A = 0.08
DEFAULT_AMB_SOLAR_B = 0.15

HEAD_CABIN_QBUS_LUTS = (
    "CHTD_FdCabinFdConvCo_M",
    "CHTD_FpCabinFpConvCo_M",
    "CHTD_FdFeetFdConvCo_M",
    "CHTD_FpFeetFpConvCo_M",
    "CHTD_FeetFdCabinFdConvCo_M",
    "CHTD_CabinFdFeetFdConvCo_M",
)


@dataclass
class Phase32ShellHeadConfig:
    """Opt-in shell-head memory coupling; disabled by default."""

    enabled: bool = False
    variant_id: str = "baseline"
    tau_shell_head_s: float = 600.0
    k_shell_head: float = 0.1
    proxy_type: str = "proxy_roof"
    k_solar_proxy: float = 15.0
    c_head_eff_scale: float = 5.0
    qbus_guard_scale: float = 1.0
    solar_integral_tau_s: float = DEFAULT_SOLAR_INTEGRAL_TAU_S
    amb_solar_a: float = DEFAULT_AMB_SOLAR_A
    amb_solar_b: float = DEFAULT_AMB_SOLAR_B

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PHASE3_2_SCHEMA,
            "phase3_2_opt_in_only": PHASE3_2_OPT_IN_ONLY,
            "enabled": self.enabled,
            "variant_id": self.variant_id,
            "tau_shell_head_s": self.tau_shell_head_s,
            "k_shell_head": self.k_shell_head,
            "proxy_type": self.proxy_type,
            "k_solar_proxy": self.k_solar_proxy,
            "c_head_eff_scale": self.c_head_eff_scale,
            "qbus_guard_scale": self.qbus_guard_scale,
            "solar_integral_tau_s": self.solar_integral_tau_s,
            "amb_solar_a": self.amb_solar_a,
            "amb_solar_b": self.amb_solar_b,
        }


@dataclass
class ShellHeadMemoryState:
    memory_fd: float
    memory_fp: float
    solar_int_fd: float = 0.0
    solar_int_fp: float = 0.0


def _console_temp(x: np.ndarray) -> float:
    return float(x[X_INDEX["ConsoleTemp"]])


def _win_avg(x: np.ndarray, side: str) -> float:
    if side == "fd":
        return float(x[X_INDEX["WinTempFd"]])
    return float(x[X_INDEX["WinTempFp"]])


def _solar_effective(u: np.ndarray, side: str) -> float:
    key = "SolarFd" if side == "fd" else "SolarFp"
    return max(0.0, float(u[U_INDEX[key]]))


def compute_instant_shell_proxy(
    x: np.ndarray,
    u: np.ndarray,
    proxy_type: str,
    aux: ShellHeadMemoryState,
    *,
    k_solar_proxy: float = 15.0,
    amb_solar_a: float = DEFAULT_AMB_SOLAR_A,
    amb_solar_b: float = DEFAULT_AMB_SOLAR_B,
) -> Tuple[float, float]:
    cabin_fd = float(x[X_INDEX["CabinTempFd"]])
    cabin_fp = float(x[X_INDEX["CabinTempFp"]])
    roof = float(x[X_INDEX["RoofTemp"]])
    win_fd = float(x[X_INDEX["WinTempFd"]])
    win_fp = float(x[X_INDEX["WinTempFp"]])
    console = _console_temp(x)
    amb = float(u[U_INDEX["AmbT"]])

    if proxy_type == "cabin":
        return cabin_fd, cabin_fp
    if proxy_type == "cabin_roof_win":
        return 0.5 * cabin_fd + 0.3 * roof + 0.2 * win_fd, 0.5 * cabin_fp + 0.3 * roof + 0.2 * win_fp
    if proxy_type == "cabin_solar_index":
        sn_fd = float(u[U_INDEX["SolarFd"]]) / SOLAR_NORM_W_M2
        sn_fp = float(u[U_INDEX["SolarFp"]]) / SOLAR_NORM_W_M2
        return cabin_fd + k_solar_proxy * sn_fd, cabin_fp + k_solar_proxy * sn_fp
    if proxy_type == "heat_soak_index":
        sn_fd = float(u[U_INDEX["SolarFd"]]) / SOLAR_NORM_W_M2
        sn_fp = float(u[U_INDEX["SolarFp"]]) / SOLAR_NORM_W_M2
        return (
            0.55 * cabin_fd + 0.30 * (cabin_fd - amb) + 0.15 * k_solar_proxy * sn_fd,
            0.55 * cabin_fp + 0.30 * (cabin_fp - amb) + 0.15 * k_solar_proxy * sn_fp,
        )
    if proxy_type == "proxy_roof":
        return roof, roof
    if proxy_type == "proxy_shell_mean":
        shell_fd = float(np.mean([roof, win_fd]))
        shell_fp = float(np.mean([roof, win_fp]))
        return shell_fd, shell_fp
    if proxy_type == "proxy_weighted_shell":
        win_avg_fd = win_fd
        win_avg_fp = win_fp
        return (
            0.5 * roof + 0.3 * win_avg_fd + 0.2 * console,
            0.5 * roof + 0.3 * win_avg_fp + 0.2 * console,
        )
    if proxy_type == "proxy_solar_integral":
        return aux.solar_int_fd, aux.solar_int_fp
    if proxy_type == "proxy_amb_solar_shell":
        return (
            cabin_fd + amb_solar_a * aux.solar_int_fd + amb_solar_b * (amb - cabin_fd),
            cabin_fp + amb_solar_a * aux.solar_int_fp + amb_solar_b * (amb - cabin_fp),
        )
    raise ValueError(f"unknown proxy_type: {proxy_type}")


def init_shell_head_memory(x: np.ndarray, u: np.ndarray, proxy_type: str) -> ShellHeadMemoryState:
    aux = ShellHeadMemoryState(memory_fd=0.0, memory_fp=0.0)
    fd, fp = compute_instant_shell_proxy(x, u, proxy_type, aux)
    return ShellHeadMemoryState(
        memory_fd=fd,
        memory_fp=fp,
        solar_int_fd=_solar_effective(u, "fd"),
        solar_int_fp=_solar_effective(u, "fp"),
    )


def update_solar_integral(aux: ShellHeadMemoryState, u: np.ndarray, dt: float, tau_s: float) -> None:
    tau = max(float(tau_s), 1e-3)
    alpha = min(1.0, dt / tau)
    aux.solar_int_fd += alpha * (_solar_effective(u, "fd") - aux.solar_int_fd)
    aux.solar_int_fp += alpha * (_solar_effective(u, "fp") - aux.solar_int_fp)


def apply_shell_head_params(params, cfg: Phase32ShellHeadConfig):
    p = copy.deepcopy(params)
    if cfg.qbus_guard_scale != 1.0:
        from hvac_sim.validation.chtd_global_calibration_phase1 import _scale_lut_params
        p = _scale_lut_params(p, HEAD_CABIN_QBUS_LUTS, cfg.qbus_guard_scale)
    return p


def apply_shell_head_post_step(
    x: np.ndarray,
    u: np.ndarray,
    mem: ShellHeadMemoryState,
    cfg: Phase32ShellHeadConfig,
    *,
    head_air_volume: float,
    head_air_cp: float,
    dt: float,
) -> ShellHeadMemoryState:
    if not cfg.enabled or cfg.tau_shell_head_s <= 0:
        return mem

    update_solar_integral(mem, u, dt, cfg.solar_integral_tau_s)

    tau = max(float(cfg.tau_shell_head_s), 1e-3)
    proxy_fd, proxy_fp = compute_instant_shell_proxy(
        x, u, cfg.proxy_type, mem,
        k_solar_proxy=cfg.k_solar_proxy,
        amb_solar_a=cfg.amb_solar_a,
        amb_solar_b=cfg.amb_solar_b,
    )
    alpha_mem = min(1.0, dt / tau)
    mem_fd = mem.memory_fd + alpha_mem * (proxy_fd - mem.memory_fd)
    mem_fp = mem.memory_fp + alpha_mem * (proxy_fp - mem.memory_fp)

    k = float(cfg.k_shell_head)
    c_scale = max(float(cfg.c_head_eff_scale), 1e-3)

    for zi, mem_t in ((X_INDEX["HeadTempFd"], mem_fd), (X_INDEX["HeadTempFp"], mem_fp)):
        head_t = float(x[zi])
        c_eff = max(cp_v_ex(head_t, float(head_air_volume), float(head_air_cp)) * c_scale, 1.0)
        delta = k * (mem_t - head_t) * dt / c_eff
        delta = float(np.clip(delta, -MAX_HEAD_DELTA_C_PER_STEP, MAX_HEAD_DELTA_C_PER_STEP))
        x[zi] = head_t + delta

    return ShellHeadMemoryState(
        memory_fd=mem_fd,
        memory_fp=mem_fp,
        solar_int_fd=mem.solar_int_fd,
        solar_int_fp=mem.solar_int_fp,
    )


# Backward-compatible alias
compute_shell_proxy = compute_instant_shell_proxy
