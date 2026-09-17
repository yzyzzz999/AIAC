"""CHTD – Cabin Heat Transfer Dynamics.

28-state Forward-Euler thermal model. T6A-0 skeleton: shape validation only.

New Simulink-aligned API:
    one_step_chtd(x, u, params, mode=AS_FOUND)  → ndarray(28,)
    compute_chtd_delta(x, u, params, mode)       → ndarray(28,)

Deprecated legacy API (pre-T6, UA-coupling approach):
    legacy_one_step_chtd(state, inp, params, UA_coupling)
    simulate_chtd(t_vec, inputs_list, params, T0, UA_coupling)

State indices: bus_index.CHTD_X_NAMES (28 zones, HeadTempSp at index 6).
Input indices: bus_index.CHTD_U_NAMES (54 signals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .bus_index import CHTD_X_NAMES, N_X_STATES, N_U, X_INDEX, U_INDEX
from .heat_terms import (
    head_sp_rear_spf_hvac_term,
    q_hvac_duct,
    q_interzone_from_bus,
    q_leakage,
    q_solar_gain,
    q_zone_convection,
    q_zone_radiation,
)
from .helpers import AS_FOUND, DefectMode, cp_m_ex, cp_v_ex, get_solar_fd_horiz, get_solar_fp_horiz, select_defect_lut
from .fast_approx import apply_fast_approx_deltas
from .lut import eval_lut_map
from .params import CHTDParams

# State names aligned with bus_index — 28 states, HeadTempSp at index 6.
_STATE_NAMES = CHTD_X_NAMES
N_STATES = N_X_STATES  # 28


@dataclass
class CHTDState:
    """28-element temperature state vector.  Access by name or index."""
    T: np.ndarray = field(default_factory=lambda: np.full(N_STATES, 20.0))

    def __post_init__(self):
        self.T = np.asarray(self.T, dtype=float)
        if self.T.shape != (N_STATES,):
            raise ValueError(f"CHTDState.T must have {N_STATES} elements")

    def __getattr__(self, name: str) -> float:
        if name in _STATE_NAMES:
            return float(self.T[_STATE_NAMES.index(name)])
        raise AttributeError(name)

    def __setattr__(self, name: str, value):
        if name != "T" and name in _STATE_NAMES:
            self.T[_STATE_NAMES.index(name)] = float(value)
        else:
            super().__setattr__(name, value)

    def as_dict(self) -> dict:
        return {n: float(self.T[i]) for i, n in enumerate(_STATE_NAMES)}

    @classmethod
    def uniform(cls, T0: float = 20.0) -> "CHTDState":
        return cls(T=np.full(N_STATES, T0))


@dataclass
class CHTDInputs:
    """Exogenous inputs for one time step (legacy API)."""
    T_amb:         float = 25.0
    T_supply_frnt: float = 20.0
    T_supply_rear: float = 20.0
    Qm_frnt:       float = 0.05
    Qm_rear:       float = 0.02
    I_solar:       float = 0.0


# ── Legacy helper functions (legacy_one_step_chtd only) ───────────────────────

def _build_default_C(p: CHTDParams) -> np.ndarray:
    """28-element capacitance vector [J/K] (legacy, deprecated)."""
    return np.array([
        p.HoodMassAtb,       # 0  HoodTemp
        p.CabinFrntMassAtb,  # 1  CabinFrntTemp
        p.ConsoleMassAtb,    # 2  ConsoleTemp
        p.HeadFdMassAtb,     # 3  HeadTempFd
        p.HeadFpMassAtb,     # 4  HeadTempFp
        p.HeadSdMassAtb,     # 5  HeadTempSd
        1.0,                 # 6  HeadTempSp — no legacy param, placeholder
        p.HeadTdMassAtb,     # 7  HeadTempTd
        p.HeadTpMassAtb,     # 8  HeadTempTp
        p.FeetFdMassAtb,     # 9  FeetTempFd
        p.FeetFpMassAtb,     # 10 FeetTempFp
        p.FeetSdMassAtb,     # 11 FeetTempSd
        p.FeetSpMassAtb,     # 12 FeetTempSp
        p.FeetTdMassAtb,     # 13 FeetTempTd
        p.FeetTpMassAtb,     # 14 FeetTempTp
        p.CabinFdMassAtb,    # 15 CabinTempFd
        p.CabinFpMassAtb,    # 16 CabinTempFp
        p.CabinSdMassAtb,    # 17 CabinTempSd
        p.CabinSpMassAtb,    # 18 CabinTempSp
        p.CabinTdMassAtb,    # 19 CabinTempTd
        p.CabinTpMassAtb,    # 20 CabinTempTp
        p.WinFdMassAtb,      # 21 WinTempFd
        p.WinFpMassAtb,      # 22 WinTempFp
        p.WinSdMassAtb,      # 23 WinTempSd
        p.WinSpMassAtb,      # 24 WinTempSp
        p.WinTdMassAtb,      # 25 WinTempTd
        p.WinTpMassAtb,      # 26 WinTempTp
        p.RoofMassAtb,       # 27 RoofTemp
    ], dtype=float)


def _build_default_UA_amb(p: CHTDParams) -> np.ndarray:
    """28-element UA-to-ambient vector [W/K] (legacy, deprecated)."""
    return np.array([
        p.HoodUAAtb,      # 0
        p.CabinFrntUAAtb, # 1
        p.ConsoleUAAtb,   # 2
        p.HeadFdUAAtb,    # 3
        p.HeadFpUAAtb,    # 4
        p.HeadSdUAAtb,    # 5
        1.0,              # 6 HeadTempSp — placeholder
        p.HeadTdUAAtb,    # 7
        p.HeadTpUAAtb,    # 8
        p.FeetFdUAAtb,    # 9
        p.FeetFpUAAtb,    # 10
        p.FeetSdUAAtb,    # 11
        p.FeetSpUAAtb,    # 12
        p.FeetTdUAAtb,    # 13
        p.FeetTpUAAtb,    # 14
        p.CabinFdUAAtb,   # 15
        p.CabinFpUAAtb,   # 16
        p.CabinSdUAAtb,   # 17
        p.CabinSpUAAtb,   # 18
        p.CabinTdUAAtb,   # 19
        p.CabinTpUAAtb,   # 20
        p.WinFdUAAtb,     # 21
        p.WinFpUAAtb,     # 22
        p.WinSdUAAtb,     # 23
        p.WinSpUAAtb,     # 24
        p.WinTdUAAtb,     # 25
        p.WinTpUAAtb,     # 26
        p.RoofUAAtb,      # 27
    ], dtype=float)


def _build_default_area(p: CHTDParams) -> np.ndarray:
    """28-element solar-absorbing area vector [m²] (legacy, deprecated)."""
    return np.array([
        p.HoodAreaAtb,       # 0
        p.CabinFrntAreaAtb,  # 1
        p.ConsoleAreaAtb,    # 2
        0.0, 0.0, 0.0, 0.0, # 3-6 HeadFd/Fp/Sd/Sp — no direct solar
        0.0, 0.0,            # 7-8 HeadTd, HeadTp
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 9-14 FeetTemp*
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 15-20 CabinTemp*
        p.WinFdAreaAtb,  # 21
        p.WinFpAreaAtb,  # 22
        p.WinSdAreaAtb,  # 23
        p.WinSpAreaAtb,  # 24
        p.WinTdAreaAtb,  # 25
        p.WinTpAreaAtb,  # 26
        p.RoofAreaAtb,   # 27
    ], dtype=float)


def _hvac_flow_vector(p: CHTDParams, inp: CHTDInputs) -> np.ndarray:
    """Per-zone HVAC mass flow [kg/s] (legacy indices, deprecated)."""
    Qm = np.zeros(N_STATES)
    front_idx = [3, 4, 8, 9, 14, 15]
    for i in front_idx:
        Qm[i] = inp.Qm_frnt / len(front_idx)
    rear_idx = [5, 10, 11, 16, 17, 18, 19, 6]
    for i in rear_idx:
        Qm[i] = inp.Qm_rear / len(rear_idx)
    return Qm


def _supply_temp_vector(inp: CHTDInputs) -> np.ndarray:
    """Per-zone supply temperature (legacy indices, deprecated)."""
    T_sup = np.full(N_STATES, inp.T_amb)
    front_idx = [3, 4, 8, 9, 14, 15]
    rear_idx   = [5, 10, 11, 16, 17, 18, 19, 6]
    for i in front_idx:
        T_sup[i] = inp.T_supply_frnt
    for i in rear_idx:
        T_sup[i] = inp.T_supply_rear
    return T_sup


# ── HeadTempSp q-bus (T6B-0 structure; physics TODO T6B-1+) ───────────────────

@dataclass(frozen=True)
class HeadTempSpQBusInputs:
    """Pre-computed q-bus heat into HeadTempSp Add4 (from neighbor subsystems).

    Maps to Simulink q-bus exports consumed by HeadTempSp:
        q_HeadTempSd.q_SdSpRadiation
        q_HeadTempSd.q_SdSpConvection
        q_HeadTempFp.q_FpSpRadiation
        q_HeadTempFp.q_FpSpConvection

    Values are raw [W] before CHTD_QgainCo_P scaling in compute_chtd_delta.
    """

    sd_sp_radiation: float   # q_HeadTempSd.q_SdSpRadiation
    sd_sp_convection: float  # q_HeadTempSd.q_SdSpConvection
    fp_sp_radiation: float   # q_HeadTempFp.q_FpSpRadiation
    fp_sp_convection: float  # q_HeadTempFp.q_FpSpConvection

    @property
    def total(self) -> float:
        return (
            self.sd_sp_radiation + self.sd_sp_convection
            + self.fp_sp_radiation + self.fp_sp_convection
        )


@dataclass(frozen=True)
class HeadTempSpQBusOutputs:
    """Raw q-bus exports from HeadTempSp consumed by other zones.

    Values are raw [W] before QgainCo/QlossCo scaling in the consumer zones.
    """

    sp_feet_temp_radiation: float
    sp_feet_temp_convection: float
    sp_cabin_sp_radiation: float
    sp_cabin_sp_convection: float
    sp_win_sp_radiation: float
    sp_win_sp_convection: float
    sp_roof_radiation: float
    sp_roof_convection: float
    sp_tp_radiation: float
    sp_tp_convection: float


# ── CabinTempFd q-bus (T6C-0; wired from HeadTempFd / FeetTempFd T8) ──────────

_PLACEHOLDER_LUT_7 = np.ones(7)


def _lut_coef_or_placeholder(
    lv: dict,
    params: CHTDParams,
    name: str,
    amb_t: float,
) -> float:
    """Return LUT coef from *lv*; placeholder 1.0 if field missing from params.py."""
    if name in lv:
        return float(lv[name])
    # TODO(T8): add missing FeetTempFd LUT fields to params.py
    from .lut import lookup_chtd_amb

    if hasattr(params, name):
        return lookup_chtd_amb(getattr(params, name), amb_t, params)
    return lookup_chtd_amb(_PLACEHOLDER_LUT_7, amb_t, params)


@dataclass(frozen=True)
class HeadTempFdQBusOutputs:
    """Raw q-bus exports from HeadTempFd (before QgainCo/QlossCo on consumer side)."""

    console_fd_radiation: float
    ws_fd_radiation: float
    fd_def_fd_flow_convection: float
    fdv_fd_flow_convection: float
    fdf_fd_flow_convection: float
    fd_fp_radiation: float
    fd_fp_convection: float
    fd_sd_radiation: float
    fd_sd_convection: float
    fd_feet_temp_radiation: float
    fd_feet_temp_convection: float
    fd_cabin_fd_radiation: float
    fd_cabin_fd_convection: float
    fd_win_fd_radiation: float
    fd_win_fd_convection: float
    fd_roof_radiation: float
    fd_roof_convection: float
    fd_human_heat_transfer: float
    fd_solar_radiation: float
    fd_heat_leakage: float


@dataclass(frozen=True)
class FeetTempFdQBusOutputs:
    """Raw q-bus exports from FeetTempFd."""

    console_feet_fd_radiation: float
    console_feet_fd_convection: float
    fdf_feet_fd_flow_convection: float
    fd_feet_temp_radiation: float
    fd_feet_temp_convection: float
    feet_fd_cabin_fd_radiation: float
    feet_fd_cabin_fd_convection: float
    feet_fd_human_heat_transfer: float
    feet_fd_heat_leakage: float


def _upstream_signal(value: float, fallback: float) -> float:
    """Treat exact 0.0 as unwired upstream; use *fallback* for equilibrium."""
    return fallback if float(value) == 0.0 else float(value)


def _hood_solar_intensity(u: np.ndarray) -> float:
    """Solar input for HoodTemp/ConsoleTemp: max(SolarFd, SolarFp)."""
    return max(
        float(u[U_INDEX["SolarFd"]]),
        float(u[U_INDEX["SolarFp"]]),
    )


@dataclass(frozen=True)
class CabinFrntTempQBusOutputs:
    """Pass-through q-bus exports from CabinFrntTemp (CabinFrntTemp_topology_spec §8)."""

    hood_cabin_frnt_radiation: float
    cabin_frnt_console_radiation: float


@dataclass(frozen=True)
class ConsoleTempQBusOutputs:
    """Raw q-bus exports from ConsoleTemp (ConsoleTemp_topology_spec §7)."""

    cabin_frnt_console_radiation: float
    console_ws_radiation: float
    console_fd_radiation: float
    console_fp_radiation: float
    console_feet_fd_radiation: float
    console_feet_fd_convection: float
    console_feet_fp_radiation: float
    console_feet_fp_convection: float
    console_solar_radiation: float


def compute_cabin_frnt_temp(
    params: CHTDParams,
    hood_cabin_frnt_radiation: float,
    cabin_frnt_console_radiation: float,
) -> tuple[float, CabinFrntTempQBusOutputs]:
    """CabinFrntTemp 2-term TRACE (CabinFrntTemp_topology_spec.md).

    No x-bus, no u-bus, no LUT. Simulink routes Q_console via a 1-step
    q_CabinFrntTemp self pass-through; Python uses the same-step ConsoleTemp
    SubRef1 raw export (``cabin_frnt_console_radiation``) — negligible for Dt=1s.
    """
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P
    C_cabin_frnt = cp_m_ex(
        params.CHTD_CabinFrntMassAtb_P, params.CHTD_CabinFrntCpAtb_P
    )

    qbus = CabinFrntTempQBusOutputs(
        hood_cabin_frnt_radiation=hood_cabin_frnt_radiation,
        cabin_frnt_console_radiation=cabin_frnt_console_radiation,
    )
    Q_net = (
        hood_cabin_frnt_radiation * _q_gain
        + cabin_frnt_console_radiation * _q_loss
    )
    delta = Q_net * params.CHTD_Dt_P / C_cabin_frnt
    return delta, qbus


def compute_console_temp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, ConsoleTempQBusOutputs]:
    """ConsoleTemp 9-term TRACE (ConsoleTemp_topology_spec.md)."""
    t_console = float(x[X_INDEX["ConsoleTemp"]])
    t_cabin_frnt = float(x[X_INDEX["CabinFrntTemp"]])
    t_head_fd = float(x[X_INDEX["HeadTempFd"]])
    t_head_fp = float(x[X_INDEX["HeadTempFp"]])
    t_feet_fd = float(x[X_INDEX["FeetTempFd"]])
    t_feet_fp = float(x[X_INDEX["FeetTempFp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    win_shd_t = _upstream_signal(float(u[U_INDEX["WinShdTEst"]]), t_console)
    frnt_fdf_flow = float(u[U_INDEX["FrntFdfFlow"]])
    frnt_fpf_flow = float(u[U_INDEX["FrntFpfFlow"]])
    solar_inten = _hood_solar_intensity(u)

    area_console = params.CHTD_ConsoleAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_console = cp_m_ex(params.CHTD_ConsoleMassAtb_P, params.CHTD_ConsoleCpAtb_P)

    cabin_frnt_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinFrntConsoleRadCo_M", amb_t
    )
    q_cabin_frnt_rad_raw = q_zone_radiation(
        t_cabin_frnt, t_console, area_console, cabin_frnt_coef, mode
    )

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_ConsoleSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_inten, area_console, solar_coef)

    ws_coef = _lut_coef_or_placeholder(lv, params, "CHTD_ConsoleWSRadCo_M", amb_t)
    q_ws_rad_raw = q_zone_radiation(
        t_console, win_shd_t, area_console, ws_coef, mode
    )

    fd_coef = _lut_coef_or_placeholder(lv, params, "CHTD_ConsoleFdRadCo_M", amb_t)
    q_fd_rad_raw = q_zone_radiation(
        t_console, t_head_fd, area_console, fd_coef, mode
    )

    fp_coef = _lut_coef_or_placeholder(lv, params, "CHTD_ConsoleFpRadCo_M", amb_t)
    q_fp_rad_raw = q_zone_radiation(
        t_console, t_head_fp, area_console, fp_coef, mode
    )

    feet_fd_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_ConsoleFeetFdRadCo_M", amb_t
    )
    q_feet_fd_rad_raw = q_zone_radiation(
        t_console, t_feet_fd, area_console, feet_fd_rad_coef, mode
    )

    feet_fd_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_ConsoleFeetFdConvCo_M", amb_t
    )
    q_feet_fd_conv_raw = q_zone_convection(
        t_console, t_feet_fd, frnt_fdf_flow, feet_fd_conv_coef
    )

    feet_fp_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_ConsoleFeetFpRadCo_M", amb_t
    )
    q_feet_fp_rad_raw = q_zone_radiation(
        t_console, t_feet_fp, area_console, feet_fp_rad_coef, mode
    )

    feet_fp_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_ConsoleFeetFpConvCo_M", amb_t
    )
    q_feet_fp_conv_raw = q_zone_convection(
        t_console, t_feet_fp, frnt_fpf_flow, feet_fp_conv_coef
    )

    qbus = ConsoleTempQBusOutputs(
        cabin_frnt_console_radiation=q_cabin_frnt_rad_raw,
        console_ws_radiation=q_ws_rad_raw,
        console_fd_radiation=q_fd_rad_raw,
        console_fp_radiation=q_fp_rad_raw,
        console_feet_fd_radiation=q_feet_fd_rad_raw,
        console_feet_fd_convection=q_feet_fd_conv_raw,
        console_feet_fp_radiation=q_feet_fp_rad_raw,
        console_feet_fp_convection=q_feet_fp_conv_raw,
        console_solar_radiation=q_solar_raw,
    )

    Q_net = (
        q_cabin_frnt_rad_raw * _q_gain
        + q_solar_raw * _q_gain
        + q_ws_rad_raw * _q_loss
        + q_fd_rad_raw * _q_loss
        + q_fp_rad_raw * _q_loss
        + q_feet_fd_rad_raw * _q_loss
        + q_feet_fd_conv_raw * _q_loss
        + q_feet_fp_rad_raw * _q_loss
        + q_feet_fp_conv_raw * _q_loss
    )

    delta = Q_net * params.CHTD_Dt_P / C_console
    return delta, qbus


def compute_head_temp_fd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    mode: DefectMode = AS_FOUND,
    console_q: Optional[ConsoleTempQBusOutputs] = None,
) -> tuple[float, HeadTempFdQBusOutputs]:
    """HeadTempFd 20-term TRACE (HeadTempFd_topology_spec.md)."""
    t_fd = float(x[X_INDEX["HeadTempFd"]])
    t_fp = float(x[X_INDEX["HeadTempFp"]])
    t_sd = float(x[X_INDEX["HeadTempSd"]])
    t_feet_fd = float(x[X_INDEX["FeetTempFd"]])
    t_cabin_fd = float(x[X_INDEX["CabinTempFd"]])
    t_win_fd = float(x[X_INDEX["WinTempFd"]])
    t_roof = float(x[X_INDEX["RoofTemp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    win_shd_t = float(u[U_INDEX["WinShdTEst"]])
    if win_shd_t == 0.0:
        # TODO(P5): WinShdTEst is upstream pre-compute; 0 means unwired → no spurious ws delta
        win_shd_t = t_fd
    frnt_def_tma = _upstream_signal(u[U_INDEX["FrntDefTmaEst"]], amb_t)
    # TODO(defrost-glass): FrntDefTmaEst is duct/supply defrost air temperature (TmaDef
    # estimate). Head-zone effective boundary should account for windshield glass
    # exchange — use glass_temp observer / post-glass defrost model (out of CORRECTED
    # low-level LUT scope; see defrost.py + glass_temp.py).
    frnt_fdv_tma = _upstream_signal(u[U_INDEX["FrntFdvTma"]], amb_t)
    frnt_fdf_tma = _upstream_signal(u[U_INDEX["FrntFdfTma"]], amb_t)
    frnt_fd_def_flow = float(u[U_INDEX["FrntFdDefFlow"]])
    frnt_fdv_flow = float(u[U_INDEX["FrntFdvFlow"]])
    frnt_fdf_flow = float(u[U_INDEX["FrntFdfFlow"]])
    frnt_fpv_flow = float(u[U_INDEX["FrntFpvFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head = float(u[U_INDEX["HumHeadFdPower"]])

    area_head = params.CHTD_HeadAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_fd = cp_v_ex(t_fd, params.CHTD_HeadAirVAtb_P, params.CHTD_AirCpAtb_P)

    # ConsoleTemp q-bus: q_ConsoleTemp.q_ConsoleFdRadiation
    q_console_rad_raw = (
        console_q.console_fd_radiation if console_q is not None else 0.0
    )

    q_ws_rad_raw = q_zone_radiation(
        win_shd_t, t_fd, area_head, lv["CHTD_WSFdRadCo_M"], mode
    )
    q_def_conv_raw = q_hvac_duct(
        frnt_def_tma, t_fd, frnt_fd_def_flow, lv["CHTD_FdDefFdConvCo_M"]
    )
    q_fdv_conv_raw = q_hvac_duct(
        frnt_fdv_tma, t_fd, frnt_fdv_flow, lv["CHTD_FdvFdConvCo_M"]
    )
    q_fdf_conv_raw = q_hvac_duct(
        frnt_fdf_tma, t_fd, frnt_fdf_flow, lv["CHTD_FdfFdConvCo_M"]
    )

    q_fd_fp_rad_raw = q_zone_radiation(
        t_fd, t_fp, area_head, lv["CHTD_FdFpRadCo_M"], mode
    )
    q_fd_fp_conv_raw = q_zone_convection(
        t_fd, t_fp, frnt_fdv_flow + frnt_fpv_flow, lv["CHTD_FdFpConvCo_M"]
    )
    q_fd_sd_rad_raw = q_zone_radiation(
        t_fd, t_sd, area_head, lv["CHTD_FdSdRadCo_M"], mode
    )
    q_fd_sd_conv_raw = q_zone_convection(
        t_fd, t_sd, frnt_fdv_flow, lv["CHTD_FdSdConvCo_M"]
    )

    q_feet_rad_raw = q_zone_radiation(
        t_feet_fd, t_fd, area_head, lv["CHTD_FdFeetFdRadCo_M"], mode
    )
    q_feet_conv_raw = q_zone_convection(
        t_feet_fd, t_fd, frnt_fdf_flow, lv["CHTD_FdFeetFdConvCo_M"]
    )

    q_cabin_rad_raw = q_zone_radiation(
        t_fd, t_cabin_fd, area_head, lv["CHTD_FdCabinFdRadCo_M"], mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_fd, t_cabin_fd, frnt_fdv_flow, lv["CHTD_FdCabinFdConvCo_M"]
    )
    q_win_rad_raw = q_zone_radiation(
        t_fd, t_win_fd, area_head, lv["CHTD_FdWinFdRadCo_M"], mode
    )
    q_win_conv_raw = q_zone_convection(
        t_fd, t_win_fd, frnt_fdv_flow, lv["CHTD_FdWinFdConvCo_M"]
    )
    q_roof_rad_raw = q_zone_radiation(
        t_fd, t_roof, area_head, lv["CHTD_FdRoofRadCo_M"], mode
    )
    q_roof_conv_raw = q_zone_convection(
        t_fd, t_roof, frnt_fdv_flow, lv["CHTD_FdRoofConvCo_M"]
    )

    q_human_raw = hum_head

    # DEFECT D1 (Policy P2): SolarFdHoriz not in Bus_CHTD_u; proxy = SolarFd
    solar_fd = get_solar_fd_horiz(u)
    q_solar_raw = q_solar_gain(
        solar_fd, area_head, lv["CHTD_FdSolarRadCo_M"]
    )

    q_leak_raw = q_leakage(
        t_fd, amb_t, veh_spd, params.CHTD_WinFdAreaAtb_P, lv["CHTD_FdLeakageCo_M"]
    )

    qbus = HeadTempFdQBusOutputs(
        console_fd_radiation=q_console_rad_raw,
        ws_fd_radiation=q_ws_rad_raw,
        fd_def_fd_flow_convection=q_def_conv_raw,
        fdv_fd_flow_convection=q_fdv_conv_raw,
        fdf_fd_flow_convection=q_fdf_conv_raw,
        fd_fp_radiation=q_fd_fp_rad_raw,
        fd_fp_convection=q_fd_fp_conv_raw,
        fd_sd_radiation=q_fd_sd_rad_raw,
        fd_sd_convection=q_fd_sd_conv_raw,
        fd_feet_temp_radiation=q_feet_rad_raw,
        fd_feet_temp_convection=q_feet_conv_raw,
        fd_cabin_fd_radiation=q_cabin_rad_raw,
        fd_cabin_fd_convection=q_cabin_conv_raw,
        fd_win_fd_radiation=q_win_rad_raw,
        fd_win_fd_convection=q_win_conv_raw,
        fd_roof_radiation=q_roof_rad_raw,
        fd_roof_convection=q_roof_conv_raw,
        fd_human_heat_transfer=q_human_raw,
        fd_solar_radiation=q_solar_raw,
        fd_heat_leakage=q_leak_raw,
    )

    Q_net = (
        q_console_rad_raw * _q_gain
        + (q_ws_rad_raw + q_def_conv_raw) * _q_gain
        + (q_fdv_conv_raw + q_fdf_conv_raw) * _q_gain
        + (q_fd_fp_rad_raw + q_fd_fp_conv_raw + q_fd_sd_rad_raw + q_fd_sd_conv_raw)
        * _q_loss
        + (q_feet_rad_raw + q_feet_conv_raw) * _q_gain
        + (
            q_cabin_rad_raw
            + q_cabin_conv_raw
            + q_win_rad_raw
            + q_win_conv_raw
            + q_roof_rad_raw
            + q_roof_conv_raw
        )
        * _q_loss
        + q_human_raw
        + q_solar_raw * _q_gain
        + q_leak_raw * _q_loss
    )

    delta = Q_net * params.CHTD_Dt_P / C_fd
    return delta, qbus


def compute_feet_temp_fd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_q: HeadTempFdQBusOutputs,
    mode: DefectMode = AS_FOUND,
    console_q: Optional[ConsoleTempQBusOutputs] = None,
) -> tuple[float, FeetTempFdQBusOutputs]:
    """FeetTempFd 9-term TRACE (FeetTempFd_topology_spec.md)."""
    t_feet = float(x[X_INDEX["FeetTempFd"]])
    t_cabin_fd = float(x[X_INDEX["CabinTempFd"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    frnt_fdf_tma = _upstream_signal(u[U_INDEX["FrntFdfTma"]], amb_t)
    frnt_fdf_flow = float(u[U_INDEX["FrntFdfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet = float(u[U_INDEX["HumFeetFdPower"]])

    area_feet = params.CHTD_FeetAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_feet = cp_v_ex(t_feet, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)

    if console_q is not None:
        q_console_rad_raw = console_q.console_feet_fd_radiation
        q_console_conv_raw = console_q.console_feet_fd_convection
    else:
        q_console_rad_raw = 0.0
        q_console_conv_raw = 0.0

    fdf_feet_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FdfFeetFdConvCo_M", amb_t
    )
    q_fdf_conv_raw = q_hvac_duct(frnt_fdf_tma, t_feet, frnt_fdf_flow, fdf_feet_coef)

    q_head_rad_raw = head_q.fd_feet_temp_radiation
    q_head_conv_raw = head_q.fd_feet_temp_convection

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetFdCabinFdRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetFdCabinFdConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_feet, t_cabin_fd, area_feet, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_feet, t_cabin_fd, frnt_fdf_flow, cabin_conv_coef
    )

    q_human_raw = hum_feet

    leak_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetFdLeakageCo_M", amb_t
    )
    q_leak_raw = q_leakage(
        t_feet, amb_t, veh_spd, params.CHTD_CabinFdAreaAtb_P, leak_coef
    )

    qbus = FeetTempFdQBusOutputs(
        console_feet_fd_radiation=q_console_rad_raw,
        console_feet_fd_convection=q_console_conv_raw,
        fdf_feet_fd_flow_convection=q_fdf_conv_raw,
        fd_feet_temp_radiation=q_head_rad_raw,
        fd_feet_temp_convection=q_head_conv_raw,
        feet_fd_cabin_fd_radiation=q_cabin_rad_raw,
        feet_fd_cabin_fd_convection=q_cabin_conv_raw,
        feet_fd_human_heat_transfer=q_human_raw,
        feet_fd_heat_leakage=q_leak_raw,
    )

    Q_net = (
        (q_console_rad_raw + q_console_conv_raw) * _q_gain
        + q_fdf_conv_raw * _q_gain
        + (q_head_rad_raw + q_head_conv_raw) * _q_loss
        + (q_cabin_rad_raw + q_cabin_conv_raw) * _q_loss
        + q_human_raw
        + q_leak_raw * _q_loss
    )

    delta = Q_net * params.CHTD_Dt_P / C_feet
    return delta, qbus


@dataclass(frozen=True)
class HeadTempFpQBusOutputs:
    """Raw q-bus exports from HeadTempFp (HeadTempFp_topology_spec §8)."""

    console_fp_radiation: float
    ws_fp_radiation: float
    fp_def_fp_flow_convection: float
    fpv_fp_flow_convection: float
    fpf_fp_flow_convection: float
    fd_fp_radiation: float
    fd_fp_convection: float
    fp_sp_radiation: float
    fp_sp_convection: float
    fp_feet_temp_radiation: float
    fp_feet_temp_convection: float
    fp_cabin_fp_radiation: float
    fp_cabin_fp_convection: float
    fp_win_fp_radiation: float
    fp_win_fp_convection: float
    fp_roof_radiation: float
    fp_roof_convection: float
    fp_human_heat_transfer: float
    fp_solar_radiation: float
    fp_heat_leakage: float


@dataclass(frozen=True)
class FeetTempFpQBusOutputs:
    """Raw q-bus exports from FeetTempFp (FeetTempFp_topology_spec §8)."""

    console_feet_fp_radiation: float
    console_feet_fp_convection: float
    fpf_feet_fp_flow_convection: float
    fp_feet_temp_radiation: float
    fp_feet_temp_convection: float
    feet_fp_cabin_fp_radiation: float
    feet_fp_cabin_fp_convection: float
    feet_fp_human_heat_transfer: float
    feet_fp_heat_leakage: float


def compute_head_temp_fp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_fd_q: HeadTempFdQBusOutputs,
    mode: DefectMode = AS_FOUND,
    console_q: Optional[ConsoleTempQBusOutputs] = None,
) -> tuple[float, HeadTempFpQBusOutputs]:
    """HeadTempFp 20-term TRACE (HeadTempFp_topology_spec.md)."""
    t_fp = float(x[X_INDEX["HeadTempFp"]])
    t_sp = float(x[X_INDEX["HeadTempSp"]])
    t_feet_fp = float(x[X_INDEX["FeetTempFp"]])
    t_cabin_fp = float(x[X_INDEX["CabinTempFp"]])
    t_win_fp = float(x[X_INDEX["WinTempFp"]])
    t_roof = float(x[X_INDEX["RoofTemp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    win_shd_t = float(u[U_INDEX["WinShdTEst"]])
    if win_shd_t == 0.0:
        win_shd_t = t_fp
    frnt_def_tma = _upstream_signal(u[U_INDEX["FrntDefTmaEst"]], amb_t)
    # TODO(defrost-glass): FrntDefTmaEst is duct/supply defrost air temperature (TmaDef
    # estimate). Head-zone effective boundary should account for windshield glass
    # exchange — use glass_temp observer / post-glass defrost model (out of CORRECTED
    # low-level LUT scope; see defrost.py + glass_temp.py).
    frnt_fpv_tma = _upstream_signal(u[U_INDEX["FrntFpvTma"]], amb_t)
    frnt_fpf_tma = _upstream_signal(u[U_INDEX["FrntFpfTma"]], amb_t)
    frnt_fp_def_flow = float(u[U_INDEX["FrntFpDefFlow"]])
    frnt_fpv_flow = float(u[U_INDEX["FrntFpvFlow"]])
    frnt_fpf_flow = float(u[U_INDEX["FrntFpfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head = float(u[U_INDEX["HumHeadFpPower"]])

    area_head = params.CHTD_HeadAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_fp = cp_v_ex(t_fp, params.CHTD_HeadAirVAtb_P, params.CHTD_AirCpAtb_P)

    q_console_rad_raw = (
        console_q.console_fp_radiation if console_q is not None else 0.0
    )

    ws_coef = _lut_coef_or_placeholder(lv, params, "CHTD_WSFpRadCo_M", amb_t)
    q_ws_rad_raw = q_zone_radiation(win_shd_t, t_fp, area_head, ws_coef, mode)

    fp_def_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpDefFpConvCo_M", amb_t
    )
    q_def_conv_raw = q_hvac_duct(
        frnt_def_tma, t_fp, frnt_fp_def_flow, fp_def_coef
    )

    fpv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_FpvFpConvCo_M", amb_t)
    q_fpv_conv_raw = q_hvac_duct(
        frnt_fpv_tma, t_fp, frnt_fpv_flow, fpv_coef
    )

    fpf_coef = _lut_coef_or_placeholder(lv, params, "CHTD_FpfFpConvCo_M", amb_t)
    q_fpf_conv_raw = q_hvac_duct(
        frnt_fpf_tma, t_fp, frnt_fpf_flow, fpf_coef
    )

    fp_sp_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpSpRadCo_M", amb_t
    )
    q_fp_sp_rad_raw = q_zone_radiation(
        t_fp, t_sp, area_head, fp_sp_rad_coef, mode
    )
    fp_sp_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpSpConvCo_M", amb_t
    )
    q_fp_sp_conv_raw = q_zone_convection(
        t_fp, t_sp, frnt_fpv_flow, fp_sp_conv_coef
    )

    fp_feet_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpFeetFpRadCo_M", amb_t
    )
    q_feet_rad_raw = q_zone_radiation(
        t_feet_fp, t_fp, area_head, fp_feet_rad_coef, mode
    )
    fp_feet_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpFeetFpConvCo_M", amb_t
    )
    q_feet_conv_raw = q_zone_convection(
        t_feet_fp, t_fp, frnt_fpf_flow, fp_feet_conv_coef
    )

    fp_cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpCabinFpRadCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_fp, t_cabin_fp, area_head, fp_cabin_rad_coef, mode
    )
    fp_cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpCabinFpConvCo_M", amb_t
    )
    q_cabin_conv_raw = q_zone_convection(
        t_fp, t_cabin_fp, frnt_fpv_flow, fp_cabin_conv_coef
    )

    fp_win_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpWinFpRadCo_M", amb_t
    )
    q_win_rad_raw = q_zone_radiation(
        t_fp, t_win_fp, area_head, fp_win_rad_coef, mode
    )
    fp_win_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpWinFpConvCo_M", amb_t
    )
    q_win_conv_raw = q_zone_convection(
        t_fp, t_win_fp, frnt_fpv_flow, fp_win_conv_coef
    )

    fp_roof_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpRoofRadCo_M", amb_t
    )
    q_roof_rad_raw = q_zone_radiation(
        t_fp, t_roof, area_head, fp_roof_rad_coef, mode
    )
    fp_roof_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpRoofConvCo_M", amb_t
    )
    q_roof_conv_raw = q_zone_convection(
        t_fp, t_roof, frnt_fpv_flow, fp_roof_conv_coef
    )

    q_human_raw = hum_head

    # DEFECT D1 (Policy P3): SolarFpHoriz not in Bus_CHTD_u; proxy = SolarFp
    solar_fp = get_solar_fp_horiz(u)
    fp_solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fp, area_head, fp_solar_coef)

    fp_leak_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpLeakageCo_M", amb_t
    )
    q_leak_raw = q_leakage(
        t_fp, amb_t, veh_spd, params.CHTD_WinFpAreaAtb_P, fp_leak_coef
    )

    qbus = HeadTempFpQBusOutputs(
        console_fp_radiation=q_console_rad_raw,
        ws_fp_radiation=q_ws_rad_raw,
        fp_def_fp_flow_convection=q_def_conv_raw,
        fpv_fp_flow_convection=q_fpv_conv_raw,
        fpf_fp_flow_convection=q_fpf_conv_raw,
        fd_fp_radiation=head_fd_q.fd_fp_radiation,
        fd_fp_convection=head_fd_q.fd_fp_convection,
        fp_sp_radiation=q_fp_sp_rad_raw,
        fp_sp_convection=q_fp_sp_conv_raw,
        fp_feet_temp_radiation=q_feet_rad_raw,
        fp_feet_temp_convection=q_feet_conv_raw,
        fp_cabin_fp_radiation=q_cabin_rad_raw,
        fp_cabin_fp_convection=q_cabin_conv_raw,
        fp_win_fp_radiation=q_win_rad_raw,
        fp_win_fp_convection=q_win_conv_raw,
        fp_roof_radiation=q_roof_rad_raw,
        fp_roof_convection=q_roof_conv_raw,
        fp_human_heat_transfer=q_human_raw,
        fp_solar_radiation=q_solar_raw,
        fp_heat_leakage=q_leak_raw,
    )

    Q_net = (
        q_console_rad_raw * _q_gain
        + (q_ws_rad_raw + q_def_conv_raw) * _q_gain
        + (q_fpv_conv_raw + q_fpf_conv_raw) * _q_gain
        + (head_fd_q.fd_fp_radiation + head_fd_q.fd_fp_convection) * _q_gain
        + (q_fp_sp_rad_raw + q_fp_sp_conv_raw) * _q_loss
        + (q_feet_rad_raw + q_feet_conv_raw) * _q_gain
        + (
            q_cabin_rad_raw
            + q_cabin_conv_raw
            + q_win_rad_raw
            + q_win_conv_raw
            + q_roof_rad_raw
            + q_roof_conv_raw
        )
        * _q_loss
        + q_human_raw
        + q_solar_raw * _q_gain
        + q_leak_raw * _q_loss
    )

    delta = Q_net * params.CHTD_Dt_P / C_fp
    return delta, qbus


def compute_feet_temp_fp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_fp_q: HeadTempFpQBusOutputs,
    mode: DefectMode = AS_FOUND,
    console_q: Optional[ConsoleTempQBusOutputs] = None,
) -> tuple[float, FeetTempFpQBusOutputs]:
    """FeetTempFp 9-term TRACE (FeetTempFp_topology_spec.md)."""
    t_feet = float(x[X_INDEX["FeetTempFp"]])
    t_cabin_fp = float(x[X_INDEX["CabinTempFp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    frnt_fpf_tma = _upstream_signal(u[U_INDEX["FrntFpfTma"]], amb_t)
    frnt_fpf_flow = float(u[U_INDEX["FrntFpfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet = float(u[U_INDEX["HumFeetFpPower"]])

    area_feet = params.CHTD_FeetAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_feet = cp_v_ex(t_feet, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)

    if console_q is not None:
        q_console_rad_raw = console_q.console_feet_fp_radiation
        q_console_conv_raw = console_q.console_feet_fp_convection
    else:
        q_console_rad_raw = 0.0
        q_console_conv_raw = 0.0

    fpf_feet_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FpfFeetFpConvCo_M", amb_t
    )
    q_fpf_conv_raw = q_hvac_duct(
        frnt_fpf_tma, t_feet, frnt_fpf_flow, fpf_feet_coef
    )

    q_head_rad_raw = head_fp_q.fp_feet_temp_radiation
    q_head_conv_raw = head_fp_q.fp_feet_temp_convection

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetFpCabinFpRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetFpCabinFpConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_feet, t_cabin_fp, area_feet, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_feet, t_cabin_fp, frnt_fpf_flow, cabin_conv_coef
    )

    q_human_raw = hum_feet

    # DEFECT D3 (AS_FOUND): FeetTempFp uses CHTD_FeetFdLeakageCo_M, not FeetFpLeakageCo_M
    leak_coef = _lut_coef_or_placeholder(
        lv,
        params,
        select_defect_lut(
            mode, "CHTD_FeetFdLeakageCo_M", "CHTD_FeetFpLeakageCo_M"
        ),
        amb_t,
    )
    q_leak_raw = q_leakage(
        t_feet, amb_t, veh_spd, params.CHTD_CabinFpAreaAtb_P, leak_coef
    )

    qbus = FeetTempFpQBusOutputs(
        console_feet_fp_radiation=q_console_rad_raw,
        console_feet_fp_convection=q_console_conv_raw,
        fpf_feet_fp_flow_convection=q_fpf_conv_raw,
        fp_feet_temp_radiation=q_head_rad_raw,
        fp_feet_temp_convection=q_head_conv_raw,
        feet_fp_cabin_fp_radiation=q_cabin_rad_raw,
        feet_fp_cabin_fp_convection=q_cabin_conv_raw,
        feet_fp_human_heat_transfer=q_human_raw,
        feet_fp_heat_leakage=q_leak_raw,
    )

    Q_net = (
        (q_console_rad_raw + q_console_conv_raw) * _q_gain
        + q_fpf_conv_raw * _q_gain
        + (q_head_rad_raw + q_head_conv_raw) * _q_loss
        + (q_cabin_rad_raw + q_cabin_conv_raw) * _q_loss
        + q_human_raw
        + q_leak_raw * _q_loss
    )

    delta = Q_net * params.CHTD_Dt_P / C_feet
    return delta, qbus


@dataclass(frozen=True)
class HeadTempSdQBusOutputs:
    """Raw q-bus exports from HeadTempSd (HeadTempSd_topology_spec §8, 21 fields)."""

    sdv_sd_flow_convection: float
    sdf_sd_flow_convection: float
    rear_sdv_sd_flow_convection: float
    rear_sdf_sd_flow_convection: float
    sd_sp_radiation: float
    sd_sp_convection: float
    fd_sd_radiation: float
    fd_sd_convection: float
    sd_td_radiation: float
    sd_td_convection: float
    sd_feet_temp_radiation: float
    sd_feet_temp_convection: float
    sd_cabin_sd_radiation: float
    sd_cabin_sd_convection: float
    sd_win_sd_radiation: float
    sd_win_sd_convection: float
    sd_roof_radiation: float
    sd_roof_convection: float
    sd_human_heat_transfer: float
    sd_solar_radiation: float
    sd_heat_leakage: float


@dataclass(frozen=True)
class FeetTempSdQBusOutputs:
    """Raw q-bus exports from FeetTempSd (FeetTempSd_topology_spec §8, 8 fields)."""

    sdf_feet_sd_flow_convection: float
    rear_sdf_feet_sd_flow_convection: float
    sd_feet_temp_radiation: float
    sd_feet_temp_convection: float
    feet_sd_cabin_sd_radiation: float
    feet_sd_cabin_sd_convection: float
    feet_sd_human_heat_transfer: float
    feet_sd_heat_leakage: float


def compute_head_temp_sd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_fd_q: HeadTempFdQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, HeadTempSdQBusOutputs]:
    """HeadTempSd 21-term TRACE (HeadTempSd_topology_spec.md)."""
    t_sd = float(x[X_INDEX["HeadTempSd"]])
    t_sp = float(x[X_INDEX["HeadTempSp"]])
    t_td = float(x[X_INDEX["HeadTempTd"]])
    t_feet_sd = float(x[X_INDEX["FeetTempSd"]])
    t_cabin_sd = float(x[X_INDEX["CabinTempSd"]])
    t_win_sd = float(x[X_INDEX["WinTempSd"]])
    t_roof = float(x[X_INDEX["RoofTemp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    frnt_sdv_tma = _upstream_signal(u[U_INDEX["FrntSdvTma"]], amb_t)
    frnt_sdf_tma = _upstream_signal(u[U_INDEX["FrntSdfTma"]], amb_t)
    rear_sdv_tma = _upstream_signal(u[U_INDEX["RearSdvTma"]], amb_t)
    rear_sdf_tma = _upstream_signal(u[U_INDEX["RearSdfTma"]], amb_t)
    frnt_sdv_flow = float(u[U_INDEX["FrntSdvFlow"]])
    frnt_sdf_flow = float(u[U_INDEX["FrntSdfFlow"]])
    rear_sdv_flow = float(u[U_INDEX["RearSdvFlow"]])
    rear_sdf_flow = float(u[U_INDEX["RearSdfFlow"]])
    frnt_spv_flow = float(u[U_INDEX["FrntSpvFlow"]])
    rear_spv_flow = float(u[U_INDEX["RearSpvFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head = float(u[U_INDEX["HumHeadSdPower"]])

    area_head = params.CHTD_HeadAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    flow_sd_sp = frnt_sdv_flow + frnt_spv_flow + rear_sdv_flow + rear_spv_flow
    flow_sd_td = frnt_sdv_flow + rear_sdv_flow
    flow_sd_sdv = frnt_sdv_flow + rear_sdv_flow
    flow_feet_sd = frnt_sdf_flow + rear_sdf_flow

    C_sd = cp_v_ex(t_sd, params.CHTD_HeadAirVAtb_P, params.CHTD_AirCpAtb_P)

    sdv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdvSdConvCo_M", amb_t)
    sdf_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdfSdConvCo_M", amb_t)
    rear_sdv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_RearSdvSdConvCo_M", amb_t)
    rear_sdf_coef = _lut_coef_or_placeholder(lv, params, "CHTD_RearSdfSdConvCo_M", amb_t)

    q_sdv_conv_raw = q_hvac_duct(frnt_sdv_tma, t_sd, frnt_sdv_flow, sdv_coef)
    q_sdf_conv_raw = q_hvac_duct(frnt_sdf_tma, t_sd, frnt_sdf_flow, sdf_coef)
    q_rear_sdv_conv_raw = q_hvac_duct(rear_sdv_tma, t_sd, rear_sdv_flow, rear_sdv_coef)
    q_rear_sdf_conv_raw = q_hvac_duct(rear_sdf_tma, t_sd, rear_sdf_flow, rear_sdf_coef)

    q_fd_sd_rad_raw = head_fd_q.fd_sd_radiation
    q_fd_sd_conv_raw = head_fd_q.fd_sd_convection

    sd_sp_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdSpRadCo_M", amb_t)
    sd_sp_conv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdSpConvCo_M", amb_t)
    q_sd_sp_rad_raw = q_zone_radiation(t_sd, t_sp, area_head, sd_sp_rad_coef, mode)
    q_sd_sp_conv_raw = q_zone_convection(t_sd, t_sp, flow_sd_sp, sd_sp_conv_coef)

    sd_td_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdTdRadCo_M", amb_t)
    sd_td_conv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdTdConvCo_M", amb_t)
    q_sd_td_rad_raw = q_zone_radiation(t_sd, t_td, area_head, sd_td_rad_coef, mode)
    q_sd_td_conv_raw = q_zone_convection(t_sd, t_td, flow_sd_td, sd_td_conv_coef)

    feet_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdFeetSdRadCo_M", amb_t)
    feet_conv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdFeetSdConvCo_M", amb_t)
    q_feet_rad_raw = q_zone_radiation(t_feet_sd, t_sd, area_head, feet_rad_coef, mode)
    q_feet_conv_raw = q_zone_convection(t_feet_sd, t_sd, flow_feet_sd, feet_conv_coef)

    cabin_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdCabinSdRadCo_M", amb_t)
    cabin_conv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdCabinSdConvCo_M", amb_t)
    q_cabin_rad_raw = q_zone_radiation(t_sd, t_cabin_sd, area_head, cabin_rad_coef, mode)
    q_cabin_conv_raw = q_zone_convection(t_sd, t_cabin_sd, flow_sd_sdv, cabin_conv_coef)

    win_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdWinSdRadCo_M", amb_t)
    win_conv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdWinSdConvCo_M", amb_t)
    q_win_rad_raw = q_zone_radiation(t_sd, t_win_sd, area_head, win_rad_coef, mode)
    q_win_conv_raw = q_zone_convection(t_sd, t_win_sd, flow_sd_sdv, win_conv_coef)

    roof_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdRoofRadCo_M", amb_t)
    roof_conv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdRoofConvCo_M", amb_t)
    q_roof_rad_raw = q_zone_radiation(t_sd, t_roof, area_head, roof_rad_coef, mode)
    q_roof_conv_raw = q_zone_convection(t_sd, t_roof, flow_sd_sdv, roof_conv_coef)

    q_human_raw = hum_head

    # DEFECT D1 (Policy P2, AS_FOUND): SolarFdHoriz not in Bus_CHTD_u; 2nd-row driver
    # zone uses driver-side horizontal solar proxy via get_solar_fd_horiz(u).
    solar_fd = get_solar_fd_horiz(u)
    solar_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdSolarRadCo_M", amb_t)
    q_solar_raw = q_solar_gain(solar_fd, area_head, solar_coef)

    leak_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdLeakageCo_M", amb_t)
    q_leak_raw = q_leakage(
        t_sd, amb_t, veh_spd, params.CHTD_WinSdAreaAtb_P, leak_coef
    )

    qbus = HeadTempSdQBusOutputs(
        sdv_sd_flow_convection=q_sdv_conv_raw,
        sdf_sd_flow_convection=q_sdf_conv_raw,
        rear_sdv_sd_flow_convection=q_rear_sdv_conv_raw,
        rear_sdf_sd_flow_convection=q_rear_sdf_conv_raw,
        sd_sp_radiation=q_sd_sp_rad_raw,
        sd_sp_convection=q_sd_sp_conv_raw,
        fd_sd_radiation=q_fd_sd_rad_raw,
        fd_sd_convection=q_fd_sd_conv_raw,
        sd_td_radiation=q_sd_td_rad_raw,
        sd_td_convection=q_sd_td_conv_raw,
        sd_feet_temp_radiation=q_feet_rad_raw,
        sd_feet_temp_convection=q_feet_conv_raw,
        sd_cabin_sd_radiation=q_cabin_rad_raw,
        sd_cabin_sd_convection=q_cabin_conv_raw,
        sd_win_sd_radiation=q_win_rad_raw,
        sd_win_sd_convection=q_win_conv_raw,
        sd_roof_radiation=q_roof_rad_raw,
        sd_roof_convection=q_roof_conv_raw,
        sd_human_heat_transfer=q_human_raw,
        sd_solar_radiation=q_solar_raw,
        sd_heat_leakage=q_leak_raw,
    )

    Q_net = (
        (q_sdv_conv_raw + q_sdf_conv_raw + q_rear_sdv_conv_raw + q_rear_sdf_conv_raw)
        * _q_gain
        + (q_fd_sd_rad_raw + q_fd_sd_conv_raw) * _q_gain
        + (q_sd_sp_rad_raw + q_sd_sp_conv_raw + q_sd_td_rad_raw + q_sd_td_conv_raw)
        * _q_loss
        + (q_feet_rad_raw + q_feet_conv_raw) * _q_gain
        + (
            q_cabin_rad_raw
            + q_cabin_conv_raw
            + q_win_rad_raw
            + q_win_conv_raw
            + q_roof_rad_raw
            + q_roof_conv_raw
        )
        * _q_loss
        + q_human_raw
        + q_solar_raw * _q_gain
        + q_leak_raw * _q_loss
    )

    delta = Q_net * params.CHTD_Dt_P / C_sd
    return delta, qbus


def compute_feet_temp_sd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sd_q: HeadTempSdQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, FeetTempSdQBusOutputs]:
    """FeetTempSd 8-term TRACE (FeetTempSd_topology_spec.md)."""
    t_feet = float(x[X_INDEX["FeetTempSd"]])
    t_cabin_sd = float(x[X_INDEX["CabinTempSd"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    frnt_sdf_tma = _upstream_signal(u[U_INDEX["FrntSdfTma"]], amb_t)
    rear_sdf_tma = _upstream_signal(u[U_INDEX["RearSdfTma"]], amb_t)
    frnt_sdf_flow = float(u[U_INDEX["FrntSdfFlow"]])
    rear_sdf_flow = float(u[U_INDEX["RearSdfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet = float(u[U_INDEX["HumFeetSdPower"]])

    area_feet = params.CHTD_FeetAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P
    flow_sdf = frnt_sdf_flow + rear_sdf_flow

    C_feet = cp_v_ex(t_feet, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)

    sdf_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SdfFeetSdConvCo_M", amb_t)
    rear_sdf_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_RearSdfFeetSdConvCo_M", amb_t
    )
    q_sdf_conv_raw = q_hvac_duct(frnt_sdf_tma, t_feet, frnt_sdf_flow, sdf_coef)
    q_rear_sdf_conv_raw = q_hvac_duct(rear_sdf_tma, t_feet, rear_sdf_flow, rear_sdf_coef)

    q_head_rad_raw = head_sd_q.sd_feet_temp_radiation
    q_head_conv_raw = head_sd_q.sd_feet_temp_convection

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetSdCabinSdRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetSdCabinSdConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_feet, t_cabin_sd, area_feet, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_feet, t_cabin_sd, flow_sdf, cabin_conv_coef
    )

    q_human_raw = hum_feet

    leak_coef = _lut_coef_or_placeholder(lv, params, "CHTD_FeetSdLeakageCo_M", amb_t)
    q_leak_raw = q_leakage(
        t_feet, amb_t, veh_spd, params.CHTD_CabinSdAreaAtb_P, leak_coef
    )

    qbus = FeetTempSdQBusOutputs(
        sdf_feet_sd_flow_convection=q_sdf_conv_raw,
        rear_sdf_feet_sd_flow_convection=q_rear_sdf_conv_raw,
        sd_feet_temp_radiation=q_head_rad_raw,
        sd_feet_temp_convection=q_head_conv_raw,
        feet_sd_cabin_sd_radiation=q_cabin_rad_raw,
        feet_sd_cabin_sd_convection=q_cabin_conv_raw,
        feet_sd_human_heat_transfer=q_human_raw,
        feet_sd_heat_leakage=q_leak_raw,
    )

    Q_net = (
        (q_sdf_conv_raw + q_rear_sdf_conv_raw) * _q_gain
        + (q_head_rad_raw + q_head_conv_raw) * _q_loss
        + (q_cabin_rad_raw + q_cabin_conv_raw) * _q_loss
        + q_human_raw
        + q_leak_raw * _q_loss
    )

    delta = Q_net * params.CHTD_Dt_P / C_feet
    return delta, qbus


def compute_head_temp_sp_qbus(
    head_fp_q: Optional[HeadTempFpQBusOutputs] = None,
    head_sd_q: Optional[HeadTempSdQBusOutputs] = None,
    x: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    params: Optional[CHTDParams] = None,
    lv: Optional[dict] = None,
    mode: DefectMode = AS_FOUND,
) -> HeadTempSpQBusInputs:
    """Return q-bus inputs for HeadTempSp Add4 block."""
    del x, u, params, lv, mode
    if isinstance(head_sd_q, HeadTempSdQBusOutputs):
        sd_sp_rad = head_sd_q.sd_sp_radiation
        sd_sp_conv = head_sd_q.sd_sp_convection
    else:
        sd_sp_rad = 0.0
        sd_sp_conv = 0.0
    if isinstance(head_fp_q, HeadTempFpQBusOutputs):
        fp_sp_rad = head_fp_q.fp_sp_radiation
        fp_sp_conv = head_fp_q.fp_sp_convection
    else:
        fp_sp_rad = 0.0
        fp_sp_conv = 0.0
    return HeadTempSpQBusInputs(
        sd_sp_radiation=sd_sp_rad,
        sd_sp_convection=sd_sp_conv,
        fp_sp_radiation=fp_sp_rad,
        fp_sp_convection=fp_sp_conv,
    )


@dataclass(frozen=True)
class CabinTempFdQBusInputs:
    """Pre-computed q-bus heat into CabinTempFd from neighbor subsystems.

    Maps to Simulink q-bus exports (CHTD_formula_spec.md §5):
        q_HeadTempFd.q_FdCabinFdRadiation
        q_HeadTempFd.q_FdCabinFdConvection
        q_FeetTempFd.q_FeetFdCabinFdRadiation
        q_FeetTempFd.q_FeetFdCabinFdConvection

    Values are raw [W] before CHTD_QgainCo_P scaling in compute_chtd_delta.
    """

    head_fd_radiation: float
    head_fd_convection: float
    feet_fd_radiation: float
    feet_fd_convection: float

    @property
    def total(self) -> float:
        return (
            self.head_fd_radiation + self.head_fd_convection
            + self.feet_fd_radiation + self.feet_fd_convection
        )


def compute_cabin_temp_fd_qbus(
    head_q: HeadTempFdQBusOutputs,
    feet_q: FeetTempFdQBusOutputs,
) -> CabinTempFdQBusInputs:
    """Return q-bus inputs for CabinTempFd from HeadTempFd / FeetTempFd exports."""
    return CabinTempFdQBusInputs(
        head_fd_radiation=head_q.fd_cabin_fd_radiation,
        head_fd_convection=head_q.fd_cabin_fd_convection,
        feet_fd_radiation=feet_q.feet_fd_cabin_fd_radiation,
        feet_fd_convection=feet_q.feet_fd_cabin_fd_convection,
    )


@dataclass(frozen=True)
class CabinTempFpQBusInputs:
    """Pre-computed q-bus heat into CabinTempFp from HeadTempFp / FeetTempFp."""

    head_fp_radiation: float
    head_fp_convection: float
    feet_fp_radiation: float
    feet_fp_convection: float

    @property
    def total(self) -> float:
        return (
            self.head_fp_radiation + self.head_fp_convection
            + self.feet_fp_radiation + self.feet_fp_convection
        )


@dataclass(frozen=True)
class CabinTempFpQBusOutputs:
    """Raw q-bus exports from CabinTempFp (CabinTempFp_topology_spec §8)."""

    fp_cabin_fp_radiation: float
    fp_cabin_fp_convection: float
    feet_fp_cabin_fp_radiation: float
    feet_fp_cabin_fp_convection: float
    cabin_fp_amb_radiation: float
    cabin_fp_amb_convection: float
    cabin_fp_human_heat_transfer: float
    cabin_fp_solar_heat_transfer: float


@dataclass(frozen=True)
class CabinTempSdQBusInputs:
    """Pre-computed q-bus heat into CabinTempSd from HeadTempSd / FeetTempSd."""

    head_sd_radiation: float
    head_sd_convection: float
    feet_sd_radiation: float
    feet_sd_convection: float

    @property
    def total(self) -> float:
        return (
            self.head_sd_radiation
            + self.head_sd_convection
            + self.feet_sd_radiation
            + self.feet_sd_convection
        )


@dataclass(frozen=True)
class CabinTempSdQBusOutputs:
    """Raw q-bus exports from CabinTempSd (CabinTempSd_topology_spec §8)."""

    sd_cabin_sd_radiation: float
    sd_cabin_sd_convection: float
    feet_sd_cabin_sd_radiation: float
    feet_sd_cabin_sd_convection: float
    cabin_sd_amb_radiation: float
    cabin_sd_amb_convection: float
    cabin_sd_human_heat_transfer: float
    cabin_sd_solar_heat_transfer: float


def compute_cabin_temp_fp_qbus(
    head_q: HeadTempFpQBusOutputs,
    feet_q: FeetTempFpQBusOutputs,
) -> CabinTempFpQBusInputs:
    """Return q-bus inputs for CabinTempFp from HeadTempFp / FeetTempFp exports."""
    return CabinTempFpQBusInputs(
        head_fp_radiation=head_q.fp_cabin_fp_radiation,
        head_fp_convection=head_q.fp_cabin_fp_convection,
        feet_fp_radiation=feet_q.feet_fp_cabin_fp_radiation,
        feet_fp_convection=feet_q.feet_fp_cabin_fp_convection,
    )


def compute_cabin_temp_sd_qbus(
    head_q: HeadTempSdQBusOutputs,
    feet_q: FeetTempSdQBusOutputs,
) -> CabinTempSdQBusInputs:
    """Return q-bus inputs for CabinTempSd from HeadTempSd / FeetTempSd exports."""
    return CabinTempSdQBusInputs(
        head_sd_radiation=head_q.sd_cabin_sd_radiation,
        head_sd_convection=head_q.sd_cabin_sd_convection,
        feet_sd_radiation=feet_q.feet_sd_cabin_sd_radiation,
        feet_sd_convection=feet_q.feet_sd_cabin_sd_convection,
    )


def compute_cabin_temp_fp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_fp_q: HeadTempFpQBusOutputs,
    feet_fp_q: FeetTempFpQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, CabinTempFpQBusOutputs]:
    """CabinTempFp 8-term TRACE (CabinTempFp_topology_spec.md)."""
    t_cabin_fp = float(x[X_INDEX["CabinTempFp"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet_fp = float(u[U_INDEX["HumFeetFpPower"]])
    # CabinTempFp uses SolarFp directly — no SolarFpHoriz alias (Head zones only).
    solar_fp = float(u[U_INDEX["SolarFp"]])
    area_cabin_fp = params.CHTD_CabinFpAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_cabin_fp = cp_m_ex(params.CHTD_CabinFpMassAtb_P, params.CHTD_CabinFpCpAtb_P)

    q_bus = compute_cabin_temp_fp_qbus(head_fp_q, feet_fp_q)
    q_from_head_fp = (
        q_interzone_from_bus(q_bus.head_fp_radiation, _q_gain)
        + q_interzone_from_bus(q_bus.head_fp_convection, _q_gain)
    )
    q_from_feet_fp = (
        q_interzone_from_bus(q_bus.feet_fp_radiation, _q_gain)
        + q_interzone_from_bus(q_bus.feet_fp_convection, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinFpAmbRadCo_M", amb_t
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinFpAmbConvCo_M", amb_t
    )
    q_cabin_rad_amb = q_zone_radiation(
        t_cabin_fp, amb_t, area_cabin_fp, amb_rad_coef, mode
    )
    q_cabin_conv_amb = q_zone_convection(
        t_cabin_fp, amb_t, veh_spd * area_cabin_fp, amb_conv_coef
    )
    q_cabin_amb = _q_loss * (q_cabin_rad_amb + q_cabin_conv_amb)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinFpSolarRadCo_M", amb_t
    )
    q_cabin_solar_raw = q_solar_gain(solar_fp, area_cabin_fp, solar_coef)
    q_cabin_solar = q_cabin_solar_raw * _q_gain
    q_cabin_human = hum_feet_fp

    Q_net_cabin_fp = (
        q_from_head_fp
        + q_from_feet_fp
        + q_cabin_amb
        + q_cabin_human
        + q_cabin_solar
    )

    qbus = CabinTempFpQBusOutputs(
        fp_cabin_fp_radiation=q_bus.head_fp_radiation,
        fp_cabin_fp_convection=q_bus.head_fp_convection,
        feet_fp_cabin_fp_radiation=q_bus.feet_fp_radiation,
        feet_fp_cabin_fp_convection=q_bus.feet_fp_convection,
        cabin_fp_amb_radiation=q_cabin_rad_amb,
        cabin_fp_amb_convection=q_cabin_conv_amb,
        cabin_fp_human_heat_transfer=q_cabin_human,
        cabin_fp_solar_heat_transfer=q_cabin_solar_raw,
    )

    delta = Q_net_cabin_fp * params.CHTD_Dt_P / C_cabin_fp
    return delta, qbus


def compute_cabin_temp_sd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sd_q: HeadTempSdQBusOutputs,
    feet_sd_q: FeetTempSdQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, CabinTempSdQBusOutputs]:
    """CabinTempSd 8-term TRACE (CabinTempSd_topology_spec.md)."""
    t_cabin_sd = float(x[X_INDEX["CabinTempSd"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet_sd = float(u[U_INDEX["HumFeetSdPower"]])
    # CabinTempSd uses SolarFd directly; SolarFdHoriz defect is head-zone only.
    solar_fd = float(u[U_INDEX["SolarFd"]])
    area_cabin_sd = params.CHTD_CabinSdAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_cabin_sd = cp_m_ex(params.CHTD_CabinSdMassAtb_P, params.CHTD_CabinSdCpAtb_P)

    q_bus = compute_cabin_temp_sd_qbus(head_sd_q, feet_sd_q)
    q_from_head_sd = (
        q_interzone_from_bus(q_bus.head_sd_radiation, _q_gain)
        + q_interzone_from_bus(q_bus.head_sd_convection, _q_gain)
    )
    q_from_feet_sd = (
        q_interzone_from_bus(q_bus.feet_sd_radiation, _q_gain)
        + q_interzone_from_bus(q_bus.feet_sd_convection, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinSdAmbRadCo_M", amb_t
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinSdAmbConvCo_M", amb_t
    )
    q_cabin_rad_amb = q_zone_radiation(
        t_cabin_sd, amb_t, area_cabin_sd, amb_rad_coef, mode
    )
    q_cabin_conv_amb = q_zone_convection(
        t_cabin_sd, amb_t, veh_spd * area_cabin_sd, amb_conv_coef
    )
    q_cabin_amb = _q_loss * (q_cabin_rad_amb + q_cabin_conv_amb)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinSdSolarRadCo_M", amb_t
    )
    q_cabin_solar_raw = q_solar_gain(solar_fd, area_cabin_sd, solar_coef)
    q_cabin_solar = q_cabin_solar_raw * _q_gain
    q_cabin_human = hum_feet_sd

    Q_net_cabin_sd = (
        q_from_head_sd
        + q_from_feet_sd
        + q_cabin_amb
        + q_cabin_human
        + q_cabin_solar
    )

    qbus = CabinTempSdQBusOutputs(
        sd_cabin_sd_radiation=q_bus.head_sd_radiation,
        sd_cabin_sd_convection=q_bus.head_sd_convection,
        feet_sd_cabin_sd_radiation=q_bus.feet_sd_radiation,
        feet_sd_cabin_sd_convection=q_bus.feet_sd_convection,
        cabin_sd_amb_radiation=q_cabin_rad_amb,
        cabin_sd_amb_convection=q_cabin_conv_amb,
        cabin_sd_human_heat_transfer=q_cabin_human,
        cabin_sd_solar_heat_transfer=q_cabin_solar_raw,
    )

    delta = Q_net_cabin_sd * params.CHTD_Dt_P / C_cabin_sd
    return delta, qbus


@dataclass(frozen=True)
class CabinTempSpQBusInputs:
    """Pre-computed q-bus heat into CabinTempSp from HeadTempSp / FeetTempSp."""

    head_sp_radiation: float
    head_sp_convection: float
    feet_sp_radiation: float
    feet_sp_convection: float

    @property
    def total(self) -> float:
        return (
            self.head_sp_radiation
            + self.head_sp_convection
            + self.feet_sp_radiation
            + self.feet_sp_convection
        )


@dataclass(frozen=True)
class CabinTempSpQBusOutputs:
    """Raw q-bus exports from CabinTempSp (CabinTempSp_topology_spec §7)."""

    sp_cabin_sp_radiation: float
    sp_cabin_sp_convection: float
    feet_sp_cabin_sp_radiation: float
    feet_sp_cabin_sp_convection: float
    cabin_sp_amb_radiation: float
    cabin_sp_amb_convection: float
    cabin_sp_human_heat_transfer: float
    cabin_sp_solar_heat_transfer: float


def compute_cabin_temp_sp_qbus(
    head_q: HeadTempSpQBusOutputs,
    feet_q: FeetTempSpQBusOutputs,
) -> CabinTempSpQBusInputs:
    """Return q-bus inputs for CabinTempSp from HeadTempSp / FeetTempSp exports."""
    return CabinTempSpQBusInputs(
        head_sp_radiation=head_q.sp_cabin_sp_radiation,
        head_sp_convection=head_q.sp_cabin_sp_convection,
        feet_sp_radiation=feet_q.feet_sp_cabin_sp_radiation,
        feet_sp_convection=feet_q.feet_sp_cabin_sp_convection,
    )


def compute_cabin_temp_sp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sp_q: HeadTempSpQBusOutputs,
    feet_sp_q: FeetTempSpQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, CabinTempSpQBusOutputs]:
    """CabinTempSp 8-term TRACE (CabinTempSp_topology_spec.md)."""
    t_cabin_sp = float(x[X_INDEX["CabinTempSp"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet_sp = float(u[U_INDEX["HumFeetSpPower"]])
    # CabinTempSp uses SolarFp directly; no SolarFpHoriz defect outside Head zones.
    solar_fp = float(u[U_INDEX["SolarFp"]])
    area_cabin_sp = params.CHTD_CabinSpAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_cabin_sp = cp_m_ex(params.CHTD_CabinSpMassAtb_P, params.CHTD_CabinSpCpAtb_P)

    q_bus = compute_cabin_temp_sp_qbus(head_sp_q, feet_sp_q)
    q_from_head_sp = (
        q_interzone_from_bus(q_bus.head_sp_radiation, _q_gain)
        + q_interzone_from_bus(q_bus.head_sp_convection, _q_gain)
    )
    q_from_feet_sp = (
        q_interzone_from_bus(q_bus.feet_sp_radiation, _q_gain)
        + q_interzone_from_bus(q_bus.feet_sp_convection, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinSpAmbRadCo_M", amb_t
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv,
        params,
        select_defect_lut(
            mode, "CHTD_CabinFpAmbConvCo_M", "CHTD_CabinSpAmbConvCo_M"
        ),
        amb_t,
    )
    q_cabin_rad_amb = q_zone_radiation(
        t_cabin_sp, amb_t, area_cabin_sp, amb_rad_coef, mode
    )
    q_cabin_conv_amb = q_zone_convection(
        t_cabin_sp, amb_t, veh_spd * area_cabin_sp, amb_conv_coef
    )
    q_cabin_amb = _q_loss * (q_cabin_rad_amb + q_cabin_conv_amb)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinSpSolarRadCo_M", amb_t
    )
    q_cabin_solar_raw = q_solar_gain(solar_fp, area_cabin_sp, solar_coef)
    q_cabin_solar = q_cabin_solar_raw * _q_gain
    q_cabin_human = hum_feet_sp

    Q_net_cabin_sp = (
        q_from_head_sp
        + q_from_feet_sp
        + q_cabin_amb
        + q_cabin_human
        + q_cabin_solar
    )

    qbus = CabinTempSpQBusOutputs(
        sp_cabin_sp_radiation=q_bus.head_sp_radiation,
        sp_cabin_sp_convection=q_bus.head_sp_convection,
        feet_sp_cabin_sp_radiation=q_bus.feet_sp_radiation,
        feet_sp_cabin_sp_convection=q_bus.feet_sp_convection,
        cabin_sp_amb_radiation=q_cabin_rad_amb,
        cabin_sp_amb_convection=q_cabin_conv_amb,
        cabin_sp_human_heat_transfer=q_cabin_human,
        cabin_sp_solar_heat_transfer=q_cabin_solar_raw,
    )

    delta = Q_net_cabin_sp * params.CHTD_Dt_P / C_cabin_sp
    return delta, qbus


@dataclass(frozen=True)
class WinTempFdQBusOutputs:
    """Raw q-bus exports from WinTempFd (WinTempFd_topology_spec §8)."""

    fd_win_fd_radiation: float
    fd_win_fd_convection: float
    win_fd_amb_radiation: float
    win_fd_amb_convection: float
    win_fd_human_heat_transfer: float
    win_fd_solar_radiation: float


@dataclass(frozen=True)
class WinTempFpQBusOutputs:
    """Raw q-bus exports from WinTempFp (WinTempFp_topology_spec §8)."""

    fp_win_fp_radiation: float
    fp_win_fp_convection: float
    win_fp_amb_radiation: float
    win_fp_amb_convection: float
    win_fp_human_heat_transfer: float
    win_fp_solar_radiation: float


@dataclass(frozen=True)
class WinTempSdQBusOutputs:
    """Raw q-bus exports from WinTempSd (WinTempSd_topology_spec §8)."""

    sd_win_sd_radiation: float
    sd_win_sd_convection: float
    win_sd_amb_radiation: float
    win_sd_amb_convection: float
    win_sd_human_heat_transfer: float
    win_sd_solar_radiation: float


@dataclass(frozen=True)
class WinTempSpQBusOutputs:
    """Raw q-bus exports from WinTempSp (WinTempSp_topology_spec §7)."""

    sp_win_sp_radiation: float
    sp_win_sp_convection: float
    win_sp_amb_radiation: float
    win_sp_amb_convection: float
    win_sp_human_heat_transfer: float
    win_sp_solar_radiation: float


@dataclass(frozen=True)
class FeetTempSpQBusOutputs:
    """Raw q-bus exports from FeetTempSp (FeetTempSp_topology_spec §7)."""

    spf_feet_sp_flow_convection: float
    rear_spf_feet_sp_flow_convection: float
    sp_feet_temp_radiation: float
    sp_feet_temp_convection: float
    feet_sp_cabin_sp_radiation: float
    feet_sp_cabin_sp_convection: float
    feet_sp_human_heat_transfer: float
    feet_sp_heat_leakage: float


def compute_win_temp_fd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_fd_q: HeadTempFdQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, WinTempFdQBusOutputs]:
    """WinTempFd 6-term TRACE (WinTempFd_topology_spec.md)."""
    t_win_fd = float(x[X_INDEX["WinTempFd"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head = float(u[U_INDEX["HumHeadFdPower"]])
    # WinTempFd uses SolarFd directly — no SolarFdHoriz alias (Head zones only).
    solar_fd = float(u[U_INDEX["SolarFd"]])
    area_win_fd = params.CHTD_WinFdAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_win_fd = cp_m_ex(params.CHTD_WinFdMassAtb_P, params.CHTD_WinFdCpAtb_P)

    q_from_head_fd = (
        q_interzone_from_bus(head_fd_q.fd_win_fd_radiation, _q_gain)
        + q_interzone_from_bus(head_fd_q.fd_win_fd_convection, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinFdAmbRadCo_M", amb_t
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinFdAmbConvCo_M", amb_t
    )
    q_amb_rad_raw = q_zone_radiation(
        t_win_fd, amb_t, area_win_fd, amb_rad_coef, mode
    )
    q_amb_conv_raw = q_zone_convection(
        t_win_fd, amb_t, veh_spd * area_win_fd, amb_conv_coef
    )
    q_amb = _q_loss * (q_amb_rad_raw + q_amb_conv_raw)

    # DEFECT D4 (AS_FOUND): WinTempFd uses CHTD_CabinFdSolarRadCo_M, not WinFdSolarRadCo_M
    solar_coef = _lut_coef_or_placeholder(
        lv,
        params,
        select_defect_lut(
            mode, "CHTD_CabinFdSolarRadCo_M", "CHTD_WinFdSolarRadCo_M"
        ),
        amb_t,
    )
    q_solar_raw = q_solar_gain(solar_fd, area_win_fd, solar_coef)
    q_solar = q_solar_raw * _q_gain

    Q_net_win_fd = q_from_head_fd + q_amb + hum_head + q_solar

    qbus = WinTempFdQBusOutputs(
        fd_win_fd_radiation=head_fd_q.fd_win_fd_radiation,
        fd_win_fd_convection=head_fd_q.fd_win_fd_convection,
        win_fd_amb_radiation=q_amb_rad_raw,
        win_fd_amb_convection=q_amb_conv_raw,
        win_fd_human_heat_transfer=hum_head,
        win_fd_solar_radiation=q_solar_raw,
    )

    delta = Q_net_win_fd * params.CHTD_Dt_P / C_win_fd
    return delta, qbus


def compute_win_temp_sd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sd_q: HeadTempSdQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, WinTempSdQBusOutputs]:
    """WinTempSd 6-term TRACE (WinTempSd_topology_spec.md)."""
    t_win_sd = float(x[X_INDEX["WinTempSd"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head = float(u[U_INDEX["HumHeadSdPower"]])
    # WinTempSd uses SolarFd directly; SolarFdHoriz defect is head-zone only.
    solar_fd = float(u[U_INDEX["SolarFd"]])
    area_win_sd = params.CHTD_WinSdAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_win_sd = cp_m_ex(params.CHTD_WinSdMassAtb_P, params.CHTD_WinSdCpAtb_P)

    q_from_head_sd = (
        q_interzone_from_bus(head_sd_q.sd_win_sd_radiation, _q_gain)
        + q_interzone_from_bus(head_sd_q.sd_win_sd_convection, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinSdAmbRadCo_M", amb_t
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinSdAmbConvCo_M", amb_t
    )
    q_amb_rad_raw = q_zone_radiation(
        t_win_sd, amb_t, area_win_sd, amb_rad_coef, mode
    )
    q_amb_conv_raw = q_zone_convection(
        t_win_sd, amb_t, veh_spd * area_win_sd, amb_conv_coef
    )
    q_amb = _q_loss * (q_amb_rad_raw + q_amb_conv_raw)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinSdSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fd, area_win_sd, solar_coef)
    q_solar = q_solar_raw * _q_gain

    Q_net_win_sd = q_from_head_sd + q_amb + hum_head + q_solar

    qbus = WinTempSdQBusOutputs(
        sd_win_sd_radiation=head_sd_q.sd_win_sd_radiation,
        sd_win_sd_convection=head_sd_q.sd_win_sd_convection,
        win_sd_amb_radiation=q_amb_rad_raw,
        win_sd_amb_convection=q_amb_conv_raw,
        win_sd_human_heat_transfer=hum_head,
        win_sd_solar_radiation=q_solar_raw,
    )

    delta = Q_net_win_sd * params.CHTD_Dt_P / C_win_sd
    return delta, qbus


def compute_win_temp_sp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sp_q: HeadTempSpQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, WinTempSpQBusOutputs]:
    """WinTempSp 6-term TRACE (WinTempSp_topology_spec.md)."""
    t_win_sp = float(x[X_INDEX["WinTempSp"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head = float(u[U_INDEX["HumHeadSpPower"]])
    # WinTempSp uses SolarFp directly; no SolarFpHoriz defect outside Head zones.
    solar_fp = float(u[U_INDEX["SolarFp"]])
    area_win_sp = params.CHTD_WinSpAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_win_sp = cp_m_ex(params.CHTD_WinSpMassAtb_P, params.CHTD_WinSpCpAtb_P)

    q_from_head_sp = (
        q_interzone_from_bus(head_sp_q.sp_win_sp_radiation, _q_gain)
        + q_interzone_from_bus(head_sp_q.sp_win_sp_convection, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_WinSpAmbRadCo_M", amb_t)
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinSpAmbConvCo_M", amb_t
    )
    q_amb_rad_raw = q_zone_radiation(
        t_win_sp, amb_t, area_win_sp, amb_rad_coef, mode
    )
    q_amb_conv_raw = q_zone_convection(
        t_win_sp, amb_t, veh_spd * area_win_sp, amb_conv_coef
    )
    q_amb = _q_loss * (q_amb_rad_raw + q_amb_conv_raw)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinSpSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fp, area_win_sp, solar_coef)
    q_solar = q_solar_raw * _q_gain

    Q_net_win_sp = q_from_head_sp + q_amb + hum_head + q_solar

    qbus = WinTempSpQBusOutputs(
        sp_win_sp_radiation=head_sp_q.sp_win_sp_radiation,
        sp_win_sp_convection=head_sp_q.sp_win_sp_convection,
        win_sp_amb_radiation=q_amb_rad_raw,
        win_sp_amb_convection=q_amb_conv_raw,
        win_sp_human_heat_transfer=hum_head,
        win_sp_solar_radiation=q_solar_raw,
    )

    delta = Q_net_win_sp * params.CHTD_Dt_P / C_win_sp
    return delta, qbus


@dataclass(frozen=True)
class HeadTempTdQBusOutputs:
    """Raw q-bus exports from HeadTempTd (HeadTempTd_topology_spec §7)."""

    td_tp_radiation: float
    td_tp_convection: float
    td_feet_temp_radiation: float
    td_feet_temp_convection: float
    td_cabin_td_radiation: float
    td_cabin_td_convection: float
    td_win_td_radiation: float
    td_win_td_convection: float
    td_roof_radiation: float
    td_roof_convection: float


def compute_head_temp_td(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sd_q: HeadTempSdQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, HeadTempTdQBusOutputs]:
    """HeadTempTd 17-term TRACE (HeadTempTd_topology_spec.md)."""
    t_td = float(x[X_INDEX["HeadTempTd"]])
    t_tp = float(x[X_INDEX["HeadTempTp"]])
    t_feet_td = float(x[X_INDEX["FeetTempTd"]])
    t_cabin_td = float(x[X_INDEX["CabinTempTd"]])
    t_win_td = float(x[X_INDEX["WinTempTd"]])
    t_roof = float(x[X_INDEX["RoofTemp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    rear_tdv_tma = _upstream_signal(u[U_INDEX["RearTdvTma"]], amb_t)
    rear_tdf_tma = _upstream_signal(u[U_INDEX["RearTdfTma"]], amb_t)
    rear_tdv_flow = float(u[U_INDEX["RearTdvFlow"]])
    rear_tdf_flow = float(u[U_INDEX["RearTdfFlow"]])
    rear_tpv_flow = float(u[U_INDEX["RearTpvFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head_td = float(u[U_INDEX["HumHeadTdPower"]])

    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P
    area_head = params.CHTD_HeadAreaAtb_P
    flow_td_tp = rear_tdv_flow + rear_tpv_flow

    C_td = cp_v_ex(t_td, params.CHTD_HeadAirVAtb_P, params.CHTD_AirCpAtb_P)

    tdv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TdvTdConvCo_M", amb_t)
    tdf_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TdfTdConvCo_M", amb_t)
    q_tdv_conv_raw = q_hvac_duct(rear_tdv_tma, t_td, rear_tdv_flow, tdv_coef)
    q_tdf_conv_raw = q_hvac_duct(rear_tdf_tma, t_td, rear_tdf_flow, tdf_coef)

    q_sd_td_rad_raw = head_sd_q.sd_td_radiation
    q_sd_td_conv_raw = head_sd_q.sd_td_convection

    tp_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TdTpRadCo_M", amb_t)
    tp_conv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TdTpConvCo_M", amb_t)
    q_tp_rad_raw = q_zone_radiation(t_td, t_tp, area_head, tp_rad_coef, mode)
    q_tp_conv_raw = q_zone_convection(t_td, t_tp, flow_td_tp, tp_conv_coef)

    feet_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdFeetTdRadCo_M", amb_t
    )
    feet_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdFeetTdConvCo_M", amb_t
    )
    q_feet_rad_raw = q_zone_radiation(
        t_feet_td, t_td, area_head, feet_rad_coef, mode
    )
    q_feet_conv_raw = q_zone_convection(
        t_feet_td, t_td, rear_tdf_flow, feet_conv_coef
    )

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdCabinTdRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdCabinTdConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_td, t_cabin_td, area_head, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_td, t_cabin_td, rear_tdv_flow, cabin_conv_coef
    )

    win_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdWinTdRadCo_M", amb_t
    )
    win_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdWinTdConvCo_M", amb_t
    )
    q_win_rad_raw = q_zone_radiation(t_td, t_win_td, area_head, win_rad_coef, mode)
    q_win_conv_raw = q_zone_convection(
        t_td, t_win_td, rear_tdv_flow, win_conv_coef
    )

    roof_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdRoofRadCo_M", amb_t
    )
    # DEFECT D1: Simulink HeadTempTd roof conv uses CHTD_SdRoofConvCo_M table.
    roof_conv_coef = _lut_coef_or_placeholder(
        lv,
        params,
        select_defect_lut(
            mode, "CHTD_SdRoofConvCo_M", "CHTD_TdRoofConvCo_M"
        ),
        amb_t,
    )
    q_roof_rad_raw = q_zone_radiation(
        t_td, t_roof, area_head, roof_rad_coef, mode
    )
    q_roof_conv_raw = q_zone_convection(
        t_td, t_roof, rear_tdv_flow, roof_conv_coef
    )

    q_human_raw = hum_head_td

    solar_fd = get_solar_fd_horiz(u)
    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fd, area_head, solar_coef)

    leak_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TdLeakageCo_M", amb_t)
    q_leak_raw = q_leakage(
        t_td, amb_t, veh_spd, params.CHTD_WinTdAreaAtb_P, leak_coef
    )

    qbus = HeadTempTdQBusOutputs(
        td_tp_radiation=q_tp_rad_raw,
        td_tp_convection=q_tp_conv_raw,
        td_feet_temp_radiation=q_feet_rad_raw,
        td_feet_temp_convection=q_feet_conv_raw,
        td_cabin_td_radiation=q_cabin_rad_raw,
        td_cabin_td_convection=q_cabin_conv_raw,
        td_win_td_radiation=q_win_rad_raw,
        td_win_td_convection=q_win_conv_raw,
        td_roof_radiation=q_roof_rad_raw,
        td_roof_convection=q_roof_conv_raw,
    )

    Q_net = (
        (q_tdv_conv_raw + q_tdf_conv_raw) * _q_gain
        + (q_sd_td_rad_raw + q_sd_td_conv_raw) * _q_gain
        + (q_tp_rad_raw + q_tp_conv_raw) * _q_loss
        + (q_feet_rad_raw + q_feet_conv_raw) * _q_gain
        + (
            q_cabin_rad_raw
            + q_cabin_conv_raw
            + q_win_rad_raw
            + q_win_conv_raw
            + q_roof_rad_raw
            + q_roof_conv_raw
        )
        * _q_loss
        + q_human_raw
        + q_solar_raw * _q_gain
        + q_leak_raw * _q_loss
    )
    delta = Q_net * params.CHTD_Dt_P / C_td
    return delta, qbus


@dataclass(frozen=True)
class FeetTempTdQBusOutputs:
    """Raw q-bus exports from FeetTempTd (FeetTempTd_topology_spec §7)."""

    feet_td_cabin_td_radiation: float
    feet_td_cabin_td_convection: float


def compute_feet_temp_td(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_td_q: HeadTempTdQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, FeetTempTdQBusOutputs]:
    """FeetTempTd 7-term TRACE (FeetTempTd_topology_spec.md)."""
    t_feet_td = float(x[X_INDEX["FeetTempTd"]])
    t_cabin_td = float(x[X_INDEX["CabinTempTd"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    rear_tdf_tma = _upstream_signal(u[U_INDEX["RearTdfTma"]], amb_t)
    rear_tdf_flow = float(u[U_INDEX["RearTdfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet_td = float(u[U_INDEX["HumFeetTdPower"]])

    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_feet_td = cp_v_ex(t_feet_td, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)

    tdf_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TdfFeetTdConvCo_M", amb_t
    )
    q_tdf_conv_raw = q_hvac_duct(rear_tdf_tma, t_feet_td, rear_tdf_flow, tdf_coef)

    q_head_td_rad_raw = head_td_q.td_feet_temp_radiation
    q_head_td_conv_raw = head_td_q.td_feet_temp_convection

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetTdCabinTdRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetTdCabinTdConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_feet_td, t_cabin_td, params.CHTD_FeetAreaAtb_P, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_feet_td, t_cabin_td, rear_tdf_flow, cabin_conv_coef
    )

    q_human_raw = hum_feet_td

    leak_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetTdLeakageCo_M", amb_t
    )
    q_leak_raw = q_leakage(
        t_feet_td, amb_t, veh_spd, params.CHTD_CabinTdAreaAtb_P, leak_coef
    )

    qbus = FeetTempTdQBusOutputs(
        feet_td_cabin_td_radiation=q_cabin_rad_raw,
        feet_td_cabin_td_convection=q_cabin_conv_raw,
    )

    Q_net = (
        q_tdf_conv_raw * _q_gain
        + (q_head_td_rad_raw + q_head_td_conv_raw) * _q_loss
        + (q_cabin_rad_raw + q_cabin_conv_raw) * _q_loss
        + q_human_raw
        + q_leak_raw * _q_loss
    )
    delta = Q_net * params.CHTD_Dt_P / C_feet_td
    return delta, qbus


@dataclass(frozen=True)
class CabinTempTdQBusOutputs:
    """Raw q-bus exports from CabinTempTd (CabinTempTd_topology_spec §7)."""

    td_cabin_td_radiation: float
    td_cabin_td_convection: float
    feet_td_cabin_td_radiation: float
    feet_td_cabin_td_convection: float
    cabin_td_amb_radiation: float
    cabin_td_amb_convection: float
    cabin_td_human_heat_transfer: float
    cabin_td_solar_heat_transfer: float


def compute_cabin_temp_td(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_td_q: HeadTempTdQBusOutputs | None = None,
    feet_td_q: FeetTempTdQBusOutputs | None = None,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, CabinTempTdQBusOutputs]:
    """CabinTempTd 8-term TRACE (CabinTempTd_topology_spec.md)."""
    t_cabin_td = float(x[X_INDEX["CabinTempTd"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet_td = float(u[U_INDEX["HumFeetTdPower"]])
    solar_fd = float(u[U_INDEX["SolarFd"]])
    area_cabin_td = params.CHTD_CabinTdAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_cabin_td = cp_m_ex(params.CHTD_CabinTdMassAtb_P, params.CHTD_CabinTdCpAtb_P)

    q_head_td_rad = head_td_q.td_cabin_td_radiation if head_td_q else 0.0
    q_head_td_conv = head_td_q.td_cabin_td_convection if head_td_q else 0.0
    q_feet_td_rad = (
        feet_td_q.feet_td_cabin_td_radiation if feet_td_q else 0.0
    )
    q_feet_td_conv = (
        feet_td_q.feet_td_cabin_td_convection if feet_td_q else 0.0
    )
    q_from_head_td = (
        q_interzone_from_bus(q_head_td_rad, _q_gain)
        + q_interzone_from_bus(q_head_td_conv, _q_gain)
    )
    q_from_feet_td = (
        q_interzone_from_bus(q_feet_td_rad, _q_gain)
        + q_interzone_from_bus(q_feet_td_conv, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinTdAmbRadCo_M", amb_t
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinTdAmbConvCo_M", amb_t
    )
    q_cabin_rad_amb = q_zone_radiation(
        t_cabin_td, amb_t, area_cabin_td, amb_rad_coef, mode
    )
    q_cabin_conv_amb = q_zone_convection(
        t_cabin_td, amb_t, veh_spd * area_cabin_td, amb_conv_coef
    )
    q_cabin_amb = _q_loss * (q_cabin_rad_amb + q_cabin_conv_amb)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinTdSolarRadCo_M", amb_t
    )
    q_cabin_solar_raw = q_solar_gain(solar_fd, area_cabin_td, solar_coef)
    q_cabin_solar = q_cabin_solar_raw * _q_gain
    q_cabin_human = hum_feet_td

    Q_net_cabin_td = (
        q_from_head_td
        + q_from_feet_td
        + q_cabin_amb
        + q_cabin_human
        + q_cabin_solar
    )

    qbus = CabinTempTdQBusOutputs(
        td_cabin_td_radiation=q_head_td_rad,
        td_cabin_td_convection=q_head_td_conv,
        feet_td_cabin_td_radiation=q_feet_td_rad,
        feet_td_cabin_td_convection=q_feet_td_conv,
        cabin_td_amb_radiation=q_cabin_rad_amb,
        cabin_td_amb_convection=q_cabin_conv_amb,
        cabin_td_human_heat_transfer=q_cabin_human,
        cabin_td_solar_heat_transfer=q_cabin_solar_raw,
    )

    delta = Q_net_cabin_td * params.CHTD_Dt_P / C_cabin_td
    return delta, qbus


@dataclass(frozen=True)
class WinTempTdQBusOutputs:
    """Raw q-bus exports from WinTempTd (WinTempTd_topology_spec §7)."""

    td_win_td_radiation: float
    td_win_td_convection: float
    win_td_amb_radiation: float
    win_td_amb_convection: float
    win_td_human_heat_transfer: float
    win_td_solar_heat_transfer: float


def compute_win_temp_td(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_td_q: HeadTempTdQBusOutputs | None = None,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, WinTempTdQBusOutputs]:
    """WinTempTd 6-term TRACE (WinTempTd_topology_spec.md)."""
    t_win_td = float(x[X_INDEX["WinTempTd"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head_td = float(u[U_INDEX["HumHeadTdPower"]])
    solar_fd = float(u[U_INDEX["SolarFd"]])
    area_win_td = params.CHTD_WinTdAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_win_td = cp_m_ex(params.CHTD_WinTdMassAtb_P, params.CHTD_WinTdCpAtb_P)

    q_head_td_rad = head_td_q.td_win_td_radiation if head_td_q else 0.0
    q_head_td_conv = head_td_q.td_win_td_convection if head_td_q else 0.0
    q_from_head_td = (
        q_interzone_from_bus(q_head_td_rad, _q_gain)
        + q_interzone_from_bus(q_head_td_conv, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_WinTdAmbRadCo_M", amb_t)
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinTdAmbConvCo_M", amb_t
    )
    q_amb_rad_raw = q_zone_radiation(
        t_win_td, amb_t, area_win_td, amb_rad_coef, mode
    )
    q_amb_conv_raw = q_zone_convection(
        t_win_td, amb_t, veh_spd * area_win_td, amb_conv_coef
    )
    q_amb = _q_loss * (q_amb_rad_raw + q_amb_conv_raw)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinTdSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fd, area_win_td, solar_coef)
    q_solar = q_solar_raw * _q_gain

    Q_net_win_td = q_from_head_td + q_amb + hum_head_td + q_solar

    qbus = WinTempTdQBusOutputs(
        td_win_td_radiation=q_head_td_rad,
        td_win_td_convection=q_head_td_conv,
        win_td_amb_radiation=q_amb_rad_raw,
        win_td_amb_convection=q_amb_conv_raw,
        win_td_human_heat_transfer=hum_head_td,
        win_td_solar_heat_transfer=q_solar_raw,
    )

    delta = Q_net_win_td * params.CHTD_Dt_P / C_win_td
    return delta, qbus


@dataclass(frozen=True)
class HeadTempTpQBusOutputs:
    """Raw q-bus exports from HeadTempTp (HeadTempTp_topology_spec §8)."""

    tp_feet_temp_radiation: float
    tp_feet_temp_convection: float
    tp_cabin_tp_radiation: float
    tp_cabin_tp_convection: float
    tp_win_tp_radiation: float
    tp_win_tp_convection: float
    tp_roof_radiation: float
    tp_roof_convection: float


def compute_head_temp_tp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sp_q: HeadTempSpQBusOutputs,
    mode: DefectMode = AS_FOUND,
    head_td_q: HeadTempTdQBusOutputs | None = None,
) -> tuple[float, HeadTempTpQBusOutputs]:
    """HeadTempTp 17-term TRACE (HeadTempTp_topology_spec.md)."""
    t_tp = float(x[X_INDEX["HeadTempTp"]])
    t_feet_tp = float(x[X_INDEX["FeetTempTp"]])
    t_cabin_tp = float(x[X_INDEX["CabinTempTp"]])
    t_win_tp = float(x[X_INDEX["WinTempTp"]])
    t_roof = float(x[X_INDEX["RoofTemp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    rear_tpv_tma = _upstream_signal(u[U_INDEX["RearTpvTma"]], amb_t)
    rear_tpf_tma = _upstream_signal(u[U_INDEX["RearTpfTma"]], amb_t)
    rear_tpv_flow = float(u[U_INDEX["RearTpvFlow"]])
    rear_tpf_flow = float(u[U_INDEX["RearTpfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head_tp = float(u[U_INDEX["HumHeadTpPower"]])

    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P
    area_head = params.CHTD_HeadAreaAtb_P

    C_tp = cp_v_ex(t_tp, params.CHTD_HeadAirVAtb_P, params.CHTD_AirCpAtb_P)

    tpv_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TpvTpConvCo_M", amb_t)
    tpf_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TpfTpConvCo_M", amb_t)
    q_tpv_conv_raw = q_hvac_duct(rear_tpv_tma, t_tp, rear_tpv_flow, tpv_coef)
    q_tpf_conv_raw = q_hvac_duct(rear_tpf_tma, t_tp, rear_tpf_flow, tpf_coef)

    q_td_tp_rad_raw = head_td_q.td_tp_radiation if head_td_q else 0.0
    q_td_tp_conv_raw = head_td_q.td_tp_convection if head_td_q else 0.0

    q_sp_tp_rad_raw = head_sp_q.sp_tp_radiation
    q_sp_tp_conv_raw = head_sp_q.sp_tp_convection

    feet_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpFeetTpRadCo_M", amb_t
    )
    feet_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpFeetTpConvCo_M", amb_t
    )
    q_feet_rad_raw = q_zone_radiation(
        t_feet_tp, t_tp, area_head, feet_rad_coef, mode
    )
    q_feet_conv_raw = q_zone_convection(
        t_feet_tp, t_tp, rear_tpf_flow, feet_conv_coef
    )

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpCabinTpRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpCabinTpConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_tp, t_cabin_tp, area_head, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_tp, t_cabin_tp, rear_tpv_flow, cabin_conv_coef
    )

    win_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpWinTpRadCo_M", amb_t
    )
    win_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpWinTpConvCo_M", amb_t
    )
    q_win_rad_raw = q_zone_radiation(
        t_tp, t_win_tp, area_head, win_rad_coef, mode
    )
    q_win_conv_raw = q_zone_convection(
        t_tp, t_win_tp, rear_tpv_flow, win_conv_coef
    )

    # DEFECT D1: Simulink CHTD_TpRoofRadCo_M block Table = CHTD_TdRoofRadCo_M.
    roof_rad_coef = _lut_coef_or_placeholder(
        lv,
        params,
        select_defect_lut(
            mode, "CHTD_TdRoofRadCo_M", "CHTD_TpRoofRadCo_M"
        ),
        amb_t,
    )
    roof_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpRoofConvCo_M", amb_t
    )
    q_roof_rad_raw = q_zone_radiation(
        t_tp, t_roof, area_head, roof_rad_coef, mode
    )
    q_roof_conv_raw = q_zone_convection(
        t_tp, t_roof, rear_tpv_flow, roof_conv_coef
    )

    q_human_raw = hum_head_tp

    solar_fp = get_solar_fp_horiz(u)
    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fp, area_head, solar_coef)

    leak_coef = _lut_coef_or_placeholder(lv, params, "CHTD_TpLeakageCo_M", amb_t)
    q_leak_raw = q_leakage(
        t_tp, amb_t, veh_spd, params.CHTD_WinTpAreaAtb_P, leak_coef
    )

    qbus = HeadTempTpQBusOutputs(
        tp_feet_temp_radiation=q_feet_rad_raw,
        tp_feet_temp_convection=q_feet_conv_raw,
        tp_cabin_tp_radiation=q_cabin_rad_raw,
        tp_cabin_tp_convection=q_cabin_conv_raw,
        tp_win_tp_radiation=q_win_rad_raw,
        tp_win_tp_convection=q_win_conv_raw,
        tp_roof_radiation=q_roof_rad_raw,
        tp_roof_convection=q_roof_conv_raw,
    )

    Q_net = (
        (q_tpv_conv_raw + q_tpf_conv_raw) * _q_gain
        + (q_td_tp_rad_raw + q_td_tp_conv_raw + q_sp_tp_rad_raw + q_sp_tp_conv_raw)
        * _q_gain
        + (q_feet_rad_raw + q_feet_conv_raw) * _q_gain
        + (
            q_cabin_rad_raw
            + q_cabin_conv_raw
            + q_win_rad_raw
            + q_win_conv_raw
            + q_roof_rad_raw
            + q_roof_conv_raw
        )
        * _q_loss
        + q_human_raw
        + q_solar_raw * _q_gain
        + q_leak_raw * _q_loss
    )
    delta = Q_net * params.CHTD_Dt_P / C_tp
    return delta, qbus


@dataclass(frozen=True)
class FeetTempTpQBusOutputs:
    """Raw q-bus exports from FeetTempTp (FeetTempTp_topology_spec §7)."""

    feet_tp_cabin_tp_radiation: float
    feet_tp_cabin_tp_convection: float


def compute_feet_temp_tp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_tp_q: HeadTempTpQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, FeetTempTpQBusOutputs]:
    """FeetTempTp 7-term TRACE (FeetTempTp_topology_spec.md)."""
    t_feet_tp = float(x[X_INDEX["FeetTempTp"]])
    t_cabin_tp = float(x[X_INDEX["CabinTempTp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    rear_tpf_tma = _upstream_signal(u[U_INDEX["RearTpfTma"]], amb_t)
    rear_tpf_flow = float(u[U_INDEX["RearTpfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet_tp = float(u[U_INDEX["HumFeetTpPower"]])

    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_feet_tp = cp_v_ex(t_feet_tp, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)

    tpf_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_TpfFeetTpConvCo_M", amb_t
    )
    q_tpf_conv_raw = q_hvac_duct(rear_tpf_tma, t_feet_tp, rear_tpf_flow, tpf_coef)

    q_head_tp_rad_raw = head_tp_q.tp_feet_temp_radiation
    q_head_tp_conv_raw = head_tp_q.tp_feet_temp_convection

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetTpCabinTpRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetTpCabinTpConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_feet_tp, t_cabin_tp, params.CHTD_FeetAreaAtb_P, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_feet_tp, t_cabin_tp, rear_tpf_flow, cabin_conv_coef
    )

    q_human_raw = hum_feet_tp

    # DEFECT: Simulink CHTD_FeetTpLeakageCo_M block Table = CHTD_FeetTdLeakageCo_M.
    leak_coef = _lut_coef_or_placeholder(
        lv,
        params,
        select_defect_lut(
            mode, "CHTD_FeetTdLeakageCo_M", "CHTD_FeetTpLeakageCo_M"
        ),
        amb_t,
    )
    q_leak_raw = q_leakage(
        t_feet_tp, amb_t, veh_spd, params.CHTD_CabinTpAreaAtb_P, leak_coef
    )

    qbus = FeetTempTpQBusOutputs(
        feet_tp_cabin_tp_radiation=q_cabin_rad_raw,
        feet_tp_cabin_tp_convection=q_cabin_conv_raw,
    )

    Q_net = (
        q_tpf_conv_raw * _q_gain
        + (q_head_tp_rad_raw + q_head_tp_conv_raw) * _q_loss
        + (q_cabin_rad_raw + q_cabin_conv_raw) * _q_loss
        + q_human_raw
        + q_leak_raw * _q_loss
    )
    delta = Q_net * params.CHTD_Dt_P / C_feet_tp
    return delta, qbus


@dataclass(frozen=True)
class CabinTempTpQBusOutputs:
    """Raw q-bus exports from CabinTempTp (CabinTempTp_topology_spec §7)."""

    tp_cabin_tp_radiation: float
    tp_cabin_tp_convection: float
    feet_tp_cabin_tp_radiation: float
    feet_tp_cabin_tp_convection: float
    cabin_tp_amb_radiation: float
    cabin_tp_amb_convection: float
    cabin_tp_human_heat_transfer: float
    cabin_tp_solar_heat_transfer: float


def compute_cabin_temp_tp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_tp_q: HeadTempTpQBusOutputs | None = None,
    feet_tp_q: FeetTempTpQBusOutputs | None = None,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, CabinTempTpQBusOutputs]:
    """CabinTempTp 8-term TRACE (CabinTempTp_topology_spec.md)."""
    t_cabin_tp = float(x[X_INDEX["CabinTempTp"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet_tp = float(u[U_INDEX["HumFeetTpPower"]])
    solar_fp = float(u[U_INDEX["SolarFp"]])
    area_cabin_tp = params.CHTD_CabinTpAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_cabin_tp = cp_m_ex(params.CHTD_CabinTpMassAtb_P, params.CHTD_CabinTpCpAtb_P)

    q_head_tp_rad = head_tp_q.tp_cabin_tp_radiation if head_tp_q else 0.0
    q_head_tp_conv = head_tp_q.tp_cabin_tp_convection if head_tp_q else 0.0
    q_feet_tp_rad = (
        feet_tp_q.feet_tp_cabin_tp_radiation if feet_tp_q else 0.0
    )
    q_feet_tp_conv = (
        feet_tp_q.feet_tp_cabin_tp_convection if feet_tp_q else 0.0
    )
    q_from_head_tp = (
        q_interzone_from_bus(q_head_tp_rad, _q_gain)
        + q_interzone_from_bus(q_head_tp_conv, _q_gain)
    )
    q_from_feet_tp = (
        q_interzone_from_bus(q_feet_tp_rad, _q_gain)
        + q_interzone_from_bus(q_feet_tp_conv, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv,
        params,
        select_defect_lut(
            mode, "CHTD_CabinSpAmbRadCo_M", "CHTD_CabinTpAmbRadCo_M"
        ),
        amb_t,
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinTpAmbConvCo_M", amb_t
    )
    q_cabin_rad_amb = q_zone_radiation(
        t_cabin_tp, amb_t, area_cabin_tp, amb_rad_coef, mode
    )
    q_cabin_conv_amb = q_zone_convection(
        t_cabin_tp, amb_t, veh_spd * area_cabin_tp, amb_conv_coef
    )
    q_cabin_amb = _q_loss * (q_cabin_rad_amb + q_cabin_conv_amb)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_CabinTpSolarRadCo_M", amb_t
    )
    q_cabin_solar_raw = q_solar_gain(solar_fp, area_cabin_tp, solar_coef)
    q_cabin_solar = q_cabin_solar_raw * _q_gain
    q_cabin_human = hum_feet_tp

    Q_net_cabin_tp = (
        q_from_head_tp
        + q_from_feet_tp
        + q_cabin_amb
        + q_cabin_human
        + q_cabin_solar
    )

    qbus = CabinTempTpQBusOutputs(
        tp_cabin_tp_radiation=q_head_tp_rad,
        tp_cabin_tp_convection=q_head_tp_conv,
        feet_tp_cabin_tp_radiation=q_feet_tp_rad,
        feet_tp_cabin_tp_convection=q_feet_tp_conv,
        cabin_tp_amb_radiation=q_cabin_rad_amb,
        cabin_tp_amb_convection=q_cabin_conv_amb,
        cabin_tp_human_heat_transfer=q_cabin_human,
        cabin_tp_solar_heat_transfer=q_cabin_solar_raw,
    )

    delta = Q_net_cabin_tp * params.CHTD_Dt_P / C_cabin_tp
    return delta, qbus


@dataclass(frozen=True)
class WinTempTpQBusOutputs:
    """Raw q-bus exports from WinTempTp (WinTempTp_topology_spec §7)."""

    tp_win_tp_radiation: float
    tp_win_tp_convection: float
    win_tp_amb_radiation: float
    win_tp_amb_convection: float
    win_tp_human_heat_transfer: float
    win_tp_solar_heat_transfer: float


def compute_win_temp_tp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_tp_q: HeadTempTpQBusOutputs | None = None,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, WinTempTpQBusOutputs]:
    """WinTempTp 6-term TRACE (WinTempTp_topology_spec.md).

    DEFECT D_WinTp (AS_FOUND): human term uses HumFeetTpPower u[53], not HumHeadTpPower.
    """
    t_win_tp = float(x[X_INDEX["WinTempTp"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    # DEFECT D_WinTp AS_FOUND: Signal Copy4 reads HumFeetTpPower (not HumHeadTpPower).
    if mode == DefectMode.CORRECTED:
        hum_win_tp = float(u[U_INDEX["HumHeadTpPower"]])
    else:
        hum_win_tp = float(u[U_INDEX["HumFeetTpPower"]])
    solar_fp = float(u[U_INDEX["SolarFp"]])
    area_win_tp = params.CHTD_WinTpAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_win_tp = cp_m_ex(params.CHTD_WinTpMassAtb_P, params.CHTD_WinTpCpAtb_P)

    q_head_tp_rad = head_tp_q.tp_win_tp_radiation if head_tp_q else 0.0
    q_head_tp_conv = head_tp_q.tp_win_tp_convection if head_tp_q else 0.0
    q_from_head_tp = (
        q_interzone_from_bus(q_head_tp_rad, _q_gain)
        + q_interzone_from_bus(q_head_tp_conv, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(lv, params, "CHTD_WinTpAmbRadCo_M", amb_t)
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinTpAmbConvCo_M", amb_t
    )
    q_amb_rad_raw = q_zone_radiation(
        t_win_tp, amb_t, area_win_tp, amb_rad_coef, mode
    )
    q_amb_conv_raw = q_zone_convection(
        t_win_tp, amb_t, veh_spd * area_win_tp, amb_conv_coef
    )
    q_amb = _q_loss * (q_amb_rad_raw + q_amb_conv_raw)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinTpSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fp, area_win_tp, solar_coef)
    q_solar = q_solar_raw * _q_gain

    Q_net_win_tp = q_from_head_tp + q_amb + hum_win_tp + q_solar

    qbus = WinTempTpQBusOutputs(
        tp_win_tp_radiation=q_head_tp_rad,
        tp_win_tp_convection=q_head_tp_conv,
        win_tp_amb_radiation=q_amb_rad_raw,
        win_tp_amb_convection=q_amb_conv_raw,
        win_tp_human_heat_transfer=hum_win_tp,
        win_tp_solar_heat_transfer=q_solar_raw,
    )

    delta = Q_net_win_tp * params.CHTD_Dt_P / C_win_tp
    return delta, qbus


def compute_feet_temp_sp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_sp_q: HeadTempSpQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, FeetTempSpQBusOutputs]:
    """FeetTempSp 8-term TRACE (FeetTempSp_topology_spec.md)."""
    t_feet_sp = float(x[X_INDEX["FeetTempSp"]])
    t_cabin_sp = float(x[X_INDEX["CabinTempSp"]])

    amb_t = float(u[U_INDEX["AmbT"]])
    frnt_spf_tma = _upstream_signal(u[U_INDEX["FrntSpfTma"]], amb_t)
    rear_spf_tma = _upstream_signal(u[U_INDEX["RearSpfTma"]], amb_t)
    frnt_spf_flow = float(u[U_INDEX["FrntSpfFlow"]])
    rear_spf_flow = float(u[U_INDEX["RearSpfFlow"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_feet = float(u[U_INDEX["HumFeetSpPower"]])

    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P
    flow_spf = frnt_spf_flow + rear_spf_flow

    C_feet_sp = cp_v_ex(t_feet_sp, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P)

    spf_coef = _lut_coef_or_placeholder(lv, params, "CHTD_SpfFeetSpConvCo_M", amb_t)
    rear_spf_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_RearSpfFeetSpConvCo_M", amb_t
    )
    q_spf_conv_raw = q_hvac_duct(frnt_spf_tma, t_feet_sp, frnt_spf_flow, spf_coef)
    q_rear_spf_conv_raw = q_hvac_duct(
        rear_spf_tma, t_feet_sp, rear_spf_flow, rear_spf_coef
    )

    q_head_rad_raw = head_sp_q.sp_feet_temp_radiation
    q_head_conv_raw = head_sp_q.sp_feet_temp_convection

    cabin_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetSpCabinSpRadCo_M", amb_t
    )
    cabin_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_FeetSpCabinSpConvCo_M", amb_t
    )
    q_cabin_rad_raw = q_zone_radiation(
        t_feet_sp, t_cabin_sp, params.CHTD_FeetAreaAtb_P, cabin_rad_coef, mode
    )
    q_cabin_conv_raw = q_zone_convection(
        t_feet_sp, t_cabin_sp, flow_spf, cabin_conv_coef
    )

    q_human_raw = hum_feet

    leak_coef = _lut_coef_or_placeholder(lv, params, "CHTD_FeetSpLeakageCo_M", amb_t)
    q_leak_raw = q_leakage(
        t_feet_sp, amb_t, veh_spd, params.CHTD_CabinSpAreaAtb_P, leak_coef
    )

    qbus = FeetTempSpQBusOutputs(
        spf_feet_sp_flow_convection=q_spf_conv_raw,
        rear_spf_feet_sp_flow_convection=q_rear_spf_conv_raw,
        sp_feet_temp_radiation=q_head_rad_raw,
        sp_feet_temp_convection=q_head_conv_raw,
        feet_sp_cabin_sp_radiation=q_cabin_rad_raw,
        feet_sp_cabin_sp_convection=q_cabin_conv_raw,
        feet_sp_human_heat_transfer=q_human_raw,
        feet_sp_heat_leakage=q_leak_raw,
    )

    Q_net = (
        (q_spf_conv_raw + q_rear_spf_conv_raw) * _q_gain
        + (q_head_rad_raw + q_head_conv_raw) * _q_loss
        + (q_cabin_rad_raw + q_cabin_conv_raw) * _q_loss
        + q_human_raw
        + q_leak_raw * _q_loss
    )
    delta = Q_net * params.CHTD_Dt_P / C_feet_sp
    return delta, qbus


def compute_win_temp_fp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    head_fp_q: HeadTempFpQBusOutputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, WinTempFpQBusOutputs]:
    """WinTempFp 6-term TRACE (WinTempFp_topology_spec.md)."""
    t_win_fp = float(x[X_INDEX["WinTempFp"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    hum_head = float(u[U_INDEX["HumHeadFpPower"]])
    # WinTempFp uses SolarFp directly — no SolarFpHoriz alias (Head zones only).
    solar_fp = float(u[U_INDEX["SolarFp"]])
    area_win_fp = params.CHTD_WinFpAreaAtb_P
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    C_win_fp = cp_m_ex(params.CHTD_WinFpMassAtb_P, params.CHTD_WinFpCpAtb_P)

    q_from_head_fp = (
        q_interzone_from_bus(head_fp_q.fp_win_fp_radiation, _q_gain)
        + q_interzone_from_bus(head_fp_q.fp_win_fp_convection, _q_gain)
    )

    amb_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinFpAmbRadCo_M", amb_t
    )
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinFpAmbConvCo_M", amb_t
    )
    q_amb_rad_raw = q_zone_radiation(
        t_win_fp, amb_t, area_win_fp, amb_rad_coef, mode
    )
    q_amb_conv_raw = q_zone_convection(
        t_win_fp, amb_t, veh_spd * area_win_fp, amb_conv_coef
    )
    q_amb = _q_loss * (q_amb_rad_raw + q_amb_conv_raw)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_WinFpSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_fp, area_win_fp, solar_coef)
    q_solar = q_solar_raw * _q_gain

    Q_net_win_fp = q_from_head_fp + q_amb + hum_head + q_solar

    qbus = WinTempFpQBusOutputs(
        fp_win_fp_radiation=head_fp_q.fp_win_fp_radiation,
        fp_win_fp_convection=head_fp_q.fp_win_fp_convection,
        win_fp_amb_radiation=q_amb_rad_raw,
        win_fp_amb_convection=q_amb_conv_raw,
        win_fp_human_heat_transfer=hum_head,
        win_fp_solar_radiation=q_solar_raw,
    )

    delta = Q_net_win_fp * params.CHTD_Dt_P / C_win_fp
    return delta, qbus


@dataclass(frozen=True)
class RoofTempQBusInputs:
    """Pre-computed q-bus heat into RoofTemp from 6 Head zones (12 raw fields)."""

    fd_roof_radiation: float
    fd_roof_convection: float
    fp_roof_radiation: float
    fp_roof_convection: float
    sd_roof_radiation: float
    sd_roof_convection: float
    sp_roof_radiation: float
    sp_roof_convection: float
    td_roof_radiation: float
    td_roof_convection: float
    tp_roof_radiation: float
    tp_roof_convection: float

    @property
    def total(self) -> float:
        return (
            self.fd_roof_radiation + self.fd_roof_convection
            + self.fp_roof_radiation + self.fp_roof_convection
            + self.sd_roof_radiation + self.sd_roof_convection
            + self.sp_roof_radiation + self.sp_roof_convection
            + self.td_roof_radiation + self.td_roof_convection
            + self.tp_roof_radiation + self.tp_roof_convection
        )


@dataclass(frozen=True)
class RoofTempQBusOutputs:
    """Raw q-bus exports from RoofTemp (RoofTemp_topology_spec §8)."""

    fd_roof_radiation: float
    fd_roof_convection: float
    fp_roof_radiation: float
    fp_roof_convection: float
    sd_roof_radiation: float
    sd_roof_convection: float
    sp_roof_radiation: float
    sp_roof_convection: float
    td_roof_radiation: float
    td_roof_convection: float
    tp_roof_radiation: float
    tp_roof_convection: float
    roof_amb_radiation: float
    roof_amb_convection: float
    roof_solar_radiation: float


def build_roof_temp_qbus_inputs(
    head_fd_q: HeadTempFdQBusOutputs,
    head_fp_q: HeadTempFpQBusOutputs,
    sp_roof_radiation: float,
    sp_roof_convection: float,
    head_sd_q: Optional[HeadTempSdQBusOutputs] = None,
    td_roof_radiation: float = 0.0,
    td_roof_convection: float = 0.0,
    tp_roof_radiation: float = 0.0,
    tp_roof_convection: float = 0.0,
) -> RoofTempQBusInputs:
    """Assemble RoofTemp q-bus inputs from implemented Head zone exports."""
    return RoofTempQBusInputs(
        fd_roof_radiation=head_fd_q.fd_roof_radiation,
        fd_roof_convection=head_fd_q.fd_roof_convection,
        fp_roof_radiation=head_fp_q.fp_roof_radiation,
        fp_roof_convection=head_fp_q.fp_roof_convection,
        sd_roof_radiation=(
            head_sd_q.sd_roof_radiation if head_sd_q is not None else 0.0
        ),
        sd_roof_convection=(
            head_sd_q.sd_roof_convection if head_sd_q is not None else 0.0
        ),
        sp_roof_radiation=sp_roof_radiation,
        sp_roof_convection=sp_roof_convection,
        td_roof_radiation=td_roof_radiation,
        td_roof_convection=td_roof_convection,
        tp_roof_radiation=tp_roof_radiation,
        tp_roof_convection=tp_roof_convection,
    )


def compute_roof_temp(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    lv: dict,
    roof_q: RoofTempQBusInputs,
    mode: DefectMode = AS_FOUND,
) -> tuple[float, RoofTempQBusOutputs]:
    """RoofTemp 15-term TRACE (RoofTemp_topology_spec.md)."""
    t_roof = float(x[X_INDEX["RoofTemp"]])
    amb_t = float(u[U_INDEX["AmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    solar_fd = float(u[U_INDEX["SolarFd"]])
    solar_fp = float(u[U_INDEX["SolarFp"]])
    solar_max = max(solar_fd, solar_fp)
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    # DEFECT D5 (AS_FOUND): CpMEx uses WinTp mass/cp, not RoofMass/RoofCp
    if mode == DefectMode.CORRECTED:
        C_roof = cp_m_ex(params.CHTD_RoofMassAtb_P, params.CHTD_RoofCpAtb_P)
    else:
        C_roof = cp_m_ex(params.CHTD_WinTpMassAtb_P, params.CHTD_WinTpCpAtb_P)

    q_from_heads = (
        q_interzone_from_bus(roof_q.fd_roof_radiation, _q_gain)
        + q_interzone_from_bus(roof_q.fd_roof_convection, _q_gain)
        + q_interzone_from_bus(roof_q.fp_roof_radiation, _q_gain)
        + q_interzone_from_bus(roof_q.fp_roof_convection, _q_gain)
        + q_interzone_from_bus(roof_q.sd_roof_radiation, _q_gain)
        + q_interzone_from_bus(roof_q.sd_roof_convection, _q_gain)
        + q_interzone_from_bus(roof_q.sp_roof_radiation, _q_gain)
        + q_interzone_from_bus(roof_q.sp_roof_convection, _q_gain)
        + q_interzone_from_bus(roof_q.td_roof_radiation, _q_gain)
        + q_interzone_from_bus(roof_q.td_roof_convection, _q_gain)
        + q_interzone_from_bus(roof_q.tp_roof_radiation, _q_gain)
        + q_interzone_from_bus(roof_q.tp_roof_convection, _q_gain)
    )

    # DEFECT D6 (AS_FOUND): ambient radiation area uses WinTpAreaAtb_P, not RoofAreaAtb_P
    if mode == DefectMode.CORRECTED:
        amb_rad_area = params.CHTD_RoofAreaAtb_P
    else:
        amb_rad_area = params.CHTD_WinTpAreaAtb_P
    roof_rad_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_RoofAreaAtb_M", amb_t
    )
    q_amb_rad_raw = q_zone_radiation(
        t_roof, amb_t, amb_rad_area, roof_rad_coef, mode
    )

    roof_flow_area = params.CHTD_RoofAreaAtb_P
    amb_conv_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_RoofAmbConvCo_M", amb_t
    )
    q_amb_conv_raw = q_zone_convection(
        t_roof, amb_t, veh_spd * roof_flow_area, amb_conv_coef
    )
    q_amb = _q_loss * (q_amb_rad_raw + q_amb_conv_raw)

    solar_coef = _lut_coef_or_placeholder(
        lv, params, "CHTD_RoofSolarRadCo_M", amb_t
    )
    q_solar_raw = q_solar_gain(solar_max, roof_flow_area, solar_coef)
    q_solar = q_solar_raw * _q_gain

    Q_net_roof = q_from_heads + q_amb + q_solar

    qbus = RoofTempQBusOutputs(
        fd_roof_radiation=roof_q.fd_roof_radiation,
        fd_roof_convection=roof_q.fd_roof_convection,
        fp_roof_radiation=roof_q.fp_roof_radiation,
        fp_roof_convection=roof_q.fp_roof_convection,
        sd_roof_radiation=roof_q.sd_roof_radiation,
        sd_roof_convection=roof_q.sd_roof_convection,
        sp_roof_radiation=roof_q.sp_roof_radiation,
        sp_roof_convection=roof_q.sp_roof_convection,
        td_roof_radiation=roof_q.td_roof_radiation,
        td_roof_convection=roof_q.td_roof_convection,
        tp_roof_radiation=roof_q.tp_roof_radiation,
        tp_roof_convection=roof_q.tp_roof_convection,
        roof_amb_radiation=q_amb_rad_raw,
        roof_amb_convection=q_amb_conv_raw,
        roof_solar_radiation=q_solar_raw,
    )

    delta = Q_net_roof * params.CHTD_Dt_P / C_roof
    return delta, qbus


# ── New Simulink-aligned API (T6A skeleton) ───────────────────────────────────

def compute_chtd_delta(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    mode: DefectMode = AS_FOUND,
) -> np.ndarray:
    """Compute xStep delta for one CHTD Forward-Euler step.

    Parameters
    ----------
    x      : 28-element state vector [°C], shape (28,)
    u      : 54-element input vector, shape (54,)
    params : CHTDParams instance
    mode   : DefectMode.AS_FOUND (default) or CORRECTED

    Returns
    -------
    delta : ndarray shape (28,) — xStep values [°C/step]

    HoodTemp [0]: dual-ambient 6-term Q_net (T6D-0, §7).
    HeadTempFd [3]: 20-term Q_net (T8, HeadTempFd_topology_spec.md).
    FeetTempFd [9]: 9-term Q_net (T8, FeetTempFd_topology_spec.md).
    HeadTempFp [4]: 20-term Q_net (T8 passenger, HeadTempFp_topology_spec.md).
    FeetTempFp [10]: 9-term Q_net (T8 passenger, FeetTempFp_topology_spec.md).
    HeadTempSp [6]: full 21-term Q_net (T6A-2); q-bus fp_sp from HeadTempFp.
    CabinTempFd [15]: 6-term Q_net (T6C-0); q-bus from HeadTempFd / FeetTempFd exports.
    CabinTempFp [16]: 8-term Q_net (T8); q-bus from HeadTempFp / FeetTempFp exports.
    WinTempFd [21]: 6-term Q_net (T8); q-bus from HeadTempFd exports.
    WinTempFp [22]: 6-term Q_net (T8); q-bus from HeadTempFp exports.
    RoofTemp [27]: 15-term Q_net (T8); q-bus from 6 Head zones (Fd/Fp/Sp live).
    Remaining zones: FAST_APPROX §8 generic (T7-0); see zone_config.py.
    TODO(T6B): HeadTempSp q-bus physics; ConsoleTemp q-bus for Head/Feet Fd
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    if x.shape != (N_X_STATES,):
        raise ValueError(
            f"compute_chtd_delta: x must have shape ({N_X_STATES},), got {x.shape}"
        )
    if u.shape != (N_U,):
        raise ValueError(
            f"compute_chtd_delta: u must have shape ({N_U},), got {u.shape}"
        )
    delta = np.zeros(N_X_STATES)

    amb_t = float(u[U_INDEX["AmbT"]])
    raw_amb_t = float(u[U_INDEX["RawAmbT"]])
    veh_spd = float(u[U_INDEX["VehSpd"]])
    lv = eval_lut_map(params, amb_t)
    _q_gain = params.CHTD_QgainCo_P
    _q_loss = params.CHTD_QlossCo_P

    # ── HoodTemp: dual-ambient Q_net (T6D-0, CHTD_formula_spec.md §7) ───────
    t_hood = float(x[X_INDEX["HoodTemp"]])
    t_cabin_frnt = float(x[X_INDEX["CabinFrntTemp"]])
    area_hood = params.CHTD_HoodAreaAtb_P
    flow_hood = veh_spd * area_hood

    C_hood = cp_m_ex(params.CHTD_HoodMassAtb_P, params.CHTD_HoodCpAtb_P)

    q_hood_raw_rad = q_zone_radiation(
        raw_amb_t, t_hood, area_hood, lv["CHTD_RawAmbHoodRadCo_M"], mode
    ) * _q_gain
    q_hood_raw_conv = q_zone_convection(
        raw_amb_t, t_hood, flow_hood, lv["CHTD_RawAmbHoodConvCo_M"]
    ) * _q_gain
    q_hood_amb_rad = q_zone_radiation(
        t_hood, amb_t, area_hood, lv["CHTD_HoodAmbRadCo_M"], mode
    ) * _q_loss
    q_hood_amb_conv = q_zone_convection(
        t_hood, amb_t, flow_hood, lv["CHTD_HoodAmbConvCo_M"]
    ) * _q_loss
    q_hood_cabin_frnt_raw = q_zone_radiation(
        t_hood, t_cabin_frnt, area_hood, lv["CHTD_HoodCabinFrntRadCo_M"], mode
    )
    q_hood_solar = q_solar_gain(
        _hood_solar_intensity(u), area_hood, lv["CHTD_HoodSolarRadCo_M"]
    ) * _q_gain

    Q_net_hood = (
        q_hood_raw_rad
        + q_hood_raw_conv
        + q_hood_amb_rad
        + q_hood_amb_conv
        + q_hood_cabin_frnt_raw * _q_loss
        + q_hood_solar
    )
    delta[X_INDEX["HoodTemp"]] = Q_net_hood * params.CHTD_Dt_P / C_hood

    # ── ConsoleTemp: 9-term Q_net (T9, ConsoleTemp_topology_spec.md) ─────────
    delta_console, console_q = compute_console_temp(x, u, params, lv, mode)
    delta[X_INDEX["ConsoleTemp"]] = delta_console

    # ── CabinFrntTemp: 2-term q-bus only (T9, CabinFrntTemp_topology_spec.md) ─
    delta_cabin_frnt, _cabin_frnt_q = compute_cabin_frnt_temp(
        params,
        hood_cabin_frnt_radiation=q_hood_cabin_frnt_raw,
        cabin_frnt_console_radiation=console_q.cabin_frnt_console_radiation,
    )
    delta[X_INDEX["CabinFrntTemp"]] = delta_cabin_frnt

    # ── HeadTempFd: 20-term Q_net (T8) ───────────────────────────────────────
    delta_head_fd, head_fd_q = compute_head_temp_fd(
        x, u, params, lv, mode, console_q=console_q
    )
    delta[X_INDEX["HeadTempFd"]] = delta_head_fd

    # ── FeetTempFd: 9-term Q_net (T8) ────────────────────────────────────────
    delta_feet_fd, feet_fd_q = compute_feet_temp_fd(
        x, u, params, lv, head_fd_q, mode, console_q=console_q
    )
    delta[X_INDEX["FeetTempFd"]] = delta_feet_fd

    # ── HeadTempFp: 20-term Q_net (T8 passenger) ─────────────────────────────
    delta_head_fp, head_fp_q = compute_head_temp_fp(
        x, u, params, lv, head_fd_q, mode, console_q=console_q
    )
    delta[X_INDEX["HeadTempFp"]] = delta_head_fp

    # ── FeetTempFp: 9-term Q_net (T8 passenger) ──────────────────────────────
    delta_feet_fp, feet_fp_q = compute_feet_temp_fp(
        x, u, params, lv, head_fp_q, mode, console_q=console_q
    )
    delta[X_INDEX["FeetTempFp"]] = delta_feet_fp

    # ── HeadTempSd: 21-term Q_net (T9) ───────────────────────────────────────
    delta_head_sd, head_sd_q = compute_head_temp_sd(
        x, u, params, lv, head_fd_q, mode
    )
    delta[X_INDEX["HeadTempSd"]] = delta_head_sd

    # ── HeadTempTd: 17-term TRACE (T11, HeadTempTd_topology_spec.md) ─────────
    delta_head_td, head_td_q = compute_head_temp_td(
        x, u, params, lv, head_sd_q, mode
    )
    delta[X_INDEX["HeadTempTd"]] = delta_head_td

    # ── FeetTempTd: 7-term TRACE (T11, FeetTempTd_topology_spec.md) ────────
    delta_feet_td, feet_td_q = compute_feet_temp_td(
        x, u, params, lv, head_td_q, mode
    )
    delta[X_INDEX["FeetTempTd"]] = delta_feet_td

    # ── CabinTempTd: 8-term TRACE (T11, CabinTempTd_topology_spec.md) ────────
    delta_cabin_td, _cabin_td_q = compute_cabin_temp_td(
        x, u, params, lv, head_td_q, feet_td_q, mode
    )
    delta[X_INDEX["CabinTempTd"]] = delta_cabin_td

    # ── WinTempTd: 6-term TRACE (T11, WinTempTd_topology_spec.md) ────────────
    delta_win_td, _win_td_q = compute_win_temp_td(
        x, u, params, lv, head_td_q, mode
    )
    delta[X_INDEX["WinTempTd"]] = delta_win_td

    # ── FeetTempSd: 8-term Q_net (T9) ────────────────────────────────────────
    delta_feet_sd, feet_sd_q = compute_feet_temp_sd(
        x, u, params, lv, head_sd_q, mode
    )
    delta[X_INDEX["FeetTempSd"]] = delta_feet_sd

    # ── HeadTempSp: full Q_net (T6A-2, 21 terms) ────────────────────────────

    t_sp      = float(x[X_INDEX["HeadTempSp"]])
    t_tp      = float(x[X_INDEX["HeadTempTp"]])
    t_feet_sp = float(x[X_INDEX["FeetTempSp"]])
    t_cabin_sp= float(x[X_INDEX["CabinTempSp"]])
    t_win_sp  = float(x[X_INDEX["WinTempSp"]])
    t_roof    = float(x[X_INDEX["RoofTemp"]])

    frnt_spv_tma  = float(u[U_INDEX["FrntSpvTma"]])
    frnt_spf_tma  = float(u[U_INDEX["FrntSpfTma"]])
    rear_spv_tma  = float(u[U_INDEX["RearSpvTma"]])
    frnt_spv_flow = float(u[U_INDEX["FrntSpvFlow"]])
    frnt_spf_flow = float(u[U_INDEX["FrntSpfFlow"]])
    rear_spv_flow = float(u[U_INDEX["RearSpvFlow"]])
    rear_spf_flow = float(u[U_INDEX["RearSpfFlow"]])
    # NOTE: HeadTempSp_topology_spec §10 lists HumHeadSpPower at 0-based index 46,
    # but bus_index.py confirms U_INDEX["HumHeadSpPower"] = 45 (Simulink u(46)).
    # bus_index.py is the authority; no hardcoded index used.
    hum_heat = float(u[U_INDEX["HumHeadSpPower"]])

    flow_spv = frnt_spv_flow + rear_spv_flow   # shared flow for SpTp/Cabin/Win/Roof convection
    flow_spf = frnt_spf_flow + rear_spf_flow   # shared flow for FeetSp convection

    C_sp = cp_v_ex(t_sp, params.CHTD_HeadAirVAtb_P, params.CHTD_AirCpAtb_P)

    # Add3: HVAC duct convection (4 terms, QgainCo)
    q_frnt_spv = q_hvac_duct(
        frnt_spv_tma, t_sp, frnt_spv_flow, lv["CHTD_SpvSpConvCo_M"]
    ) * params.CHTD_QgainCo_P
    q_frnt_spf = q_hvac_duct(
        frnt_spf_tma, t_sp, frnt_spf_flow, lv["CHTD_SpfSpConvCo_M"]
    ) * params.CHTD_QgainCo_P
    q_rear_spv = q_hvac_duct(
        rear_spv_tma, t_sp, rear_spv_flow, lv["CHTD_RearSpvSpConvCo_M"]
    ) * params.CHTD_QgainCo_P
    # DEFECT D1 (Policy P4): T_src = HeadTempSd AS_FOUND, HeadTempSp CORRECTED
    q_rear_spf = head_sp_rear_spf_hvac_term(x, u, params, lv, mode)

    # Add4: q-bus coupling from HeadTempSd + HeadTempFp (4 terms, QgainCo)
    q_bus = compute_head_temp_sp_qbus(head_fp_q=head_fp_q, head_sd_q=head_sd_q)
    q_sd_sp_rad  = q_interzone_from_bus(q_bus.sd_sp_radiation, _q_gain)
    q_sd_sp_conv = q_interzone_from_bus(q_bus.sd_sp_convection, _q_gain)
    q_fp_sp_rad  = q_interzone_from_bus(q_bus.fp_sp_radiation, _q_gain)
    q_fp_sp_conv = q_interzone_from_bus(q_bus.fp_sp_convection, _q_gain)

    # Add4: HeadTempTp coupling (2 terms, QlossCo)
    q_sp_tp_rad_raw = q_zone_radiation(
        t_sp, t_tp, params.CHTD_HeadAreaAtb_P, lv["CHTD_SpTpRadCo_M"], mode
    )
    q_sp_tp_conv_raw = q_zone_convection(
        t_sp, t_tp, flow_spv, lv["CHTD_SpTpConvCo_M"]
    )
    q_sp_tp_rad = q_sp_tp_rad_raw * params.CHTD_QlossCo_P
    q_sp_tp_conv = q_sp_tp_conv_raw * params.CHTD_QlossCo_P

    # Add6: FeetTempSp coupling (2 terms, QgainCo)
    # SubRef18 as-found anomaly: Convection block receives CHTD_HeadAreaAtb_P as
    # flow input (not an area); ConvCoBaseFlowEx(HeadAreaAtb_P) is computed.
    # CHTD_SpFeetSpRadCo_M is used as convection coef (not radiation block).
    q_feet_sp_conv_block = q_zone_convection(
        t_feet_sp, t_sp, params.CHTD_HeadAreaAtb_P, lv["CHTD_SpFeetSpRadCo_M"]
    ) * params.CHTD_QgainCo_P
    q_feet_sp_conv = q_zone_convection(
        t_feet_sp, t_sp, flow_spf, lv["CHTD_SpFeetSpConvCo_M"]
    ) * params.CHTD_QgainCo_P

    # Exports to FeetTempSp: HeadSp → FeetSp q-bus (raw, consumer applies QlossCo).
    q_sp_feet_temp_rad_raw = q_zone_radiation(
        t_sp, t_feet_sp, params.CHTD_HeadAreaAtb_P, lv["CHTD_SpFeetSpRadCo_M"], mode
    )
    q_sp_feet_temp_conv_raw = q_zone_convection(
        t_sp, t_feet_sp, flow_spf, lv["CHTD_SpFeetSpConvCo_M"]
    )

    # Add7: CabinTempSp / WinTempSp / RoofTemp coupling (6 terms, QlossCo)
    q_cabin_sp_rad_raw = q_zone_radiation(
        t_sp, t_cabin_sp, params.CHTD_HeadAreaAtb_P, lv["CHTD_SpCabinSpRadCo_M"], mode
    )
    q_cabin_sp_conv_raw = q_zone_convection(
        t_sp, t_cabin_sp, flow_spv, lv["CHTD_SpCabinSpConvCo_M"]
    )
    q_cabin_sp_rad = q_cabin_sp_rad_raw * params.CHTD_QlossCo_P
    q_cabin_sp_conv = q_cabin_sp_conv_raw * params.CHTD_QlossCo_P
    q_win_sp_rad_raw = q_zone_radiation(
        t_sp, t_win_sp, params.CHTD_HeadAreaAtb_P, lv["CHTD_SpWinSpRadCo_M"], mode
    )
    q_win_sp_conv_raw = q_zone_convection(
        t_sp, t_win_sp, flow_spv, lv["CHTD_SpWinSpConvCo_M"]
    )
    q_win_sp_rad = q_win_sp_rad_raw * params.CHTD_QlossCo_P
    q_win_sp_conv = q_win_sp_conv_raw * params.CHTD_QlossCo_P
    q_roof_rad_raw = q_zone_radiation(
        t_sp, t_roof, params.CHTD_HeadAreaAtb_P, lv["CHTD_SpRoofRadCo_M"], mode
    )
    q_roof_conv_raw = q_zone_convection(
        t_sp, t_roof, flow_spv, lv["CHTD_SpRoofConvCo_M"]
    )
    q_roof_rad = q_roof_rad_raw * params.CHTD_QlossCo_P
    q_roof_conv = q_roof_conv_raw * params.CHTD_QlossCo_P

    # Human heat: direct addition, no QgainCo multiplier
    q_human = hum_heat

    # Solar (DEFECT D2 / Policy P3): SolarFpHoriz not in Bus_CHTD_u; alias = SolarFp
    solar_fp = get_solar_fp_horiz(u)
    q_solar = q_solar_gain(
        solar_fp, params.CHTD_HeadAreaAtb_P, lv["CHTD_SpSolarRadCo_M"]
    ) * params.CHTD_QgainCo_P

    # Leakage through WinTempSp (QlossCo); flow proxy = VehSpd × CHTD_WinSpAreaAtb_P
    q_leak = q_leakage(
        t_sp, amb_t, veh_spd, params.CHTD_WinSpAreaAtb_P, lv["CHTD_SpLeakageCo_M"]
    ) * params.CHTD_QlossCo_P

    Q_net = (
        q_frnt_spv + q_frnt_spf + q_rear_spv + q_rear_spf
        + q_sd_sp_rad + q_sd_sp_conv + q_fp_sp_rad + q_fp_sp_conv
        + q_sp_tp_rad + q_sp_tp_conv
        + q_feet_sp_conv_block + q_feet_sp_conv
        + q_cabin_sp_rad + q_cabin_sp_conv
        + q_win_sp_rad + q_win_sp_conv
        + q_roof_rad + q_roof_conv
        + q_human + q_solar + q_leak
    )
    delta[X_INDEX["HeadTempSp"]] = Q_net * params.CHTD_Dt_P / C_sp

    head_sp_q = HeadTempSpQBusOutputs(
        sp_feet_temp_radiation=q_sp_feet_temp_rad_raw,
        sp_feet_temp_convection=q_sp_feet_temp_conv_raw,
        sp_cabin_sp_radiation=q_cabin_sp_rad_raw,
        sp_cabin_sp_convection=q_cabin_sp_conv_raw,
        sp_win_sp_radiation=q_win_sp_rad_raw,
        sp_win_sp_convection=q_win_sp_conv_raw,
        sp_roof_radiation=q_roof_rad_raw,
        sp_roof_convection=q_roof_conv_raw,
        sp_tp_radiation=q_sp_tp_rad_raw,
        sp_tp_convection=q_sp_tp_conv_raw,
    )

    # ── HeadTempTp: 17-term TRACE (T11, HeadTempTp_topology_spec.md) ─────────
    delta_head_tp, head_tp_q = compute_head_temp_tp(
        x, u, params, lv, head_sp_q, mode, head_td_q
    )
    delta[X_INDEX["HeadTempTp"]] = delta_head_tp

    # ── FeetTempTp: 7-term TRACE (T11, FeetTempTp_topology_spec.md) ──────────
    delta_feet_tp, feet_tp_q = compute_feet_temp_tp(
        x, u, params, lv, head_tp_q, mode
    )
    delta[X_INDEX["FeetTempTp"]] = delta_feet_tp

    # ── CabinTempTp: 8-term TRACE (T11, CabinTempTp_topology_spec.md) ────────
    delta_cabin_tp, _cabin_tp_q = compute_cabin_temp_tp(
        x, u, params, lv, head_tp_q, feet_tp_q, mode
    )
    delta[X_INDEX["CabinTempTp"]] = delta_cabin_tp

    # ── WinTempTp: 6-term TRACE (T11, WinTempTp_topology_spec.md) ────────────
    delta_win_tp, _win_tp_q = compute_win_temp_tp(x, u, params, lv, head_tp_q, mode)
    delta[X_INDEX["WinTempTp"]] = delta_win_tp

    # ── FeetTempSp: 8-term Q_net (T9, FeetTempSp_topology_spec.md) ───────────
    delta_feet_sp, feet_sp_q = compute_feet_temp_sp(x, u, params, lv, head_sp_q, mode)
    delta[X_INDEX["FeetTempSp"]] = delta_feet_sp

    # ── CabinTempSp: 8-term Q_net (T9, CabinTempSp_topology_spec.md) ───────────
    delta_cabin_sp, _cabin_sp_q = compute_cabin_temp_sp(
        x, u, params, lv, head_sp_q, feet_sp_q, mode
    )
    delta[X_INDEX["CabinTempSp"]] = delta_cabin_sp

    # ── WinTempSp: 6-term Q_net (T9, WinTempSp_topology_spec.md) ─────────────
    delta_win_sp, _win_sp_q = compute_win_temp_sp(x, u, params, lv, head_sp_q, mode)
    delta[X_INDEX["WinTempSp"]] = delta_win_sp

    # ── CabinTempFd: 6-term Q_net (T6C-0, CHTD_formula_spec.md §5) ──────────
    t_cabin_fd = float(x[X_INDEX["CabinTempFd"]])
    hum_feet_fd = float(u[U_INDEX["HumFeetFdPower"]])
    solar_fd = float(u[U_INDEX["SolarFd"]])
    area_cabin_fd = params.CHTD_CabinFdAreaAtb_P

    C_cabin_fd = cp_m_ex(params.CHTD_CabinFdMassAtb_P, params.CHTD_CabinFdCpAtb_P)

    q_cabin_rad_amb = q_zone_radiation(
        t_cabin_fd, amb_t, area_cabin_fd, lv["CHTD_CabinFdAmbRadCo_M"], mode
    )
    q_cabin_conv_amb = q_zone_convection(
        t_cabin_fd, amb_t, veh_spd * area_cabin_fd, lv["CHTD_CabinFdAmbConvCo_M"]
    )
    q_cabin_amb = params.CHTD_QlossCo_P * (q_cabin_rad_amb + q_cabin_conv_amb)

    q_bus_cabin = compute_cabin_temp_fd_qbus(head_fd_q, feet_fd_q)
    q_from_head_fd = (
        q_interzone_from_bus(q_bus_cabin.head_fd_radiation, _q_gain)
        + q_interzone_from_bus(q_bus_cabin.head_fd_convection, _q_gain)
    )
    q_from_feet_fd = (
        q_interzone_from_bus(q_bus_cabin.feet_fd_radiation, _q_gain)
        + q_interzone_from_bus(q_bus_cabin.feet_fd_convection, _q_gain)
    )

    q_cabin_solar = q_solar_gain(
        solar_fd, area_cabin_fd, lv["CHTD_CabinFdSolarRadCo_M"]
    ) * params.CHTD_QgainCo_P
    q_cabin_human = hum_feet_fd

    Q_net_cabin_fd = (
        q_cabin_amb
        + q_from_head_fd
        + q_from_feet_fd
        + q_cabin_human
        + q_cabin_solar
    )
    delta[X_INDEX["CabinTempFd"]] = Q_net_cabin_fd * params.CHTD_Dt_P / C_cabin_fd

    # ── CabinTempFp: 8-term Q_net (T8, CabinTempFp_topology_spec.md) ────────
    delta_cabin_fp, _cabin_fp_q = compute_cabin_temp_fp(
        x, u, params, lv, head_fp_q, feet_fp_q, mode
    )
    delta[X_INDEX["CabinTempFp"]] = delta_cabin_fp

    # ── CabinTempSd: 8-term Q_net (T9, CabinTempSd_topology_spec.md) ─────────
    delta_cabin_sd, _cabin_sd_q = compute_cabin_temp_sd(
        x, u, params, lv, head_sd_q, feet_sd_q, mode
    )
    delta[X_INDEX["CabinTempSd"]] = delta_cabin_sd

    # ── WinTempFd: 6-term Q_net (T8, WinTempFd_topology_spec.md) ─────────────
    delta_win_fd, _win_fd_q = compute_win_temp_fd(
        x, u, params, lv, head_fd_q, mode
    )
    delta[X_INDEX["WinTempFd"]] = delta_win_fd

    # ── WinTempFp: 6-term Q_net (T8, WinTempFp_topology_spec.md) ─────────────
    delta_win_fp, _win_fp_q = compute_win_temp_fp(
        x, u, params, lv, head_fp_q, mode
    )
    delta[X_INDEX["WinTempFp"]] = delta_win_fp

    # ── WinTempSd: 6-term Q_net (T9, WinTempSd_topology_spec.md) ─────────────
    delta_win_sd, _win_sd_q = compute_win_temp_sd(
        x, u, params, lv, head_sd_q, mode
    )
    delta[X_INDEX["WinTempSd"]] = delta_win_sd

    roof_q = build_roof_temp_qbus_inputs(
        head_fd_q,
        head_fp_q,
        q_roof_rad_raw,
        q_roof_conv_raw,
        head_sd_q=head_sd_q,
        td_roof_radiation=head_td_q.td_roof_radiation,
        td_roof_convection=head_td_q.td_roof_convection,
        tp_roof_radiation=head_tp_q.tp_roof_radiation,
        tp_roof_convection=head_tp_q.tp_roof_convection,
    )
    delta_roof, _roof_q_out = compute_roof_temp(
        x, u, params, lv, roof_q, mode
    )
    delta[X_INDEX["RoofTemp"]] = delta_roof

    apply_fast_approx_deltas(delta, x, u, params, mode)
    return delta


def one_step_chtd(
    x: np.ndarray,
    u: np.ndarray,
    params: CHTDParams,
    mode: DefectMode = AS_FOUND,
) -> np.ndarray:
    """Advance the 28-state CHTD model by one Forward-Euler step.

    Parameters
    ----------
    x      : 28-element state vector [°C], shape (28,)
    u      : 54-element input vector, shape (54,)
    params : CHTDParams instance
    mode   : DefectMode.AS_FOUND (default) or CORRECTED

    Returns
    -------
    x_next : ndarray shape (28,) — x + xStep
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (N_X_STATES,):
        raise ValueError(
            f"one_step_chtd: x must have shape ({N_X_STATES},), got {x.shape}"
        )
    delta = compute_chtd_delta(x, u, params, mode)
    return x + delta


# ── Legacy API (deprecated) ───────────────────────────────────────────────────

def legacy_one_step_chtd(
    state: CHTDState,
    inp: CHTDInputs,
    params: CHTDParams,
    UA_coupling: Optional[np.ndarray] = None,
) -> CHTDState:
    """Advance CHTD by one step — DEPRECATED legacy UA-coupling API.

    Does NOT match the Simulink model. Retained for backward compatibility.
    Use one_step_chtd(x, u, params, mode) for all new code.
    """
    T = state.T.copy()
    Dt = params.Dt

    C      = _build_default_C(params)
    UA_amb = _build_default_UA_amb(params)
    A_sol  = _build_default_area(params)
    Qm_vec = _hvac_flow_vector(params, inp)
    T_sup  = _supply_temp_vector(inp)

    Q_conv_amb = UA_amb * (inp.T_amb - T)
    Q_solar    = params.QgainCo * A_sol * inp.I_solar
    Q_hvac     = Qm_vec * params.AirCpAtb * (T_sup - T)

    if UA_coupling is not None:
        delta_T    = T[np.newaxis, :] - T[:, np.newaxis]
        Q_exchange = np.sum(UA_coupling * (-delta_T), axis=1)
    else:
        Q_exchange = np.zeros(N_STATES)

    Q_net  = Q_conv_amb + Q_solar + Q_hvac + Q_exchange
    C_safe = np.where(C > 0, C, 1.0)
    T_new  = T + Dt * Q_net / C_safe

    return CHTDState(T=T_new)


def simulate_chtd(
    t_vec: np.ndarray,
    inputs_list: List[CHTDInputs],
    params: CHTDParams,
    T0: float = 20.0,
    UA_coupling: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run CHTD over a time vector using the legacy API.

    Returns
    -------
    T_hist : (N × 28) temperature history array
    """
    N = len(t_vec)
    state = CHTDState.uniform(T0)
    T_hist = np.empty((N, N_STATES))
    T_hist[0] = state.T

    for k in range(1, N):
        state = legacy_one_step_chtd(state, inputs_list[k - 1], params, UA_coupling)
        T_hist[k] = state.T

    return T_hist


STATE_NAMES = _STATE_NAMES
