"""CHTD calibration parameters — Simulink-aligned, Phase 3.5 schema.

Canonical fields follow Simulink UpdateParam.m naming convention
(信号命名规范(1).xlsx):
  CHTD_{Name}_P  — scalar parameter
  CHTD_{Name}_M  — 1-D LUT map (shape (7,), breakpoints = CHTD_AmbT_X)
  CHTD_{Name}_X  — LUT breakpoint axis

Deprecated fields (no CHTD_ prefix) are kept verbatim for backward
compatibility with old thermal.py (pre-T6 rewrite).  They are synchronised
with their canonical counterparts in __post_init__.

Only CHTD_AirCpAtb_P = 1006.0 is a confirmed physical constant.
All other numeric values are placeholder = 1.0 until calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional

import numpy as np


# ── Module-level helpers ─────────────────────────────────────────────────────

_N_LUT = 7   # length of all LUT map arrays


def ones_lut() -> np.ndarray:
    """Return a placeholder all-ones LUT of length 7."""
    return np.ones(_N_LUT, dtype=float)


def as_lut(value: float) -> np.ndarray:
    """Convert scalar to a constant LUT of length 7."""
    return np.full(_N_LUT, float(value), dtype=float)


def validate_lut_shapes(p: "CHTDParams") -> None:
    """Raise ValueError if any *_M field does not have shape (7,)."""
    for f in dc_fields(p):
        if f.name.endswith("_M"):
            v = getattr(p, f.name)
            if not isinstance(v, np.ndarray) or v.shape != (_N_LUT,):
                raise ValueError(
                    f"CHTDParams.{f.name}: expected ndarray shape ({_N_LUT},), "
                    f"got {type(v).__name__} shape {getattr(v, 'shape', '?')}"
                )


def _lut() -> np.ndarray:
    return ones_lut()


# ── Parameter dataclass ──────────────────────────────────────────────────────

@dataclass
class CHTDParams:
    """CHTD calibration parameters, Phase 3.5 Simulink-aligned schema."""

    # ── LUT breakpoint axis ──────────────────────────────────────────────
    CHTD_AmbT_X: np.ndarray = field(
        default_factory=lambda: np.array(
            [-20., -10., 0., 10., 20., 30., 40.], dtype=float
        )
    )

    # ── Global scalar parameters ─────────────────────────────────────────
    CHTD_Dt_P:          float = 1.0
    CHTD_QgainCo_P:     float = 1.0
    CHTD_QlossCo_P:     float = -1.0
    CHTD_CHTDSdlTi_P:   float = 1.0
    CHTD_AirCpAtb_P:    float = 1006.0   # confirmed physical constant
    CHTD_HeadAirVAtb_P: float = 1.0
    CHTD_FeetAirVAtb_P: float = 1.0

    # ── Shared zone geometry ─────────────────────────────────────────────
    CHTD_HeadAreaAtb_P: float = 1.0   # shared for all head-zone inter-zone couplings
    CHTD_FeetAreaAtb_P: float = 1.0   # shared for all feet-zone inter-zone couplings

    # ── Hood zone ────────────────────────────────────────────────────────
    CHTD_HoodMassAtb_P: float = 1.0
    CHTD_HoodCpAtb_P:   float = 1.0
    CHTD_HoodAreaAtb_P: float = 1.0

    # ── CabinFrnt zone ───────────────────────────────────────────────────
    CHTD_CabinFrntMassAtb_P: float = 1.0
    CHTD_CabinFrntCpAtb_P:   float = 1.0
    CHTD_CabinFrntAreaAtb_P: float = 1.0

    # ── Console zone ─────────────────────────────────────────────────────
    CHTD_ConsoleMassAtb_P: float = 1.0
    CHTD_ConsoleCpAtb_P:   float = 1.0
    CHTD_ConsoleAreaAtb_P: float = 1.0

    # ── ConsoleTemp LUT maps (9) — Phase 4.12 ────────────────────────────
    CHTD_CabinFrntConsoleRadCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleSolarRadCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleWSRadCo_M:        np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleFdRadCo_M:        np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleFpRadCo_M:        np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleFeetFdRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleFeetFdConvCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleFeetFpRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_ConsoleFeetFpConvCo_M:   np.ndarray = field(default_factory=_lut)

    # ── Cabin air zones (6) ──────────────────────────────────────────────
    CHTD_CabinFdMassAtb_P: float = 1.0
    CHTD_CabinFdCpAtb_P:   float = 1.0
    CHTD_CabinFdAreaAtb_P: float = 1.0
    CHTD_CabinFpMassAtb_P: float = 1.0
    CHTD_CabinFpCpAtb_P:   float = 1.0
    CHTD_CabinFpAreaAtb_P: float = 1.0
    CHTD_CabinSdMassAtb_P: float = 1.0
    CHTD_CabinSdCpAtb_P:   float = 1.0
    CHTD_CabinSdAreaAtb_P: float = 1.0
    CHTD_CabinSpMassAtb_P: float = 1.0
    CHTD_CabinSpCpAtb_P:   float = 1.0
    CHTD_CabinSpAreaAtb_P: float = 1.0
    CHTD_CabinTdMassAtb_P: float = 1.0
    CHTD_CabinTdCpAtb_P:   float = 1.0
    CHTD_CabinTdAreaAtb_P: float = 1.0
    CHTD_CabinTpMassAtb_P: float = 1.0
    CHTD_CabinTpCpAtb_P:   float = 1.0
    CHTD_CabinTpAreaAtb_P: float = 1.0

    # ── Window zones (6) ─────────────────────────────────────────────────
    CHTD_WinFdMassAtb_P: float = 1.0
    CHTD_WinFdCpAtb_P:   float = 1.0
    CHTD_WinFdAreaAtb_P: float = 1.0
    CHTD_WinFpMassAtb_P: float = 1.0
    CHTD_WinFpCpAtb_P:   float = 1.0
    CHTD_WinFpAreaAtb_P: float = 1.0
    CHTD_WinSdMassAtb_P: float = 1.0
    CHTD_WinSdCpAtb_P:   float = 1.0
    CHTD_WinSdAreaAtb_P: float = 1.0
    CHTD_WinSpMassAtb_P: float = 1.0
    CHTD_WinSpCpAtb_P:   float = 1.0
    CHTD_WinSpAreaAtb_P: float = 1.0   # also used as HeadTempSp leakage flow scale
    CHTD_WinTdMassAtb_P: float = 1.0
    CHTD_WinTdCpAtb_P:   float = 1.0
    CHTD_WinTdAreaAtb_P: float = 1.0
    CHTD_WinTpMassAtb_P: float = 1.0
    CHTD_WinTpCpAtb_P:   float = 1.0
    CHTD_WinTpAreaAtb_P: float = 1.0

    # ── Roof zone ────────────────────────────────────────────────────────
    CHTD_RoofMassAtb_P: float = 1.0
    CHTD_RoofCpAtb_P:   float = 1.0
    CHTD_RoofAreaAtb_P: float = 1.0

    # ── HoodTemp LUT maps (6) ────────────────────────────────────────────
    CHTD_RawAmbHoodRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_RawAmbHoodConvCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_HoodAmbRadCo_M:       np.ndarray = field(default_factory=_lut)
    CHTD_HoodAmbConvCo_M:      np.ndarray = field(default_factory=_lut)
    CHTD_HoodCabinFrntRadCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_HoodSolarRadCo_M:     np.ndarray = field(default_factory=_lut)

    # ── HeadTempFd LUT maps (20) ─────────────────────────────────────────
    CHTD_FdvFdConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_FdfFdConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_FdDefFdConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_FdWinFdConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_FdWinFdRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_WSFdRadCo_M:      np.ndarray = field(default_factory=_lut)
    CHTD_FdCabinFdConvCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_FdCabinFdRadCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_FdFeetFdConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_FdFeetFdRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_FdFpConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_FdFpRadCo_M:      np.ndarray = field(default_factory=_lut)
    CHTD_FdSdConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_FdSdRadCo_M:      np.ndarray = field(default_factory=_lut)
    CHTD_FdRoofConvCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_FdRoofRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_FdAmbConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_FdAmbRadCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_FdSolarRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_FdLeakageCo_M:    np.ndarray = field(default_factory=_lut)
    # FeetTempFp CORRECTED leakage (AS_FOUND uses FeetFdLeakageCo — defect D3).
    CHTD_FeetFpLeakageCo_M: np.ndarray = field(default_factory=_lut)

    # ── CabinTempFd LUT maps (3) ─────────────────────────────────────────
    CHTD_CabinFdAmbRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_CabinFdAmbConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_CabinFdSolarRadCo_M: np.ndarray = field(default_factory=_lut)
    # WinTempFd CORRECTED solar (AS_FOUND uses CabinFdSolarRadCo — defect D4).
    CHTD_WinFdSolarRadCo_M: np.ndarray = field(default_factory=_lut)

    # ── CabinTempFp LUT maps (3) ─────────────────────────────────────────
    # NOTE: Simulink AS_FOUND contains a known cross-zone LUT copy defect in Cabin
    # non-Fd zones. Python uses zone-specific canonical names below (corrected),
    # even when placeholder values are all-ones.
    CHTD_CabinFpAmbRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_CabinFpAmbConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_CabinFpSolarRadCo_M: np.ndarray = field(default_factory=_lut)

    # ── CabinTempSd LUT maps (3) ─────────────────────────────────────────
    CHTD_CabinSdAmbRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_CabinSdAmbConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_CabinSdSolarRadCo_M: np.ndarray = field(default_factory=_lut)

    # ── CabinTempSp LUT maps (3) ─────────────────────────────────────────
    CHTD_CabinSpAmbRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_CabinSpAmbConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_CabinSpSolarRadCo_M: np.ndarray = field(default_factory=_lut)

    # ── CabinTempTp LUT maps (3) ─────────────────────────────────────────
    # Simulink AS_FOUND (Defect D_CabinTp): CHTD_CabinTpAmbRadCo_M block table reads
    # CHTD_CabinSpAmbRadCo_M. Python uses zone-specific names below; placeholder=1.0
    # is numerically equivalent until real LUT data is loaded.
    CHTD_CabinTpAmbRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_CabinTpAmbConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_CabinTpSolarRadCo_M: np.ndarray = field(default_factory=_lut)

    # ── WinTempTp LUT maps (3) ───────────────────────────────────────────
    CHTD_WinTpAmbRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_WinTpAmbConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_WinTpSolarRadCo_M: np.ndarray = field(default_factory=_lut)

    # Legacy/AS_FOUND defect name (missing "_M") for CabinSp ambient convection.
    # This is NOT a canonical Python parameter; use CHTD_CabinSpAmbConvCo_M.
    CHTD_CabinSpAmbConvCo: Optional[np.ndarray] = None

    # ── HeadTempSp LUT maps (16) — Phase 3.5 ─────────────────────────────
    # §5.1 HVAC duct convection (4 ducts)
    CHTD_SpvSpConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_SpfSpConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_RearSpvSpConvCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_RearSpfSpConvCo_M: np.ndarray = field(default_factory=_lut)  # DEFECT D1: T_src=HeadTempSd
    # §5.3 HeadTempTp coupling (2)
    CHTD_SpTpRadCo_M:       np.ndarray = field(default_factory=_lut)
    CHTD_SpTpConvCo_M:      np.ndarray = field(default_factory=_lut)
    # §5.4 FeetTempSp coupling (2)
    CHTD_SpFeetSpRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_SpFeetSpConvCo_M:  np.ndarray = field(default_factory=_lut)
    # §5.5 CabinSp + WinSp + Roof coupling (6)
    CHTD_SpCabinSpRadCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_SpCabinSpConvCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_SpWinSpRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_SpWinSpConvCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_SpRoofRadCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_SpRoofConvCo_M:    np.ndarray = field(default_factory=_lut)
    # §5.7 Solar (1)
    CHTD_SpSolarRadCo_M:    np.ndarray = field(default_factory=_lut)
    # §5.8 Leakage through WinSp (1)
    CHTD_SpLeakageCo_M:     np.ndarray = field(default_factory=_lut)

    # ── HeadTempTd LUT maps (13) — T11 (HeadTempTd_topology_spec.md) ────────
    # DEFECT D1 (AS_FOUND): roof convection block uses CHTD_SdRoofConvCo_M (Sd zone
    # copy-paste); CHTD_TdRoofConvCo_M does not exist in Simulink. Python reads
    # CHTD_SdRoofConvCo_M for HeadTempTd roof conv; placeholder=1.0 equivalent.
    CHTD_TdvTdConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_TdfTdConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_TdTpRadCo_M:       np.ndarray = field(default_factory=_lut)
    CHTD_TdTpConvCo_M:      np.ndarray = field(default_factory=_lut)
    CHTD_TdFeetTdRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_TdFeetTdConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_TdCabinTdRadCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_TdCabinTdConvCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_TdWinTdRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_TdWinTdConvCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_TdRoofRadCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_TdRoofConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_SdRoofConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_TdSolarRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_TdLeakageCo_M:     np.ndarray = field(default_factory=_lut)

    # ── FeetTempTd LUT maps (4) — T11 (FeetTempTd_topology_spec.md) ──────────
    CHTD_TdfFeetTdConvCo_M:      np.ndarray = field(default_factory=_lut)
    CHTD_FeetTdCabinTdRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_FeetTdCabinTdConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_FeetTdLeakageCo_M:      np.ndarray = field(default_factory=_lut)

    # ── CabinTempTd / WinTempTd LUT maps (6) — T11 (Cabin/Win Td topology specs) ─
    CHTD_CabinTdAmbRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_CabinTdAmbConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_CabinTdSolarRadCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_WinTdAmbRadCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_WinTdAmbConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_WinTdSolarRadCo_M:   np.ndarray = field(default_factory=_lut)

    # ── HeadTempTp LUT maps (12) — T11 (HeadTempTp_topology_spec.md) ───────
    # Simulink AS_FOUND (Defect D1): CHTD_TpRoofRadCo_M block Table = CHTD_TdRoofRadCo_M.
    # Python uses zone-specific CHTD_TpRoofRadCo_M; placeholder=1.0 is numerically equivalent.
    CHTD_TpvTpConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_TpfTpConvCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_TpFeetTpRadCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_TpFeetTpConvCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_TpCabinTpRadCo_M:  np.ndarray = field(default_factory=_lut)
    CHTD_TpCabinTpConvCo_M: np.ndarray = field(default_factory=_lut)
    CHTD_TpWinTpRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_TpWinTpConvCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_TpRoofRadCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_TpRoofConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_TpSolarRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_TpLeakageCo_M:     np.ndarray = field(default_factory=_lut)

    # ── FeetTempSp LUT maps (5) — Phase 4.10 (FeetTempSp_topology_spec.md) ─
    # NOTE: Simulink AS_FOUND contains cross-zone LUT table copy defects:
    # - RearSpf→FeetSp convection uses an Sd LUT in Simulink; Python uses corrected
    #   CHTD_RearSpfFeetSpConvCo_M.
    # - FeetSp→CabinSp convection table reads Sd LUT in Simulink; Python uses corrected
    #   CHTD_FeetSpCabinSpConvCo_M.
    CHTD_SpfFeetSpConvCo_M:        np.ndarray = field(default_factory=_lut)
    CHTD_RearSpfFeetSpConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_FeetSpCabinSpRadCo_M:     np.ndarray = field(default_factory=_lut)
    CHTD_FeetSpCabinSpConvCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_FeetSpLeakageCo_M:        np.ndarray = field(default_factory=_lut)

    # ── FeetTempTp LUT maps (4) — T11 (FeetTempTp_topology_spec.md) ────────
    # Simulink AS_FOUND: CHTD_FeetTpLeakageCo_M block Table = CHTD_FeetTdLeakageCo_M.
    # Python uses zone-specific CHTD_FeetTpLeakageCo_M; placeholder=1.0 equivalent.
    CHTD_TpfFeetTpConvCo_M:       np.ndarray = field(default_factory=_lut)
    CHTD_FeetTpCabinTpRadCo_M:    np.ndarray = field(default_factory=_lut)
    CHTD_FeetTpCabinTpConvCo_M:   np.ndarray = field(default_factory=_lut)
    CHTD_FeetTpLeakageCo_M:       np.ndarray = field(default_factory=_lut)

    # ── DEPRECATED: scalar aliases (old thermal.py backward compat) ──────
    # Set deprecated → canonical (or canonical → deprecated) in __post_init__.
    Dt:          Optional[float] = None   # alias: CHTD_Dt_P
    QgainCo:     Optional[float] = None   # alias: CHTD_QgainCo_P
    QlossCo:     Optional[float] = None   # alias: CHTD_QlossCo_P
    SdlTi:       Optional[float] = None   # alias: CHTD_CHTDSdlTi_P
    AirCpAtb:    Optional[float] = None   # alias: CHTD_AirCpAtb_P
    HeadAirVAtb: Optional[float] = None   # alias: CHTD_HeadAirVAtb_P
    FeetAirVAtb: Optional[float] = None   # alias: CHTD_FeetAirVAtb_P

    # DEPRECATED: per-zone lumped mass [J/K] used by old thermal.py._build_default_C
    HoodMassAtb:      float = 1.0
    CabinFrntMassAtb: float = 1.0
    ConsoleMassAtb:   float = 1.0
    HeadFdMassAtb:    float = 1.0
    HeadFpMassAtb:    float = 1.0
    HeadSdMassAtb:    float = 1.0
    HeadTdMassAtb:    float = 1.0
    HeadTpMassAtb:    float = 1.0
    FeetFdMassAtb:    float = 1.0
    FeetFpMassAtb:    float = 1.0
    FeetSdMassAtb:    float = 1.0
    FeetSpMassAtb:    float = 1.0
    FeetTdMassAtb:    float = 1.0
    FeetTpMassAtb:    float = 1.0
    CabinFdMassAtb:   float = 1.0
    CabinFpMassAtb:   float = 1.0
    CabinSdMassAtb:   float = 1.0
    CabinSpMassAtb:   float = 1.0
    CabinTdMassAtb:   float = 1.0
    CabinTpMassAtb:   float = 1.0
    WinFdMassAtb:     float = 1.0
    WinFpMassAtb:     float = 1.0
    WinSdMassAtb:     float = 1.0
    WinSpMassAtb:     float = 1.0
    WinTdMassAtb:     float = 1.0
    WinTpMassAtb:     float = 1.0
    RoofMassAtb:      float = 1.0

    # DEPRECATED: specific heats (not accessed by old thermal.py directly)
    HoodCpAtb:      float = 1.0
    CabinFrntCpAtb: float = 1.0
    ConsoleCpAtb:   float = 1.0
    CabinFdCpAtb:   float = 1.0
    CabinFpCpAtb:   float = 1.0
    CabinSdCpAtb:   float = 1.0
    CabinSpCpAtb:   float = 1.0
    CabinTdCpAtb:   float = 1.0
    CabinTpCpAtb:   float = 1.0
    WinFdCpAtb:     float = 1.0
    WinFpCpAtb:     float = 1.0
    WinSdCpAtb:     float = 1.0
    WinSpCpAtb:     float = 1.0
    WinTdCpAtb:     float = 1.0
    WinTpCpAtb:     float = 1.0
    RoofCpAtb:      float = 1.0

    # DEPRECATED: surface areas — used by old thermal.py._build_default_area
    HoodAreaAtb:      float = 1.0
    CabinFrntAreaAtb: float = 1.0
    ConsoleAreaAtb:   float = 1.0
    CabinFdAreaAtb:   float = 1.0
    CabinFpAreaAtb:   float = 1.0
    CabinSdAreaAtb:   float = 1.0
    CabinSpAreaAtb:   float = 1.0
    CabinTdAreaAtb:   float = 1.0
    CabinTpAreaAtb:   float = 1.0
    WinFdAreaAtb:     float = 1.0
    WinFpAreaAtb:     float = 1.0
    WinSdAreaAtb:     float = 1.0
    WinSpAreaAtb:     float = 1.0
    WinTdAreaAtb:     float = 1.0
    WinTpAreaAtb:     float = 1.0
    RoofAreaAtb:      float = 1.0

    # DEPRECATED: UA fields — NOT in Simulink; used by old thermal.py._build_default_UA_amb
    HoodUAAtb:      float = 1.0
    CabinFrntUAAtb: float = 1.0
    ConsoleUAAtb:   float = 1.0
    HeadFdUAAtb:    float = 1.0
    HeadFpUAAtb:    float = 1.0
    HeadSdUAAtb:    float = 1.0
    HeadTdUAAtb:    float = 1.0
    HeadTpUAAtb:    float = 1.0
    FeetFdUAAtb:    float = 1.0
    FeetFpUAAtb:    float = 1.0
    FeetSdUAAtb:    float = 1.0
    FeetSpUAAtb:    float = 1.0
    FeetTdUAAtb:    float = 1.0
    FeetTpUAAtb:    float = 1.0
    CabinFdUAAtb:   float = 1.0
    CabinFpUAAtb:   float = 1.0
    CabinSdUAAtb:   float = 1.0
    CabinSpUAAtb:   float = 1.0
    CabinTdUAAtb:   float = 1.0
    CabinTpUAAtb:   float = 1.0
    WinFdUAAtb:     float = 1.0
    WinFpUAAtb:     float = 1.0
    WinSdUAAtb:     float = 1.0
    WinSpUAAtb:     float = 1.0
    WinTdUAAtb:     float = 1.0
    WinTpUAAtb:     float = 1.0
    RoofUAAtb:      float = 1.0

    def __post_init__(self) -> None:
        # Sync deprecated scalar aliases with canonical _P fields.
        # If the deprecated field was explicitly set (not None), it takes
        # precedence and overwrites the canonical value.  Otherwise the
        # deprecated field is populated from the canonical default.
        _DEP_SYNC = [
            ("Dt",          "CHTD_Dt_P"),
            ("QgainCo",     "CHTD_QgainCo_P"),
            ("QlossCo",     "CHTD_QlossCo_P"),
            ("SdlTi",       "CHTD_CHTDSdlTi_P"),
            ("AirCpAtb",    "CHTD_AirCpAtb_P"),
            ("HeadAirVAtb", "CHTD_HeadAirVAtb_P"),
            ("FeetAirVAtb", "CHTD_FeetAirVAtb_P"),
        ]
        for dep, canonical in _DEP_SYNC:
            v = getattr(self, dep)
            if v is not None:
                setattr(self, canonical, float(v))
            else:
                setattr(self, dep, float(getattr(self, canonical)))

        validate_lut_shapes(self)
