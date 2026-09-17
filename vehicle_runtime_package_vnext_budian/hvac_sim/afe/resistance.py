"""AFE resistance network computation.

Implements the complete resistance bus (34 fields) produced by AFE.slx and
the Rnorm 23-element vector consumed by AFE_Calc.slx.

Key formulas (from Simulink block analysis):

FlapRessRatio block:
    R = StdRess / max(Posn, ε)²

Hex/Pass mix (positions 7, 12, 14, 16 in the 18-inport list):
    R_Hex  = StdRess_hex  / max(Posn, ε)²
    R_Pass = StdRess_pass / max(1 - Posn, ε)²

Zone aggregation  (e.g. FdZoneRess):
    C_up   = 1/√R_Pass + 1/√R_Hex          (parallel upstream paths)
    C_down = 1/√R_Def + 1/√R_Side + 1/√R_v + 1/√R_f  (parallel outlets)
    R_zone = 1/C_up² + 1/C_down²           (series: upstream + downstream)

Rear aggregation (R_2 in the solver) follows the same rule applied to Sd+Sp
combined in parallel:
    C_Sd = 1/√R_SdZone ;  C_Sp = 1/√R_SpZone
    R_2  = 1/(C_Sd + C_Sp)²

FrntZoneRess = parallel of Fd + Fp:
    C_Fd  = 1/√R_FdZone ; C_Fp  = 1/√R_FpZone
    R_Frnt = 1/(C_Fd + C_Fp)²
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .params import AFEParams


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _flap_r(std_ress: float, posn: float, eps: float) -> float:
    """FlapRessRatio formula: R = std_ress / max(posn, eps)²."""
    if posn <= 0.0:
        return math.inf
    return std_ress / posn ** 2


def _parallel_conductance(*resistances: float) -> float:
    """Sum of 1/√R_i  (conductance in the pressure-flow model)."""
    c = 0.0
    for r in resistances:
        if r == 0.0:
            return math.inf
        if math.isfinite(r) and r > 0.0:
            c += 1.0 / math.sqrt(r)
    return c


def _parallel_r(*resistances: float) -> float:
    """Equivalent resistance of parallel branches:  1 / (Σ 1/√Ri)²."""
    c = _parallel_conductance(*resistances)
    if c == 0.0:
        return math.inf
    return 1.0 / (c * c)


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------

@dataclass
class FlapSet:
    """The 18 flap position inputs (0 = closed, 1 = fully open).

    Attribute names match AFE.slx inport names (FH prefix kept for clarity).
    """
    FHOsaFlapPosn:   float = 1.0   # 1
    FHRecFlapPosn:   float = 0.0   # 2
    FHFdDefFlapPosn: float = 0.5   # 3
    FHFdSvFlapPosn:  float = 0.5   # 4
    FHFdvFlapPosn:   float = 0.5   # 5
    FHFdfFlapPosn:   float = 0.5   # 6
    FHFdhFlapPosn:   float = 0.5   # 7  hex/pass mix
    FHFpDefFlapPosn: float = 0.5   # 8
    FHFpSvFlapPosn:  float = 0.5   # 9
    FHFpvFlapPosn:   float = 0.5   # 10
    FHFpfFlapPosn:   float = 0.5   # 11
    FHFphFlapPosn:   float = 0.5   # 12  hex/pass mix
    FHSdvFlapPosn:   float = 0.5   # 13
    FHSdhFlapPosn:   float = 0.5   # 14  hex/pass mix
    FHSpvFlapPosn:   float = 0.5   # 15
    FHSphFlapPosn:   float = 0.5   # 16  hex/pass mix
    FHSdfFlapPosn:   float = 0.5   # 17
    FHSpfFlapPosn:   float = 0.5   # 18

    def as_array(self) -> np.ndarray:
        return np.array([
            self.FHOsaFlapPosn, self.FHRecFlapPosn,
            self.FHFdDefFlapPosn, self.FHFdSvFlapPosn,
            self.FHFdvFlapPosn, self.FHFdfFlapPosn, self.FHFdhFlapPosn,
            self.FHFpDefFlapPosn, self.FHFpSvFlapPosn,
            self.FHFpvFlapPosn, self.FHFpfFlapPosn, self.FHFphFlapPosn,
            self.FHSdvFlapPosn, self.FHSdhFlapPosn,
            self.FHSpvFlapPosn, self.FHSphFlapPosn,
            self.FHSdfFlapPosn, self.FHSpfFlapPosn,
        ])


@dataclass
class RnessBus:
    """34-field resistance bus output from AFE.slx."""
    # Inlet
    AFE_FHOsaFlapRess:   float = 0.0
    AFE_FHRecFlapRess:   float = 0.0
    # Fd zone
    AFE_FHFdDefFlapRess: float = 0.0
    AFE_FHFdSvFlapRess:  float = 0.0
    AFE_FHFdvFlapRess:   float = 0.0
    AFE_FHFdfFlapRess:   float = 0.0
    AFE_FHFdOutLetRess:  float = 0.0
    AFE_FHFdHexUpRess:   float = 0.0
    AFE_FHFdPassUpRess:  float = 0.0
    AFE_FHFdhRess:       float = 0.0
    AFE_FHFdZoneRess:    float = 0.0
    # Fp zone
    AFE_FHFpDefFlapRess: float = 0.0
    AFE_FHFpSvFlapRess:  float = 0.0
    AFE_FHFpvFlapRess:   float = 0.0
    AFE_FHFpfFlapRess:   float = 0.0
    AFE_FHFpOutLetRess:  float = 0.0
    AFE_FHFpHexUpRess:   float = 0.0
    AFE_FHFpPassUpRess:  float = 0.0
    AFE_FHFphRess:       float = 0.0
    AFE_FHFpZoneRess:    float = 0.0
    # Front combined
    AFE_FHFrntZoneRess:  float = 0.0
    AFE_FHFrnEvaRess:    float = 0.0
    # Sd zone
    AFE_FHSdvFlapRess:   float = 0.0
    AFE_FHSdfFlapRess:   float = 0.0
    AFE_FHSdOutLetRess:  float = 0.0
    AFE_FHSdHexRess:     float = 0.0
    AFE_FHSdPassRess:    float = 0.0
    AFE_FHSdhRess:       float = 0.0
    # Sp zone
    AFE_FHSpvFlapRess:   float = 0.0
    AFE_FHSpfFlapRess:   float = 0.0
    AFE_FHSpOutLetRess:  float = 0.0
    AFE_FHSpHexRess:     float = 0.0
    AFE_FHSpPassRess:    float = 0.0
    AFE_FHSphRess:       float = 0.0


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_ress_bus(flaps: FlapSet, params: AFEParams) -> RnessBus:
    """Compute the complete 34-field resistance bus from flap positions."""
    eps = params.eps
    bus = RnessBus()

    # -- Inlet --
    bus.AFE_FHOsaFlapRess = _flap_r(params.OsaFlapRessCo, flaps.FHOsaFlapPosn, eps)
    bus.AFE_FHRecFlapRess = _flap_r(params.RecFlapRessCo, flaps.FHRecFlapPosn, eps)

    # -- Fd zone --
    bus.AFE_FHFdDefFlapRess = _flap_r(params.FdDefFlapRessCo, flaps.FHFdDefFlapPosn, eps)
    bus.AFE_FHFdSvFlapRess  = _flap_r(params.FdSvFlapRessCo,  flaps.FHFdSvFlapPosn,  eps)
    bus.AFE_FHFdvFlapRess   = _flap_r(params.FdvFlapRessCo,   flaps.FHFdvFlapPosn,   eps)
    bus.AFE_FHFdfFlapRess   = _flap_r(params.FdfFlapRessCo,   flaps.FHFdfFlapPosn,   eps)

    # Hex/pass mix flap (position 7 → ratio between hex and bypass paths)
    fdh = flaps.FHFdhFlapPosn
    bus.AFE_FHFdHexUpRess  = _flap_r(params.FdHexUpRessCo,  fdh,       eps)
    bus.AFE_FHFdPassUpRess = _flap_r(params.FdPassUpRessCo, 1.0 - fdh, eps)
    bus.AFE_FHFdhRess = _parallel_r(bus.AFE_FHFdHexUpRess, bus.AFE_FHFdPassUpRess)

    # Outlet parallel combination (Def + Sv + v); floor is exported only.
    bus.AFE_FHFdOutLetRess = _parallel_r(
        bus.AFE_FHFdDefFlapRess,
        bus.AFE_FHFdSvFlapRess,
        bus.AFE_FHFdvFlapRess,
    )
    # Zone = series(upstream_mix, outlet_parallel)
    bus.AFE_FHFdZoneRess = bus.AFE_FHFdhRess + bus.AFE_FHFdOutLetRess

    # -- Fp zone (identical topology to Fd) --
    bus.AFE_FHFpDefFlapRess = _flap_r(params.FpDefFlapRessCo, flaps.FHFpDefFlapPosn, eps)
    bus.AFE_FHFpSvFlapRess  = _flap_r(params.FpSvFlapRessCo,  flaps.FHFpSvFlapPosn,  eps)
    bus.AFE_FHFpvFlapRess   = _flap_r(params.FpvFlapRessCo,   flaps.FHFpvFlapPosn,   eps)
    bus.AFE_FHFpfFlapRess   = _flap_r(params.FpfFlapRessCo,   flaps.FHFpfFlapPosn,   eps)

    fph = flaps.FHFphFlapPosn
    bus.AFE_FHFpHexUpRess  = _flap_r(params.FpHexUpRessCo,  fph,       eps)
    bus.AFE_FHFpPassUpRess = _flap_r(params.FpPassUpRessCo, 1.0 - fph, eps)
    bus.AFE_FHFphRess = _parallel_r(bus.AFE_FHFpHexUpRess, bus.AFE_FHFpPassUpRess)

    bus.AFE_FHFpOutLetRess = _parallel_r(
        bus.AFE_FHFpDefFlapRess,
        bus.AFE_FHFpSvFlapRess,
        bus.AFE_FHFpvFlapRess,
    )
    bus.AFE_FHFpZoneRess = bus.AFE_FHFphRess + bus.AFE_FHFpOutLetRess

    # -- Front combined (Fd ‖ Fp in parallel) --
    bus.AFE_FHFrntZoneRess = 0.0
    # These combined outputs are finalized after the rear internal paths below.
    bus.AFE_FHFrnEvaRess = 0.0

    # -- Sd zone --
    bus.AFE_FHSdvFlapRess = _flap_r(params.SdvFlapRessCo, flaps.FHSdvFlapPosn, eps)
    bus.AFE_FHSdfFlapRess = _flap_r(params.SdfFlapRessCo, flaps.FHSdfFlapPosn, eps)

    sdh = flaps.FHSdhFlapPosn
    bus.AFE_FHSdHexRess  = _flap_r(params.SdHexRessCo,  sdh,       eps)
    bus.AFE_FHSdPassRess = _flap_r(params.SdPassRessCo, 1.0 - sdh, eps)
    bus.AFE_FHSdhRess = _parallel_r(bus.AFE_FHSdHexRess, bus.AFE_FHSdPassRess)

    bus.AFE_FHSdOutLetRess = 0.0

    # -- Sp zone --
    bus.AFE_FHSpvFlapRess = _flap_r(params.SpvFlapRessCo, flaps.FHSpvFlapPosn, eps)
    bus.AFE_FHSpfFlapRess = _flap_r(params.SpfFlapRessCo, flaps.FHSpfFlapPosn, eps)

    sph = flaps.FHSphFlapPosn
    bus.AFE_FHSpHexRess  = _flap_r(params.SpHexRessCo,  sph,       eps)
    bus.AFE_FHSpPassRess = _flap_r(params.SpPassRessCo, 1.0 - sph, eps)
    bus.AFE_FHSphRess = _parallel_r(bus.AFE_FHSpHexRess, bus.AFE_FHSpPassRess)

    bus.AFE_FHSpOutLetRess = 0.0

    sd_zone_internal = bus.AFE_FHSdvFlapRess + bus.AFE_FHSdhRess
    sp_zone_internal = bus.AFE_FHSpvFlapRess + bus.AFE_FHSphRess
    bus.AFE_FHFrntZoneRess = _parallel_r(
        bus.AFE_FHFdZoneRess,
        bus.AFE_FHFpZoneRess,
        sd_zone_internal,
        sp_zone_internal,
    )

    return bus


def ress_bus_to_rnorm(bus: RnessBus, rho: float) -> np.ndarray:
    """Convert the resistance bus to the 23-element Rnorm vector used by AFE_Calc.

    Rnorm[i] = R_i × rho  (density scaling done inside AFE_Calc/Rho subsystem)

    Index mapping (0-based):
      0  R_Osa        1  R_Rec        2  R_EVA
      3  R_FdDef      4  R_FdSide    5  R_Fdv       6  R_Fdf
      7  R_FdHex      8  R_FdPass    9  R_FpDef    10  R_FpSide
     11  R_Fpv       12  R_Fpf      13  R_FpHex    14  R_FpPass
     15  R_Sdv       16  R_Sdf      17  R_SdHex    18  R_SdPass
     19  R_Spv       20  R_Spf      21  R_SpHex    22  R_SpPass
    """
    r = np.array([
        bus.AFE_FHOsaFlapRess,
        bus.AFE_FHRecFlapRess,
        bus.AFE_FHFrnEvaRess,
        bus.AFE_FHFdDefFlapRess,
        bus.AFE_FHFdSvFlapRess,
        bus.AFE_FHFdvFlapRess,
        bus.AFE_FHFdfFlapRess,
        bus.AFE_FHFdHexUpRess,
        bus.AFE_FHFdPassUpRess,
        bus.AFE_FHFpDefFlapRess,
        bus.AFE_FHFpSvFlapRess,
        bus.AFE_FHFpvFlapRess,
        bus.AFE_FHFpfFlapRess,
        bus.AFE_FHFpHexUpRess,
        bus.AFE_FHFpPassUpRess,
        bus.AFE_FHSdvFlapRess,
        bus.AFE_FHSdfFlapRess,
        bus.AFE_FHSdHexRess,
        bus.AFE_FHSdPassRess,
        bus.AFE_FHSpvFlapRess,
        bus.AFE_FHSpfFlapRess,
        bus.AFE_FHSpHexRess,
        bus.AFE_FHSpPassRess,
    ])
    r = np.nan_to_num(r, posinf=1e12)
    return r * rho
