"""AFE Simulink.Parameter definitions.

Each field corresponds to one _P workspace variable consumed by AFE.slx.
All resistance values carry units of [Pa·s²/m⁶].

Names match exactly the Simulink workspace variables (AFE_FH prefix and _P
suffix stripped).  21 of the 23 used parameters come from param.xlsx;
OsaFlapRessCo and RecFlapRessCo are not in param.xlsx and use placeholder
values pending bench calibration.

Naming note: front-zone hex/pass mix uses "HexUp"/"PassUp"
(AFE_FHFdHexUpRessCo_P), but rear-zone uses "Hex"/"Pass" without "Up"
(AFE_FHSdHexRessCo_P) — this asymmetry exists in the Simulink model itself.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AFEParams:
    """Standard resistance coefficients for every flap and fixed element.

    Resistance formula:  R = StdRess / max(Posn, ε)²
    For hex/pass mix flaps:
        R_Hex  = StdRess / max(Posn, ε)²
        R_Pass = StdRess / max(1 - Posn, ε)²
    """

    # --- Inlet flaps (not in param.xlsx — placeholders) ---
    OsaFlapRessCo: float = 300.0    # AFE_FHOsaFlapRessCo_P  ← missing in param.xlsx
    RecFlapRessCo: float = 300.0    # AFE_FHRecFlapRessCo_P  ← missing in param.xlsx

    # --- EVA core ---
    EvaRessCo: float = 300.0        # AFE_FHEvaRessCo_P = 300

    # --- Front-driver (Fd) zone flaps ---
    FdDefFlapRessCo: float = 30.0   # AFE_FHFdDefFlapRessCo_P
    FdSvFlapRessCo:  float = 1.0    # AFE_FHFdSvFlapRessCo_P
    FdvFlapRessCo:   float = 30.0   # AFE_FHFdvFlapRessCo_P
    FdfFlapRessCo:   float = 20.0   # AFE_FHFdfFlapRessCo_P
    FdHexUpRessCo:   float = 500.0  # AFE_FHFdHexUpRessCo_P  (front uses "Up")
    FdPassUpRessCo:  float = 5.0    # AFE_FHFdPassUpRessCo_P

    # --- Front-passenger (Fp) zone flaps ---
    FpDefFlapRessCo: float = 30.0   # AFE_FHFpDefFlapRessCo_P
    FpSvFlapRessCo:  float = 1.0    # AFE_FHFpSvFlapRessCo_P
    FpvFlapRessCo:   float = 30.0   # AFE_FHFpvFlapRessCo_P
    FpfFlapRessCo:   float = 20.0   # AFE_FHFpfFlapRessCo_P
    FpHexUpRessCo:   float = 500.0  # AFE_FHFpHexUpRessCo_P
    FpPassUpRessCo:  float = 5.0    # AFE_FHFpPassUpRessCo_P

    # --- Side-driver (Sd) zone flaps ---
    SdvFlapRessCo:  float = 150.0   # AFE_FHSdvFlapRessCo_P
    SdfFlapRessCo:  float = 150.0   # AFE_FHSdfFlapRessCo_P
    SdHexRessCo:    float = 500.0   # AFE_FHSdHexRessCo_P    (rear uses no "Up")
    SdPassRessCo:   float = 5.0     # AFE_FHSdPassRessCo_P

    # --- Side-passenger (Sp) zone flaps ---
    SpvFlapRessCo:  float = 150.0   # AFE_FHSpvFlapRessCo_P
    SpfFlapRessCo:  float = 150.0   # AFE_FHSpfFlapRessCo_P
    SpHexRessCo:    float = 500.0   # AFE_FHSpHexRessCo_P
    SpPassRessCo:   float = 5.0     # AFE_FHSpPassRessCo_P

    # Minimum flap position to avoid division-by-zero
    eps: float = 1e-4
