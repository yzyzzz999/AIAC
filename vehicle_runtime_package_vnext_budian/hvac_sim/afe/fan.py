"""Fan characteristic model.

Mirrors the 'n-K_F1' subsystem in AFE_Calc.slx.

Fan pressure-rise curve (quadratic in volumetric flow Q):
    dp(Q) = k[0]·Q² + k[1]·Q + k[2]

Coefficients derived from speed n and fan parameters:
    k[0] = -F_a0 · ρ² · n²     (negative: pressure falls with increasing flow)
    k[1] = 0
    k[2] =  F_p0 · n²           (stall pressure at Q = 0)

This gives:  dp = n² · (F_p0  -  F_a0 · ρ² · Q²)
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class FanParams:
    """Characteristic parameters for one fan unit.

    F_p0 : stall-pressure coefficient  [Pa / (rev/s)²]
    F_a0 : flow-resistance coefficient  [Pa·s²/m⁶ / (rev/s)²·(kg/m³)²]
    """
    n0: float = 1000.0  # reference speed, same unit as n
    p0: float = 10.0    # reference stall pressure [Pa]
    a0: float = 1.0     # reference flow-resistance coefficient


def fan_k_coeffs(fan: FanParams, n: float, rho: float) -> np.ndarray:
    """Return the 3-element polynomial coefficient vector [k0, k1, k2].

    Parameters
    ----------
    fan  : fan characteristic parameters (n0, p0, a0 from the Simulink model)
    n    : current fan speed [rev/s]  (converted from gear/duty in the caller)
    rho  : air density [kg/m³]
    """
    if fan.n0 <= 0.0 or n <= 0.0 or rho <= 0.0:
        return np.array([-np.inf, 0.0, 0.0])

    k0 = -fan.a0 * (fan.n0 / n) ** 2 / rho ** 2
    k1 = 0.0
    k2 = fan.p0 * (n / fan.n0) ** 2
    return np.array([k0, k1, k2])


@dataclass
class FanSet:
    """Three fan units in the HVAC system (F1=front-driver, F2=front-pass, S=rear)."""
    F1: FanParams = None   # type: ignore[assignment]
    F2: FanParams = None   # type: ignore[assignment]
    S:  FanParams = None   # type: ignore[assignment]

    def __post_init__(self):
        if self.F1 is None:
            self.F1 = FanParams()
        if self.F2 is None:
            self.F2 = FanParams()
        if self.S is None:
            self.S = FanParams()
