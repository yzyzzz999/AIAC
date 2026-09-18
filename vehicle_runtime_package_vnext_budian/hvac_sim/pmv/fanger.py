"""Fanger PMV / PPD per ISO 7730.

Independent implementation from the standard heat-balance equations.
"""

from __future__ import annotations

import math

# ISO 7730: 1 met = 58.15 W/m2 (ASHRAE 55 / project reference PMV doc)
_MET_TO_W_M2 = 58.15

# Clothing insulation: 1 clo = 0.155 m2K/W
_CLO_TO_M2K_W = 0.155

# Radiation constant in ISO 7730 clothing exchange [W/(m2*K4)]
_RAD_CONST = 3.96e-8

_MAX_TCL_ITER = 150
_TCL_TOL_C = 0.00015


def _vapour_pressure_pa(air_temp_c: float, relative_humidity_pct: float) -> float:
    """Partial water vapour pressure [Pa] from dry-bulb temperature and RH."""
    rh = max(0.0, min(100.0, relative_humidity_pct))
    return rh * 10.0 * math.exp(16.6536 - 4030.183 / (air_temp_c + 235.0))


def _clothing_area_factor(clo: float) -> tuple[float, float]:
    """Return clothing insulation Icl [m2K/W] and clothing area factor fcl."""
    icl = max(0.0, clo) * _CLO_TO_M2K_W
    if icl <= 0.078:
        fcl = 1.0 + 1.29 * icl
    else:
        fcl = 1.05 + 0.645 * icl
    return icl, fcl


def _solve_clothing_temp_c(
    ta: float,
    tr: float,
    vel: float,
    mw: float,
    icl: float,
    fcl: float,
) -> tuple[float, float]:
    """Iteratively solve clothing surface temperature [deg C] and hc."""
    if icl <= 0.0:
        hcf = 12.1 * math.sqrt(max(0.0, vel))
        hcn = 2.38 * abs(35.7 - 0.028 * mw - ta) ** 0.25
        return ta, max(hcf, hcn)

    ta_k = ta + 273.15
    tr_k = tr + 273.15
    hcf = 12.1 * math.sqrt(max(0.0, vel))
    tcla = ta_k + (35.5 - ta) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * ta_k
    p5 = 308.7 - 0.028 * mw + p2 * (tr_k / 100.0) ** 4

    xn = tcla / 100.0
    xf = xn
    hc = hcf
    for _ in range(_MAX_TCL_ITER):
        xf = 0.5 * (xf + xn)
        hcn = 2.38 * abs(100.0 * xf - ta_k) ** 0.25
        hc = max(hcf, hcn)
        xn_next = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        if not math.isfinite(xn_next):
            break
        if abs(xn_next - xf) < _TCL_TOL_C:
            xn = xn_next
            break
        xn = xn_next

    return 100.0 * xn - 273.15, hc


def _ppd_from_pmv(pmv: float) -> float:
    """PPD [%] from PMV (ISO 7730)."""
    pmv = max(-10.0, min(10.0, pmv))
    exponent = -0.03353 * pmv**4 - 0.2179 * pmv**2
    exponent = max(exponent, -700.0)
    return 100.0 - 95.0 * math.exp(exponent)


def pmv_ppd(
    ta: float,
    tr: float,
    vel: float,
    rh: float,
    met: float,
    clo: float,
) -> tuple[float, float]:
    """Compute Fanger PMV and PPD.

    Units:
        ta/tr: deg C
        vel: m/s
        rh: %
        met: met
        clo: clo
    """
    ta = float(ta)
    tr = float(tr)
    vel = max(0.01, float(vel))
    met = max(0.5, float(met))
    clo = max(0.0, float(clo))

    pa = _vapour_pressure_pa(ta, rh)
    mw = met * _MET_TO_W_M2
    icl, fcl = _clothing_area_factor(clo)

    tcl, hc = _solve_clothing_temp_c(ta, tr, vel, mw, icl, fcl)
    tr_k = tr + 273.15
    tcl_k = tcl + 273.15

    # ISO 7730 heat-loss terms with external work W=0:
    # skin diffusion, regulatory sweating, latent respiration, dry respiration,
    # radiation, and convection.
    hl1 = 3.05e-3 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - _MET_TO_W_M2) if mw > _MET_TO_W_M2 else 0.0
    hl3 = 1.7e-5 * mw * (5867.0 - pa)
    hl4 = 1.4e-3 * mw * (34.0 - ta)
    hl5 = _RAD_CONST * fcl * (tcl_k**4 - tr_k**4)
    hl6 = fcl * hc * (tcl - ta)

    thermal_load = mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6
    pmv = (0.303 * math.exp(-0.036 * mw) + 0.028) * thermal_load
    ppd = _ppd_from_pmv(pmv)
    return pmv, ppd
