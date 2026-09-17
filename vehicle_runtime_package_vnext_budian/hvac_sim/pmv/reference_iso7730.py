"""Independent ISO 7730 Fanger PMV/PPD reference (project-local cross-check).

Derived from project documentation:
  - PMV综合模型任务分解.md (heat-balance terms, PPD)
  - 标定计划与完成情况分析/参考PMV算法项目分析.md (smoke-test golden values)

Uses met→W/m² factor 58.15 (reference PMV doc / ASHRAE 55 convention) and the
ISO 7730 clothing-temperature iteration in Kelvin form.  Intentionally omits
the production ``fanger.py`` minimum air-speed clamp (vel ≥ 0.01 m/s).

Not used in the runtime pipeline — regression tests only.
"""

from __future__ import annotations

import math

_MET_W_M2 = 58.15
_CLO_M2K_W = 0.155
_RAD_CONST = 3.96e-8
_MAX_TCL_ITER = 150
_TCL_TOL_C = 0.00015


def _vapour_pressure_pa(air_temp_c: float, relative_humidity_pct: float) -> float:
    rh = max(0.0, min(100.0, relative_humidity_pct))
    return rh * 10.0 * math.exp(16.6536 - 4030.183 / (air_temp_c + 235.0))


def _clothing_area_factor(clo: float) -> tuple[float, float]:
    icl = max(0.0, clo) * _CLO_M2K_W
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
    pmv = max(-10.0, min(10.0, pmv))
    exponent = -0.03353 * pmv**4 - 0.2179 * pmv**2
    exponent = max(exponent, -700.0)
    return 100.0 - 95.0 * math.exp(exponent)


def pmv_ppd_reference(
    ta: float,
    tr: float,
    vel: float,
    rh: float,
    met: float,
    clo: float,
    wme: float = 0.0,
) -> tuple[float, float]:
    """Reference Fanger PMV/PPD (ISO 7730, project golden alignment)."""
    del wme  # reference doc uses wme=0; external work not modelled in pipeline
    ta = float(ta)
    tr = float(tr)
    vel = max(0.0, float(vel))
    met = max(0.5, float(met))
    clo = max(0.0, float(clo))

    pa = _vapour_pressure_pa(ta, rh)
    mw = met * _MET_W_M2
    icl, fcl = _clothing_area_factor(clo)

    tcl, hc = _solve_clothing_temp_c(ta, tr, vel, mw, icl, fcl)
    tr_k = tr + 273.15
    tcl_k = tcl + 273.15

    hl1 = 3.05e-3 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - _MET_W_M2) if mw > _MET_W_M2 else 0.0
    hl3 = 1.7e-5 * mw * (5867.0 - pa)
    hl4 = 1.4e-3 * mw * (34.0 - ta)
    hl5 = _RAD_CONST * fcl * (tcl_k**4 - tr_k**4)
    hl6 = fcl * hc * (tcl - ta)

    thermal_load = mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6
    pmv = (0.303 * math.exp(-0.036 * mw) + 0.028) * thermal_load
    ppd = _ppd_from_pmv(pmv)
    return pmv, ppd
