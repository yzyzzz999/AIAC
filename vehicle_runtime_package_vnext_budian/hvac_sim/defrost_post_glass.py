"""Post-glass defrost effective temperature (minimal mixing model).

Estimates the air temperature *after* passing over the windshield glass,
for use in head-zone effective-temperature diagnostics. Does **not** modify
CHTD ``HeadTempFd`` equations directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_TEMP_MIN_C = -40.0
_TEMP_MAX_C = 90.0


@dataclass
class DefrostPostGlassInputs:
    """Inputs for post-glass defrost mixing."""

    tma_def_outlet_c: float
    windshield_glass_temp_c: float
    defrost_flow_m3h: float
    cabin_air_temp_c: Optional[float] = None
    vehicle_speed_kph: Optional[float] = None


@dataclass
class DefrostPostGlassParams:
    """Mixing efficiency parameters (showcase defaults)."""

    eta_base: float = 0.0
    eta_flow_gain: float = 0.0015
    eta_min: float = 0.0
    eta_max: float = 0.85
    vehicle_speed_gain: float = 0.0008


@dataclass(frozen=True)
class DefrostPostGlassResult:
    """Post-glass defrost air temperature estimate."""

    t_after_glass_c: float
    eta: float
    provenance: Dict[str, Any] = field(default_factory=dict)


def _finite_or(value: Optional[float], fallback: float) -> float:
    if value is None:
        return float(fallback)
    v = float(value)
    if not math.isfinite(v):
        return float(fallback)
    return v


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def estimate_defrost_post_glass_temperature(
    inputs: DefrostPostGlassInputs,
    params: Optional[DefrostPostGlassParams] = None,
) -> DefrostPostGlassResult:
    """Mix defrost outlet air with windshield glass surface temperature.

    .. math::

        \\eta = \\mathrm{clamp}(\\eta_{base} + g_f Q + g_v v, \\eta_{min}, \\eta_{max})
        T_{after} = T_{glass} + \\eta (T_{ma,def} - T_{glass})

    Non-finite inputs fall back to the last known good scalar (glass → outlet → 20 °C).
    """
    p = params or DefrostPostGlassParams()
    fallback_chain = [
        inputs.windshield_glass_temp_c,
        inputs.tma_def_outlet_c,
        inputs.cabin_air_temp_c,
        20.0,
    ]
    base_fallback = next(
        (float(v) for v in fallback_chain if v is not None and math.isfinite(float(v))),
        20.0,
    )

    t_outlet = _finite_or(inputs.tma_def_outlet_c, base_fallback)
    t_glass = _finite_or(inputs.windshield_glass_temp_c, t_outlet)
    flow = max(0.0, _finite_or(inputs.defrost_flow_m3h, 0.0))
    v_kph = max(0.0, _finite_or(inputs.vehicle_speed_kph, 0.0))

    eta_raw = (
        float(p.eta_base)
        + float(p.eta_flow_gain) * flow
        + float(p.vehicle_speed_gain) * v_kph
    )
    eta = _clamp(eta_raw, float(p.eta_min), float(p.eta_max))

    t_after = t_glass + eta * (t_outlet - t_glass)
    t_after = _clamp(t_after, _TEMP_MIN_C, _TEMP_MAX_C)

    provenance: Dict[str, Any] = {
        "method": "defrost_post_glass_mix",
        "eta": eta,
        "eta_raw": eta_raw,
        "t_glass_c": t_glass,
        "tma_def_outlet_c": t_outlet,
        "defrost_flow_m3h": flow,
        "vehicle_speed_kph": v_kph,
        "params": {
            "eta_base": p.eta_base,
            "eta_flow_gain": p.eta_flow_gain,
            "eta_min": p.eta_min,
            "eta_max": p.eta_max,
            "vehicle_speed_gain": p.vehicle_speed_gain,
        },
    }
    if not math.isfinite(float(inputs.tma_def_outlet_c)):
        provenance["fallback"] = "non_finite_tma_def_outlet"
    if not math.isfinite(float(inputs.windshield_glass_temp_c)):
        provenance["fallback_glass"] = "non_finite_glass_temp"

    return DefrostPostGlassResult(
        t_after_glass_c=float(t_after),
        eta=float(eta),
        provenance=provenance,
    )
