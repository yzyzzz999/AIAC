"""Runtime signal conditioning utilities (vehicle-grade helpers).

Standalone tools — not wired into the main PMV pipeline yet. Each function
returns a result dataclass with provenance for downstream diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from hvac_sim.chtd.bus_index import N_X_STATES, X_INDEX

_TEMP_MIN_C = -40.0
_TEMP_MAX_C = 90.0
_FLOW_MAX_M3H = 600.0

# Ambient filter: below this speed (km/h) upward ambient changes are slowed.
_AMB_LOW_SPEED_KPH = 15.0
_AMB_HIGH_SPEED_KPH = 60.0

# Solar: fast rise / slow decay time constants (seconds).
_SOLAR_TAU_UP_S = 3.0
_SOLAR_TAU_DOWN_S = 45.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _finite_or(value: float, fallback: float) -> float:
    v = float(value)
    return v if math.isfinite(v) else float(fallback)


@dataclass(frozen=True)
class ConditioningResult:
    """Scalar conditioning output."""

    value: float
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateInitResult:
    """Cabin state vector after key-off conditioning."""

    x_init: np.ndarray
    provenance: Dict[str, Any] = field(default_factory=dict)


def clamp_temperature(
    value: float,
    *,
    min_c: float = _TEMP_MIN_C,
    max_c: float = _TEMP_MAX_C,
) -> ConditioningResult:
    """Clamp temperature to vehicle sensor plausible range."""
    raw = _finite_or(value, min_c)
    out = _clamp(raw, float(min_c), float(max_c))
    return ConditioningResult(
        value=out,
        provenance={
            "method": "clamp_temperature",
            "raw": raw,
            "min_c": min_c,
            "max_c": max_c,
            "clamped": out != raw,
        },
    )


def clamp_flow_m3h(
    value: float,
    *,
    min_m3h: float = 0.0,
    max_m3h: float = _FLOW_MAX_M3H,
) -> ConditioningResult:
    """Clamp volumetric flow to non-negative, bounded range."""
    raw = _finite_or(value, min_m3h)
    out = _clamp(raw, float(min_m3h), float(max_m3h))
    return ConditioningResult(
        value=out,
        provenance={
            "method": "clamp_flow_m3h",
            "raw": raw,
            "min_m3h": min_m3h,
            "max_m3h": max_m3h,
            "clamped": out != raw,
        },
    )


def rate_limit(
    prev: float,
    raw: float,
    max_up_per_s: float,
    max_down_per_s: float,
    dt: float,
) -> ConditioningResult:
    """Asymmetric slew-rate limiter."""
    p = _finite_or(prev, raw)
    r = _finite_or(raw, p)
    step = float(dt)
    if step <= 0.0:
        return ConditioningResult(value=p, provenance={"method": "rate_limit", "dt_invalid": True})

    delta = r - p
    if delta > 0.0:
        limited = min(delta, float(max_up_per_s) * step)
    else:
        limited = max(delta, -float(max_down_per_s) * step)
    out = p + limited
    return ConditioningResult(
        value=out,
        provenance={
            "method": "rate_limit",
            "prev": p,
            "raw": r,
            "delta_requested": delta,
            "delta_applied": limited,
            "max_up_per_s": max_up_per_s,
            "max_down_per_s": max_down_per_s,
            "dt_s": step,
        },
    )


def first_order_relax(
    prev: float,
    target: float,
    tau_s: float,
    dt: float,
) -> ConditioningResult:
    """First-order low-pass toward ``target``."""
    p = _finite_or(prev, target)
    t = _finite_or(target, p)
    step = max(float(dt), 0.0)
    tau = max(float(tau_s), 1e-6)
    alpha = 1.0 - math.exp(-step / tau) if step > 0.0 else 0.0
    out = p + alpha * (t - p)
    return ConditioningResult(
        value=out,
        provenance={
            "method": "first_order_relax",
            "prev": p,
            "target": t,
            "tau_s": tau,
            "dt_s": step,
            "alpha": alpha,
        },
    )


def asymmetric_filter(
    prev: float,
    raw: float,
    tau_up_s: float,
    tau_down_s: float,
    dt: float,
) -> ConditioningResult:
    """First-order filter with separate rise/fall time constants."""
    p = _finite_or(prev, raw)
    r = _finite_or(raw, p)
    tau = float(tau_up_s) if r >= p else float(tau_down_s)
    return first_order_relax(p, r, tau, dt)


def ambient_temp_filter(
    prev: float,
    raw: float,
    vehicle_speed_kph: float,
    dt: float,
    *,
    tau_low_speed_s: float = 120.0,
    tau_high_speed_s: float = 25.0,
    max_rise_low_speed_per_s: float = 0.08,
) -> ConditioningResult:
    """Ambient temperature filter with speed-dependent upward allowance.

    At low vehicle speed, ambient readings cannot rise quickly (heat-soak /
    sensor lag). At high speed, the filter relaxes faster toward ``raw``.
    """
    p = _finite_or(prev, raw)
    r = _finite_or(raw, p)
    spd = max(0.0, _finite_or(vehicle_speed_kph, 0.0))
    step = max(float(dt), 0.0)

    if _AMB_HIGH_SPEED_KPH <= _AMB_LOW_SPEED_KPH:
        speed_factor = 1.0 if spd >= _AMB_LOW_SPEED_KPH else 0.0
    else:
        speed_factor = _clamp(
            (spd - _AMB_LOW_SPEED_KPH) / (_AMB_HIGH_SPEED_KPH - _AMB_LOW_SPEED_KPH),
            0.0,
            1.0,
        )

    tau = tau_low_speed_s + (tau_high_speed_s - tau_low_speed_s) * speed_factor
    relaxed = first_order_relax(p, r, tau, step).value

    if r > p and spd < _AMB_LOW_SPEED_KPH:
        max_up = float(max_rise_low_speed_per_s) * step
        out = min(relaxed, p + max_up)
        limited_rise = True
    else:
        out = relaxed
        limited_rise = False

    out = clamp_temperature(out).value
    return ConditioningResult(
        value=out,
        provenance={
            "method": "ambient_temp_filter",
            "prev": p,
            "raw": r,
            "vehicle_speed_kph": spd,
            "speed_factor": speed_factor,
            "tau_s": tau,
            "limited_rise_at_low_speed": limited_rise,
            "dt_s": step,
        },
    )


def solar_fast_up_slow_down(
    prev: float,
    raw: float,
    dt: float,
    *,
    tau_up_s: float = _SOLAR_TAU_UP_S,
    tau_down_s: float = _SOLAR_TAU_DOWN_S,
    max_memory_w_m2: Optional[float] = None,
) -> ConditioningResult:
    """Solar irradiance: fast rise, slow decay; optional peak memory."""
    p = max(0.0, _finite_or(prev, 0.0))
    r = max(0.0, _finite_or(raw, p))
    if max_memory_w_m2 is not None:
        r = min(r, max(0.0, float(max_memory_w_m2)))
    filtered = asymmetric_filter(p, r, tau_up_s, tau_down_s, dt)
    out = max(0.0, filtered.value)
    prov = dict(filtered.provenance)
    prov["method"] = "solar_fast_up_slow_down"
    prov["max_memory_w_m2"] = max_memory_w_m2
    return ConditioningResult(value=out, provenance=prov)


def initialize_cabin_state_after_keyoff(
    previous_x: Sequence[float],
    cabin_sensor_temp: float,
    amb_t: float,
    keyoff_duration_s: float,
    tau_s: float,
) -> StateInitResult:
    """Blend previous CHTD state toward sensor / ambient after key-off.

    Longer ``keyoff_duration_s`` moves states closer to cabin sensor (interior
    zones) or ambient (exterior shell / glass / roof).
    """
    if len(previous_x) != N_X_STATES:
        raise ValueError(f"previous_x must have length {N_X_STATES}, got {len(previous_x)}")

    sensor = _finite_or(cabin_sensor_temp, amb_t)
    amb = _finite_or(amb_t, sensor)
    duration = max(0.0, _finite_or(keyoff_duration_s, 0.0))
    tau = max(float(tau_s), 1e-6)
    blend = 1.0 - math.exp(-duration / tau)

    prev = np.asarray(previous_x, dtype=float)
    if not np.all(np.isfinite(prev)):
        prev = np.where(np.isfinite(prev), prev, sensor)

    exterior = {
        "HoodTemp",
        "WinTempFd",
        "WinTempFp",
        "WinTempSd",
        "WinTempSp",
        "WinTempTd",
        "WinTempTp",
        "RoofTemp",
    }

    x_init = prev.copy()
    targets: Dict[str, float] = {}
    for name, idx in X_INDEX.items():
        if name in exterior:
            target = amb
        elif name in ("CabinFrntTemp", "ConsoleTemp"):
            target = 0.5 * sensor + 0.5 * amb
        else:
            target = sensor
        targets[name] = target
        x_init[idx] = prev[idx] * (1.0 - blend) + target * blend

    x_init = np.clip(x_init, _TEMP_MIN_C, _TEMP_MAX_C)
    return StateInitResult(
        x_init=x_init,
        provenance={
            "method": "initialize_cabin_state_after_keyoff",
            "blend_factor": blend,
            "keyoff_duration_s": duration,
            "tau_s": tau,
            "cabin_sensor_temp_c": sensor,
            "amb_t_c": amb,
            "n_states": N_X_STATES,
            "exterior_zones": sorted(exterior),
        },
    )
