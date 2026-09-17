"""Windshield glass temperature first-order observer (WinShdTEst)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WindshieldGlassParams:
    """Lumped first-order windshield thermal parameters (showcase defaults)."""

    heat_capacity_j_k: float = 5.0e4
    h_def_w_k_per_m3h: float = 0.08
    h_cabin_w_k: float = 18.0
    h_ext_w_k: float = 28.0
    k_speed_per_kph: float = 0.015
    k_solar_w_per_wm2: float = 0.35
    h_rad_dash_w_k: float = 10.0


@dataclass(frozen=True)
class WindshieldGlassStepResult:
    """One Forward-Euler step of windshield glass temperature."""

    glass_temp_next_c: float
    delta_c: float
    heat_terms: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def estimate_windshield_glass_temp_step(
    glass_temp_c: float,
    tma_def_c: float,
    defrost_flow_m3h: float,
    cabin_temp_c: float,
    dashboard_temp_c: float,
    ambient_temp_c: float,
    vehicle_speed_kph: float,
    solar_w_m2: float,
    params: Optional[WindshieldGlassParams] = None,
    dt_s: float = 1.0,
) -> WindshieldGlassStepResult:
    r"""Advance windshield glass temperature one step.

    .. math::

        C \\frac{dT}{dt} =
          h_{def} \\cdot flow \\cdot (T_{maDef} - T_{glass})
        + h_{cabin} (T_{cabin} - T_{glass})
        + h_{ext} (1 + k_{speed} V_{veh}) (T_{amb} - T_{glass})
        + k_{solar} \\cdot solar
        + h_{rad,dash} (T_{dash} - T_{glass})
    """
    p = params or WindshieldGlassParams()
    tg = float(glass_temp_c)
    for name, val in (
        ("glass_temp_c", glass_temp_c),
        ("tma_def_c", tma_def_c),
        ("cabin_temp_c", cabin_temp_c),
        ("dashboard_temp_c", dashboard_temp_c),
        ("ambient_temp_c", ambient_temp_c),
    ):
        if not math.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")

    flow = max(0.0, float(defrost_flow_m3h))
    veh = max(0.0, float(vehicle_speed_kph))
    solar = max(0.0, float(solar_w_m2))
    dt = max(0.0, float(dt_s))
    c = max(p.heat_capacity_j_k, 1.0)

    q_def = p.h_def_w_k_per_m3h * flow * (float(tma_def_c) - tg)
    q_cabin = p.h_cabin_w_k * (float(cabin_temp_c) - tg)
    q_ext = p.h_ext_w_k * (1.0 + p.k_speed_per_kph * veh) * (
        float(ambient_temp_c) - tg
    )
    q_solar = p.k_solar_w_per_wm2 * solar
    q_dash = p.h_rad_dash_w_k * (float(dashboard_temp_c) - tg)
    q_net = q_def + q_cabin + q_ext + q_solar + q_dash

    delta = dt * q_net / c
    t_next = tg + delta

    if not math.isfinite(t_next):
        raise ValueError("windshield glass step produced non-finite temperature")

    return WindshieldGlassStepResult(
        glass_temp_next_c=float(t_next),
        delta_c=float(delta),
        heat_terms={
            "q_defrost_w": float(q_def),
            "q_cabin_w": float(q_cabin),
            "q_external_w": float(q_ext),
            "q_solar_w": float(q_solar),
            "q_dashboard_w": float(q_dash),
            "q_net_w": float(q_net),
        },
        diagnostics={
            "dt_s": dt,
            "heat_capacity_j_k": c,
            "vehicle_speed_kph": veh,
            "defrost_flow_m3h": flow,
        },
    )
