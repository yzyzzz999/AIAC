"""IR head-surface temperature → equivalent air temperature (showcase observer)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class IRCompensationResult:
    """Compensated air-equivalent temperature from IR surface measurement."""

    observed_air_temp_c: float
    offset_used_c: float
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def compensate_ir_surface_to_air(
    ir_surface_temp_c: float,
    model_air_temp_c: float,
    mean_radiant_temp_c: Optional[float] = None,
    air_speed_m_s: float = 0.0,
    *,
    offset_c: float = 2.0,
    k_air_speed: float = 0.0,
    k_radiation: float = 0.0,
) -> IRCompensationResult:
    """Map IR surface temperature to an air-equivalent observation for fusion.

    Base (v1):
        T_obs_air = ir_surface_temp_c - offset_c

    Enhanced (when k_* non-zero):
        dynamic_offset = offset_c
            + k_air_speed * air_speed_m_s
            + k_radiation * (ir_surface_temp_c - mean_radiant_temp_c)
    """
    if not math.isfinite(ir_surface_temp_c):
        raise ValueError("ir_surface_temp_c must be finite")
    if not math.isfinite(model_air_temp_c):
        raise ValueError("model_air_temp_c must be finite")

    dynamic_offset = float(offset_c)
    enhancement_terms: Dict[str, float] = {}

    if k_air_speed != 0.0:
        spd = max(0.0, float(air_speed_m_s))
        term = float(k_air_speed) * spd
        dynamic_offset += term
        enhancement_terms["air_speed_term_c"] = term

    if k_radiation != 0.0 and mean_radiant_temp_c is not None:
        if math.isfinite(mean_radiant_temp_c):
            term = float(k_radiation) * (
                float(ir_surface_temp_c) - float(mean_radiant_temp_c)
            )
            dynamic_offset += term
            enhancement_terms["radiation_term_c"] = term
        else:
            enhancement_terms["radiation_term_c"] = 0.0
            enhancement_terms["radiation_skipped"] = 1.0

    observed = float(ir_surface_temp_c) - dynamic_offset
    method = "constant_offset"
    if enhancement_terms:
        method = "dynamic_offset"

    return IRCompensationResult(
        observed_air_temp_c=observed,
        offset_used_c=dynamic_offset,
        method=method,
        diagnostics={
            "base_offset_c": float(offset_c),
            "model_air_temp_c": float(model_air_temp_c),
            "mean_radiant_temp_c": (
                float(mean_radiant_temp_c)
                if mean_radiant_temp_c is not None
                else None
            ),
            "air_speed_m_s": float(air_speed_m_s),
            "k_air_speed": float(k_air_speed),
            "k_radiation": float(k_radiation),
            **enhancement_terms,
        },
    )
