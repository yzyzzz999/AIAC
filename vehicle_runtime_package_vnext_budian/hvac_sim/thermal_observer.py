"""Thermal observers for showcase runtime integration (IR, TmaDef, windshield)."""

from __future__ import annotations

from hvac_sim.defrost import (
    TmaDefResult,
    estimate_posn_fdh_from_blend,
    estimate_tma_def,
)
from hvac_sim.glass_temp import (
    WindshieldGlassParams,
    WindshieldGlassStepResult,
    estimate_windshield_glass_temp_step,
)
from hvac_sim.ir_compensation import IRCompensationResult, compensate_ir_surface_to_air

__all__ = [
    "IRCompensationResult",
    "TmaDefResult",
    "WindshieldGlassParams",
    "WindshieldGlassStepResult",
    "compensate_ir_surface_to_air",
    "estimate_posn_fdh_from_blend",
    "estimate_tma_def",
    "estimate_windshield_glass_temp_step",
]
