"""Occupant / image-module input schema for PMV pipeline (T7-8).

Lightweight schema for RGB/DMS/FTE/IR module outputs mapped to driver/passenger
met/clo/valid and IR diagnostics. IR head-surface temperature is diagnostic
only; it does not feed CHTD or EKF in this stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_CLOTHING_CLASS_CLO = {
    "light": 0.3,
    "summer": 0.5,
    "normal": 0.7,
    "winter": 1.0,
}

_MET_MIN = 0.8
_MET_MAX = 2.0
_CLO_MIN = 0.0
_CLO_MAX = 2.0


@dataclass(frozen=True)
class SeatOccupantInput:
    """Per-seat occupant signals from RGB/DMS/FTE/IR modules."""

    occupied: Optional[bool] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    clothing_clo: Optional[float] = None
    clothing_class: Optional[str] = None
    activity_met: Optional[float] = None
    ir_head_surface_temp_c: Optional[float] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class ImageModuleInputs:
    """Driver/passenger image-module bundle."""

    driver: SeatOccupantInput = field(default_factory=SeatOccupantInput)
    passenger: SeatOccupantInput = field(default_factory=SeatOccupantInput)


@dataclass(frozen=True)
class SeatOccupantResolved:
    """Resolved PMV inputs and diagnostics for one seat."""

    occupied: bool
    met: float
    clo: float
    ir_head_surface_temp_c: Optional[float]
    confidence: Optional[float]
    source: Dict[str, Any] = field(default_factory=dict)


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, float(value)))


def _resolve_clo(
    occupant: SeatOccupantInput, default_clo: float
) -> tuple[float, str, Optional[str]]:
    if occupant.clothing_clo is not None:
        return _clamp(occupant.clothing_clo, _CLO_MIN, _CLO_MAX), "clothing_clo", None
    if occupant.clothing_class is not None:
        key = occupant.clothing_class.strip().lower()
        if key in _CLOTHING_CLASS_CLO:
            return _CLOTHING_CLASS_CLO[key], "clothing_class", key
        return _clamp(default_clo, _CLO_MIN, _CLO_MAX), "default_clo", key
    return _clamp(default_clo, _CLO_MIN, _CLO_MAX), "default_clo", None


def _resolve_met(occupant: SeatOccupantInput, default_met: float) -> tuple[float, str]:
    if occupant.activity_met is not None:
        return _clamp(occupant.activity_met, _MET_MIN, _MET_MAX), "activity_met"
    return _clamp(default_met, _MET_MIN, _MET_MAX), "default_met"


def resolve_seat_occupant(
    occupant: SeatOccupantInput,
    default_met: float = 1.0,
    default_clo: float = 0.5,
) -> SeatOccupantResolved:
    """Resolve one seat's PMV met/clo/valid from image-module input."""
    occupied = True if occupant.occupied is None else bool(occupant.occupied)
    met, met_source = _resolve_met(occupant, default_met)
    clo, clo_source, clothing_class_used = _resolve_clo(occupant, default_clo)

    source: Dict[str, Any] = {
        "occupied": "default_true" if occupant.occupied is None else "input",
        "met": met_source,
        "clo": clo_source,
    }
    if clothing_class_used is not None:
        source["clothing_class"] = clothing_class_used

    return SeatOccupantResolved(
        occupied=occupied,
        met=met,
        clo=clo,
        ir_head_surface_temp_c=occupant.ir_head_surface_temp_c,
        confidence=occupant.confidence,
        source=source,
    )


def resolve_image_module_inputs(
    image_inputs: ImageModuleInputs,
    driver_default_met: float = 1.0,
    driver_default_clo: float = 0.5,
    passenger_default_met: float = 1.0,
    passenger_default_clo: float = 0.5,
) -> tuple[SeatOccupantResolved, SeatOccupantResolved]:
    """Resolve driver and passenger occupant inputs."""
    driver = resolve_seat_occupant(
        image_inputs.driver,
        default_met=driver_default_met,
        default_clo=driver_default_clo,
    )
    passenger = resolve_seat_occupant(
        image_inputs.passenger,
        default_met=passenger_default_met,
        default_clo=passenger_default_clo,
    )
    return driver, passenger
