"""Local air-speed estimation for PMV inputs (T7-2).

Estimates driver/passenger air velocity from volumetric flow, outlet area,
distance attenuation, and distribution ratio. Intended for
``VehicleComfortInputs.air_velocity_m_s`` in the runnable PMV pipeline.

TODO(calibration): outlet areas, seat distances, distribution ratios, and
attenuation k must be calibrated against cabin air-speed measurements.
Missing rear-foot and side-defrost flows degrade to zero contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


PMV_AIR_SPEED_SCALE = 0.7


@dataclass(frozen=True)
class AirSpeedInputs:
    """Inputs for driver/passenger PMV air-speed estimation.

    Primary face-vent flows, such as FrntFdvFlow / FrntFpvFlow, drive the
    estimate. Optional flows default to None and are treated as 0.
    """

    driver_face_flow: float = 0.0
    passenger_face_flow: float = 0.0
    flow_unit: str = "m3h"

    driver_face_outlet_area_m2: float = 0.02
    passenger_face_outlet_area_m2: float = 0.02
    driver_face_distance_m: float = 0.45
    passenger_face_distance_m: float = 0.45
    driver_face_distribution_ratio: float = 1.0
    passenger_face_distribution_ratio: float = 1.0

    driver_floor_flow: Optional[float] = None
    passenger_floor_flow: Optional[float] = None
    driver_floor_outlet_area_m2: float = 0.015
    passenger_floor_outlet_area_m2: float = 0.015
    driver_floor_distance_m: float = 0.55
    passenger_floor_distance_m: float = 0.55
    driver_floor_distribution_ratio: float = 0.5
    passenger_floor_distribution_ratio: float = 0.5

    driver_defrost_flow: Optional[float] = None
    passenger_defrost_flow: Optional[float] = None
    driver_defrost_outlet_area_m2: float = 0.01
    passenger_defrost_outlet_area_m2: float = 0.01
    driver_defrost_distance_m: float = 0.6
    passenger_defrost_distance_m: float = 0.6
    driver_defrost_distribution_ratio: float = 0.2
    passenger_defrost_distribution_ratio: float = 0.2

    rear_driver_foot_flow: Optional[float] = None
    rear_passenger_foot_flow: Optional[float] = None
    rear_foot_outlet_area_m2: float = 0.012
    rear_foot_distance_m: float = 0.7
    rear_foot_distribution_ratio: float = 0.3

    attenuation_k: float = 1.0


@dataclass(frozen=True)
class AirSpeedResult:
    """Estimated local air speeds for Fanger PMV [m/s]."""

    driver_air_speed_m_s: float
    passenger_air_speed_m_s: float


def _clamp_distribution_ratio(ratio: float) -> float:
    if not math.isfinite(ratio):
        return 0.0
    return max(0.0, min(1.0, ratio))


def _safe_nonneg_flow(flow: float) -> float:
    if not math.isfinite(flow):
        return 0.0
    return max(0.0, flow)


def _optional_flow(flow: Optional[float]) -> float:
    if flow is None:
        return 0.0
    return _safe_nonneg_flow(flow)


def _finite_or_zero(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, value)


def flow_to_m3s(flow: float, unit: str = "m3h") -> float:
    """Convert volumetric flow to m3/s."""
    q = _safe_nonneg_flow(flow)
    unit_key = unit.lower()
    if unit_key in ("m3s", "m3/s"):
        return q
    if unit_key in ("m3h", "m3/h"):
        return q / 3600.0
    raise ValueError(f"unsupported flow unit: {unit!r}")


def outlet_velocity(flow: float, outlet_area_m2: float, unit: str = "m3h") -> float:
    """Mean outlet exit velocity [m/s] from volumetric flow and area."""
    if outlet_area_m2 <= 0.0 or not math.isfinite(outlet_area_m2):
        return 0.0
    return flow_to_m3s(flow, unit) / outlet_area_m2


def attenuation(distance_m: float, k: float = 1.0) -> float:
    """Distance attenuation factor: 1 / (1 + k * d^2)."""
    d = max(0.0, distance_m) if math.isfinite(distance_m) else 0.0
    kk = k if math.isfinite(k) and k >= 0.0 else 1.0
    return 1.0 / (1.0 + kk * d * d)


def local_air_speed(
    flow: float,
    outlet_area_m2: float,
    distance_m: float,
    distribution_ratio: float = 1.0,
    unit: str = "m3h",
    k: float = 1.0,
) -> float:
    """Local air speed at occupant [m/s]."""
    v_out = outlet_velocity(flow, outlet_area_m2, unit=unit)
    att = attenuation(distance_m, k=k)
    ratio = _clamp_distribution_ratio(distribution_ratio)
    return _finite_or_zero(v_out * att * ratio)


def _combine_speeds(*speeds: float) -> float:
    """Combine channel speeds by quadrature to avoid double-counting magnitude."""
    total = 0.0
    for v in speeds:
        vv = _finite_or_zero(v)
        total += vv * vv
    return math.sqrt(total)


def estimate_driver_passenger_air_speed(inputs: AirSpeedInputs) -> AirSpeedResult:
    """Estimate driver and passenger PMV air speeds from vent flows."""
    k = inputs.attenuation_k
    unit = inputs.flow_unit

    driver_face = local_air_speed(
        inputs.driver_face_flow,
        inputs.driver_face_outlet_area_m2,
        inputs.driver_face_distance_m,
        inputs.driver_face_distribution_ratio,
        unit=unit,
        k=k,
    )
    driver_floor = local_air_speed(
        _optional_flow(inputs.driver_floor_flow),
        inputs.driver_floor_outlet_area_m2,
        inputs.driver_floor_distance_m,
        inputs.driver_floor_distribution_ratio,
        unit=unit,
        k=k,
    )
    driver_defrost = local_air_speed(
        _optional_flow(inputs.driver_defrost_flow),
        inputs.driver_defrost_outlet_area_m2,
        inputs.driver_defrost_distance_m,
        inputs.driver_defrost_distribution_ratio,
        unit=unit,
        k=k,
    )
    driver_rear_foot = local_air_speed(
        _optional_flow(inputs.rear_driver_foot_flow),
        inputs.rear_foot_outlet_area_m2,
        inputs.rear_foot_distance_m,
        inputs.rear_foot_distribution_ratio,
        unit=unit,
        k=k,
    )

    passenger_face = local_air_speed(
        inputs.passenger_face_flow,
        inputs.passenger_face_outlet_area_m2,
        inputs.passenger_face_distance_m,
        inputs.passenger_face_distribution_ratio,
        unit=unit,
        k=k,
    )
    passenger_floor = local_air_speed(
        _optional_flow(inputs.passenger_floor_flow),
        inputs.passenger_floor_outlet_area_m2,
        inputs.passenger_floor_distance_m,
        inputs.passenger_floor_distribution_ratio,
        unit=unit,
        k=k,
    )
    passenger_defrost = local_air_speed(
        _optional_flow(inputs.passenger_defrost_flow),
        inputs.passenger_defrost_outlet_area_m2,
        inputs.passenger_defrost_distance_m,
        inputs.passenger_defrost_distribution_ratio,
        unit=unit,
        k=k,
    )
    passenger_rear_foot = local_air_speed(
        _optional_flow(inputs.rear_passenger_foot_flow),
        inputs.rear_foot_outlet_area_m2,
        inputs.rear_foot_distance_m,
        inputs.rear_foot_distribution_ratio,
        unit=unit,
        k=k,
    )

    driver_air_speed = _combine_speeds(
        driver_face, driver_floor, driver_defrost, driver_rear_foot
    )
    passenger_air_speed = _combine_speeds(
        passenger_face, passenger_floor, passenger_defrost, passenger_rear_foot
    )

    return AirSpeedResult(
        driver_air_speed_m_s=driver_air_speed * PMV_AIR_SPEED_SCALE,
        passenger_air_speed_m_s=passenger_air_speed * PMV_AIR_SPEED_SCALE,
    )
