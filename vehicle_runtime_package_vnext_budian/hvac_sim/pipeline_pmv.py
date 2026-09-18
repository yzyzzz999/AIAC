"""Small, side-effect-free helpers for evaluating one seat's PMV."""

from __future__ import annotations

from dataclasses import dataclass

from hvac_sim.pmv.interface import VehicleComfortInputs, VehicleComfortResult, compute_vehicle_pmv


@dataclass(frozen=True)
class SeatPmvConditions:
    """Resolved physical inputs for one occupant position."""

    air_temp_c: float
    mean_radiant_temp_c: float
    air_speed_m_s: float
    relative_humidity_pct: float
    metabolic_rate_met: float
    clothing_insulation_clo: float


def evaluate_seat_pmv(conditions: SeatPmvConditions) -> VehicleComfortResult:
    """Evaluate Fanger PMV/PPD without knowing about pipeline modes."""
    return compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=conditions.air_temp_c,
            mean_radiant_temp_c=conditions.mean_radiant_temp_c,
            air_velocity_m_s=conditions.air_speed_m_s,
            relative_humidity_pct=conditions.relative_humidity_pct,
            metabolic_rate_met=conditions.metabolic_rate_met,
            clothing_insulation_clo=conditions.clothing_insulation_clo,
        )
    )
