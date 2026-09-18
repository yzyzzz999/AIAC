"""Translate cached CAN signals into the public PMV runtime input mapping."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from hvac_sim.afe.actuator_score_decoder import decode_hvac_actuator_scores
from hvac_sim.air_speed import AirSpeedInputs, estimate_driver_passenger_air_speed


DBC_TO_PMV = {
    "VIU_AmbT": "amb_t_c",
    "RSM_RelHum": "rh_percent",
    "IPB_VehicleSpeed": "vehicle_speed_kph",
    "RSM_LeSolarInten": "solar_driver_w_m2",
    "RSM_RiSolarInten": "solar_passenger_w_m2",
    "AC_FEvapCurrentTemp": "eva_t_c",
    "AC_FBlowSpeedLevel_raw": "front_blower_level",
    "AC_FBlowSpeedLevel": "front_blower_level",
    "AC_FrntInCarT": "ict_c",
    "AC_DrvrFaceVentActT": "driver_face_tma",
    "AC_PassFaceVentActT": "passenger_face_tma",
    "AC_DrvrFootVentActT": "driver_foot_tma",
    "AC_PassFootVentActT": "passenger_foot_tma",
    "AC_Forward_AirFlowTarget": "total_airflow",
    "AC_BLOW_FaceVentilaPosn": "face_vent_posn",
    "AC_FrantFootVentPosn": "foot_vent_posn",
    "AC_DefrostVentilaPosn": "defrost_vent_posn",
    "TA_FdHeadTempLe": "driver_head_left_temp",
    "TA_FdHeadTempRi": "driver_head_right_temp",
    "TA_FpHeadTempLe": "passenger_head_left_temp",
    "TA_FpHeadTempRi": "passenger_head_right_temp",
    "TS_FrntWidTemp": "front_windshield_temp_c",
}


def mean_finite(*values: Optional[float]) -> Optional[float]:
    """Average present finite values, preserving the existing one-sided fallback."""
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def build_air_speed_overrides(signal_cache: Dict[str, float]) -> Dict[str, float]:
    """Estimate symmetric per-seat speed from total m³/h and vent openings."""
    total_flow = signal_cache.get("AC_Forward_AirFlowTarget")
    face_position = signal_cache.get("AC_BLOW_FaceVentilaPosn")
    foot_position = signal_cache.get("AC_FrantFootVentPosn")
    defrost_position = signal_cache.get("AC_DefrostVentilaPosn")
    raw_inputs = (total_flow, face_position, foot_position, defrost_position)
    if not all(value is not None and math.isfinite(value) for value in raw_inputs):
        return {}
    if total_flow <= 0:
        return {}

    decoded = decode_hvac_actuator_scores(face_position, foot_position, defrost_position)
    face_open = decoded["face_open_score"]
    foot_open = decoded["foot_open_score"]
    defrost_open = decoded["defrost_open_score"]
    total_open = face_open + foot_open + defrost_open
    if total_open <= 0:
        return {}

    # No left/right duct-flow signals are available. Preserve the explicit
    # fallback of splitting each branch equally between the two front seats.
    per_seat_scale = float(total_flow) / (2.0 * total_open)
    result = estimate_driver_passenger_air_speed(
        AirSpeedInputs(
            driver_face_flow=per_seat_scale * face_open,
            passenger_face_flow=per_seat_scale * face_open,
            driver_floor_flow=per_seat_scale * foot_open,
            passenger_floor_flow=per_seat_scale * foot_open,
            driver_defrost_flow=per_seat_scale * defrost_open,
            passenger_defrost_flow=per_seat_scale * defrost_open,
            flow_unit="m3h",
        )
    )
    return {
        "driver_air_speed_override_m_s": result.driver_air_speed_m_s,
        "passenger_air_speed_override_m_s": result.passenger_air_speed_m_s,
    }


def set_bypass_overrides(
    pmv: Dict[str, Any],
    signal_cache: Dict[str, float],
    measured_head_air: Dict[str, float],
) -> None:
    """Populate direct-PMV overrides while preserving existing fallbacks."""
    driver_head = mean_finite(
        measured_head_air.get("driver_left"),
        measured_head_air.get("driver_right"),
    )
    passenger_head = mean_finite(
        measured_head_air.get("passenger_left"),
        measured_head_air.get("passenger_right"),
    )
    if driver_head is not None:
        pmv["driver_air_temp_override_c"] = driver_head
    if passenger_head is not None:
        pmv["passenger_air_temp_override_c"] = passenger_head

    in_car_temp = signal_cache.get("AC_FrntInCarT")
    if in_car_temp is not None and math.isfinite(in_car_temp):
        pmv["driver_mrt_override_c"] = in_car_temp
        pmv["passenger_mrt_override_c"] = in_car_temp

    pmv.update(build_air_speed_overrides(signal_cache))


def build_pmv_input(signal_cache: Dict[str, float]) -> Dict[str, Any]:
    """Build the backward-compatible runtime payload from the signal cache."""
    pmv: Dict[str, Any] = {"use_next_state": True, "bypass_models": True}
    tma: Dict[str, float] = {}
    measured_head_air: Dict[str, float] = {}

    tma_fields = {
        "driver_face_tma": "FrntFdvTma",
        "passenger_face_tma": "FrntFpvTma",
        "driver_foot_tma": "FrntFdfTma",
        "passenger_foot_tma": "FrntFpfTma",
    }
    head_fields = {
        "driver_head_left_temp": "driver_left",
        "driver_head_right_temp": "driver_right",
        "passenger_head_left_temp": "passenger_left",
        "passenger_head_right_temp": "passenger_right",
    }
    copied_fields = {
        "front_blower_level",
        "front_windshield_temp_c",
        "face_vent_posn",
        "foot_vent_posn",
        "defrost_vent_posn",
    }

    for dbc_name, value in signal_cache.items():
        pmv_field = DBC_TO_PMV.get(dbc_name)
        if pmv_field is None:
            continue
        if pmv_field in tma_fields:
            tma[tma_fields[pmv_field]] = value
        elif pmv_field in head_fields:
            measured_head_air[head_fields[pmv_field]] = value
        elif pmv_field == "total_airflow":
            # The direct-PMV air-speed builder consumes the canonical CAN name.
            continue
        elif pmv_field in copied_fields:
            pmv[pmv_field] = value
        else:
            pmv[pmv_field] = value

    if tma:
        pmv["tma"] = tma
    if measured_head_air:
        pmv["measured_head_air_temp_c"] = measured_head_air

    set_bypass_overrides(pmv, signal_cache, measured_head_air)
    return pmv
