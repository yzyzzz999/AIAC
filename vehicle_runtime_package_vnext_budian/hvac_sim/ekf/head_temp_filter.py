"""1D Kalman fusion of CHTD head air temperature with IR surface temperature (T9-4 demo)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from hvac_sim.ir_compensation import compensate_ir_surface_to_air
from hvac_sim.occupant import ImageModuleInputs, SeatOccupantInput


@dataclass(frozen=True)
class HeadTempFilterResult:
    """Fused head air-temperature estimate for one seat."""

    fused_temp_c: float
    model_temp_c: float
    observation_temp_c: Optional[float]
    innovation_c: Optional[float]
    kalman_gain: float
    variance: float
    valid: bool
    source: str


class HeadTempKalmanFilter:
    """Scalar Kalman filter with CHTD model temperature as the predict step."""

    def __init__(self, initial_variance: float = 1.0) -> None:
        self._initial_variance = max(0.0, float(initial_variance))
        self._x: Optional[float] = None
        self._variance = self._initial_variance

    @property
    def variance(self) -> float:
        return self._variance

    def predict(self, model_temp: float, process_var: float) -> None:
        """Set state to the CHTD model air temperature and add process noise."""
        if not math.isfinite(model_temp):
            raise ValueError("model_temp must be finite")
        q = max(0.0, float(process_var))
        self._x = float(model_temp)
        self._variance = self._variance + q

    def update(
        self,
        ir_surface_temp_c: float,
        measurement_var: float,
        surface_to_air_offset_c: float,
        *,
        mean_radiant_temp_c: Optional[float] = None,
        air_speed_m_s: float = 0.0,
        k_air_speed: float = 0.0,
        k_radiation: float = 0.0,
    ) -> HeadTempFilterResult:
        """Fuse IR surface temperature after compensating to equivalent air temp."""
        if self._x is None:
            raise RuntimeError("predict must be called before update")
        if not math.isfinite(ir_surface_temp_c):
            raise ValueError("ir_surface_temp_c must be finite")

        model_temp = self._x
        comp = compensate_ir_surface_to_air(
            ir_surface_temp_c,
            model_temp,
            mean_radiant_temp_c,
            air_speed_m_s,
            offset_c=surface_to_air_offset_c,
            k_air_speed=k_air_speed,
            k_radiation=k_radiation,
        )
        z_air = comp.observed_air_temp_c
        r = max(float(measurement_var), 1e-12)
        innovation = z_air - model_temp
        kalman_gain = self._variance / (self._variance + r)
        fused = model_temp + kalman_gain * innovation
        self._variance = (1.0 - kalman_gain) * self._variance
        self._x = fused

        return HeadTempFilterResult(
            fused_temp_c=fused,
            model_temp_c=model_temp,
            observation_temp_c=z_air,
            innovation_c=innovation,
            kalman_gain=kalman_gain,
            variance=self._variance,
            valid=True,
            source="model_ir_fused",
        )


def _model_only_result(
    model_temp_c: float,
    *,
    variance: float,
    valid: bool,
) -> HeadTempFilterResult:
    return HeadTempFilterResult(
        fused_temp_c=float(model_temp_c),
        model_temp_c=float(model_temp_c),
        observation_temp_c=None,
        innovation_c=None,
        kalman_gain=0.0,
        variance=float(variance),
        valid=valid,
        source="model_only",
    )


def _ir_usable(
    ir_surface_temp_c: Optional[float],
    confidence: Optional[float],
    min_confidence: float,
) -> bool:
    if ir_surface_temp_c is None:
        return False
    if not math.isfinite(ir_surface_temp_c):
        return False
    if confidence is None:
        return True
    if not math.isfinite(confidence):
        return False
    return float(confidence) >= float(min_confidence)


def fuse_head_temperature(
    model_temp_c: float,
    ir_surface_temp_c: Optional[float] = None,
    confidence: Optional[float] = None,
    *,
    process_var: float = 1.0,
    measurement_var: float = 1.0,
    surface_to_air_offset_c: float = 2.0,
    min_confidence: float = 0.5,
    initial_variance: float = 1.0,
    mean_radiant_temp_c: Optional[float] = None,
    air_speed_m_s: float = 0.0,
    k_air_speed: float = 0.0,
    k_radiation: float = 0.0,
) -> HeadTempFilterResult:
    """Fuse one seat's CHTD head air temperature with optional IR observation."""
    if not math.isfinite(model_temp_c):
        return _model_only_result(
            model_temp_c,
            variance=initial_variance + max(0.0, process_var),
            valid=False,
        )

    filt = HeadTempKalmanFilter(initial_variance=initial_variance)
    filt.predict(model_temp_c, process_var)

    if not _ir_usable(ir_surface_temp_c, confidence, min_confidence):
        return _model_only_result(
            model_temp_c,
            variance=filt.variance,
            valid=True,
        )

    return filt.update(
        float(ir_surface_temp_c),
        measurement_var,
        surface_to_air_offset_c,
        mean_radiant_temp_c=mean_radiant_temp_c,
        air_speed_m_s=air_speed_m_s,
        k_air_speed=k_air_speed,
        k_radiation=k_radiation,
    )


def fuse_front_row_heads(
    driver_model_temp_c: float,
    passenger_model_temp_c: float,
    image_inputs: Optional[ImageModuleInputs] = None,
    *,
    process_var: float = 1.0,
    measurement_var: float = 1.0,
    surface_to_air_offset_c: float = 2.0,
    min_confidence: float = 0.5,
    initial_variance: float = 1.0,
    mean_radiant_temp_c: Optional[float] = None,
    air_speed_m_s: float = 0.0,
    k_air_speed: float = 0.0,
    k_radiation: float = 0.0,
) -> Tuple[HeadTempFilterResult, HeadTempFilterResult]:
    """Fuse driver and passenger head temperatures from model temps and image inputs."""
    driver_occ = SeatOccupantInput()
    passenger_occ = SeatOccupantInput()
    if image_inputs is not None:
        driver_occ = image_inputs.driver
        passenger_occ = image_inputs.passenger

    fusion_kw = dict(
        process_var=process_var,
        measurement_var=measurement_var,
        surface_to_air_offset_c=surface_to_air_offset_c,
        min_confidence=min_confidence,
        initial_variance=initial_variance,
        mean_radiant_temp_c=mean_radiant_temp_c,
        air_speed_m_s=air_speed_m_s,
        k_air_speed=k_air_speed,
        k_radiation=k_radiation,
    )
    driver = fuse_head_temperature(
        driver_model_temp_c,
        driver_occ.ir_head_surface_temp_c,
        driver_occ.confidence,
        **fusion_kw,
    )
    passenger = fuse_head_temperature(
        passenger_model_temp_c,
        passenger_occ.ir_head_surface_temp_c,
        passenger_occ.confidence,
        **fusion_kw,
    )
    return driver, passenger
