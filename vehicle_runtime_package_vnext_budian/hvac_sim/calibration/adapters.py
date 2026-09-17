"""Model adapters for generic calibration objectives (T9-7)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from hvac_sim.air_speed import AirSpeedInputs, estimate_driver_passenger_air_speed
from hvac_sim.chtd.bus_index import N_U, N_X_STATES
from hvac_sim.ekf.head_temp_filter import fuse_head_temperature
from hvac_sim.occupant import ImageModuleInputs, SeatOccupantInput
from hvac_sim.pipeline import (
    PipelineInputs,
    air_speed_inputs_from_chtd_u,
    run_comfort_pipeline_dict,
)

from .schema import CalibrationSchemaError

DEFAULT_RH = 50.0

AIR_SPEED_ADAPTER_PARAM_FIELDS: Dict[str, str] = {
    "attenuation_k": "attenuation_k",
    "driver_face_area_m2": "driver_face_outlet_area_m2",
    "passenger_face_area_m2": "passenger_face_outlet_area_m2",
    "driver_face_distribution_ratio": "driver_face_distribution_ratio",
    "passenger_face_distribution_ratio": "passenger_face_distribution_ratio",
    "driver_floor_distribution_ratio": "driver_floor_distribution_ratio",
    "passenger_floor_distribution_ratio": "passenger_floor_distribution_ratio",
    "driver_defrost_distribution_ratio": "driver_defrost_distribution_ratio",
    "passenger_defrost_distribution_ratio": "passenger_defrost_distribution_ratio",
}

PMV_ADAPTER_PARAM_FIELDS: Dict[str, str] = AIR_SPEED_ADAPTER_PARAM_FIELDS

EKF_ADAPTER_PARAM_FIELDS: Dict[str, str] = {
    "process_var": "process_var",
    "measurement_var": "measurement_var",
    "surface_to_air_offset_c": "surface_to_air_offset_c",
    "initial_variance": "initial_variance",
    "min_confidence": "min_confidence",
}

EKF_ADAPTER_DEFAULTS: Dict[str, float] = {
    "process_var": 1.0,
    "measurement_var": 1.0,
    "surface_to_air_offset_c": 2.0,
    "initial_variance": 1.0,
    "min_confidence": 0.5,
}


class AdapterError(CalibrationSchemaError):
    """Invalid adapter input or parameter mapping."""


class ModelAdapter(ABC):
    """Bridge between prepared calibration samples and a simulation model."""

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """Return ordered parameter names bound to the optimizer vector."""

    @abstractmethod
    def apply_parameters(self, params: np.ndarray) -> None:
        """Store the current optimizer parameter vector."""

    @abstractmethod
    def predict(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """Run the model for one prepared sample and return prediction fields."""


def _normalize_parameter_names(
    parameter_names: Union[Sequence[str], Mapping[str, int]],
) -> List[str]:
    if isinstance(parameter_names, Mapping):
        size = len(parameter_names)
        ordered = [""] * size
        for name, idx in parameter_names.items():
            if not isinstance(idx, int) or idx < 0 or idx >= size:
                raise AdapterError(f"parameter index for {name!r} out of range")
            ordered[idx] = str(name)
        if any(not name for name in ordered):
            raise AdapterError("parameter mapping must cover contiguous indices")
        return ordered
    names = [str(name) for name in parameter_names]
    if len(names) != len(set(names)):
        raise AdapterError("parameter names must be unique")
    return names


def _vector_to_param_dict(
    params: np.ndarray,
    param_names: Sequence[str],
    field_map: Mapping[str, str],
    defaults: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    vec = np.asarray(params, dtype=float).reshape(-1)
    if vec.shape[0] != len(param_names):
        raise AdapterError(
            f"parameter vector length {vec.shape[0]} != "
            f"parameter count {len(param_names)}"
        )
    values = dict(defaults or {})
    for idx, name in enumerate(param_names):
        if name not in field_map:
            raise AdapterError(
                f"unsupported calibration parameter {name!r}; "
                f"supported: {sorted(field_map)}"
            )
        values[field_map[name]] = float(vec[idx])
    return values


def air_speed_inputs_from_prepared_sample(sample: Mapping[str, Any]) -> AirSpeedInputs:
    """Build ``AirSpeedInputs`` from a prepared sample dict."""
    raw = sample.get("air_speed_inputs")
    if not isinstance(raw, Mapping):
        raise AdapterError("prepared sample missing 'air_speed_inputs' object")
    return AirSpeedInputs(
        driver_face_flow=float(raw.get("driver_face_flow") or 0.0),
        passenger_face_flow=float(raw.get("passenger_face_flow") or 0.0),
        driver_floor_flow=float(raw.get("driver_floor_flow") or 0.0),
        passenger_floor_flow=float(raw.get("passenger_floor_flow") or 0.0),
        driver_defrost_flow=float(raw.get("driver_defrost_flow") or 0.0),
        passenger_defrost_flow=float(raw.get("passenger_defrost_flow") or 0.0),
        rear_driver_foot_flow=raw.get("rear_driver_foot_flow"),
        rear_passenger_foot_flow=raw.get("rear_passenger_foot_flow"),
    )


def _apply_air_speed_overrides(
    base: AirSpeedInputs,
    overrides: Mapping[str, float],
) -> AirSpeedInputs:
    return replace(base, **dict(overrides))


class AirSpeedCalibrationAdapter(ModelAdapter):
    """Adapter for ``air_speed.estimate_driver_passenger_air_speed``."""

    def __init__(
        self,
        parameter_names: Union[Sequence[str], Mapping[str, int]],
        *,
        param_fields: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._param_names = _normalize_parameter_names(parameter_names)
        self._param_fields = dict(param_fields or AIR_SPEED_ADAPTER_PARAM_FIELDS)
        self._params = np.zeros(len(self._param_names), dtype=float)

    def parameter_names(self) -> List[str]:
        return list(self._param_names)

    def apply_parameters(self, params: np.ndarray) -> None:
        _vector_to_param_dict(params, self._param_names, self._param_fields)
        self._params = np.asarray(params, dtype=float).reshape(-1)

    def predict(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        base = air_speed_inputs_from_prepared_sample(sample)
        overrides = _vector_to_param_dict(
            self._params, self._param_names, self._param_fields
        )
        air_inputs = _apply_air_speed_overrides(base, overrides)
        result = estimate_driver_passenger_air_speed(air_inputs)
        return {
            "driver_air_speed_m_s": float(result.driver_air_speed_m_s),
            "passenger_air_speed_m_s": float(result.passenger_air_speed_m_s),
        }


def _parse_vector(name: str, value: Any, expected_len: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_len:
        raise AdapterError(f"{name} must be a list of length {expected_len}")
    return np.asarray(value, dtype=float)


def _parse_seat_occupant(raw: Any, seat: str) -> SeatOccupantInput:
    if raw is None:
        return SeatOccupantInput()
    if not isinstance(raw, dict):
        raise AdapterError(f"image_inputs.{seat} must be an object")
    return SeatOccupantInput(
        occupied=raw.get("occupied"),
        age=raw.get("age"),
        gender=raw.get("gender"),
        height_cm=raw.get("height_cm"),
        weight_kg=raw.get("weight_kg"),
        bmi=raw.get("bmi"),
        clothing_clo=raw.get("clothing_clo"),
        clothing_class=raw.get("clothing_class"),
        activity_met=raw.get("activity_met"),
        ir_head_surface_temp_c=raw.get("ir_head_surface_temp_c"),
        confidence=raw.get("confidence"),
    )


def _parse_image_inputs(raw: Any) -> Optional[ImageModuleInputs]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AdapterError("image_inputs must be an object when present")
    return ImageModuleInputs(
        driver=_parse_seat_occupant(raw.get("driver"), "driver"),
        passenger=_parse_seat_occupant(raw.get("passenger"), "passenger"),
    )


class PMVCalibrationAdapter(ModelAdapter):
    """Adapter for ``run_comfort_pipeline_dict`` PMV outputs."""

    def __init__(
        self,
        parameter_names: Union[Sequence[str], Mapping[str, int]],
        *,
        base_air_speed: Optional[AirSpeedInputs] = None,
        param_fields: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._param_names = _normalize_parameter_names(parameter_names)
        self._param_fields = dict(param_fields or PMV_ADAPTER_PARAM_FIELDS)
        self._base_air_speed = base_air_speed
        self._params = np.zeros(len(self._param_names), dtype=float)

    def parameter_names(self) -> List[str]:
        return list(self._param_names)

    def apply_parameters(self, params: np.ndarray) -> None:
        _vector_to_param_dict(params, self._param_names, self._param_fields)
        self._params = np.asarray(params, dtype=float).reshape(-1)

    def _pipeline_inputs(self, sample: Mapping[str, Any]) -> PipelineInputs:
        x = _parse_vector("x", sample["x"], N_X_STATES)
        u = _parse_vector("u", sample["u"], N_U)
        rh_raw = sample.get("rh", DEFAULT_RH)
        try:
            rh = float(rh_raw)
        except (TypeError, ValueError) as exc:
            raise AdapterError("rh must be numeric when present") from exc
        image_inputs = _parse_image_inputs(sample.get("image_inputs"))
        air_base = (
            self._base_air_speed
            if self._base_air_speed is not None
            else air_speed_inputs_from_chtd_u(u)
        )
        overrides = _vector_to_param_dict(
            self._params, self._param_names, self._param_fields
        )
        air_speed = _apply_air_speed_overrides(air_base, overrides)
        return PipelineInputs(
            x=x,
            u=u,
            rh=rh,
            image_inputs=image_inputs,
            air_speed=air_speed,
        )

    def predict(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        payload = run_comfort_pipeline_dict(self._pipeline_inputs(sample))
        return {
            "driver": dict(payload["driver"]),
            "passenger": dict(payload["passenger"]),
            "trace_status": payload["trace_status"],
        }


class HeadTempEKFCalibrationAdapter(ModelAdapter):
    """Adapter for ``ekf.head_temp_filter.fuse_head_temperature``."""

    def __init__(
        self,
        parameter_names: Union[Sequence[str], Mapping[str, int]],
        *,
        param_fields: Optional[Mapping[str, str]] = None,
        defaults: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._param_names = _normalize_parameter_names(parameter_names)
        self._param_fields = dict(param_fields or EKF_ADAPTER_PARAM_FIELDS)
        self._defaults = dict(defaults or EKF_ADAPTER_DEFAULTS)
        self._params = np.zeros(len(self._param_names), dtype=float)

    def parameter_names(self) -> List[str]:
        return list(self._param_names)

    def apply_parameters(self, params: np.ndarray) -> None:
        _vector_to_param_dict(
            params,
            self._param_names,
            self._param_fields,
            defaults=self._defaults,
        )
        self._params = np.asarray(params, dtype=float).reshape(-1)

    def predict(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        model_temp = float(sample["model_temp_c"])
        ir_surface = sample.get("ir_surface_temp_c")
        confidence = sample.get("confidence")
        kwargs = dict(_vector_to_param_dict(
            self._params,
            self._param_names,
            self._param_fields,
            defaults=self._defaults,
        ))
        result = fuse_head_temperature(
            model_temp,
            ir_surface,
            confidence,
            process_var=kwargs["process_var"],
            measurement_var=kwargs["measurement_var"],
            surface_to_air_offset_c=kwargs["surface_to_air_offset_c"],
            min_confidence=kwargs["min_confidence"],
            initial_variance=kwargs["initial_variance"],
        )
        return {
            "fused_temp_c": float(result.fused_temp_c),
            "model_temp_c": float(result.model_temp_c),
            "observation_temp_c": result.observation_temp_c,
            "source": result.source,
            "valid": bool(result.valid),
        }
