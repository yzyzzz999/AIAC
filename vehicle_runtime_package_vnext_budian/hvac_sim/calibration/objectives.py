"""Calibration objective functions (T9-3 / T9-7)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.air_speed import AirSpeedInputs

from .adapters import (
    AIR_SPEED_ADAPTER_PARAM_FIELDS,
    PMVCalibrationAdapter,
    ModelAdapter,
)
from .schema import CalibrationSchemaError

DEFAULT_RH = 50.0

AIR_SPEED_PARAM_FIELDS: Dict[str, str] = dict(AIR_SPEED_ADAPTER_PARAM_FIELDS)


class PMVObjectiveError(CalibrationSchemaError):
    """Invalid PMV calibration sample data."""


def _require_samples(samples: Union[Sequence[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(samples, dict):
        if "samples" in samples:
            raw = samples["samples"]
        else:
            raise PMVObjectiveError("samples dict must contain a 'samples' list")
    else:
        raw = samples

    if not isinstance(raw, list):
        raise PMVObjectiveError("samples must be a list")
    if len(raw) == 0:
        raise PMVObjectiveError("samples must not be empty")
    for idx, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise PMVObjectiveError(f"samples[{idx}] must be an object")
    return raw


def _get_nested_field(obj: Any, field_path: str) -> Any:
    current = obj
    for part in field_path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(field_path)
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def make_mse_objective(
    adapter: ModelAdapter,
    samples: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    target_field: str,
    prediction_field: str,
) -> Callable[[np.ndarray], float]:
    """Build scalar MSE objective for one target/prediction field pair."""
    return make_multi_output_mse_objective(
        adapter,
        samples,
        [(target_field, prediction_field)],
    )


def make_multi_output_mse_objective(
    adapter: ModelAdapter,
    samples: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    field_pairs: Sequence[Tuple[str, str]],
) -> Callable[[np.ndarray], float]:
    """Build MSE objective over multiple target/prediction field pairs.

    Each pair maps ``(sample[target_field], prediction[prediction_field])``.
    ``prediction_field`` supports dot paths such as ``driver.pmv``.
    """
    if not field_pairs:
        raise CalibrationSchemaError("field_pairs must not be empty")

    sample_list = _require_samples(samples)

    def objective(params: np.ndarray) -> float:
        adapter.apply_parameters(params)
        total = 0.0
        count = 0
        for sample in sample_list:
            prediction = adapter.predict(sample)
            for target_field, prediction_field in field_pairs:
                target = float(_get_nested_field(sample, target_field))
                pred = float(_get_nested_field(prediction, prediction_field))
                total += (pred - target) ** 2
                count += 1
        return total / count

    return objective


def make_pmv_mse_objective(
    samples: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    parameter_mapping: Union[Sequence[str], Mapping[str, int]],
    *,
    base_air_speed: Optional[AirSpeedInputs] = None,
) -> Callable[[np.ndarray], float]:
    """Build a mean-squared-error objective over PMV targets.

    Backward-compatible wrapper around ``PMVCalibrationAdapter`` and the generic
    multi-output MSE helper.
    """
    sample_list = _require_samples(samples)
    adapter = PMVCalibrationAdapter(
        parameter_mapping,
        base_air_speed=base_air_speed,
    )
    return make_multi_output_mse_objective(
        adapter,
        sample_list,
        [
            ("target_driver_pmv", "driver.pmv"),
            ("target_passenger_pmv", "passenger.pmv"),
        ],
    )
