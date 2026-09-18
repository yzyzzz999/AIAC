"""JSON-safe serialization for comfort-pipeline results."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _seat_comfort_to_dict(seat: Any) -> Dict[str, Any]:
    return {
        "air_temp_c": float(seat.air_temp_c),
        "mean_radiant_temp_c": float(seat.mean_radiant_temp_c),
        "air_speed_m_s": float(seat.air_speed_m_s),
        "pmv": float(seat.pmv),
        "ppd": float(seat.ppd),
        "valid": bool(seat.valid),
    }


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return [float(item) for item in np.asarray(value, dtype=float).tolist()]
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return int(item)
        return float(item)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"pipeline: value not JSON-serializable: {type(value)!r}")


def serialize_pipeline_result(result: Any) -> Dict[str, Any]:
    """Convert a pipeline result to the stable public dictionary contract."""
    x_delta = np.asarray(result.x_delta, dtype=float)
    x_next = np.asarray(result.x_next, dtype=float)
    return {
        "trace_status": result.trace_status,
        "driver": _seat_comfort_to_dict(result.driver),
        "passenger": _seat_comfort_to_dict(result.passenger),
        "x_delta": [float(item) for item in x_delta.tolist()],
        "x_next": [float(item) for item in x_next.tolist()],
        "x_delta_summary": {
            "min": float(np.min(x_delta)),
            "max": float(np.max(x_delta)),
            "mean": float(np.mean(x_delta)),
        },
        "implementation_modes": {
            mode: list(names) for mode, names in result.implementation_modes.items()
        },
        "inputs_used": _to_json_safe(result.inputs_used),
    }
