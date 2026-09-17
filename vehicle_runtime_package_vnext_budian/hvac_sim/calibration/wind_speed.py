"""Wind-speed experiment conversion for local air-speed prediction validation.

The May 2026 experiment stores one steady operating condition per CSV file.
Each CSV contains time-series columns such as ``风速1(m/s)`` and ``温度1(℃)``.
This module summarizes each file into one **validation sample** for checking
``air_speed.py`` predictions. It is not formal airflow or wind-speed calibration.
See ``LOCAL_AIR_SPEED_PREDICTION_VALIDATION.md``.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from hvac_sim.air_speed import AirSpeedInputs

from .adapters import (
    AIR_SPEED_ADAPTER_PARAM_FIELDS,
    AirSpeedCalibrationAdapter,
    air_speed_inputs_from_prepared_sample,
)
from .objectives import make_multi_output_mse_objective


SENSOR_MAP: Dict[int, str] = {
    1: "passenger_front_floor",
    2: "rear_driver_center",
    3: "rear_driver_face_side",
    5: "driver_front_floor",
    6: "front_windshield_defrost_a",
    7: "front_windshield_defrost_b",
    8: "passenger_face_right_side",
    9: "driver_face_right_center_low_bias",
    10: "passenger_face_left_center_reference",
    11: "rear_passenger_face_side",
    12: "driver_face_left_side",
    13: "rear_passenger_center",
}

DRIVER_FACE_SENSORS = (9, 12)
PASSENGER_FACE_SENSORS = (8, 10)
DRIVER_FLOOR_SENSORS = (5,)
PASSENGER_FLOOR_SENSORS = (1,)
FRONT_DEFROST_SENSORS = (6, 7)

FACE_SHARE = 0.93
SIDE_DEFROST_SHARE = 0.07


@dataclass(frozen=True)
class WindCondition:
    """Parsed condition text for one CSV condition."""

    sequence_index: int
    label_index: Optional[int]
    raw: str
    front_level: Optional[int]
    rear_level: Optional[int]
    front_mode: Optional[str]
    rear_mode: Optional[str]
    thermal_mode: Optional[str]
    circulation: Optional[str]


@dataclass(frozen=True)
class WindSpeedSample:
    """One steady-state wind-speed calibration sample."""

    source_file: str
    condition: Optional[WindCondition]
    n_rows: int
    time_start: Optional[str]
    time_end: Optional[str]
    sensor_speed_mean_m_s: Dict[str, float]
    sensor_speed_std_m_s: Dict[str, float]
    sensor_temp_mean_c: Dict[str, float]
    measured_driver_air_speed_m_s: float
    measured_passenger_air_speed_m_s: float
    measured_driver_floor_speed_m_s: float
    measured_passenger_floor_speed_m_s: float
    measured_front_defrost_speed_m_s: float
    estimated_driver_side_defrost_speed_m_s: float
    estimated_passenger_side_defrost_speed_m_s: float
    air_speed_inputs: Dict[str, Optional[float]]
    notes: List[str]


class WindSpeedDataError(ValueError):
    """Invalid wind-speed experiment input."""


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return 0.0
    return float(np.mean(finite))


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) <= 1:
        return 0.0
    return float(np.std(finite, ddof=0))


def _safe_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _parse_condition_line(raw: str, sequence_index: int) -> WindCondition:
    text = raw.strip()
    match = re.match(r"^(?:(\d+)\.\s*)?(.*)$", text)
    label = int(match.group(1)) if match and match.group(1) else None
    body = match.group(2).strip() if match else text
    tokens = body.split()

    front_level: Optional[int] = None
    rear_level: Optional[int] = None
    front_mode: Optional[str] = None
    rear_mode: Optional[str] = None
    thermal_mode: Optional[str] = None
    circulation: Optional[str] = None

    front_match = re.search(r"前\s*(\d+)", body)
    rear_match = re.search(r"后\s*(\d+)", body)
    if front_match:
        front_level = int(front_match.group(1))
    if rear_match:
        rear_level = int(rear_match.group(1))

    non_level_tokens = []
    for token in tokens:
        if re.fullmatch(r"[前后]\d+", token):
            continue
        if token in ("前", "后") or token.isdigit():
            continue
        non_level_tokens.append(token)
    if non_level_tokens:
        front_mode = non_level_tokens[0]
    if len(non_level_tokens) >= 2 and non_level_tokens[1] not in ("冷", "热", "内循环", "外循环"):
        rear_mode = non_level_tokens[1]
    for token in non_level_tokens:
        if token in ("冷", "热"):
            thermal_mode = token
        if token in ("内循环", "外循环"):
            circulation = token

    return WindCondition(
        sequence_index=sequence_index,
        label_index=label,
        raw=body,
        front_level=front_level,
        rear_level=rear_level,
        front_mode=front_mode,
        rear_mode=rear_mode,
        thermal_mode=thermal_mode,
        circulation=circulation,
    )


def parse_condition_file(path: Union[str, Path]) -> List[WindCondition]:
    """Parse the experiment condition text into sequential CSV conditions."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    conditions: List[WindCondition] = []
    in_conditions = False
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("对应模式"):
            in_conditions = True
            continue
        if text.startswith("##"):
            break
        if not in_conditions:
            continue
        if text == "0-31":
            continue
        if "前" not in text or "后" not in text:
            continue
        conditions.append(_parse_condition_line(text, len(conditions)))
    return conditions


def _speed_col(sensor_id: int) -> str:
    return f"风速{sensor_id}(m/s)"


def _temp_col(sensor_id: int) -> str:
    return f"温度{sensor_id}(℃)"


def _csv_condition_index(path: Path) -> int:
    match = re.search(r"_(\d+)(?:\(\d+\))?\.csv$", path.name)
    if not match:
        raise WindSpeedDataError(f"cannot parse condition index from {path.name!r}")
    return int(match.group(1))


def _flow_from_speed(speed_m_s: float, area_m2: float) -> float:
    return max(0.0, float(speed_m_s)) * area_m2 * 3600.0


def _sensor_group_mean(sensor_means: Mapping[str, float], sensors: Iterable[int]) -> float:
    return _mean([sensor_means.get(str(sensor), float("nan")) for sensor in sensors])


def summarize_wind_csv(
    csv_path: Union[str, Path],
    *,
    condition: Optional[WindCondition] = None,
    nominal_face_area_m2: float = 0.02,
    nominal_floor_area_m2: float = 0.015,
    nominal_defrost_area_m2: float = 0.01,
) -> WindSpeedSample:
    """Summarize one steady CSV into a calibration sample."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise WindSpeedDataError(f"{path} contains no data rows")

    speed_mean: Dict[str, float] = {}
    speed_std: Dict[str, float] = {}
    temp_mean: Dict[str, float] = {}
    for sensor_id in SENSOR_MAP:
        s_col = _speed_col(sensor_id)
        t_col = _temp_col(sensor_id)
        if s_col in rows[0]:
            values = [_safe_float(row.get(s_col)) for row in rows]
            speed_mean[str(sensor_id)] = _mean(values)
            speed_std[str(sensor_id)] = _std(values)
        if t_col in rows[0]:
            values = [_safe_float(row.get(t_col)) for row in rows]
            temp_mean[str(sensor_id)] = _mean(values)

    driver_face = _sensor_group_mean(speed_mean, DRIVER_FACE_SENSORS)
    passenger_face = _sensor_group_mean(speed_mean, PASSENGER_FACE_SENSORS)
    driver_floor = _sensor_group_mean(speed_mean, DRIVER_FLOOR_SENSORS)
    passenger_floor = _sensor_group_mean(speed_mean, PASSENGER_FLOOR_SENSORS)
    front_defrost = _sensor_group_mean(speed_mean, FRONT_DEFROST_SENSORS)

    total_face_speed_proxy = driver_face + passenger_face
    side_total = total_face_speed_proxy * SIDE_DEFROST_SHARE / FACE_SHARE if FACE_SHARE > 0 else 0.0
    driver_side = side_total / 2.0
    passenger_side = side_total / 2.0

    air_inputs = {
        "driver_face_flow": _flow_from_speed(driver_face, nominal_face_area_m2),
        "passenger_face_flow": _flow_from_speed(passenger_face, nominal_face_area_m2),
        "driver_floor_flow": _flow_from_speed(driver_floor, nominal_floor_area_m2),
        "passenger_floor_flow": _flow_from_speed(passenger_floor, nominal_floor_area_m2),
        "driver_defrost_flow": _flow_from_speed(driver_side, nominal_defrost_area_m2),
        "passenger_defrost_flow": _flow_from_speed(passenger_side, nominal_defrost_area_m2),
        "rear_driver_foot_flow": None,
        "rear_passenger_foot_flow": None,
    }

    notes = [
        "side defrost estimated from front face speed proxy: total side defrost = face_total * 0.07 / 0.93",
        "rear foot flow reserved as None until rear-foot experiment data is available",
    ]
    if "头" in (condition.front_mode if condition else ""):
        notes.append("front face mode active; side-defrost estimate follows front face distribution rule")

    return WindSpeedSample(
        source_file=str(path),
        condition=condition,
        n_rows=len(rows),
        time_start=rows[0].get("时间"),
        time_end=rows[-1].get("时间"),
        sensor_speed_mean_m_s=speed_mean,
        sensor_speed_std_m_s=speed_std,
        sensor_temp_mean_c=temp_mean,
        measured_driver_air_speed_m_s=driver_face,
        measured_passenger_air_speed_m_s=passenger_face,
        measured_driver_floor_speed_m_s=driver_floor,
        measured_passenger_floor_speed_m_s=passenger_floor,
        measured_front_defrost_speed_m_s=front_defrost,
        estimated_driver_side_defrost_speed_m_s=driver_side,
        estimated_passenger_side_defrost_speed_m_s=passenger_side,
        air_speed_inputs=air_inputs,
        notes=notes,
    )


def convert_wind_experiment_dir(
    csv_dir: Union[str, Path],
    *,
    condition_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert an experiment directory into calibration JSON payload."""
    base = Path(csv_dir)
    if condition_file is None:
        condition_path = base / "多站点记录.txt"
    else:
        condition_path = Path(condition_file)
    conditions = parse_condition_file(condition_path) if condition_path.exists() else []
    condition_by_index = {c.sequence_index: c for c in conditions}

    samples: List[WindSpeedSample] = []
    for csv_path in sorted(base.glob("*.csv"), key=_csv_condition_index):
        idx = _csv_condition_index(csv_path)
        samples.append(
            summarize_wind_csv(csv_path, condition=condition_by_index.get(idx))
        )

    return {
        "schema": "wind_speed_experiment_v1",
        "purpose": "local_air_speed_prediction_validation",
        "classification": "demo_validation_dataset",
        "not_formal_calibration": True,
        "source_dir": str(base),
        "condition_file": str(condition_path) if condition_path.exists() else None,
        "notes": [
            "validation dataset for local air speed prediction from vent flows and geometry",
            "not a formal calibration dataset — anemometer probes validate prediction only",
            "formal PSO targets: AFE/flow resistance, CHTD thermal, PMV correction, EKF Q/R",
        ],
        "sensor_map": {str(k): v for k, v in SENSOR_MAP.items()},
        "distribution_policy": {
            "front_face_share": FACE_SHARE,
            "front_side_defrost_total_share": SIDE_DEFROST_SHARE,
            "side_defrost_rule": "total_side_defrost = (driver_face + passenger_face) * 0.07 / 0.93; split equally",
            "rear_foot_flow": "reserved_null_until_data_available",
        },
        "samples": [sample_to_dict(s) for s in samples],
    }


def sample_to_dict(sample: WindSpeedSample) -> Dict[str, Any]:
    data = asdict(sample)
    if sample.condition is not None:
        data["condition"] = asdict(sample.condition)
    return data


def air_speed_inputs_from_wind_sample(sample: Mapping[str, Any]) -> AirSpeedInputs:
    """Backward-compatible alias for prepared-sample air-speed input parsing."""
    return air_speed_inputs_from_prepared_sample(sample)


WIND_SPEED_PARAM_FIELDS: Dict[str, str] = {
    key: AIR_SPEED_ADAPTER_PARAM_FIELDS[key]
    for key in (
        "attenuation_k",
        "driver_face_distribution_ratio",
        "passenger_face_distribution_ratio",
        "driver_floor_distribution_ratio",
        "passenger_floor_distribution_ratio",
        "driver_defrost_distribution_ratio",
        "passenger_defrost_distribution_ratio",
    )
}


def make_wind_speed_mse_objective(
    wind_payload_or_samples: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    parameter_mapping: Sequence[str],
) -> Callable[[np.ndarray], float]:
    """Build MSE objective against measured driver/passenger air speeds.

    Thin wrapper over ``AirSpeedCalibrationAdapter`` for **local air-speed
    prediction validation**; expects *prepared* samples, not raw CSV.
    """
    if isinstance(wind_payload_or_samples, Mapping):
        samples = wind_payload_or_samples.get("samples")
    else:
        samples = wind_payload_or_samples
    if not isinstance(samples, list) or not samples:
        raise WindSpeedDataError("wind-speed samples must be a non-empty list")

    adapter = AirSpeedCalibrationAdapter(
        parameter_mapping,
        param_fields=WIND_SPEED_PARAM_FIELDS,
    )
    return make_multi_output_mse_objective(
        adapter,
        samples,
        [
            ("measured_driver_air_speed_m_s", "driver_air_speed_m_s"),
            ("measured_passenger_air_speed_m_s", "passenger_air_speed_m_s"),
        ],
    )
