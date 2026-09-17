"""主副驾推荐合成策略。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from src.utils.base_recommendation_override import apply_base_recommendation_override


def clamp_wind_weight(value: Any, default: float = 0.5) -> float:
    """把风量权重K限制到0~1。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def load_runtime_wind_weight(
    runtime_path: Path,
    default: float = 0.5,
) -> float:
    """读取页面写入的运行时K；文件不存在或损坏时使用配置默认值。"""
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        return clamp_wind_weight(payload.get("occupant_wind_weight_k"), default)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return clamp_wind_weight(default)


def compose_occupant_recommendation(
    result: Mapping[str, Any],
    *,
    driver_id: Optional[str],
    passenger_id: Optional[str],
    driver_exists: bool,
    passenger_exists: bool,
    overrides: Mapping[str, Any],
    wind_weight_k: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """按乘员存在状态合成温度、风量和模式，并返回页面展示元数据。"""
    base = dict(result)
    driver_candidate = apply_base_recommendation_override(
        base, driver_id, overrides, temperature_key="driver_temp"
    )
    passenger_candidate = apply_base_recommendation_override(
        base, passenger_id, overrides, temperature_key="passenger_temp"
    )
    k = clamp_wind_weight(wind_weight_k)

    driver_temp = float(driver_candidate["driver_temp"])
    passenger_temp = float(passenger_candidate["passenger_temp"])
    driver_wind = int(float(driver_candidate["wind_speed"]))
    passenger_wind = int(float(passenger_candidate["wind_speed"]))
    driver_mode = float(driver_candidate["air_mode"])
    passenger_mode = float(passenger_candidate["air_mode"])

    composed = dict(base)
    if driver_exists and passenger_exists:
        blended_wind = driver_wind * k + passenger_wind * (1.0 - k)
        output_wind = max(1, min(9, int(blended_wind)))
        composed.update({
            "driver_temp": driver_temp,
            "passenger_temp": passenger_temp,
            "wind_speed": float(output_wind),
            "air_mode": driver_mode,
        })
    elif driver_exists:
        blended_wind = float(driver_wind)
        output_wind = driver_wind
        passenger_wind = driver_wind
        composed.update({
            "driver_temp": driver_temp,
            "passenger_temp": driver_temp,
            "wind_speed": float(output_wind),
            "air_mode": driver_mode,
        })
    elif passenger_exists:
        blended_wind = float(passenger_wind)
        output_wind = passenger_wind
        driver_wind = passenger_wind
        composed.update({
            "driver_temp": passenger_temp,
            "passenger_temp": passenger_temp,
            "wind_speed": float(output_wind),
            "air_mode": passenger_mode,
        })
    else:
        blended_wind = float(base.get("wind_speed", 0.0) or 0.0)
        output_wind = int(blended_wind)

    metadata = {
        "driver_wind": driver_wind,
        "passenger_wind": passenger_wind,
        "wind_weight_k": k,
        "blended_wind": blended_wind,
        "rounded_wind": output_wind,
        "driver_exists": bool(driver_exists),
        "passenger_exists": bool(passenger_exists),
    }
    return composed, metadata
