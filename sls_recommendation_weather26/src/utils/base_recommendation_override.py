"""按主驾 FaceID 调整基础空调推荐。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.utils.constants import TEMP_MAX, TEMP_MIN


def apply_base_recommendation_override(
    result: Mapping[str, Any],
    driver_id: Optional[str],
    overrides: Mapping[str, Any],
    temperature_key: str = "driver_temp",
) -> Dict[str, Any]:
    """返回应用人员FaceID专属偏移后的基础推荐，不修改传入字典。"""
    adjusted = dict(result)
    if not driver_id:
        return adjusted

    override = overrides.get(str(driver_id))
    if not isinstance(override, Mapping):
        return adjusted

    temperature_delta = override.get(
        "temperature_delta", override.get("driver_temp_delta")
    )
    if temperature_delta is not None:
        try:
            temperature = float(adjusted[temperature_key]) + float(temperature_delta)
            adjusted[temperature_key] = round(
                max(TEMP_MIN, min(TEMP_MAX, temperature)) * 2
            ) / 2
        except (KeyError, TypeError, ValueError):
            pass

    if "wind_speed_delta" in override:
        try:
            wind_speed_delta = int(override["wind_speed_delta"])
            wind_speed = int(float(adjusted["wind_speed"])) + wind_speed_delta
            adjusted["wind_speed"] = float(max(1, min(9, wind_speed)))
        except (KeyError, TypeError, ValueError):
            pass

    return adjusted
