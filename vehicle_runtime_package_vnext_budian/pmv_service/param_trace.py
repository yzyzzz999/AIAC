"""Human-readable PMV parameter trace writer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


PARAM_LABELS = {
    "amb_t_c": "外界环境温度, degC",
    "rh_percent": "相对湿度, %",
    "vehicle_speed_kph": "车速, km/h",
    "solar_driver_w_m2": "主驾侧太阳强度, W/m2",
    "solar_passenger_w_m2": "副驾侧太阳强度, W/m2",
    "eva_t_c": "蒸发器当前温度, degC",
    "ict_c": "前排车内温度传感器, degC",
    "front_blower_level": "前排风机档位 raw, 1-10 有效",
    "front_windshield_temp_c": "前挡玻璃温度布点 TS_FrntWidTemp, degC",
    "tma": "出风口温度字典, degC",
    "flows": "按座位/出风模式估算的风量字典, m3/h",
    "measured_head_air_temp_c": "主副驾头部左右侧布点温度, degC",
    "image_inputs": "图像模块乘员/衣着/代谢率输入",
    "bypass_models": "CHTD+风量模型旁路开关",
    "driver_air_temp_override_c": "主驾空气温度替代值, degC (bypass模式)",
    "passenger_air_temp_override_c": "副驾空气温度替代值, degC (bypass模式)",
    "driver_mrt_override_c": "主驾平均辐射温度替代值, degC (bypass模式)",
    "passenger_mrt_override_c": "副驾平均辐射温度替代值, degC (bypass模式)",
    "driver_air_speed_override_m_s": "主驾风速替代值, m/s (bypass模式)",
    "passenger_air_speed_override_m_s": "副驾风速替代值, m/s (bypass模式)",
}

DIAGNOSTIC_KEYS = (
    "bypass_models",
    "driver_air_temp_c",
    "passenger_air_temp_c",
    "driver_air_temp_source",
    "passenger_air_temp_source",
    "driver_mrt_c",
    "passenger_mrt_c",
    "driver_mrt_source",
    "passenger_mrt_source",
    "driver_air_speed_m_s",
    "passenger_air_speed_m_s",
    "rh_percent",
    "met_driver",
    "met_passenger",
    "clo_driver",
    "clo_passenger",
    "driver_tsv_mrt_formula",
    "passenger_tsv_mrt_formula",
)

DIAGNOSTIC_LABELS = {
    "bypass_models": "CHTD+风量模型旁路开关 (true=直接输入, false=模型计算)",
    "driver_air_temp_c": "主驾 PMV 空气温度 air_temp_c, degC",
    "passenger_air_temp_c": "副驾 PMV 空气温度 air_temp_c, degC",
    "driver_air_temp_source": "主驾空气温度来源",
    "passenger_air_temp_source": "副驾空气温度来源",
    "driver_mrt_c": "主驾 PMV 平均辐射温度 mean_radiant_temp_c, degC",
    "passenger_mrt_c": "副驾 PMV 平均辐射温度 mean_radiant_temp_c, degC",
    "driver_mrt_source": "主驾平均辐射温度来源",
    "passenger_mrt_source": "副驾平均辐射温度来源",
    "driver_air_speed_m_s": "主驾 PMV 风速 air_velocity_m_s, m/s, Fanger 内部最低按 0.01 处理",
    "passenger_air_speed_m_s": "副驾 PMV 风速 air_velocity_m_s, m/s, Fanger 内部最低按 0.01 处理",
    "rh_percent": "PMV 相对湿度, %",
    "met_driver": "主驾代谢率 met",
    "met_passenger": "副驾代谢率 met",
    "clo_driver": "主驾服装热阻 clo",
    "clo_passenger": "副驾服装热阻 clo",
    "driver_tsv_mrt_formula": "主驾 TSV MRT 公式诊断",
    "passenger_tsv_mrt_formula": "副驾 TSV MRT 公式诊断",
}


def _format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def append_param_log(
    log_file: str,
    *,
    run_index: int,
    timestamp: str,
    signal_cache: Dict[str, float],
    pmv_input: Dict[str, Any],
    pmv_output: Dict[str, Any],
) -> None:
    """Append one backward-compatible PMV parameter trace block."""
    if not log_file:
        return

    diagnostics = pmv_output.get("diagnostics", {}) or {}
    pmv = pmv_output.get("pmv", {}) or {}
    lines = [
        "",
        "=" * 100,
        f"运行序号 run_index: {run_index}",
        f"时间 timestamp: {timestamp}",
        f"状态 status: {pmv_output.get('status')}",
        "-" * 100,
        "PMV/PPD 输出:",
        f"  主驾 PMV: {_format_json(pmv.get('pmv_driver'))}",
        f"  主驾 PPD: {_format_json(pmv.get('ppd_driver'))} %",
        f"  副驾 PMV: {_format_json(pmv.get('pmv_passenger'))}",
        f"  副驾 PPD: {_format_json(pmv.get('ppd_passenger'))} %",
        "-" * 100,
        "PMV 实际入参/派生参数:",
    ]
    for key in DIAGNOSTIC_KEYS:
        label = DIAGNOSTIC_LABELS.get(key, "诊断字段")
        lines.append(f"  {key} ({label}): {_format_json(diagnostics.get(key))}")
    lines.extend(["-" * 100, "PMV 输入字典 pmv_input:"])
    for key in sorted(pmv_input):
        label = PARAM_LABELS.get(key, "运行时输入字段")
        lines.append(f"  {key} ({label}): {_format_json(pmv_input.get(key))}")
    lines.extend(["-" * 100, "最新 CAN 信号缓存 signal_cache:"])
    for key in sorted(signal_cache):
        lines.append(f"  {key}: {_format_json(signal_cache.get(key))}")
    lines.extend(["=" * 100, ""])

    try:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    except OSError as exc:
        LOGGER.warning("Parameter log write failed: %s", exc)
