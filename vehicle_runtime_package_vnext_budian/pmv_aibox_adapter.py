#!/usr/bin/env python3
"""
PMV → API 简化适配层。

提供 wrap_pmv_for_api() 将 PMV 全量输出精简为 API 响应所需的字段:
主副驾 PMV / PPD / 头部温度 / 脚部温度。
"""

from datetime import datetime
from typing import Any, Dict, Optional


def wrap_pmv_for_api(
    pmv_output: Dict[str, Any],
    *,
    run_index: int = 0,
    amb_t_c: Optional[float] = None,
) -> Dict[str, Any]:
    """从 PMV 全量输出提取 API 精简字段。"""
    def _safe(v):
        if v is None:
            return None
        try:
            import math
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    return {
        "driver": {
            "pmv":          _safe(pmv_output.get("pmv", {}).get("pmv_driver")),
            "ppd":          _safe(pmv_output.get("pmv", {}).get("ppd_driver")),
            "head_temp_c":  _safe(pmv_output.get("diagnostics", {}).get("driver_air_temp_c")),
            "feet_temp_c":  _safe(pmv_output.get("model_state", {}).get("driver_feet_temp_c")),
        },
        "passenger": {
            "pmv":          _safe(pmv_output.get("pmv", {}).get("pmv_passenger")),
            "ppd":          _safe(pmv_output.get("pmv", {}).get("ppd_passenger")),
            "head_temp_c":  _safe(pmv_output.get("diagnostics", {}).get("passenger_air_temp_c")),
            "feet_temp_c":  _safe(pmv_output.get("model_state", {}).get("passenger_feet_temp_c")),
        },
        "cabin_temp_c": _safe(pmv_output.get("diagnostics", {}).get("driver_air_temp_c")),
        "amb_temp_c":   _safe(amb_t_c),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": pmv_output.get("status", "error"),
        "run_index": run_index,
    }
