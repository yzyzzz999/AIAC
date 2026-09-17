# 标定状态说明

**calibration_status**: `midterm_preview_not_vehicle_calibration`
**chtd_param_mode**: `safe_preview`
**日期**: 2026-06-15

## 状态定义

| 字段 | 值 | 含义 |
|------|----|------|
| `calibration_status` | `midterm_preview_not_vehicle_calibration` | 中期预览，非量产标定 |
| `chtd_param_mode` | `safe_preview` | 使用 safe_preview 参数包（非默认，非 PSO 优化）|
| `production_ready` | false | 不允许量产标注 |
| `from_invalid_pso` | false | 未使用已失效的 PSO 参数 |

## 已知风险

- `known_risk: calibrated_before_airflow_v5_fix`
  - 当前参数在 airflow v5 修复前标定
  - 风量模型已在 cache_v5 修复（face_pure 模式检测），但参数尚未基于 v5 重新 PSO 优化
  - PMV 精度属于工程预览级，不代表量产验收标准

## 参数来源

参数模块: `hvac_sim/chtd/safe_preview_params.py`
基础: `physical_baseline_v2` + HVAC 对流缩放 + 太阳辐射 LUT 缩放
不来自: Stage2 V2 PSO（已标注 INVALID_AIRFLOW_FALLBACK_BUG）

## 下一步标定路径

1. Task D: v5 baseline 对标（等 Cursor 提供 replay 结果）
2. V3 PSO: 基于 cache_v5 重新优化（条件允许后启动）
3. 量产标定: PSO 收敛 + 工程验收后更新 calibration_status

## 不得做什么

- 不得将 `midterm_preview_not_vehicle_calibration` 标注为生产就绪
- 不得使用 Stage2 V2 PSO 参数（已记录 INVALID）
- 不得跳过 Task D baseline 验证直接推 V3 PSO
