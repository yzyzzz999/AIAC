# 冒烟测试报告

**日期**: 2026-06-15
**测试脚本**: `tests/smoke_test_runtime.py`
**输入**: `tests/sample_input.json`
**结果**: **22/22 PASS**

---

## 测试用例

| 测试 | 项目 | 结果 |
|------|------|------|
| T1 | sample_input.json 可读 | PASS |
| T2 | 无禁止输入键 | PASS |
| T3 | run() 无异常退出 | PASS |
| T4-a | 顶层字段 status 存在 | PASS |
| T4-b | 顶层字段 model_state 存在 | PASS |
| T4-c | 顶层字段 pmv 存在 | PASS |
| T4-d | 顶层字段 diagnostics 存在 | PASS |
| T4-e | 顶层字段 provenance 存在 | PASS |
| T4-f | model_state.driver_head_temp_c 存在 | PASS |
| T4-g | model_state.passenger_head_temp_c 存在 | PASS |
| T4-h | model_state.driver_feet_temp_c 存在 | PASS |
| T4-i | model_state.passenger_feet_temp_c 存在 | PASS |
| T4-j | pmv.pmv_driver 存在 | PASS |
| T4-k | pmv.pmv_passenger 存在 | PASS |
| T4-l | diagnostics.forbidden_input_detected 存在 | PASS |
| T4-m | diagnostics.fallback_used 存在 | PASS |
| T5 | forbidden_input_detected == False | PASS |
| T6 | status 为 ok 或 degraded | PASS (ok) |
| T7-a | calibration_status 正确 | PASS |
| T7-b | chtd_param_mode = safe_preview | PASS |
| T8-a | use_next_state=True | PASS |
| T8-b | x_next 字段存在 | PASS |

---

## 关键输出数值

| 字段 | 值 |
|------|-----|
| `status` | `ok` |
| `pmv_driver` | 4.503 |
| `pmv_passenger` | 4.822 |
| `driver_head_temp_c` | 37.289 °C |
| `driver_feet_temp_c` | 37.970 °C |
| `forbidden_input_detected` | false |
| `trace_status` | CHTD 28 states: 28 TRACE, 0 FAST_APPROX, 0 GAP |
| `chtd_param_mode` | safe_preview |
| `calibration_status` | midterm_preview_not_vehicle_calibration |

**注**: PMV≈4.5 属于预期范围（夏季高温 38°C 工况，冷机启动初始状态，无热启动 x_init）。
实际运行需提供热启动状态向量（x_init）以获得稳态估算结果。

---

## 输入场景（tests/sample_input.json）

- 工况: 夏季面部冷却，环境温度 38°C
- 相对湿度: 45%，车速: 60 km/h
- 太阳辐射: 驾驶员 600 W/m²，副驾驶 400 W/m²
- 蒸发器出口: 8°C，驾驶员面部风量: 45 m³/h
- `chtd_param_mode: safe_preview`，`use_next_state: true`
- 无禁止热电偶输入

---

*冒烟测试通过 — SMOKE_TEST_VERDICT: PASS*
