# B3: 禁止运行时输入清单

**文档版本**: vehicle_runtime_package_vnext_v1
**日期**: 2026-06-15

## 核心原则

**HeadTemp / FeetTemp / CabinTemp 真值严禁作为运行时输入。**

运行时推理的目的是估算这些温度，不能用其作为输入（循环依赖 + 数据来源不可靠）。

## 禁止字段清单

| 字段名 | 类型 | 禁止原因 |
|--------|------|---------|
| `TA_FdHeadTempLe` | 热电偶 | 驾驶员头部热电偶，只允许离线标定/验证 |
| `TA_FpHeadTempLe` | 热电偶 | 副驾驶头部热电偶，只允许离线标定/验证 |
| `TA_FdFloorTemp1` | 热电偶 | 驾驶员脚部热电偶 1 |
| `TA_FdFloorTemp2` | 热电偶 | 驾驶员脚部热电偶 2 |
| `TA_FpFloorTemp1` | 热电偶 | 副驾驶脚部热电偶 1 |
| `TA_FpFloorTemp2` | 热电偶 | 副驾驶脚部热电偶 2 |
| `TA_CarbinFrntTempLe` | 热电偶 | 前排舱内热电偶（左）|
| `TA_CarbinFrntTempRi` | 热电偶 | 前排舱内热电偶（右）|
| `TS_*` | 测试热电偶 | 所有 TS_ 前缀测试温度传感器 |
| `V_*` | 验证数据 | 验证测试专用，禁止运行时 |
| `measured_head_temperature` | 真值 | 绝对禁止 |
| `measured_feet_temperature` | 真值 | 绝对禁止 |
| `measured_cabin_temperature` | 真值 | 绝对禁止 |

## 允许的温度信号

| 字段名 | 来源 | 用途 |
|--------|------|------|
| `amb_t_c` | SIG/VIU_AmbT | 环境温度（外部）|
| `eva_t_c` | FTE | 蒸发器出口温度（热系统侧）|
| `hct_c` | FTE | 加热器芯出口温度 |
| `ict_c` | AC_FrntInCarT / SIG | 舱内设定目标温度（非热电偶）|

## 审计检查

`run_vehicle_pmv.py` 中内置禁止字段检查：

```python
out = run(raw_input)
assert not out["diagnostics"]["forbidden_input_detected"], "禁止输入被检测到！"
```

冒烟测试（`tests/smoke_test_runtime.py`）验证 T5：`forbidden_input_detected == False`。
