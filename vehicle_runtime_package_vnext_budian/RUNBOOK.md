# Vehicle PMV Runtime — 运行手册

## 环境要求

- Python 3.9+
- numpy, scipy（标准科学计算库）
- 工作目录: `vehicle_runtime_package_vnext/`

## 步骤 1：准备输入 JSON

输入必须只包含 SIG/FTE 信号。最小必填字段：

```json
{
  "amb_t_c": 35.0,
  "chtd_param_mode": "safe_preview",
  "use_next_state": true
}
```

参考完整示例: `runtime_input_example_full.json`

禁止字段: `TA_FdHeadTempLe`, `TA_FpHeadTempLe`, `TA_*` 热电偶，见 `FORBIDDEN_RUNTIME_INPUTS.md`

## 步骤 2：运行推理

```bash
python run_vehicle_pmv.py --input tests/sample_input.json --output result.json
```

## 步骤 3：读取输出

```python
import json
with open("result.json") as f:
    out = json.load(f)

print("PMV 驾驶员:", out["pmv"]["pmv_driver"])
print("驾驶员头部温度:", out["model_state"]["driver_head_temp_c"])
print("输入审计通过:", not out["diagnostics"]["forbidden_input_detected"])
```

## 步骤 4：状态传递（热启动）

```python
import numpy as np, json
from run_vehicle_pmv import run

# 第一步
out1 = run({"amb_t_c": 35.0, "use_next_state": True, "chtd_param_mode": "safe_preview"})
x_next = out1.get("x_next")  # list of 28 floats

# 第二步（热启动）
raw2 = {"amb_t_c": 36.0, "x_init": x_next, "use_next_state": True, "chtd_param_mode": "safe_preview"}
out2 = run(raw2)
```

## 输出结构速查

```
status:           "ok" / "degraded" / "error"
model_state:
  driver_head_temp_c       驾驶员头部区域温度 (°C)
  passenger_head_temp_c    副驾驶头部区域温度 (°C)
  driver_feet_temp_c       驾驶员脚部区域温度 (°C)
  passenger_feet_temp_c    副驾驶脚部区域温度 (°C)
pmv:
  pmv_driver               驾驶员 PMV (-3~+3)
  pmv_passenger            副驾驶 PMV (-3~+3)
diagnostics:
  forbidden_input_detected  必须为 False
  fallback_used             是否使用了 fallback
  trace_status              CHTD 状态追踪信息
provenance:
  calibration_status        midterm_preview_not_vehicle_calibration
  chtd_param_mode           safe_preview
```

## 已知阻塞

- `run_vehicle_pmv_guarded.py` 依赖 `examples.run_runtime_pmv`（不在本包内），guarded 入口在本版本中无法单独使用
- `parameter_master_table.csv` 必须在 `../simulink_conversion_package/python_targets/` 相对于包父目录
- 标定未完成，PMV 数值为预览精度

详见 `DEPLOYMENT_LIMITATIONS.md`
