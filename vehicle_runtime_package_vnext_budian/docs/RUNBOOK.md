# Vehicle PMV Runtime 运行手册

## 环境

统一使用：

```bash
/home/data/miniconda3/envs/AIAC/bin/python
```

## 离线运行

最小输入：

```json
{
  "amb_t_c": 35.0,
  "chtd_param_mode": "default",
  "use_next_state": true
}
```

运行：

```bash
cd /home/data/AIAC/vehicle_runtime_package_vnext_budian
/home/data/miniconda3/envs/AIAC/bin/python run_vehicle_pmv.py \
  --input tests/sample_input.json \
  --output /tmp/pmv_result.json
```

`run_vehicle_pmv.py` 未显式收到 `params_bundle_path` 时，会加载当前 v8 L1 candidate 参数包。

## 在线服务

```bash
bash start_pmv_service.sh
curl -s http://127.0.0.1:7861/pmv | python -m json.tool
bash stop_pmv_service.sh
```

日志位于 `/tmp/pmv_service_logs/`，PID 位于 `/tmp/pmv_service_pids/`。

## 状态传递

```python
from run_vehicle_pmv import run

first = run({"amb_t_c": 35.0})
second = run({"amb_t_c": 36.0}, previous_state=first["x_next"])
```

在线服务由 `pmv_service.runner` 维护 `x_next`，并保留现有车内温度状态校正策略。

## 参数模式

- `default`：默认参数或显式 `params_bundle_path`；当前正式入口使用此模式。
- `safe_preview`：仅用于工程预览，非车辆标定。
- `phase3_engineering_preview`：实验性 phase3 参数，必须显式启用。

## 验证

```bash
/home/data/miniconda3/envs/AIAC/bin/python -m pytest -q tests
/home/data/miniconda3/envs/AIAC/bin/python tests/smoke_test_runtime.py
bash -n start_pmv_service.sh stop_pmv_service.sh tests/integration/run_e2e_test.sh
```
