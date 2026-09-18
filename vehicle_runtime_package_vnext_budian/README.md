# Vehicle PMV Runtime

车端 PMV 服务从 CAN Socket 接收工程量，构造运行时输入，计算主副驾 PMV/PPD，并通过 HTTP 提供最新结果。

## 当前状态

- Python 环境：`/home/data/miniconda3/envs/AIAC/bin/python`
- HTTP：`http://127.0.0.1:7861/pmv`
- 参数包：`config/initial_calibration_params_v8_l1_candidate.json`
- 标定分类：`spring_summer_v8_l1_candidate_not_vehicle_calibration`
- 当前参数是工程候选，不得标记为量产标定。

## 在线链路

```text
CAN → can0_service_v1.3.8 → Unix Socket
    → pmv_socket_consumer.py
    → pmv_service.socket_client
    → pmv_service.input_builder / image_inputs
    → hvac_sim.runtime_input / runtime_adapter
    → hvac_sim.pipeline / pipeline_pmv
    → pmv_service.api → /pmv → dashboard
```

`pmv_socket_consumer.py` 是兼容 CLI；网络、输入映射、服务循环、参数日志和 HTTP 契约分别位于 `pmv_service/`。核心计算不执行网络或文件 I/O。

## 启动

```bash
cd /home/data/AIAC/vehicle_runtime_package_vnext_budian
bash start_pmv_service.sh
bash stop_pmv_service.sh
```

可配置环境变量：`API_PORT`、`PMV_INTERVAL`、`PMV_LOG_LEVEL`、`SOCKET_PATH`、`PMV_PARAM_LOG`。

离线运行：

```bash
/home/data/miniconda3/envs/AIAC/bin/python run_vehicle_pmv.py \
  --input tests/sample_input.json
```

## 主要目录

```text
pmv_service/       在线服务及外部 I/O
hvac_sim/          CHTD、风速、乘员和 PMV 核心
config/            当前运行参数与接口配置
tests/             pytest、smoke 和集成回放工具
```

## 测试

```bash
/home/data/miniconda3/envs/AIAC/bin/python -m pytest -q tests
/home/data/miniconda3/envs/AIAC/bin/python tests/smoke_test_runtime.py
```

HTTP 字段见 `docs/README_API.md` 和 `config/schemas/pmv_http_api_schema.json`。
运行时输入见 `docs/RUNBOOK.md`、`config/schemas/runtime_input_schema.json` 和
`examples/`。其余信号、标定与部署说明统一位于 `docs/`，根目录只保留项目入口。

## 已知边界

- 在线 consumer 默认使用旁路 PMV：头温来自四路实测点平均，MRT 当前使用前排车内温度代理，局部风速由总风量和风门位置估算。
- `feet_temp_c` 是旧客户端兼容字段，旁路模式下取模型状态，可能不随每次 PMV 计算变化。
- `cabin_temp_c` 当前兼容映射为主驾 PMV 空气温度，名称与语义存在偏差，尚未调整。
- 后排 PMV 未接入。
