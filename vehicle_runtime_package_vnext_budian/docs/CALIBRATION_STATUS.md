# 标定状态

当前运行参数：

| 字段 | 值 |
|---|---|
| 参数文件 | `config/initial_calibration_params_v8_l1_candidate.json` |
| `calibration_status` | `spring_summer_v8_l1_candidate_not_vehicle_calibration` |
| `production_ready` | `false` |
| 默认参数模式 | `default` + 显式参数包 |

该参数包是 v8 L1 工程候选，只允许功能集成、回放和实车对比，不代表量产标定。

运行输出的 `provenance.calibration_status` 必须与参数包一致。`safe_preview` 和 `phase3_engineering_preview` 是显式 opt-in 工程模式，不能覆盖正式车辆标定结论。
