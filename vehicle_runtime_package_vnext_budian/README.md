# Vehicle PMV Runtime Package — vnext

**包版本**: `vehicle_runtime_package_vnext_v1`
**标定状态**: `midterm_preview_not_vehicle_calibration`
**日期**: 2026-06-16

---

## 架构

```
CAN 总线 (can2)
  │
  ├── can0_service_v1.3.2        ← 已有，读 CAN → DBC 解析 → Unix Socket 广播
  │     ./sock/can0_bus.sock
  │
  └── pmv_socket_consumer.py     ← 新增：连 socket，定时跑 PMV
        ├── run_vehicle_pmv.py   ← PMV 主入口
        └── pmv_aibox_adapter.py ← AIBox 格式包装
```

## 快速开始

**离线模式**（JSON 文件输入）:
```bash
conda activate AIAC
python run_vehicle_pmv.py --input tests/sample_input.json
```

**在线模式**（CAN 实时数据）:
```bash
conda activate AIAC
python pmv_socket_consumer.py --interval 1.0
```

**启停脚本**:
```bash
bash start_pmv_service.sh dev     # 开发模式 (vcan + BLF 回放)
bash start_pmv_service.sh prod    # 生产模式
bash stop_pmv_service.sh
```

---

## PMV 使用的 CAN 信号

消费者订阅 6 个 CAN ID，映射 10 个 DBC 信号：

| CAN ID | 消息名 | DBC 信号 | PMV 输入字段 | 用途 | 状态 |
|--------|--------|----------|-------------|------|------|
| 0x35B | VIU_35B | `VIU_AmbT` | `amb_t_c` | 环境温度 | ✓ 有效 |
| 0x3B9 | VIU_LIN1 | `RSM_RelHum` | `rh_percent` | 相对湿度 | ✓ 有效 |
| 0x3B9 | VIU_LIN1 | `RSM_LeSolarInten` | `solar_driver_w_m2` | 主驾太阳辐照 | ✓ 有效 |
| 0x3B9 | VIU_LIN1 | `RSM_RiSolarInten` | `solar_passenger_w_m2` | 副驾太阳辐照 | ✓ 有效 |
| 0x18F | VIU_CHA_18F | `IPB_VehicleSpeed` | `vehicle_speed_kph` | 车速 | ✓ 有效 |
| 0x33E | AC_TEMP2 | `AC_FEvapCurrentTemp` | `eva_t_c` | 蒸发器出口温度 | ✓ 有效 |
| 0x338 | AC_TEMP1 | `AC_FrntInCarT` | `ict_c` | 车内设定温度 | ✓ 有效 |
| 0x338 | AC_TEMP1 | `AC_Forward_BlwPwmOut` | `eva_t_c` | 蒸发器温度(备用) | ✓ 有效 |
| 0x33F | AC_TEMP3 | `AC_DrvrFaceVentActT` | `tma.FrntFdvFlow` | 主驾吹面出风温度 | ✓ 有效 |
| 0x33F | AC_TEMP3 | `AC_PassFaceVentActT` | `tma.FrntFpvFlow` | 副驾吹面出风温度 | ✓ 有效 |

> **注意**: can0_service 已对上述信号做 DBC 物理量缩放（温度→°C、速度→km/h 等），消费者直接使用无需再次缩放。

---

## 输出字段有效性

### 有效字段

| 字段 | 路径 | 类型 | 说明 |
|------|------|------|------|
| 主驾 PMV | `comfort.driver.pmv` | float | -3 ~ +3，当前范围 0.2~1.2 |
| 副驾 PMV | `comfort.passenger.pmv` | float | -3 ~ +3，当前范围 0.2~1.2 |
| 主驾是否舒适 | `comfort.driver.comfortable` | bool | PMV 在 ±0.5 内为 true |
| 副驾是否舒适 | `comfort.passenger.comfortable` | bool | 同上 |
| 主驾头部温度 | `comfort.driver.head_temp_c` | float | °C |
| 副驾头部温度 | `comfort.passenger.head_temp_c` | float | °C |
| 主驾脚部温度 | `comfort.driver.feet_temp_c` | float | °C |
| 副驾脚部温度 | `comfort.passenger.feet_temp_c` | float | °C |
| 前排座舱温度 | `comfort.cabin_temp_c` | float/null | °C |
| 后排头部温度 | pmv_full.model_state | float | 2nd row driver/passenger head |
| 后排脚部温度 | pmv_full.model_state | float | 2nd row driver/passenger feet |
| 运行状态 | `status` | string | ok / error |
| 禁止输入检测 | pmv_full.diagnostics.forbidden_input_detected | bool | |
| 诊断错误 | pmv_full.diagnostics.error | string/null | |
| 适配器警告 | pmv_full.diagnostics.adapter_warnings | list | 缺失信号告警 |
| 下一帧状态 | pmv_full.x_next | list[28] | CHTD 状态向量(热启动用) |

### 无效/占位字段 (暂不可用)

| 字段 | 路径 | 原因 |
|------|------|------|
| 后排 PMV | pmv_full.pmv.pmv_second_row_driver/passenger | **硬编码 null** — 后排脚部流量信号未接入 |
| 风量来源 | pmv_full.diagnostics.airflow_distribution_source | 固定 "unknown" — 无执行器位置电压信号 |
| 风口分配比 | pmv_full.diagnostics.{face,foot,defrost}_fraction | 固定 null — 无 FTE 风量信号 |
| 左右分配源 | pmv_full.diagnostics.lr_split_source | 固定 "unknown" — 默认 50:50 |
| 座舱温度 | model_state.front_cabin_temp_c | 可能 null — 依赖状态向量初始化 |

---

---

## 入口文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `run_vehicle_pmv.py` | 主运行入口（JSON 文件输入） | ✓ |
| `pmv_socket_consumer.py` | Socket 消费者（CAN 实时） | ✓ 新增 |
| `pmv_aibox_adapter.py` | AIBox 格式适配 | ✓ 新增 |
| `start_pmv_service.sh` | 一键启动脚本 | ✓ 新增 |
| `stop_pmv_service.sh` | 一键停止脚本 | ✓ 新增 |
| `run_vehicle_pmv_guarded.py` | 守护入口（依赖外部模块，标注阻塞）| ⚠ |

## 测试工具

| 文件 | 用途 |
|------|------|
| `test/blf_to_vcan.py` | BLF 录包 → vcan 回放 |
| `test/can_to_socket_simple.py` | 简化版 CAN→Socket 服务（单进程测试用） |
| `test/blf_to_pmv_direct.py` | BLF 直接跑 PMV（无需 vcan/socket） |
| `test/run_e2e_test.sh` | 端到端回放测试 |
| `tests/smoke_test_runtime.py` | 冒烟测试 (22/22 PASS) |
| `config/pmv_aibox_output_example.json` | AIBox 输出格式示例 |

## 文档索引

| 文档 | 说明 |
|------|------|
| `RUNBOOK.md` | 运行手册 |
| `SIGNAL_REQUIREMENTS.md` | SIG/FTE 信号需求 (B1) |
| `CAN_SIGNAL_EXCEPTION_REPORT.md` | CAN 信号例外清单 (B2) |
| `FORBIDDEN_RUNTIME_INPUTS.md` | 禁止输入清单 (B3) |
| `CALIBRATION_STATUS.md` | 标定状态说明 |
| `DEPLOYMENT_LIMITATIONS.md` | 部署局限性 |
| `SMOKE_TEST_REPORT.md` | 冒烟测试报告 |
| `DEPLOYMENT_READINESS_REPORT.md` | 就绪性裁定 |

## 已知限制

1. **后排 PMV 不可用** — 缺少后排风量信号，`pmv_second_row_*` 固定 null
2. **标定非量产** — `calibration_status = midterm_preview_not_vehicle_calibration`
3. **缺少部分 FTE 信号** — 无脚部/除霜出风温度、执行器位置电压 → Tma/风量使用 fallback
4. **参数 CSV 依赖** — 需要 `config/parameter_master_table.csv` (已包含在包内)
5. **CHTD 模型 AS_FOUND** — 未启用 corrected_chtd / runtime_conditioning 等优化特征
