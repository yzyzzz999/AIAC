# 车载 PMV Runtime 包就绪性裁定报告

**包**: `vehicle_runtime_package_vnext`
**日期**: 2026-06-15
**裁定**: **RUNTIME_PACKAGE_SCHEMA_READY_MODEL_NOT_FINAL**

---

## 裁定含义

| 裁定 | 含义 |
|------|------|
| RUNTIME_PACKAGE_READY_MIDTERM_PREVIEW | 包完整、冒烟通过、可用于中期预览演示 |
| **RUNTIME_PACKAGE_SCHEMA_READY_MODEL_NOT_FINAL** | **本次裁定** — 接口/Schema 就绪，模型参数标定未完成 |
| RUNTIME_PACKAGE_BLOCKED_FOR_MISSING_SIG | SIG/FTE 信号缺失阻断，本次不适用 |

---

## 就绪条件检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| hvac_sim 模型库完整 | PASS | 52 个 .py 文件，含 afe/ekf/chtd/pmv/calibration/config/core/interface |
| 冒烟测试 22/22 PASS | PASS | `tests/smoke_test_runtime.py` 全通过 |
| 禁止输入审计 | PASS | `forbidden_input_detected=False` |
| 输出 Schema 符合规范 | PASS | 所有必须字段存在 |
| CAN 例外文档化 | PASS | 14 个 CAN 信号已列入 B2 |
| 禁止输入文档化 | PASS | 见 B3 `FORBIDDEN_RUNTIME_INPUTS.md` |
| calibration_status 正确标注 | PASS | `midterm_preview_not_vehicle_calibration` |
| 不含 PSO INVALID 参数 | PASS | safe_preview 参数包，非 Stage2 V2 |
| SIG/FTE 信号覆盖 | PARTIAL | 14 个 CAN 例外，SIG/FTE 完整接入待完成 |
| 量产标定完成 | FAIL (P0) | V3 PSO 未执行，Task D baseline 待定 |

---

## 阻塞项（写入报告，不阻断 zip 生成）

### P0 阻塞（量产必须解决）

1. **标定未完成**: V3 PSO 未执行，`calibration_status=midterm_preview_not_vehicle_calibration`
   - 影响: PMV 绝对精度无保证，不允许量产标注
   - 解决路径: 完成 Task D baseline 对标 → V3 PSO → 量产标定

2. **Task D 待定**: v5 baseline 与 v3 对标未完成
   - 影响: 无法确认 airflow v5 修复后模型精度未退化
   - 解决路径: Cursor 提供 `cache_v5_baseline_metrics_by_zone.csv` 后 CC 独立对标

3. **guarded 入口受阻**: `run_vehicle_pmv_guarded.py` 依赖 `examples.run_runtime_pmv`（不在包内）
   - 影响: guarded 版本无法在本包独立运行
   - 解决路径: 将 `examples/run_runtime_pmv.py` 纳入包，或重构为包内独立函数

### P1 限制（中期预览允许）

- LR 50:50 假设（影响头部温度 ±0.3~0.8°C）
- H6 工况执行器混合位置警告
- 无状态持久化管理器（需上层管理 x_next）

---

## 包完整性

| 模块 | 文件数 | 状态 |
|------|--------|------|
| `hvac_sim/` (顶层) | 14 | PASS |
| `hvac_sim/chtd/` | 26 | PASS |
| `hvac_sim/pmv/` | 5 | PASS |
| `hvac_sim/interface/` | 2 | PASS |
| `hvac_sim/config/` | 5 | PASS |
| `hvac_sim/core/` | 3 | PASS |
| `hvac_sim/afe/` | 20+ | PASS |
| `hvac_sim/ekf/` | 2 | PASS（补充）|
| `hvac_sim/calibration/` | 13 | PASS（补充）|
| `config/` JSON | 4 | PASS |
| `tests/` | 3 | PASS |
| Schemas | 2 | PASS |
| 文档 MDs | 11 | PASS |

---

## 最终裁定

```
verdict = RUNTIME_PACKAGE_SCHEMA_READY_MODEL_NOT_FINAL

含义:
  - 包可运行: 冒烟测试 22/22 PASS
  - 接口已定义: 输入/输出 Schema、SIG/FTE 映射、CAN 例外清单
  - 模型参数: safe_preview（非量产），标定状态明确标注
  - 阻塞项: 已写入本报告（未阻断 zip 生成）
  - 下一步: Task D baseline → V3 PSO → 量产标定
```

---

*TDC-VEHICLE-RUNTIME-DEPLOYMENT-PACKAGE-VNEXT | 2026-06-15*
