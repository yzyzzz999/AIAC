# 部署局限性

**文档版本**: vehicle_runtime_package_vnext_v1
**日期**: 2026-06-15

## P0 — 量产阻断

| 编号 | 类型 | 说明 | 影响 |
|------|------|------|------|
| P0-1 | 标定未完成 | `calibration_status=midterm_preview_not_vehicle_calibration` | PMV 绝对精度无法保证 |
| P0-2 | V3 PSO 未执行 | cache_v5 尚未完成 PSO 优化 | 参数非最优 |
| P0-3 | Task D 待定 | v5 baseline 对标未完成 | 无法确认 v5 较 v3 改善 |

## P1 — 已知限制（不阻断中期预览）

| 编号 | 类型 | 说明 | 影响量级 |
|------|------|------|---------|
| P1-1 | LR 50:50 假设 | 左右风量按 50:50 分配（PDF 无左右分流数据）| 头部温度偏差 ±0.3~0.8°C |
| P1-2 | H6 模式混合 | H6 工况 face_score=0.41/foot_score=0.42（混合位置）| 该工况风量分配可能有偏 |
| P1-3 | guarded 入口 | `run_vehicle_pmv_guarded.py` 依赖 `examples.run_runtime_pmv`，不在本包内 | guarded 版无法独立运行 |
| P1-4 | x_next 状态 | 无持久化状态管理，每次调用需手动传递 x_next | 长时序推理需上层管理状态 |
| P1-5 | 图像占用模块 | `enable_image_occupancy=True` 但 image_inputs 为可选 | 无图像输入时退化为默认乘员 |

## 已修复（本包确认）

| 项 | 状态 |
|----|------|
| airflow 均匀 fallback bug (v4 1/10 分配) | 已修复 (v5) |
| face_pure 模式检测 | 已修复 (v5，执行器优先策略) |
| stage2 V2 PSO INVALID 参数 | 未使用 (safe_preview 参数包) |

## 文件依赖

- `parameter_master_table.csv`: 必须位于包父目录 `../simulink_conversion_package/python_targets/`
  - 本包以 `F:\TempWork\SeriesPMV\001_APP\delivery\simulink_conversion_package\python_targets\` 形式提供
  - 量产部署需确认路径或修改 `hvac_sim/chtd/physical_params.py` 中的 `_DEFAULT_PARAM_TABLE`
