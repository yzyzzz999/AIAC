# PMV 调试记录 — CAN In-Car Temp 对标

## 问题

CHTD 模型 cabin/head 温度与 CAN 实测车内温度偏差很大。

## 关键发现

### 2026-06-20 — CAN 实测 vs CHTD 模型

```
CAN AC_FrntInCarT = 26.5°C  ← 真实车内温度（前排）
CHTD cabin        = 20.2°C  ← 模型算的，差 6.3°C
CHTD head         = 19.6°C  ← PMV 用的 ta，差 6.9°C
外温 VIU_AmbT     = 23.0°C
```

### PMV 偏差

| 温度源 | 值 | PMV (clo=0.4) | 体感 |
|--------|-----|---------------|------|
| CAN 实测 | 26.5°C | **+0.16** | 中性 ✓ |
| CHTD head | 19.6°C | **−3.32** | 很冷 ✗ |

### 根因

v8_L1 标定参数（`stage2_v4_calibrated` mode）与当前车辆不匹配：
- CHTD head 区（x[3] HeadTempFd）被车窗/车顶冷表面强耦合拉低
- Cabin 区（x[15] CabinTempFd）稍好但仍偏低
- `init_temp_c` 从 ict_c 起步后，模型在 20-30 步内跌回 20°C

### 尝试的修复

1. **在线自修正**：每步用 CAN 实测值 nudge CHTD 状态
   - Cabin 区: 5%/step → 收敛到 ~26°C ✓
   - Head 区: 15%/step → 勉强到 24°C（CHTD 动态持续拉低）
   - PMV 从 -3.3 → -1.3，有改善但仍偏离

2. **后续方向**：
   - PMV ta 直接取 CAN 实测 `AC_FrntInCarT`，CHTD 模型只做热动态
   - 或联系师兄确认 v8_L1 标定数据来源车辆是否匹配

### 已完成的代码改动（当前分支）

| 文件 | 改动 |
|------|------|
| `run_vehicle_pmv.py` | +`previous_state` +`init_temp_c` 参数；补 PPD；修 cabin X_INDEX |
| `pmv_socket_consumer.py` | TMA key 修正；flow PWM 缩放；ict_c 初始化；在线自修正 |
| `occupant.py` | v8_L1 自带 met 简化，不需额外改 |
| `hvac_sim/` | 全量替换为 v8_L1 |
| `config/` | 替换为 v8_L1 标定参数 |
