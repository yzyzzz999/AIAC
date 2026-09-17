#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 修复白盒测试: 采集期rf_result=None导致0样本卡死。

复现依据: 2026-08-03 14:29 接管→COLLECTING, 14分钟0样本未触发训练。
原因: P3清空_last_command_result后rf_result=None, 样本生成条件
"rf_result is not None"永不满足。修复: 采集期每5s补一次纯RF推理
(不发送), 为样本提供当前工况真实base预测。
"""
import sys, time, threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.preference_learning.preference_layer_manager import (
    PreferenceLayerManager, PreferenceState,
)

FEATS = {f"f{i}": float(i) for i in range(26)}
RF = {"driver_temp": 22.5, "passenger_temp": 24.5, "wind_speed": 4.0, "air_mode": 2.0}
HUMAN = {"driver_temp": 25.0, "passenger_temp": 23.0, "wind_speed": 1.0, "air_mode": 1.0}


def make():
    m = PreferenceLayerManager.__new__(PreferenceLayerManager)
    m._lock = threading.RLock()
    m.SAMPLE_INTERVAL = 5.0
    m.COLLECTION_DURATION = 300.0
    m.MIN_TRAINING_SAMPLES = 30
    m._feature_columns = [f"f{i}" for i in range(26)]
    m._state = PreferenceState.COLLECTING
    m._collection_start = time.time()
    m._last_sample_time = 0.0
    m._samples = []
    m._last_human_actions = None
    m._takeover_start = 0.0
    m._takeover_confirmed = False
    return m


ok = fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print(f"  PASS {name}")
    else: fail += 1; print(f"  FAIL {name}")


# 1. 采集期+实时RF预测 → 样本正常生成(P4核心)
m = make()
r = m._handle_collecting(FEATS, RF, HUMAN)
check("1 采集期rf_result非空→生成样本",
      len(m._samples) == 1 and m._samples[0].base_predictions["base_driver_temperature"] == 22.5)

# 2. 复现P3引入的卡死: rf_result=None → 不生成样本(修复前14分钟0样本)
m = make()
r = m._handle_collecting(FEATS, None, HUMAN)
check("2 rf_result=None→不生成样本(卡死场景复现)", len(m._samples) == 0)

# 3. 采样间隔门控: 5s内第二次调用不重复生成
m = make()
m._handle_collecting(FEATS, RF, HUMAN)
m._handle_collecting(FEATS, RF, HUMAN)
check("3 5s采样间隔门控生效", len(m._samples) == 1)

# 4. 时间满300s+样本≥30 → 触发训练
m = make()
m._collection_start = time.time() - 301
m._samples = [m._make_sample(FEATS, RF, HUMAN) for _ in range(30)]
r = m._handle_collecting(FEATS, RF, HUMAN)
check("4 300s+30样本→触发TRAINING",
      m._state == PreferenceState.TRAINING and r.get("trigger_training") is True)

# 5. state只读属性可用(P4主循环路由依赖)
m = make()
check("5 state属性返回COLLECTING", m.state == PreferenceState.COLLECTING)

print(f"\n结果: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
