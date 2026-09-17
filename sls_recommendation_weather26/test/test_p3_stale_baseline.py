#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 修复白盒测试: 训练后过期命令基线导致秒级phantom接管。

复现依据: 2026-08-03 12:06 车端日志 —— 训练完成→JOINT_ACTIVE 1秒后,
系统拿人工设定(24.0/风量3)对比接管前最后AI命令(19.5/风量6),
phantom不一致 → 5秒后二次接管, 形成死循环。
"""
import sys, time, threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.preference_learning.preference_layer_manager import (
    PreferenceLayerManager, PreferenceState,
)


def make():
    m = PreferenceLayerManager.__new__(PreferenceLayerManager)
    m._lock = threading.RLock()
    m.TEMP_TOLERANCE = 0.5
    m.COMMAND_ACK_TIMEOUT = 8.0
    m.ACK_MISMATCH_FALLBACK_SECONDS = 4.0
    m.TAKEOVER_CONFIRM_SECONDS = 5.0
    m._expected_command = None
    m._expected_command_time = 0.0
    m._expected_command_acknowledged = False
    m._command_ack_timeout_logged = False
    m._post_train_cooldown = 0.0
    m._takeover_start = 0.0
    m._takeover_confirmed = False
    m._state = PreferenceState.JOINT_ACTIVE
    m._samples = []
    m._last_human_actions = None
    m._collection_start = 0.0
    m._current_user_id = "u1"
    m._last_training_report = None
    return m


# 12:00:36 接管前最后一条AI命令(已被CAN确认)
STALE = {"driver_temp": 19.5, "passenger_temp": 24.5, "wind_speed": 6.0, "air_mode": 2.0}
# 她的人工设定
HUMAN = {"driver_temp": 24.0, "passenger_temp": 24.5, "wind_speed": 3.0, "air_mode": 2.0}
# 训练后首轮联合推理应发出的值(贴近人工)
FRESH = {"driver_temp": 24.0, "passenger_temp": 24.5, "wind_speed": 3.0, "air_mode": 2.0}
# 她再次手动改温
CHANGED = {"driver_temp": 25.5, "passenger_temp": 24.5, "wind_speed": 3.0, "air_mode": 2.0}

ok = fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print(f"  PASS {name}")
    else: fail += 1; print(f"  FAIL {name}")


# 1. 训练完成(TRAINING→JOINT_ACTIVE)后, 过期命令基线必须被清空
m = make()
m.note_command_sent(STALE, time.time() - 310)   # 接管前最后命令
m._detect_takeover(STALE, STALE)                # CAN确认, ack开放
m._state = PreferenceState.TRAINING
m._load_mlp_for_user = lambda uid: True         # 跳过磁盘加载
m.on_training_done({"success": True, "user_id": "u1",
                    "base_score": 1.04, "updated_score": 0.02})
check("1 训练完成→基线清空+进入JOINT_ACTIVE",
      m._state == PreferenceState.JOINT_ACTIVE and m._expected_command is None
      and not m._expected_command_acknowledged)

# 2. 复现12:06: 恢复推理后主循环仍传入过期rf_result(STALE), 不得phantom接管
r = m._handle_joint_active({}, STALE, HUMAN)
check("2 过期基线+人工反馈→不触发接管(核心场景)",
      not r.get("takeover_pending", False) and m._state == PreferenceState.JOINT_ACTIVE)

# 3. 门卫独立验证: 本阶段未真实发送任何命令时, 一律不做接管检测
m = make()
r = m._handle_joint_active({}, STALE, HUMAN)
check("3 _expected_command为空→跳过接管检测",
      not r.get("takeover_pending", False) and m._expected_command is None)

# 4. 回归: 新命令真实发送+ack开放后, 人工再调温 → 正常触发接管
m = make()
m.note_command_sent(FRESH, time.time())
r1 = m._handle_joint_active({}, FRESH, FRESH)   # CAN回读匹配 → ack开放
r2 = m._handle_joint_active({}, FRESH, CHANGED) # 她改到25.5
check("4 发送+ack后人工改动→正常判接管pending",
      not r1.get("takeover_pending", False) and r2.get("takeover_pending") is True)

# 5. 回归: P2的ack兜底不受P3影响 —— 命令发出≥4s未获确认且差异持续 → 判接管
m = make()
m.note_command_sent(FRESH, time.time() - 5)
check("5 ack挂起≥4s差异持续→P2兜底仍生效",
      m._detect_takeover(FRESH, CHANGED) is True)

# 6. 训练失败回退RF_ONLY: 基线同样清空, 且RF_ONLY门卫生效
m = make()
m.note_command_sent(STALE, time.time() - 310)
m._detect_takeover(STALE, STALE)
m._state = PreferenceState.TRAINING
m.on_training_done({"success": False, "reason": "score_no_improve"})
r = m._handle_rf_only({}, STALE, HUMAN)
check("6 训练失败回退RF_ONLY→基线清空+不phantom接管",
      m._state == PreferenceState.RF_ONLY and m._expected_command is None
      and not r.get("takeover_pending", False))

print(f"\n结果: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
