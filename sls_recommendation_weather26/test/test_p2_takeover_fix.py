#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 接管检测修复白盒测试: ack兜底 + 冷却窗移除。"""
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
    return m

AI = {"driver_temp": 19.0, "passenger_temp": 23.5, "wind_speed": 6.0, "air_mode": 2.0}
HUMAN = {"driver_temp": 22.0, "passenger_temp": 23.5, "wind_speed": 6.0, "air_mode": 2.0}

ok = fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print(f"  PASS {name}")
    else: fail += 1; print(f"  FAIL {name}")

# 1. 反馈匹配命令 → ack开放, 不判接管
m = make()
m.note_command_sent(AI, time.time())
check("1 反馈匹配→ack开放无接管", m._detect_takeover(AI, AI) is False and m._expected_command_acknowledged)

# 2. ack挂起+差异+未满4s → 不判接管
m = make()
m.note_command_sent(AI, time.time())
check("2 ack挂起<4s→不判接管", m._detect_takeover(AI, HUMAN) is False)

# 3. ack挂起+差异持续≥4s → P2兜底判接管
m = make()
m.note_command_sent(AI, time.time() - 5)
check("3 ack挂起≥4s差异持续→兜底判接管(P2)", m._detect_takeover(AI, HUMAN) is True)

# 4. ack已开放后用户改动 → 判接管(原有行为回归)
m = make()
m.note_command_sent(AI, time.time())
m._detect_takeover(AI, AI)  # 打开ack
check("4 ack开放后差异→判接管", m._detect_takeover(AI, HUMAN) is True)

# 5. 冷却窗内(P2移除后)仍执行接管检测
m = make()
m._post_train_cooldown = time.time() + 60.0  # 冷却中
m.note_command_sent(AI, time.time())
m._detect_takeover(AI, AI)  # 打开ack
r = m._handle_joint_active({}, AI, HUMAN)
check("5 冷却窗内差异→仍判接管pending(P2)", r.get("takeover_pending") is True)

# 6. 冷却窗内反馈正常 → 不误判
m = make()
m._post_train_cooldown = time.time() + 60.0
m.note_command_sent(AI, time.time())
r = m._handle_joint_active({}, AI, AI)
check("6 冷却窗内反馈正常→不误判", not r.get("takeover_pending", False))

# 7. 反馈值非法(越界) → 不判接管(原有保护回归)
m = make()
m.note_command_sent(AI, time.time() - 5)
bad = dict(HUMAN); bad["driver_temp"] = 99.0
check("7 反馈越界→不判接管", m._detect_takeover(AI, bad) is False)

print(f"\n结果: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
