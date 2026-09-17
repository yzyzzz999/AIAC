#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 链路门控逻辑白盒测试：绕过 __init__ 直接构造实例测试三个新方法。"""
import sys, time
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.main_controller import MainController as C  # noqa

def make():
    c = C.__new__(C)
    c.can_fetcher = MagicMock()
    c._can_link_was_connected = True
    c._can_link_down_time = 0.0
    c._can_link_up_time = 0.0
    c._can_recover_guard_seconds = 10.0
    c._send_temp_jump_tolerance = 2.0
    c._last_command_result = None
    return c

ok = fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print(f"  PASS {name}")
    else: fail += 1; print(f"  FAIL {name}")

# --- _update_can_link_state 沿检测 ---
c = make()
c.can_fetcher.connected = False
c._update_can_link_state()
check("断开沿记录down_time", c._can_link_down_time > 0 and c._can_link_up_time == 0)
c.can_fetcher.connected = True
c._update_can_link_state()
check("恢复沿记录up_time", c._can_link_up_time >= c._can_link_down_time)

# --- 健康门控 ---
c = make()
c.can_fetcher.connected = False
check("断开时不健康", c._can_link_healthy_for_send() is False)

c = make()
c.can_fetcher.connected = True
c.can_fetcher.is_signal_fresh.return_value = True
check("连接且信号新鲜=健康", c._can_link_healthy_for_send() is True)

c.can_fetcher.is_signal_fresh.side_effect = lambda n: n != "AC_FBlowSpeedLevel"
check("任一反馈信号过期=不健康", c._can_link_healthy_for_send() is False)

# --- 跳变校验 ---
c = make()
# 模拟刚经历一次 断开->恢复
c._can_link_down_time = time.time() - 8
c._can_link_up_time = time.time() - 3   # 恢复3秒, 在10s观察窗内
c._last_command_result = {"driver_temp": 22.0, "passenger_temp": 23.0,
                          "wind_speed": 4.0, "air_mode": 2.0}
check("恢复窗内跳变+3.5°C=可疑", c._send_jump_suspect({"driver_temp": 25.5, "passenger_temp": 23.0}) is True)
check("恢复窗内跳变+1.5°C=正常", c._send_jump_suspect({"driver_temp": 23.5, "passenger_temp": 23.0}) is False)
c._can_link_up_time = time.time() - 15  # 恢复15秒, 已过观察窗
check("超过观察窗不校验", c._send_jump_suspect({"driver_temp": 26.0, "passenger_temp": 23.0}) is False)
c = make()
check("无历史命令不校验", c._send_jump_suspect({"driver_temp": 26.0}) is False)

print(f"\n结果: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
