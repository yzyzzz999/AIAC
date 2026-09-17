#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - CAN Signal Fetcher
==============================================
CAN信号获取模块，从CAN总线获取实车存在的11个信号。

连接机制向 canid_signal_fetcher.py 看齐：
- Unix Socket连接（AF_UNIX）
- 心跳机制（客户端发送心跳，服务端发送heartbeat时回复pong）
- 自动重连（连接失败/断开后自动重试）
- 信号新鲜度检测（超过超时时间认为信号失效）
- 行缓冲JSON解析（处理粘包）
- 演示模式支持（缺失信号使用默认值）

信号列表（实车存在的11个）：
1.  实时车速
2.  外温（整车）
3.  内温（整车）
4.  阳光传感器
5.  车内相对湿度
6.  前挡风玻璃温度
7.  主驾吹面出风口温度
8.  主驾吹脚出风口温度
9.  副驾吹面出风口温度
10. 副驾吹脚出风口温度
11. 后排中央吹面出风口温度

职责边界:
- 通过Unix Socket连接CAN服务端
- 订阅指定CAN ID并解析信号
- 支持心跳机制和自动重连
- 提供信号就绪状态查询
- 支持演示模式（缺失信号使用默认值）
- 支持天气信号透传

接口:
- CanSignalFetcher: CAN信号获取器
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.config_manager import get_config
from src.utils.constants import (
    ACTIVE_CAN_SIGNAL_NAMES,
    CAN_SIGNAL_MAP,
    FEATURE_DEFAULTS,
    WEATHER_SIGNAL_NAMES,
)
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# ============================================================
# CAN信号获取器
# ============================================================
class CanSignalFetcher:
    """从CAN总线获取信号的模块，支持自动重连和心跳"""

    # 默认配置常量
    CDC_HVAC_SUBSCRIPTION_ID = "0x387"
    AI_STATE_SUBSCRIPTION_ID = "0xF001"
    DEFAULT_SOCKET_PATH = "/home/data/AIAC/can_service/sock/can0_bus.sock"
    DEFAULT_CAN_IDS = [
        "0x18F", "0x338", "0x33F", "0x358", "0x371", "0x3B9"
    ]
    DEFAULT_CLIENT_ID = "recommendation_client"
    DEFAULT_HEARTBEAT_INTERVAL = 5.0
    DEFAULT_HEARTBEAT_TIMEOUT = 15.0
    DEFAULT_RETRY_INTERVAL = 5.0
    DEFAULT_SIGNAL_TIMEOUT = 5.0

    def __init__(
        self,
        socket_path: Optional[str] = None,
        can_ids: Optional[List[int]] = None,
        client_id: Optional[str] = None,
    ):
        cfg = get_config()
        can_cfg = cfg.get_section("can")

        self.socket_path = socket_path or can_cfg.get("socket_path", self.DEFAULT_SOCKET_PATH)
        configured_can_ids = can_ids or can_cfg.get("can_ids", self.DEFAULT_CAN_IDS)
        self.can_ids = list(configured_can_ids)
        subscribed_ids = {
            int(can_id, 0) if isinstance(can_id, str) else int(can_id)
            for can_id in self.can_ids
        }
        for required_id, required_text in (
            (0x387, self.CDC_HVAC_SUBSCRIPTION_ID),
            (0xF001, self.AI_STATE_SUBSCRIPTION_ID),
        ):
            if required_id not in subscribed_ids:
                self.can_ids.append(required_text)
        self.client_id = client_id or can_cfg.get("client_id", self.DEFAULT_CLIENT_ID)
        self.heartbeat_interval = can_cfg.get("heartbeat_interval", self.DEFAULT_HEARTBEAT_INTERVAL)
        self.heartbeat_timeout = can_cfg.get("heartbeat_timeout", self.DEFAULT_HEARTBEAT_TIMEOUT)
        self.retry_interval = can_cfg.get("retry_interval", self.DEFAULT_RETRY_INTERVAL)
        self.signal_timeout = can_cfg.get("signal_timeout", self.DEFAULT_SIGNAL_TIMEOUT)

        self.sock: Optional[socket.socket] = None
        self.buffer = ""
        self.connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_heartbeat = 0.0

        # 信号存储
        self._signals: Dict[str, float] = {}
        self._signal_timestamps: Dict[str, float] = {}
        self._ai_state = "on"
        self._ai_state_timestamp = 0.0
        self._cdc_hvac_actions: Dict[str, float] = {}
        self._cdc_action_timestamps: Dict[str, float] = {}
        self._cdc_wind_10_logged = False
        self._lock = threading.Lock()

    # ---------------------------
    # 连接管理
    # ---------------------------
    def connect(self) -> bool:
        """连接CAN Socket服务端"""
        self._cleanup_socket()
        try:
            if hasattr(socket, "AF_UNIX"):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(self.socket_path)
            else:
                import platform
                if platform.system() == "Windows":
                    logger.warning("Windows平台不支持Unix Socket，CAN获取将使用模拟模式")
                    self.connected = False
                    return False
                else:
                    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self.sock.connect(self.socket_path)

            self.connected = True
            self.last_heartbeat = time.time()
            logger.info(f"CAN信号获取器已连接到 {self.socket_path}")

            # 发送订阅请求
            if not self._send_subscribe():
                self._cleanup_socket()
                return False
            return True

        except FileNotFoundError:
            logger.error(f"CAN Socket文件不存在: {self.socket_path}")
            return False
        except ConnectionRefusedError:
            logger.error("CAN连接被拒绝，请确保CAN服务端正在运行")
            return False
        except Exception as e:
            logger.error(f"CAN连接失败: {e}")
            return False

    def _cleanup_socket(self) -> None:
        """清理Socket连接"""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.connected = False
        if hasattr(self, "_lock"):
            with self._lock:
                self._ai_state = "on"
                self._ai_state_timestamp = 0.0
                self._cdc_hvac_actions.clear()
                self._cdc_action_timestamps.clear()
                self._cdc_wind_10_logged = False

    def _send_subscribe(self) -> bool:
        """发送订阅请求"""
        try:
            msg = json.dumps({
                "client_id": self.client_id,
                "filters": self.can_ids,
                "version": "1.0"
            })
            self.sock.sendall((msg + "\n").encode("utf-8"))
            resp = self.sock.recv(1024).decode("utf-8")
            logger.info(f"CAN服务器响应: {resp.strip()}")
            return True
        except Exception as e:
            logger.error(f"CAN发送订阅失败: {e}")
            return False

    def _send_heartbeat(self) -> None:
        """发送心跳"""
        try:
            heartbeat_msg = json.dumps({
                "type": "heartbeat",
                "timestamp": time.time()
            })
            self.sock.sendall((heartbeat_msg + "\n").encode("utf-8"))
            self.last_heartbeat = time.time()
        except Exception as e:
            logger.warning(f"CAN发送心跳失败: {e}")

    def _reply_pong(self) -> None:
        """回复心跳pong"""
        try:
            pong_msg = json.dumps({
                "type": "pong",
                "timestamp": time.time()
            })
            self.sock.sendall((pong_msg + "\n").encode("utf-8"))
        except Exception as e:
            logger.warning(f"CAN发送pong失败: {e}")

    # ---------------------------
    # 信号接收循环
    # ---------------------------
    def run(self) -> None:
        """主循环：接收CAN数据（阻塞）"""
        logger.info("CAN信号获取线程启动")
        self._running = True

        while self._running:
            if not self.connect():
                time.sleep(self.retry_interval)
                continue

            logger.info("CAN开始接收数据...")
            heartbeat_timer = time.time()

            try:
                while self._running and self.connected:
                    # 发送心跳
                    if time.time() - heartbeat_timer >= self.heartbeat_interval:
                        self._send_heartbeat()
                        heartbeat_timer = time.time()

                    # 检查心跳超时
                    if time.time() - self.last_heartbeat > self.heartbeat_timeout:
                        logger.warning("CAN心跳超时，重新连接...")
                        break

                    # 接收数据（短超时避免buffer堵死）
                    self.sock.settimeout(0.05)
                    try:
                        data = self.sock.recv(65536).decode("utf-8")
                        if not data:
                            logger.warning("CAN服务端断开")
                            break

                        self.last_heartbeat = time.time()
                        self.buffer += data

                        # 解析JSON行
                        while "\n" in self.buffer:
                            line, self.buffer = self.buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                msg = json.loads(line)
                                self._process_message(msg)
                            except json.JSONDecodeError:
                                logger.warning(f"CAN JSON解析失败: {line[:100]}")

                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.error(f"CAN接收异常: {e}")
                        break

            finally:
                self._cleanup_socket()

    def start(self) -> None:
        """在后台线程启动信号接收"""
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        logger.info("CAN信号接收线程已启动")

    def stop(self) -> None:
        """停止信号接收"""
        self._running = False
        self._cleanup_socket()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("CAN信号获取已停止")

    # ---------------------------
    # 数据解析
    # ---------------------------
    def _process_message(self, msg: Dict[str, Any]) -> None:
        """处理单条CAN消息"""
        msg_type = msg.get("type")
        if msg_type == "heartbeat":
            self._reply_pong()
            return

        data = msg.get("data", {})
        raw = msg.get("raw", {})
        timestamp = msg.get("timestamp", time.time())
        self.last_heartbeat = time.time()

        if msg_type == "ai_state":
            ai_state = data.get("ai") if isinstance(data, dict) else None
            if ai_state in ("on", "off"):
                with self._lock:
                    previous = self._ai_state
                    self._ai_state = ai_state
                    self._ai_state_timestamp = timestamp
                    if previous == "on" and ai_state == "off":
                        self._cdc_hvac_actions.clear()
                        self._cdc_action_timestamps.clear()
                        self._cdc_wind_10_logged = False
                if previous != ai_state:
                    logger.info(f"CAN AI状态变更: {previous} -> {ai_state}")
            return

        if isinstance(data, dict) and "CDC_DriverTempCSet" in data:
            self._process_cdc_hvac_message(data, raw, timestamp)

        with self._lock:
            for signal_name, value in data.items():
                if value is None:
                    continue
                try:
                    float_value = float(value)
                except (ValueError, TypeError):
                    continue
                if np.isnan(float_value):
                    continue

                self._signals[signal_name] = float_value
                self._signal_timestamps[signal_name] = timestamp

    def _process_cdc_hvac_message(
        self,
        data: Dict[str, Any],
        raw: Dict[str, Any],
        timestamp: float,
    ) -> None:
        """锁存AI关闭期间0x387中的有效操作请求。"""
        with self._lock:
            if self._ai_state != "off":
                return

            for signal_name, action_name in (
                ("CDC_DriverTempCSet", "driver_temp"),
                ("CDC_PassengerTempCSet", "passenger_temp"),
            ):
                try:
                    raw_value = int(raw.get(signal_name))
                except (TypeError, ValueError):
                    continue
                if 0 <= raw_value <= 30:
                    self._cdc_hvac_actions[action_name] = 16.0 + raw_value * 0.5
                    self._cdc_action_timestamps[action_name] = timestamp

            try:
                wind = int(raw.get("CDC_FHvacBlowLvSet"))
            except (TypeError, ValueError):
                wind = 0
            if 1 <= wind <= 10:
                if wind == 10 and not self._cdc_wind_10_logged:
                    logger.warning("CDC风量10超出偏好模型范围，训练标签按9记录")
                    self._cdc_wind_10_logged = True
                self._cdc_hvac_actions["wind_speed"] = float(min(wind, 9))
                self._cdc_action_timestamps["wind_speed"] = timestamp

            try:
                air_mode = int(raw.get("CDC_FHvacModeSet"))
            except (TypeError, ValueError):
                air_mode = 0
            if 1 <= air_mode <= 7:
                self._cdc_hvac_actions["air_mode"] = float(air_mode)
                self._cdc_action_timestamps["air_mode"] = timestamp

    # ---------------------------
    # 信号查询
    # ---------------------------
    def get_signal(self, name: str) -> Optional[float]:
        """获取单个信号值"""
        with self._lock:
            return self._signals.get(name)

    def get_ai_state(self) -> str:
        """获取CAN服务推送的AI开关；首次推送前默认按on处理。"""
        with self._lock:
            return self._ai_state

    def get_cdc_hvac_actions(
        self, ac_fallback: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, float]]:
        """返回锁存的CDC人工动作，未操作字段由当前AC状态补齐。"""
        required = ("driver_temp", "passenger_temp", "wind_speed", "air_mode")
        with self._lock:
            merged = dict(ac_fallback or {})
            merged.update(self._cdc_hvac_actions)
            if not all(name in merged and merged[name] is not None for name in required):
                return None
            try:
                actions = {name: float(merged[name]) for name in required}
            except (TypeError, ValueError):
                return None
            if not (16.0 <= actions["driver_temp"] <= 31.0 and
                    16.0 <= actions["passenger_temp"] <= 31.0 and
                    1.0 <= actions["wind_speed"] <= 10.0 and
                    1.0 <= actions["air_mode"] <= 7.0):
                return None
            actions["wind_speed"] = min(actions["wind_speed"], 9.0)
            return actions

    def get_feature_value(self, feature_name: str) -> float:
        """获取特征值（支持多信号取平均，如阳光传感器左右取平均）"""
        signal_names = CAN_SIGNAL_MAP.get(feature_name, [feature_name])
        values = []
        with self._lock:
            for sig in signal_names:
                val = self._signals.get(sig)
                if val is not None and not np.isnan(val):
                    values.append(val)
        if values:
            return float(np.mean(values))
        return FEATURE_DEFAULTS.get(feature_name, 0.0)

    def is_signal_fresh(self, signal_name: str, timeout: Optional[float] = None) -> bool:
        """检查信号是否在指定时间内更新"""
        timeout = timeout or self.signal_timeout
        with self._lock:
            ts = self._signal_timestamps.get(signal_name)
            if ts is None:
                return False
            return (time.time() - ts) <= timeout

    def is_feature_fresh(self, feature_name: str, timeout: Optional[float] = None) -> bool:
        """检查特征是否有新鲜信号"""
        signal_names = CAN_SIGNAL_MAP.get(feature_name, [feature_name])
        for sig in signal_names:
            if self.is_signal_fresh(sig, timeout):
                return True
        return False

    # ---------------------------
    # 演示模式支持
    # ---------------------------
    def get_all_features(self, demo_mode: bool = False, weather_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """获取所有特征值（含天气信号）"""
        result = {}
        used_defaults = []

        # 1. 处理实车存在的信号
        for feature_name in CAN_SIGNAL_MAP.keys():
            fresh = self.is_feature_fresh(feature_name)
            if fresh:
                result[feature_name] = self.get_feature_value(feature_name)
            elif demo_mode:
                default_val = FEATURE_DEFAULTS.get(feature_name, 0.0)
                result[feature_name] = default_val
                used_defaults.append(feature_name)

        # 2. 添加天气信号（如提供）
        if weather_features:
            for name in WEATHER_SIGNAL_NAMES:
                if name in weather_features:
                    result[name] = weather_features[name]

        # 3. 演示模式：天气信号缺失用默认值
        if demo_mode:
            for name in WEATHER_SIGNAL_NAMES:
                if name not in result or result[name] is None:
                    result[name] = FEATURE_DEFAULTS.get(name, 0.0)

        if demo_mode and used_defaults:
            logger.info(f"[演示模式] CAN信号使用默认值: {len(used_defaults)}/{len(CAN_SIGNAL_MAP)}, "
                       f"默认信号: {', '.join(used_defaults[:5])}{'...' if len(used_defaults) > 5 else ''}")

        return result

    def get_ready_signals(self, demo_mode: bool = False) -> Tuple[List[str], List[str]]:
        """获取已就绪和缺失的信号列表"""
        if demo_mode:
            return list(CAN_SIGNAL_MAP.keys()), []

        ready = []
        missing = []
        for feature_name in CAN_SIGNAL_MAP.keys():
            if self.is_feature_fresh(feature_name):
                ready.append(feature_name)
            else:
                missing.append(feature_name)
        return ready, missing

    def is_all_ready(self, demo_mode: bool = False) -> Tuple[bool, List[str]]:
        """检查全部实车信号是否都已就绪"""
        if demo_mode:
            return True, []
        ready, missing = self.get_ready_signals()
        return len(missing) == 0, missing

    def get_feature_count(self, demo_mode: bool = False) -> Tuple[int, int]:
        """获取已就绪的特征数量"""
        if demo_mode:
            return len(CAN_SIGNAL_MAP), len(CAN_SIGNAL_MAP)
        ready, _ = self.get_ready_signals()
        return len(ready), len(CAN_SIGNAL_MAP)
