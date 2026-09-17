#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Result Sender
===========================================
结果发送模块，将推理结果发送回CAN服务端。

职责边界:
- 通过CAN Socket连接发送结果
- 将温度、风速、模式编码为CAN帧格式
- 支持多次发送确保可靠性
- 提供发送状态查询

接口:
- ResultSender: 结果发送器
"""

from __future__ import annotations

import json
import select
import socket
import threading
import time
from typing import Any, Dict, Optional

from src.utils.config_manager import get_config
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# ============================================================
# 结果发送器
# ============================================================
class ResultSender:
    """将推理结果发送回CAN服务端"""

    def __init__(
        self,
        socket_path: Optional[str] = None,
        can_id: Optional[str] = None,
    ):
        cfg = get_config()
        result_cfg = cfg.get_section("result")

        self.socket_path = socket_path or result_cfg.get("socket_path", "/home/data/AIAC/can_service/sock/can0_bus.sock")
        self.can_id = can_id or result_cfg.get("can_id", "0x387")
        self.send_count = result_cfg.get("send_count", 5)
        self.send_interval_ms = result_cfg.get("send_interval_ms", 20)
        self.enabled = result_cfg.get("enabled", True)

        self.sock: Optional[socket.socket] = None
        self.connected = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_running = False

    def connect(self) -> bool:
        """连接CAN Socket服务端（先发订阅握手，再收ACK）"""
        if not self.enabled:
            logger.info("结果发送已禁用")
            return False

        try:
            if hasattr(socket, "AF_UNIX"):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(self.socket_path)
            else:
                import platform
                if platform.system() == "Windows":
                    logger.warning("Windows平台不支持Unix Socket，结果发送将使用模拟模式")
                    self.connected = False
                    return False
                else:
                    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self.sock.connect(self.socket_path)

            # 发送订阅消息（协议要求第一条消息必须是 client_id + filters）
            sub = json.dumps({"client_id": "result_sender", "filters": []})
            self.sock.sendall(sub.encode() + b"\n")

            # 读取 ACK（服务端处理其他客户端可能耗时，给10秒超时）
            self.sock.settimeout(10)
            ack = self.sock.recv(4096)
            ack_data = json.loads(ack.decode().strip())
            if ack_data.get("status") == "ok":
                self.sock.settimeout(None)
                self.connected = True
                # 启动心跳响应线程（不发pong会被服务端1.5s踢掉）
                self._heartbeat_running = True
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, daemon=True, name="result_sender_hb"
                )
                self._heartbeat_thread.start()
                logger.info(f"结果发送器已连接到 {self.socket_path}")
                return True
            else:
                logger.error(f"连接被拒绝: {ack_data}")
                self.sock.close()
                self.sock = None
                return False

        except FileNotFoundError:
            logger.error(f"CAN Socket文件不存在: {self.socket_path}")
            self.connected = False
            return False
        except ConnectionRefusedError:
            logger.error("CAN连接被拒绝，结果发送不可用")
            self.connected = False
            return False
        except socket.timeout:
            logger.error("等待服务端ACK超时")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"结果发送器连接失败: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """断开连接"""
        self._heartbeat_running = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1)
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.connected = False
        logger.info("结果发送器已断开")

    # 内部参数名 → DBC信号名 映射
    SIGNAL_MAP = {
        "driver_temp": "CDC_DriverTempCSet",
        "passenger_temp": "CDC_PassengerTempCSet",
        "wind_speed": "CDC_FHvacBlowLvSet",
        "air_mode": "CDC_FHvacModeSet",
    }

    def send(self, result: Dict[str, Any]) -> bool:
        """
        发送推理结果（按信号名+物理值，服务端自动DBC编码+路由）

        Args:
            result: 包含 driver_temp, passenger_temp, wind_speed, air_mode 的字典

        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            return False

        if not self.connected and not self.connect():
            return False

        try:
            # 转换为 DBC 信号名 → 物理值
            signals = {}
            for key, dbc_name in self.SIGNAL_MAP.items():
                val = result.get(key)
                if val is not None:
                    # 风量裁剪 1-9, 模式裁剪 1-7
                    if key == "wind_speed":
                        val = max(2, min(9, int(val)))
                    elif key == "air_mode":
                        val = max(1, min(7, int(val)))
                        # val = 1
                    # 温度裁剪 DBC有效范围: LO(0x00/16.0°C) ~ HI(0x1E/31.0°C)
                    elif key in ("driver_temp", "passenger_temp"):
                        val = max(16.0, min(31.0, float(val)))
                    signals[dbc_name] = val

            # 设置开关状态为 ON（发送推荐即表示开启），DBC: 2=ON
            signals["CDC_ACSystemOnOffSet"] = 2
            signals["CDC_ACSet"] = 2

            # 新协议: type=send_signal, 服务端自动DBC编码+路由
            msg = {
                "type": "send_signal",
                "signals": signals,
                "count": self.send_count,
                "interval_ms": self.send_interval_ms,
            }
            self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))

            logger.info(
                f"发送空调推荐结果: 主驾温度={signals.get('CDC_DriverTempCSet')}°C, "
                f"副驾温度={signals.get('CDC_PassengerTempCSet')}°C, "
                f"风量={signals.get('CDC_FHvacBlowLvSet')}, "
                f"模式={signals.get('CDC_FHvacModeSet')}"
            )
            logger.debug(f"完整发送信号: {signals}")
            return True

        except Exception as e:
            logger.error(f"结果发送失败: {e}")
            self.connected = False
            return False

    def send_from_result_dict(self, result: Dict[str, Any]) -> bool:
        """从推理结果字典发送"""
        return self.send({
            "driver_temp": result.get("driver_temp"),
            "passenger_temp": result.get("passenger_temp"),
            "wind_speed": result.get("wind_speed"),
            "air_mode": result.get("air_mode"),
        })

    def _heartbeat_loop(self) -> None:
        """心跳响应循环：回复服务端心跳，防止被踢"""
        buf = ""
        while self._heartbeat_running and self.sock:
            try:
                r, _, _ = select.select([self.sock], [], [], 0.5)
                if r:
                    data = self.sock.recv(4096)
                    if not data:
                        break  # 服务端关闭连接
                    buf += data.decode(errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            msg = json.loads(line)
                            if msg.get("type") == "heartbeat":
                                self.sock.sendall(json.dumps({"type": "pong"}).encode() + b"\n")
                        except json.JSONDecodeError:
                            pass
            except Exception:
                break
        self.connected = False

