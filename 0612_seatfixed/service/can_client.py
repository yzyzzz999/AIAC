"""
service/can_client.py
====================
CAN 总线座椅位置客户端。

连接到 can0_service (v1.3.2+) 的 Unix Socket，订阅 0x552 (主驾) 和 0x553 (副驾)
的座椅电机位置信号，实时更新 HeightEstimator 的座椅参数。

协议：
  1. 连接 → 发送订阅 {client_id, filters: [0x552, 0x553]}
  2. 收到 ACK → 进入接收循环
  3. 收到 heartbeat → 回复 pong
  4. 收到 data → 提取 RTE 信号名 → set_seat_params()
  5. 断线自动重连
"""

from __future__ import annotations

import json
import os
import socket
import time
import logging
import threading
from typing import Optional

log = logging.getLogger("can_client")

# ── Socket 路径（基于脚本自身位置，不依赖工作目录）──
CAN_SOCKET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "can_service", "sock", "can0_bus.sock",
)

# ── 客户端标识 ──
CLIENT_ID = "face_recognition"

# ── 订阅的 CAN ID ──
CID_DSM_552 = 0x552  # 主驾座椅电机位置
CID_PSM_553 = 0x553  # 副驾座椅电机位置

# ── RTE 信号名（can0_service v1.3.2 转换后的 key）──
SIG_DRIVER_SLIDE = "DriverSeat_SlidePosition"
SIG_DRIVER_BACKREST = "DriverSeat_BackrestPosition"
SIG_PASSENGER_SLIDE = "PassengerSeat_SlidePosition"
SIG_PASSENGER_BACKREST = "PassengerSeat_BackrestPosition"


class CANSeatClient:
    """CAN 座椅位置客户端：后台线程连接 CAN socket，订阅并实时更新 height_estimator。"""

    def __init__(self, height_estimator):
        self._he = height_estimator
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cache = {"dx": None, "dy": None, "px": None, "py": None}
        self._first_data = True
        self._buf = ""  # ACK 读多出的数据暂存

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="CANSeatClient"
        )
        self._thread.start()
        log.info("CAN seat client started")

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    # ── 连接 + 订阅 ─────────────────────────────────────────────────────

    def _connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect(CAN_SOCKET_PATH)

            sub_msg = json.dumps({
                "client_id": CLIENT_ID,
                "filters": [CID_DSM_552, CID_PSM_553],
                "version": "1.0",
            }) + "\n"
            self._sock.sendall(sub_msg.encode("utf-8"))

            ack_raw = self._sock.recv(1024).decode("utf-8")
            lines = ack_raw.split("\n")
            ack_line = lines[0].strip()
            ack = json.loads(ack_line)
            if ack.get("status") == "ok":
                log.info("CAN seat client connected, subscribed 0x552/0x553")
                self._sock.settimeout(1.0)
                self._buf = "\n".join(lines[1:])  # 多读的消息暂存
                return True
            log.warning("CAN seat client unexpected ACK: %s", ack)
            return False
        except Exception as e:
            log.debug("CAN seat client connect failed: %s", e)
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
            return False

    # ── 主循环 ──────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            if not self._connect():
                time.sleep(3.0)
                continue

            buf = self._buf  # 先消费 ACK 时多读的数据
            self._buf = ""
            try:
                while self._running:
                    try:
                        data = self._sock.recv(4096).decode("utf-8")
                        if not data:
                            raise ConnectionError("empty recv")
                        buf += data
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            self._handle_message(line.strip())
                    except socket.timeout:
                        continue
            except Exception as e:
                log.debug("CAN seat client disconnected: %s", e)
            finally:
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                time.sleep(3.0)

    # ── 消息处理 ────────────────────────────────────────────────────────

    def _handle_message(self, line: str):
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")
        if msg_type == "heartbeat":
            self._send_pong()
        elif msg_type == "data":
            self._handle_data(msg.get("data", {}))

    def _send_pong(self):
        if self._sock:
            try:
                pong = json.dumps({"type": "pong"}) + "\n"
                self._sock.sendall(pong.encode("utf-8"))
            except Exception:
                pass

    def _handle_data(self, data: dict):
        """提取 RTE 座椅信号并推送到 height_estimator。"""
        dx_raw = data.get(SIG_DRIVER_SLIDE)
        dy_raw = data.get(SIG_DRIVER_BACKREST)
        px_raw = data.get(SIG_PASSENGER_SLIDE)
        py_raw = data.get(SIG_PASSENGER_BACKREST)

        dx = dx_raw / 100.0 if dx_raw is not None else None
        dy = dy_raw / 100.0 if dy_raw is not None else None
        px = px_raw / 100.0 if px_raw is not None else None
        py = py_raw / 100.0 if py_raw is not None else None

        if any(v is None for v in (dx, dy, px, py)):
            return

        # 仅值变化时才更新（避免频繁重置累加器）
        if (dx == self._cache["dx"] and dy == self._cache["dy"] and
                px == self._cache["px"] and py == self._cache["py"]):
            return

        self._cache = {"dx": dx, "dy": dy, "px": px, "py": py}
        try:
            self._he.set_seat_params(dx, dy, px, py)
            if self._first_data:
                self._first_data = False
                self._he.mark_can_received()
            log.debug("Seat updated: d=(%.3f,%.3f) p=(%.3f,%.3f)", dx, dy, px, py)
        except Exception as e:
            log.debug("set_seat_params failed: %s", e)
