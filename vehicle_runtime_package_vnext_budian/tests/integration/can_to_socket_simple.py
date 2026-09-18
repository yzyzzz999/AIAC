#!/usr/bin/env python3
"""简化版 CAN→Socket 服务（单进程，用于测试验证）。"""
import sys
import time
import json
import socket
import os
import threading
from pathlib import Path

import can

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOCKET_PATH = str(PROJECT_ROOT.parent / "can_service" / "sock" / "can0_bus.sock")
CHANNEL = sys.argv[1] if len(sys.argv) > 1 else "can2"

# RTE 信号映射（来自 can0_service 的 RTESignalMapper）
RTE_MAP = {
    # 0x35B VIU_35B
    "VIU_AmbT": "AmbientTemp",
    "VIU_AmbTVld": "AmbientTemp_Valid",
    # 0x3B9 VIU_LIN1
    "RSM_RelHum": "Relative_Humidity",
    "RSM_LeSolarInten": "Left_Solar_Intensity",
    "RSM_RiSolarInten": "Right_Solar_Intensity",
    # 0x18F VIU_CHA_18F
    "IPB_VehicleSpeed": "VehicleSpeed",
    "IPB_VehicleSpeedValid": "VehicleSpeed_Valid",
    # 0x33E AC_TEMP2
    "AC_FEvapCurrentTemp": "FrontEvap_CurrentTemp",
    "AC_FEvapTargetTemp": "FrontEvap_TargetTemp",
    "AC_THS_RelHum": "THS_RelativeHumidity",
    "AC_cabinCoolingLevel": "Cabin_CoolingLevel",
    "AC_cabinheatingLevel": "Cabin_HeatingLevel",
    # 0x33F AC_TEMP3
    "AC_DrvrFaceVentActT": "Driver_FaceVent_ActualTemp",
    "AC_PassFaceVentActT": "Passenger_FaceVent_ActualTemp",
    "AC_DrvrFaceVentTargetT": "Driver_FaceVent_TargetTemp",
    "AC_PassFaceVentTargetT": "Passenger_FaceVent_TargetTemp",
    # 0x338 AC_TEMP1
    "AC_FrntInCarT": "FrontInCarTemp",
    "AC_Forward_BlwPwmOut": "ChillerOutletTemp",
}

# 仅处理 PMV 相关的 CAN ID
TARGET_IDS = {0x18F, 0x338, 0x33E, 0x33F, 0x35B, 0x3B9}


def parse_signal_value(value):
    """提取数值，兼容字符串类型信号值。"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


class SimpleSocketServer:
    def __init__(self, sock_path):
        self.sock_path = sock_path
        self.clients = []
        self.lock = threading.Lock()

    def start(self):
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.sock_path)
        self.server.listen(5)
        self.server.settimeout(0.5)
        self.running = True
        print(f"[Socket] Listening on {self.sock_path}")
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self.server.accept()
                init = client.recv(1024).decode().strip()
                print(f"[Socket] New client: {init[:80]}")
                with self.lock:
                    self.clients.append(client)
                # 发 ACK
                ack = json.dumps({"status": "ok", "server_version": "test_1.0"})
                client.sendall(ack.encode() + b"\n")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Socket] Accept error: {e}")

    def broadcast(self, data: dict):
        with self.lock:
            dead = []
            for c in self.clients:
                try:
                    msg = json.dumps(data, ensure_ascii=False) + "\n"
                    c.sendall(msg.encode())
                except Exception:
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)


def main():
    print(f"[Service] Opening CAN bus: {CHANNEL}")
    can_bus = can.interface.Bus(channel=CHANNEL, interface="socketcan")
    print(f"[Service] CAN bus opened successfully")

    ss = SimpleSocketServer(SOCKET_PATH)
    ss.start()

    from cantools.database import load_file
    dbc_path = "/home/data/data_collection/CAN_utf8.dbc"
    db = load_file(dbc_path)
    print(f"[Service] DBC loaded: {dbc_path}")

    frame_count = 0
    broadcast_count = 0
    t0 = time.monotonic()

    print(f"[Service] Waiting for CAN data on {CHANNEL}...")
    while True:
        msg = can_bus.recv(1.0)
        if msg is None:
            continue

        if msg.arbitration_id not in TARGET_IDS:
            continue

        frame_count += 1

        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data, decode_choices=False)
        except Exception:
            continue

        # 提取 RTE 信号
        rte_data = {}
        for sig_name, sig_value in decoded.items():
            rte_name = RTE_MAP.get(sig_name)
            if rte_name is None:
                continue
            num_val = parse_signal_value(sig_value)
            if num_val is not None:
                rte_data[rte_name] = num_val

        if not rte_data:
            continue

        output = {
            "version": "1.0",
            "timestamp": msg.timestamp,
            "data": rte_data,
            "type": "data",
        }
        ss.broadcast(output)
        broadcast_count += 1

        elapsed = time.monotonic() - t0
        if elapsed >= 5.0:
            print(f"[Service] {frame_count} frames, {broadcast_count} broadcasts, {len(ss.clients)} clients")
            t0 = time.monotonic()


if __name__ == "__main__":
    main()
