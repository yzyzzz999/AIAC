#!/usr/bin/env python3
"""
CAN Socket 客户端例程
连接 can0 service 的 Unix Socket，实时读取 CAN 信号数据
"""

import socket
import json
import time
import threading
import os
import sys

# ============================================================
# 配置
# ============================================================
SOCKET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sock", "can0_bus.sock")

# 订阅的 CAN ID 列表（空列表 = 接收全部）
# 例如只关注 0x100 和 0x387:
# SUBSCRIBE_FILTERS = [0x100, 0x387]
SUBSCRIBE_FILTERS = [0x100]   # 只订阅 0x100 (CAN-Send-2)

# 客户端标识
CLIENT_ID = "example_client"


class CanSocketClient:
    def __init__(self, socket_path=SOCKET_PATH, client_id=CLIENT_ID, filters=None):
        self.socket_path = socket_path
        self.client_id = client_id
        self.filters = filters or []
        self.sock = None
        self.running = False
        self._recv_thread = None
        self._heartbeat_thread = None

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)

        # 发送注册消息
        init_msg = json.dumps({
            "client_id": self.client_id,
            "filters": self.filters,
            "version": "1.0"
        }) + "\n"
        self.sock.sendall(init_msg.encode("utf-8"))

        # 等待 ACK
        self.sock.settimeout(2.0)
        ack_data = self.sock.recv(4096).decode("utf-8").strip()
        ack = json.loads(ack_data)
        if ack.get("status") != "ok":
            raise RuntimeError(f"Connection rejected: {ack}")
        print(f"✓ Connected | server={ack.get('server_name')} v{ack.get('server_version')}")

        self.running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _recv_loop(self):
        buf = b""
        while self.running:
            try:
                self.sock.settimeout(0.5)
                chunk = self.sock.recv(4096)
                if not chunk:
                    print("✗ Server closed connection")
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode("utf-8"))
                        self._handle_message(msg)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"✗ Receive error: {e}")
                break
        self.running = False

    def _handle_message(self, msg):
        msg_type = msg.get("type", "")

        if msg_type == "data":
            can_data = msg.get("data", {})
            raw_data = msg.get("raw", {})
            ts = msg.get("timestamp", 0)

            print(f"\n[{ts:.3f}] CAN Data:")
            for name, value in can_data.items():
                raw = raw_data.get(name, "?")
                print(f"  {name:30s} = {value:<10} (raw={raw})")

        elif msg_type == "heartbeat":
            # 回复 pong
            try:
                pong = json.dumps({"type": "pong"}) + "\n"
                self.sock.sendall(pong.encode("utf-8"))
            except Exception:
                pass

    def _heartbeat_loop(self):
        pass  # pong 已在 _handle_message 中回复

    def disconnect(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        print("Disconnected")


def main():
    client = CanSocketClient(filters=SUBSCRIBE_FILTERS)

    if not os.path.exists(SOCKET_PATH):
        print(f"✗ Socket not found: {SOCKET_PATH}")
        print("  Make sure can0 service is running first.")
        sys.exit(1)

    try:
        client.connect()
        print(f"Listening for CAN data (filters={[hex(f) for f in SUBSCRIBE_FILTERS] or 'all'})...")
        print("Press Ctrl+C to stop.\n")
        while client.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    except FileNotFoundError:
        print(f"✗ Cannot connect to {SOCKET_PATH}")
    except ConnectionRefusedError:
        print(f"✗ Connection refused — is can0 service running?")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
