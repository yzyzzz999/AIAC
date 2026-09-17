"""
service/ipc.py
=============
进程间通信模块：GPU 推理进程 ↔ API 进程

- 共享内存：标注帧（~1MB）+ 识别结果 JSON（32KB），零拷贝
- Unix Socket：API → GPU 命令通道（register/pause/resume/seat_params 等）
"""

from __future__ import annotations

import json
import os
import socket
import time
import logging
from multiprocessing import shared_memory
from typing import Optional, Dict, Any

import numpy as np
import cv2

log = logging.getLogger("ipc")

# ── 共享内存名称 ──
SHM_FRAME_NAME = "vr_frame"
SHM_RESULT_NAME = "vr_result"
SHM_RAW_NAME = "vr_raw"

# ── 原始流分辨率 ──
RAW_FRAME_W = 800
RAW_FRAME_H = 600

# ── 命令 socket（Windows 用 TCP） ──
CMD_SOCKET_HOST = "127.0.0.1"
CMD_SOCKET_PORT = 7863

# ── 缓冲区大小 ──
RESULT_BUFFER_SIZE = 32768  # 32KB

# ── 默认帧尺寸（运行时可能覆盖）──
DEFAULT_FRAME_W = 1280
DEFAULT_FRAME_H = 720
DEFAULT_FRAME_C = 3


# ═══════════════════════════════════════════════════════════════════════
# GPU 进程侧：创建并写入
# ═══════════════════════════════════════════════════════════════════════

class IPCServer:
    """GPU 推理进程侧：创建共享内存、启动命令 socket 监听。"""

    def __init__(self, frame_w: int = DEFAULT_FRAME_W, frame_h: int = DEFAULT_FRAME_H):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.frame_nbytes = frame_w * frame_h * DEFAULT_FRAME_C
        self.raw_nbytes = RAW_FRAME_W * RAW_FRAME_H * DEFAULT_FRAME_C

        # 清理可能残留的同名共享内存
        for name in (SHM_FRAME_NAME, SHM_RESULT_NAME, SHM_RAW_NAME):
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        self._shm_frame = shared_memory.SharedMemory(
            name=SHM_FRAME_NAME, create=True, size=self.frame_nbytes
        )
        self._shm_result = shared_memory.SharedMemory(
            name=SHM_RESULT_NAME, create=True, size=RESULT_BUFFER_SIZE
        )
        self._shm_raw = shared_memory.SharedMemory(
            name=SHM_RAW_NAME, create=True, size=self.raw_nbytes
        )

        self._frame_buf = np.ndarray(
            (frame_h, frame_w, DEFAULT_FRAME_C),
            dtype=np.uint8,
            buffer=self._shm_frame.buf,
        )
        self._raw_buf = np.ndarray(
            (RAW_FRAME_H, RAW_FRAME_W, DEFAULT_FRAME_C),
            dtype=np.uint8,
            buffer=self._shm_raw.buf,
        )
        self._result_buf = self._shm_result.buf

        self._cmd_socket: Optional[socket.socket] = None
        self._cmd_running = False

        log.info("IPC Server created: frame=%dx%d raw=%dx%d result=%d bytes",
                 frame_w, frame_h, RAW_FRAME_W, RAW_FRAME_H, RESULT_BUFFER_SIZE)

    # ── 帧写入 ────────────────────────────────────────────────────────

    def write_frame(self, frame: np.ndarray):
        """写入标注帧到共享内存。帧应已缩放到缓冲区尺寸，否则自动缩放。"""
        if frame.shape[0] != self.frame_h or frame.shape[1] != self.frame_w:
            frame = cv2.resize(frame, (self.frame_w, self.frame_h))
        np.copyto(self._frame_buf, frame)

    def write_raw(self, frame: np.ndarray):
        """写入原始帧到共享内存。自动缩放到 800x600。"""
        if frame.shape[0] != RAW_FRAME_H or frame.shape[1] != RAW_FRAME_W:
            frame = cv2.resize(frame, (RAW_FRAME_W, RAW_FRAME_H))
        np.copyto(self._raw_buf, frame)

    # ── 结果写入 ──────────────────────────────────────────────────────

    def write_result(self, result: Dict[str, Any]):
        """写入识别结果 JSON 到共享内存。"""
        data = json.dumps(result, ensure_ascii=False).encode("utf-8")
        n = min(len(data), RESULT_BUFFER_SIZE - 1)
        self._result_buf[:n] = data[:n]
        self._result_buf[n] = 0

    # ── 命令 socket ───────────────────────────────────────────────────

    def start_cmd_server(self, handler):
        """启动命令 socket 监听线程。handler(cmd: dict) -> dict 同步处理。"""
        self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._cmd_socket.bind((CMD_SOCKET_HOST, CMD_SOCKET_PORT))
        self._cmd_socket.listen(2)
        self._cmd_socket.settimeout(1.0)
        self._cmd_running = True
        self._cmd_handler = handler
        log.info("Command socket listening at %s:%d", CMD_SOCKET_HOST, CMD_SOCKET_PORT)

    def accept_cmd_loop(self):
        """主线程调用：循环接受命令连接并处理。"""
        while self._cmd_running:
            try:
                client, _ = self._cmd_socket.accept()
            except socket.timeout:
                continue
            try:
                data = client.recv(4096).decode("utf-8").strip()
                if not data:
                    client.close()
                    continue
                cmd = json.loads(data)
                resp = self._cmd_handler(cmd)
                resp_bytes = json.dumps(resp, ensure_ascii=False).encode("utf-8")
                client.sendall(resp_bytes)
            except json.JSONDecodeError:
                client.sendall(json.dumps({"error": "invalid json"}).encode())
            except Exception as exc:
                log.exception("Command handler error")
                try:
                    client.sendall(json.dumps({"error": str(exc)}).encode())
                except Exception:
                    pass
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    def stop_cmd_server(self):
        """停止命令 socket 监听。"""
        self._cmd_running = False
        if self._cmd_socket:
            try:
                self._cmd_socket.close()
            except Exception:
                pass

    # ── 清理 ──────────────────────────────────────────────────────────

    def cleanup(self):
        """释放共享内存和 socket。"""
        self.stop_cmd_server()
        for shm in (self._shm_frame, self._shm_result, self._shm_raw):
            try:
                shm.close()
            except Exception:
                pass
        for name in (SHM_FRAME_NAME, SHM_RESULT_NAME, SHM_RAW_NAME):
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
        log.info("IPC Server cleaned up")


# ═══════════════════════════════════════════════════════════════════════
# API 进程侧：连接并读取
# ═══════════════════════════════════════════════════════════════════════

class IPCClient:
    """API 进程侧：连接共享内存，读取帧和结果。"""

    def __init__(self, frame_w: int = DEFAULT_FRAME_W, frame_h: int = DEFAULT_FRAME_H):
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._shm_frame: Optional[shared_memory.SharedMemory] = None
        self._shm_result: Optional[shared_memory.SharedMemory] = None
        self._shm_raw: Optional[shared_memory.SharedMemory] = None
        self._frame_buf: Optional[np.ndarray] = None
        self._raw_buf: Optional[np.ndarray] = None
        self._result_buf = None
        self._connected = False

    def connect(self, timeout: float = 10.0) -> bool:
        """等待并连接共享内存。返回 True 表示连接成功。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._shm_frame = shared_memory.SharedMemory(name=SHM_FRAME_NAME)
                self._shm_result = shared_memory.SharedMemory(name=SHM_RESULT_NAME)
                self._shm_raw = shared_memory.SharedMemory(name=SHM_RAW_NAME)
                self._frame_buf = np.ndarray(
                    (self.frame_h, self.frame_w, DEFAULT_FRAME_C),
                    dtype=np.uint8,
                    buffer=self._shm_frame.buf,
                )
                self._raw_buf = np.ndarray(
                    (RAW_FRAME_H, RAW_FRAME_W, DEFAULT_FRAME_C),
                    dtype=np.uint8,
                    buffer=self._shm_raw.buf,
                )
                self._result_buf = self._shm_result.buf
                self._connected = True
                log.info("IPC Client connected")
                return True
            except FileNotFoundError:
                time.sleep(0.5)
        return False

    @property
    def connected(self) -> bool:
        return self._connected

    def read_frame(self) -> Optional[np.ndarray]:
        """读取当前标注帧副本。"""
        if not self._connected or self._frame_buf is None:
            return None
        return self._frame_buf.copy()

    def read_raw(self) -> Optional[np.ndarray]:
        """读取当前原始帧副本。800x600 BGR。"""
        if not self._connected or self._raw_buf is None:
            return None
        return self._raw_buf.copy()

    def read_result(self) -> Dict[str, Any]:
        """读取当前识别结果。"""
        if not self._connected or self._result_buf is None:
            return {}
        raw = bytes(self._result_buf[:RESULT_BUFFER_SIZE])
        null_pos = raw.find(b"\0")
        if null_pos >= 0:
            raw = raw[:null_pos]
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def send_cmd(self, cmd: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        """发送命令到 GPU 进程并等待响应。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((CMD_SOCKET_HOST, CMD_SOCKET_PORT))
            data = json.dumps(cmd, ensure_ascii=False).encode("utf-8")
            sock.sendall(data)
            resp = sock.recv(4096).decode("utf-8").strip()
            return json.loads(resp)
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            return {"error": str(e)}
        except json.JSONDecodeError:
            return {"error": "invalid response"}
        finally:
            sock.close()

    def close(self):
        """断开共享内存连接。"""
        self._connected = False
        for shm in (self._shm_frame, self._shm_result, self._shm_raw):
            if shm:
                try:
                    shm.close()
                except Exception:
                    pass
