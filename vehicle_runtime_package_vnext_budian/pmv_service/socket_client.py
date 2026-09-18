"""Unix-socket protocol helpers for the CAN service."""

from __future__ import annotations

import json
import logging
import math
import socket
import time
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)
SOCKET_RETRY_DELAY_S = 2.0


def is_socket_disconnect_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError),
    )


def connect_socket(
    sock_path: str,
    client_id: str,
    can_filters: list[int],
    retry_delay_s: float = SOCKET_RETRY_DELAY_S,
) -> socket.socket:
    """Connect and subscribe, retrying until the CAN service is available."""
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(sock_path)
            sock.sendall(
                json.dumps(
                    {"client_id": client_id, "filters": can_filters, "version": "1.0"}
                ).encode("utf-8")
            )
            sock.settimeout(2.0)
            ack = json.loads(sock.recv(4096).decode("utf-8").strip())
            if ack.get("status") != "ok":
                sock.close()
                raise RuntimeError(f"Connection rejected: {ack}")
            LOGGER.info(
                "Connected to %s, server=%s v%s",
                sock_path,
                ack.get("server_name"),
                ack.get("server_version"),
            )
            sock.settimeout(1.0)
            return sock
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            LOGGER.warning(
                "Socket %s not ready (%s), retrying in %.1fs",
                sock_path,
                exc,
                retry_delay_s,
            )
            time.sleep(retry_delay_s)


def pong_handler(sock: socket.socket, data_str: str) -> None:
    """Reply to heartbeat messages; malformed messages are ignored."""
    try:
        message = json.loads(data_str)
        if message.get("type") == "heartbeat":
            pong = json.dumps({"type": "pong", "timestamp": time.time()})
            sock.sendall(pong.encode("utf-8") + b"\n")
    except Exception:
        return


def parse_data_message(data_str: str) -> Optional[Dict[str, float]]:
    """Parse a CAN service data envelope into finite-or-numeric signals."""
    try:
        message = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    if message.get("type") != "data" or not isinstance(message.get("data"), dict):
        return None

    signals: Dict[str, float] = {}
    for key, value in message["data"].items():
        if isinstance(value, (int, float)):
            converted = float(value)
            if math.isfinite(converted):
                signals[key] = converted
        elif isinstance(value, str):
            try:
                converted = float(value)
            except ValueError:
                continue
            if math.isfinite(converted):
                signals[key] = converted
    return signals
