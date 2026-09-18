#!/usr/bin/env python3
"""Backward-compatible CLI entry point for the PMV socket service."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_vehicle_pmv import run as run_pmv
from pmv_service.api import LatestApiData, build_api_data, start_api_server as _start_api_server
from pmv_service.image_inputs import (
    CLO_ENUM_MAP,
    DEFAULT_CLO,
    IMAGE_FETCH_TIMEOUT_S,
    IMAGE_STATS_URL,
    fetch_image_stats,
    inject_image_to_pmv_input,
    parse_seat as _parse_seat,
)
from pmv_service.input_builder import DBC_TO_PMV, build_pmv_input
from pmv_service.param_trace import append_param_log
from pmv_service.runner import PmvServiceConfig, run_service
from pmv_service.socket_client import (
    SOCKET_RETRY_DELAY_S,
    connect_socket,
    is_socket_disconnect_error,
    parse_data_message,
    pong_handler,
)


DEFAULT_PARAM_LOG_FILE = "/tmp/pmv_service_logs/pmv_param_trace.log"
SOCKET_PATH = str(ROOT.parent / "can_service" / "sock" / "can0_bus.sock")
SUBSCRIBE_CAN_IDS = [
    0x33A,
    0x35B,
    0x3B9,
    0x18F,
    0x33E,
    0x33F,
    0x338,
    0x370,
    0x371,
    0x541,
    0x100,
]

_api_state = LatestApiData()


def start_api_server(port: int):
    """Start the legacy global-state API server used by this CLI."""
    server = _start_api_server(port, _api_state)
    logging.getLogger(__name__).info("HTTP API listening on http://0.0.0.0:%d", port)
    return server


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PMV Socket Consumer")
    parser.add_argument("--socket", default=SOCKET_PATH, help="Unix socket path")
    parser.add_argument("--interval", type=float, default=1.0, help="PMV 最小运行间隔 (秒)")
    parser.add_argument("--output-dir", default=None, help="输出 JSON 目录 (可选)")
    parser.add_argument("--client-id", default=f"pmv_consumer_{os.getpid()}", help="客户端标识")
    parser.add_argument("--api-port", type=int, default=7861, help="HTTP API 端口 (默认 7861)")
    parser.add_argument(
        "--param-log",
        default=os.environ.get("PMV_PARAM_LOG_FILE", DEFAULT_PARAM_LOG_FILE),
        help=f"PMV 参数明细日志文件 (默认 {DEFAULT_PARAM_LOG_FILE})",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("PMV_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="服务日志级别 (默认 INFO)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start_api_server(args.api_port)
    run_service(
        PmvServiceConfig(
            socket_path=args.socket,
            client_id=args.client_id,
            can_filters=SUBSCRIBE_CAN_IDS,
            interval_s=args.interval,
            param_log_file=args.param_log,
            output_dir=args.output_dir,
        ),
        _api_state,
        run_pmv,
    )


if __name__ == "__main__":
    main()
