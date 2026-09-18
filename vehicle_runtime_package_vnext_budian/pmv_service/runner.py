"""Long-running orchestration loop for the PMV socket service."""

from __future__ import annotations

import datetime
import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pmv_service.api import LatestApiData, build_api_data
from pmv_service.image_inputs import fetch_image_stats, inject_image_to_pmv_input
from pmv_service.input_builder import build_pmv_input
from pmv_service.param_trace import append_param_log
from pmv_service.socket_client import (
    connect_socket,
    is_socket_disconnect_error,
    parse_data_message,
    pong_handler,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PmvServiceConfig:
    socket_path: str
    client_id: str
    can_filters: list[int]
    interval_s: float = 1.0
    param_log_file: str = ""
    output_dir: Optional[str] = None


def correct_previous_state(previous_state: Any, in_car_temp_c: Optional[float]) -> Any:
    """Preserve the deployed online state correction policy."""
    if in_car_temp_c is None:
        return previous_state
    for index in (1, 15, 16):
        previous_state[index] = previous_state[index] * 0.95 + in_car_temp_c * 0.05
    for index in (3, 4):
        previous_state[index] = previous_state[index] * 0.85 + in_car_temp_c * 0.15
    return previous_state


def _write_optional_output(output_dir: Optional[str], run_index: int, output: Dict[str, Any]) -> None:
    if not output_dir:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"pmv_output_{run_index:05d}.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _log_run(run_index: int, timestamp: str, pmv_input: Dict[str, Any], output: Dict[str, Any]) -> None:
    if run_index == 1 or run_index % 20 == 0:
        LOGGER.info(
            "%4s %12s %5s %5s %7s %7s %8s %8s %6s",
            "#", "time", "amb", "spd", "solar_d", "solar_p", "PMV_drv", "PMV_pass", "status",
        )
    pmv = output.get("pmv", {}) or {}
    LOGGER.info(
        "%4d %12s %5.1f %5.0f %7.0f %7.0f %8s %8s %6s",
        run_index,
        timestamp[-15:-1],
        pmv_input.get("amb_t_c", 0),
        pmv_input.get("vehicle_speed_kph", 0),
        pmv_input.get("solar_driver_w_m2", 0),
        pmv_input.get("solar_passenger_w_m2", 0),
        _format_pmv(pmv.get("pmv_driver")),
        _format_pmv(pmv.get("pmv_passenger")),
        output.get("status", "?"),
    )


def _format_pmv(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def run_service(
    config: PmvServiceConfig,
    api_state: LatestApiData,
    run_pmv: Callable[..., Dict[str, Any]],
) -> None:
    """Consume CAN envelopes forever and publish the latest PMV result."""
    sock = connect_socket(config.socket_path, config.client_id, config.can_filters)
    signal_cache: Dict[str, float] = {}
    previous_state = None
    last_pmv_run = 0.0
    run_count = 0
    buffer = ""

    LOGGER.info(
        "Subscribed to %d CAN IDs, PMV interval=%.3fs",
        len(config.can_filters),
        config.interval_s,
    )
    LOGGER.info("Parameter trace log: %s", config.param_log_file)
    LOGGER.info("Waiting for CAN data")

    while True:
        try:
            data = sock.recv(4096)
            if not data:
                LOGGER.warning("Server closed connection, reconnecting")
                sock.close()
                sock = connect_socket(config.socket_path, config.client_id, config.can_filters)
                buffer = ""
                continue
        except socket.timeout:
            continue
        except Exception as exc:
            if not is_socket_disconnect_error(exc):
                LOGGER.exception("Receive error")
                break
            LOGGER.warning("Receive error: %s; reconnecting", exc)
            try:
                sock.close()
            except Exception:
                pass
            sock = connect_socket(config.socket_path, config.client_id, config.can_filters)
            buffer = ""
            continue

        buffer += data.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            pong_handler(sock, line)
            signals = parse_data_message(line)
            if signals is None:
                continue
            signal_cache.update(signals)

            now = time.monotonic()
            if now - last_pmv_run < config.interval_s:
                continue
            if "VIU_AmbT" not in signal_cache:
                continue
            if previous_state is None and "AC_FrntInCarT" not in signal_cache:
                continue

            last_pmv_run = now
            run_count += 1
            try:
                pmv_input = build_pmv_input(signal_cache)
                inject_image_to_pmv_input(pmv_input, fetch_image_stats())
                init_temp = pmv_input.get("ict_c") if previous_state is None else None
                pmv_output = run_pmv(
                    pmv_input,
                    previous_state=previous_state,
                    init_temp_c=init_temp,
                )
                if "x_next" in pmv_output:
                    previous_state = correct_previous_state(
                        pmv_output["x_next"], pmv_input.get("ict_c")
                    )
            except Exception:
                LOGGER.exception("PMV run #%d failed", run_count)
                continue

            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            _log_run(run_count, timestamp, pmv_input, pmv_output)
            append_param_log(
                config.param_log_file,
                run_index=run_count,
                timestamp=timestamp,
                signal_cache=signal_cache,
                pmv_input=pmv_input,
                pmv_output=pmv_output,
            )
            api_state.update(
                build_api_data(
                    pmv_output,
                    run_count,
                    amb_t_c=pmv_input.get("amb_t_c", 0),
                )
            )
            _write_optional_output(config.output_dir, run_count, pmv_output)

    sock.close()
    LOGGER.info("Stopped after %d PMV runs", run_count)
