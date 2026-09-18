"""HTTP response contract and in-memory state for the PMV service."""

from __future__ import annotations

import datetime
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def build_api_data(
    pmv_output: Dict[str, Any],
    run_index: int,
    amb_t_c: Optional[float] = None,
) -> Dict[str, Any]:
    """Project the full runtime output onto the stable HTTP contract."""
    pmv = pmv_output.get("pmv", {}) or {}
    diagnostics = pmv_output.get("diagnostics", {}) or {}
    model_state = pmv_output.get("model_state", {}) or {}

    def seat(prefix: str) -> Dict[str, Optional[float]]:
        return {
            "pmv": safe_float(pmv.get(f"pmv_{prefix}")),
            "ppd": safe_float(pmv.get(f"ppd_{prefix}")),
            "head_temp_c": safe_float(diagnostics.get(f"{prefix}_air_temp_c")),
            # Compatibility field retained for older PMV API consumers. In
            # bypass mode this remains the (possibly unchanged) model state.
            "feet_temp_c": safe_float(model_state.get(f"{prefix}_feet_temp_c")),
            "mrt_c": safe_float(diagnostics.get(f"{prefix}_mrt_c")),
            "rh_percent": safe_float(diagnostics.get("rh_percent")),
            "air_speed_m_s": safe_float(
                diagnostics.get(f"{prefix}_air_speed_m_s")
            ),
        }

    return {
        "driver": seat("driver"),
        "passenger": seat("passenger"),
        # Preserve the current deployed mapping for compatibility. Its naming
        # mismatch is tracked separately from this structural refactor.
        "cabin_temp_c": safe_float(diagnostics.get("driver_air_temp_c")),
        "amb_temp_c": safe_float(amb_t_c),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": pmv_output.get("status", "error"),
        "run_index": run_index,
    }


class LatestApiData:
    """Thread-safe holder shared by the producer loop and HTTP handler."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {"status": "waiting", "run_index": 0}
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def update(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self._data = dict(data)


def make_api_handler(state: LatestApiData) -> type[BaseHTTPRequestHandler]:
    class PMVApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/pmv"):
                self.send_response(404)
                self.end_headers()
                return

            body = json.dumps(state.snapshot(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return PMVApiHandler


def start_api_server(port: int, state: LatestApiData) -> HTTPServer:
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", port), make_api_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
