#!/usr/bin/env python3
"""
PMV Socket Consumer — 从 can0_service Unix Socket 读取 RTE 信号，定时运行 PMV 模块，
并通过 HTTP API 提供最新 PMV 结果。

用法: python pmv_socket_consumer.py [--interval 1.0] [--output-dir /tmp/pmv_out]
          [--api-port 7861]
"""

import sys
import json
import time
import socket
import argparse
import datetime
import os
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_vehicle_pmv import run as run_pmv

IMAGE_STATS_URL = "http://127.0.0.1:7860/stats"
IMAGE_FETCH_TIMEOUT_S = 0.5
DEFAULT_PARAM_LOG_FILE = "/tmp/pmv_service_logs/pmv_param_trace.log"

SOCKET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "can_service", "sock", "can0_bus.sock")

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

DBC_TO_PMV = {
    "VIU_AmbT":                   "amb_t_c",
    "RSM_RelHum":                 "rh_percent",
    "IPB_VehicleSpeed":           "vehicle_speed_kph",
    "RSM_LeSolarInten":           "solar_driver_w_m2",
    "RSM_RiSolarInten":           "solar_passenger_w_m2",
    "AC_FEvapCurrentTemp":        "eva_t_c",
    "AC_FBlowSpeedLevel_raw":     "front_blower_level",
    "AC_FBlowSpeedLevel":         "front_blower_level",
    "AC_FrntInCarT":              "ict_c",
    "AC_DrvrFaceVentActT":        "driver_face_tma",
    "AC_PassFaceVentActT":        "passenger_face_tma",
    "AC_DrvrFootVentActT":        "driver_foot_tma",
    "AC_PassFootVentActT":        "passenger_foot_tma",
    "AC_Forward_AirFlowTarget":   "total_airflow",
    "AC_BLOW_FaceVentilaPosn":    "face_vent_posn",
    "AC_FrantFootVentPosn":       "foot_vent_posn",
    "AC_DefrostVentilaPosn":      "defrost_vent_posn",
    "TA_FdHeadTempLe":            "driver_head_left_temp",
    "TA_FdHeadTempRi":            "driver_head_right_temp",
    "TA_FpHeadTempLe":            "passenger_head_left_temp",
    "TA_FpHeadTempRi":            "passenger_head_right_temp",
    "TS_FrntWidTemp":             "front_windshield_temp_c",
}

SOCKET_RETRY_DELAY_S = 2.0

# 服装 enum → 热阻 clo 对照表（来自 combind_in_future/TSV预测）
CLO_ENUM_MAP = {
    0: 1.00,   # 西装外套
    1: 0.70,   # 薄夹克
    2: 1.20,   # 长款大衣
    3: 1.50,   # 羽绒服
    4: 0.70,   # 长袖针织毛衣
    5: 0.60,   # 长袖衬衫 (默认)
    6: 0.40,   # 短袖
    7: 0.25,   # 背心
    8: 0.50,   # 长袖
    9: 0.70,   # 连帽卫衣
}
DEFAULT_CLO = 0.60  # 默认服装热阻，对应长袖衬衫

# Shared state: latest PMV result for HTTP API
_latest_api_data: Dict[str, Any] = {"status": "waiting", "run_index": 0}
_api_lock = threading.Lock()


def is_socket_disconnect_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            OSError,
        ),
    )


def connect_socket(sock_path: str, client_id: str, can_filters: list[int]) -> socket.socket:
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(sock_path)

            subscribe_msg = json.dumps({
                "client_id": client_id,
                "filters": can_filters,
                "version": "1.0",
            })
            sock.sendall(subscribe_msg.encode("utf-8"))

            sock.settimeout(2.0)
            ack_data = sock.recv(4096)
            ack = json.loads(ack_data.decode("utf-8").strip())
            if ack.get("status") != "ok":
                sock.close()
                raise RuntimeError(f"Connection rejected: {ack}")

            print(f"[consumer] Connected to {sock_path}, server={ack.get('server_name')} v{ack.get('server_version')}")
            sock.settimeout(1.0)
            return sock

        except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
            print(f"[consumer] Socket {sock_path} not ready ({e}), retrying in {SOCKET_RETRY_DELAY_S}s...")
            time.sleep(SOCKET_RETRY_DELAY_S)


def build_pmv_input(signal_cache: Dict[str, float]) -> Dict[str, Any]:
    pmv: Dict[str, Any] = {
        "use_next_state": True,
        "bypass_models": True,
    }

    tma: Dict[str, float] = {}
    flows: Dict[str, float] = {}
    measured_head_air: Dict[str, float] = {}
    total_airflow_val: Optional[float] = None

    for dbc_name, value in signal_cache.items():
        pmv_field = DBC_TO_PMV.get(dbc_name)
        if pmv_field is None:
            continue

        scaled = value

        if pmv_field == "driver_face_tma":
            tma["FrntFdvTma"] = scaled
        elif pmv_field == "passenger_face_tma":
            tma["FrntFpvTma"] = scaled
        elif pmv_field == "driver_foot_tma":
            tma["FrntFdfTma"] = scaled
        elif pmv_field == "passenger_foot_tma":
            tma["FrntFpfTma"] = scaled
        elif pmv_field == "driver_head_left_temp":
            measured_head_air["driver_left"] = scaled
        elif pmv_field == "driver_head_right_temp":
            measured_head_air["driver_right"] = scaled
        elif pmv_field == "passenger_head_left_temp":
            measured_head_air["passenger_left"] = scaled
        elif pmv_field == "passenger_head_right_temp":
            measured_head_air["passenger_right"] = scaled
        elif pmv_field == "total_airflow":
            total_airflow_val = scaled
        elif pmv_field == "front_blower_level":
            pmv["front_blower_level"] = scaled
        elif pmv_field == "front_windshield_temp_c":
            pmv["front_windshield_temp_c"] = scaled
        elif pmv_field in ("face_vent_posn", "foot_vent_posn", "defrost_vent_posn"):
            pmv[pmv_field] = scaled
        else:
            pmv[pmv_field] = scaled

    if tma:
        pmv["tma"] = tma

    if measured_head_air:
        pmv["measured_head_air_temp_c"] = measured_head_air

    # --- bypass_models: compute override values from CAN signals ---
    _set_bypass_overrides(pmv, signal_cache, measured_head_air)

    return pmv


def _set_bypass_overrides(
    pmv: Dict[str, Any],
    signal_cache: Dict[str, float],
    measured_head_air: Dict[str, float],
) -> None:
    """Populate bypass override fields from available CAN signals.

    When bypass_models=True the pipeline skips CHTD and air-speed estimation.
    Overrides are derived from:
      - air_temp  ← measured head temperature (left/right average per seat)
      - mrt       ← in-car temperature sensor (ICT) as MRT proxy
      - air_speed ← simple estimate from total airflow / vent positions
    """
    import math

    def _avg(*values: Optional[float]) -> Optional[float]:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        return sum(finite) / len(finite) if finite else None

    # air_temp override: from measured head temps
    driver_head = _avg(
        measured_head_air.get("driver_left"),
        measured_head_air.get("driver_right"),
    )
    passenger_head = _avg(
        measured_head_air.get("passenger_left"),
        measured_head_air.get("passenger_right"),
    )
    if driver_head is not None:
        pmv["driver_air_temp_override_c"] = driver_head
    if passenger_head is not None:
        pmv["passenger_air_temp_override_c"] = passenger_head

    # mrt override: use ICT as proxy
    ict = signal_cache.get("AC_FrntInCarT")
    if ict is not None and math.isfinite(ict):
        pmv["driver_mrt_override_c"] = ict
        pmv["passenger_mrt_override_c"] = ict

    # air_speed override: simple per-person flow → velocity estimate
    total_flow = signal_cache.get("AC_Forward_AirFlowTarget")
    if total_flow and total_flow > 0:
        blower_pwm = signal_cache.get("AC_Forward_BlwPwmOut", 100.0)
        pwm_ratio = max(0.0, min(100.0, blower_pwm)) / 100.0
        effective_total = total_flow * pwm_ratio

        face_v = signal_cache.get("AC_BLOW_FaceVentilaPosn", 0.0)
        foot_v = signal_cache.get("AC_FrantFootVentPosn", 0.0)
        face_open = max(0.0, face_v)
        foot_open = max(0.0, foot_v)
        total_open = face_open + foot_open
        if total_open > 0:
            face_frac = face_open / total_open
            foot_frac = foot_open / total_open
        else:
            face_frac = 0.5
            foot_frac = 0.5

        # per-person flow [m³/h] → outlet velocity [m/s] with nominal geometry
        per_person = effective_total / 2.0
        # outlet_velocity = (flow_m3h / 3600) / area_m2
        face_v_out = (per_person * face_frac / 3600.0) / 0.02
        foot_v_out = (per_person * foot_frac / 3600.0) / 0.015
        # attenuation at 0.45 m (face) / 0.55 m (foot)
        face_att = 1.0 / (1.0 + 0.45 ** 2)
        foot_att = 1.0 / (1.0 + 0.55 ** 2)
        # combine by quadrature, apply global scale
        raw_speed = math.sqrt((face_v_out * face_att) ** 2 + (foot_v_out * foot_att) ** 2)
        air_speed = raw_speed * 0.7  # PMV_AIR_SPEED_SCALE

        pmv["driver_air_speed_override_m_s"] = air_speed
        pmv["passenger_air_speed_override_m_s"] = air_speed


def pong_handler(sock: socket.socket, data_str: str) -> None:
    try:
        msg = json.loads(data_str)
        if msg.get("type") == "heartbeat":
            pong = json.dumps({"type": "pong", "timestamp": time.time()})
            sock.sendall(pong.encode("utf-8") + b"\n")
    except Exception:
        pass


def parse_data_message(data_str: str) -> Optional[Dict[str, Any]]:
    try:
        msg = json.loads(data_str)
    except json.JSONDecodeError:
        return None

    if msg.get("type") != "data":
        return None

    payload = msg.get("data")
    if not isinstance(payload, dict):
        return None

    signals: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            signals[key] = float(value)
        elif isinstance(value, str):
            try:
                signals[key] = float(value)
            except ValueError:
                pass
    return signals


_image_api_ok = False


def fetch_image_stats(timeout: float = IMAGE_FETCH_TIMEOUT_S) -> Optional[Dict[str, Any]]:
    """从 Image API 获取 stats，返回完整 dict（含 driver / passenger）。

    返回格式: { "driver": {...}, "passenger": {...}, "status": "ok" }
    失败/超时返回 None。
    """
    global _image_api_ok
    try:
        resp = requests.get(IMAGE_STATS_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        if _image_api_ok:
            print("[consumer] Image API unreachable, using default clothing/met")
            _image_api_ok = False
        return None

    if not isinstance(data, dict):
        return None
    if data.get("status") != "ok":
        return None
    if not _image_api_ok:
        print("[consumer] Image API connected, using real clothing/met")
        _image_api_ok = True
    return data


def _parse_seat(stats_seat: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """将 stats API 返回的单座信息转为 pipeline 所需的 SeatOccupantInput 字段。

    关键映射：cloth enum → clothing_clo (热阻)
    """
    if stats_seat is None or not isinstance(stats_seat, dict):
        return {"occupied": False}

    cloth_enum = stats_seat.get("cloth")
    clothing_clo = None
    if cloth_enum is not None:
        try:
            clothing_clo = CLO_ENUM_MAP.get(int(cloth_enum))
        except (TypeError, ValueError):
            pass

    gender_raw = stats_seat.get("gender")
    if gender_raw is not None:
        try:
            gender_str = "male" if int(gender_raw) == 1 else "female"
        except (TypeError, ValueError):
            gender_str = None
    else:
        gender_str = None

    seat: Dict[str, Any] = {
        "occupied":     True,
        "gender":       gender_str,
        "age":          stats_seat.get("age"),
        "height_cm":    stats_seat.get("height"),
        "bmi":          stats_seat.get("bmi"),
        "clothing_clo": clothing_clo,          # 枚举 → clo 映射后的值
        "cloth_enum":   cloth_enum,            # 保留原始枚举用于诊断
    }
    return seat


def inject_image_to_pmv_input(pmv_input: Dict[str, Any], stats_data: Optional[Dict[str, Any]]) -> None:
    """将 Image API 返回的 stats 注入 PMV pipeline 输入。

    处理主驾 + 副驾，将 cloth enum 映射到 clothing_clo（热阻）。
    """
    if stats_data is None:
        pmv_input["image_inputs"] = {
            "driver": {"occupied": False},
            "passenger": {"occupied": False},
        }
        return

    pmv_input["image_inputs"] = {
        "driver":    _parse_seat(stats_data.get("driver")),
        "passenger": _parse_seat(stats_data.get("passenger")),
    }


def build_api_data(pmv_output: Dict[str, Any], run_index: int, amb_t_c: Optional[float] = None) -> Dict[str, Any]:
    """从 PMV 全量输出提取 API 响应所需的精简字段。"""
    return {
        "driver": {
            "pmv":          _safe(pmv_output.get("pmv", {}).get("pmv_driver")),
            "ppd":          _safe(pmv_output.get("pmv", {}).get("ppd_driver")),
            "head_temp_c":  _safe(pmv_output.get("diagnostics", {}).get("driver_air_temp_c")),
            "feet_temp_c":  _safe(pmv_output.get("model_state", {}).get("driver_feet_temp_c")),
        },
        "passenger": {
            "pmv":          _safe(pmv_output.get("pmv", {}).get("pmv_passenger")),
            "ppd":          _safe(pmv_output.get("pmv", {}).get("ppd_passenger")),
            "head_temp_c":  _safe(pmv_output.get("diagnostics", {}).get("passenger_air_temp_c")),
            "feet_temp_c":  _safe(pmv_output.get("model_state", {}).get("passenger_feet_temp_c")),
        },
        "cabin_temp_c": _safe(pmv_output.get("diagnostics", {}).get("driver_air_temp_c")),
        "amb_temp_c":   _safe(amb_t_c),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": pmv_output.get("status", "error"),
        "run_index": run_index,
    }


PARAM_LABELS = {
    "amb_t_c": "外界环境温度, degC",
    "rh_percent": "相对湿度, %",
    "vehicle_speed_kph": "车速, km/h",
    "solar_driver_w_m2": "主驾侧太阳强度, W/m2",
    "solar_passenger_w_m2": "副驾侧太阳强度, W/m2",
    "eva_t_c": "蒸发器当前温度, degC",
    "ict_c": "前排车内温度传感器, degC",
    "front_blower_level": "前排风机档位 raw, 1-10 有效",
    "front_windshield_temp_c": "前挡玻璃温度布点 TS_FrntWidTemp, degC",
    "tma": "出风口温度字典, degC",
    "flows": "按座位/出风模式估算的风量字典, m3/h",
    "measured_head_air_temp_c": "主副驾头部左右侧布点温度, degC",
    "image_inputs": "图像模块乘员/衣着/代谢率输入",
    "bypass_models": "CHTD+风量模型旁路开关",
    "driver_air_temp_override_c": "主驾空气温度替代值, degC (bypass模式)",
    "passenger_air_temp_override_c": "副驾空气温度替代值, degC (bypass模式)",
    "driver_mrt_override_c": "主驾平均辐射温度替代值, degC (bypass模式)",
    "passenger_mrt_override_c": "副驾平均辐射温度替代值, degC (bypass模式)",
    "driver_air_speed_override_m_s": "主驾风速替代值, m/s (bypass模式)",
    "passenger_air_speed_override_m_s": "副驾风速替代值, m/s (bypass模式)",
}

DIAG_LABELS = {
    "bypass_models": "CHTD+风量模型旁路开关 (true=直接输入, false=模型计算)",
    "driver_air_temp_c": "主驾 PMV 空气温度 air_temp_c, degC",
    "passenger_air_temp_c": "副驾 PMV 空气温度 air_temp_c, degC",
    "driver_air_temp_source": "主驾空气温度来源",
    "passenger_air_temp_source": "副驾空气温度来源",
    "driver_mrt_c": "主驾 PMV 平均辐射温度 mean_radiant_temp_c, degC",
    "passenger_mrt_c": "副驾 PMV 平均辐射温度 mean_radiant_temp_c, degC",
    "driver_mrt_source": "主驾平均辐射温度来源",
    "passenger_mrt_source": "副驾平均辐射温度来源",
    "driver_air_speed_m_s": "主驾 PMV 风速 air_velocity_m_s, m/s, Fanger 内部最低按 0.01 处理",
    "passenger_air_speed_m_s": "副驾 PMV 风速 air_velocity_m_s, m/s, Fanger 内部最低按 0.01 处理",
    "rh_percent": "PMV 相对湿度, %",
    "met_driver": "主驾代谢率 met",
    "met_passenger": "副驾代谢率 met",
    "clo_driver": "主驾服装热阻 clo",
    "clo_passenger": "副驾服装热阻 clo",
    "driver_tsv_mrt_formula": "主驾 TSV MRT 公式诊断",
    "passenger_tsv_mrt_formula": "副驾 TSV MRT 公式诊断",
}


def _format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def append_param_log(
    log_file: str,
    *,
    run_index: int,
    timestamp: str,
    signal_cache: Dict[str, float],
    pmv_input: Dict[str, Any],
    pmv_output: Dict[str, Any],
) -> None:
    """Append one human-readable PMV parameter trace block."""
    if not log_file:
        return

    diagnostics = pmv_output.get("diagnostics", {}) or {}
    pmv = pmv_output.get("pmv", {}) or {}
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "",
        "=" * 100,
        f"运行序号 run_index: {run_index}",
        f"时间 timestamp: {timestamp}",
        f"状态 status: {pmv_output.get('status')}",
        "-" * 100,
        "PMV/PPD 输出:",
        f"  主驾 PMV: {_format_json(pmv.get('pmv_driver'))}",
        f"  主驾 PPD: {_format_json(pmv.get('ppd_driver'))} %",
        f"  副驾 PMV: {_format_json(pmv.get('pmv_passenger'))}",
        f"  副驾 PPD: {_format_json(pmv.get('ppd_passenger'))} %",
        "-" * 100,
        "PMV 实际入参/派生参数:",
    ]
    for key in (
        "bypass_models",
        "driver_air_temp_c",
        "passenger_air_temp_c",
        "driver_air_temp_source",
        "passenger_air_temp_source",
        "driver_mrt_c",
        "passenger_mrt_c",
        "driver_mrt_source",
        "passenger_mrt_source",
        "driver_air_speed_m_s",
        "passenger_air_speed_m_s",
        "rh_percent",
        "met_driver",
        "met_passenger",
        "clo_driver",
        "clo_passenger",
        "driver_tsv_mrt_formula",
        "passenger_tsv_mrt_formula",
    ):
        lines.append(f"  {key} ({DIAG_LABELS.get(key, '诊断字段')}): {_format_json(diagnostics.get(key))}")

    lines.extend(["-" * 100, "PMV 输入字典 pmv_input:"])
    for key in sorted(pmv_input):
        lines.append(f"  {key} ({PARAM_LABELS.get(key, '运行时输入字段')}): {_format_json(pmv_input.get(key))}")

    lines.extend(["-" * 100, "最新 CAN 信号缓存 signal_cache:"])
    for key in sorted(signal_cache):
        lines.append(f"  {key}: {_format_json(signal_cache.get(key))}")

    lines.extend(["=" * 100, ""])
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as exc:
        print(f"[consumer] Parameter log write failed: {exc}")


def _safe(v):
    if v is None:
        return None
    try:
        import math
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


class PMVApiHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/pmv"):
            with _api_lock:
                data = dict(_latest_api_data)
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress HTTP access logs


def start_api_server(port: int) -> HTTPServer:
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", port), PMVApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[api] HTTP API listening on http://0.0.0.0:{port}")
    return server


def main():
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
    args = parser.parse_args()

    global _latest_api_data
    _api_server = start_api_server(args.api_port)

    sock = connect_socket(args.socket, args.client_id, SUBSCRIBE_CAN_IDS)

    signal_cache: Dict[str, float] = {}
    last_pmv_run = 0.0
    run_count = 0
    previous_state = None  # CHTD x[28]; None → init from sensor readings

    print(f"[consumer] Subscribed to {len(SUBSCRIBE_CAN_IDS)} CAN IDs, PMV interval={args.interval}s")
    print(f"[consumer] Parameter trace log: {args.param_log}")
    print("[consumer] Waiting for CAN data...")

    buf = ""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                print("[consumer] Server closed connection, reconnecting...")
                sock.close()
                sock = connect_socket(args.socket, args.client_id, SUBSCRIBE_CAN_IDS)
                buf = ""
                continue
        except socket.timeout:
            continue
        except Exception as e:
            if is_socket_disconnect_error(e):
                print(f"[consumer] Receive error: {e}; reconnecting...")
                try:
                    sock.close()
                except Exception:
                    pass
                sock = connect_socket(args.socket, args.client_id, SUBSCRIBE_CAN_IDS)
                buf = ""
                continue
            print(f"[consumer] Receive error: {e}")
            break

        buf += data.decode("utf-8")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            pong_handler(sock, line)

            signals = parse_data_message(line)
            if signals is None:
                continue

            signal_cache.update(signals)

            now = time.monotonic()
            if now - last_pmv_run < args.interval:
                continue

            if "VIU_AmbT" not in signal_cache:
                continue
            # Wait for in-car temp sensor before first run (used as CHTD init state)
            if previous_state is None and "AC_FrntInCarT" not in signal_cache:
                continue

            last_pmv_run = now
            run_count += 1

            try:
                pmv_input = build_pmv_input(signal_cache)
                stats_data = fetch_image_stats()
                inject_image_to_pmv_input(pmv_input, stats_data)
                init_temp = pmv_input.get("ict_c") if previous_state is None else None
                pmv_output = run_pmv(pmv_input, previous_state=previous_state, init_temp_c=init_temp)
                if "x_next" in pmv_output:
                    previous_state = pmv_output["x_next"]
                    # Online correction: nudge cabin/head states toward CAN measured temp
                    ict_c = pmv_input.get("ict_c")
                    if ict_c is not None:
                        # Cabin states: gentle correction
                        for i in [1, 15, 16]:  # CabinFrntTemp, CabinFd, CabinFp
                            previous_state[i] = previous_state[i] * 0.95 + ict_c * 0.05
                        # Head states: stronger correction (CHTD pulls head down toward glass)
                        for i in [3, 4]:  # HeadTempFd, HeadTempFp
                            previous_state[i] = previous_state[i] * 0.85 + ict_c * 0.15
            except Exception as e:
                print(f"[consumer] PMV run #{run_count} error: {e}")
                continue

            ts = datetime.datetime.utcnow().isoformat() + "Z"
            status = pmv_output.get("status", "?")
            driver_pmv = pmv_output.get("pmv", {}).get("pmv_driver")
            passenger_pmv = pmv_output.get("pmv", {}).get("pmv_passenger")
            amb = pmv_input.get("amb_t_c", 0)
            speed = pmv_input.get("vehicle_speed_kph", 0)
            solar_d = pmv_input.get("solar_driver_w_m2", 0)
            solar_p = pmv_input.get("solar_passenger_w_m2", 0)

            if run_count == 1 or run_count % 20 == 0:
                print(f"{'#':>4s} {'time':>12s} {'amb':>5s} {'spd':>5s} {'solar_d':>7s} {'solar_p':>7s} {'PMV_drv':>8s} {'PMV_pass':>8s} {'status':>6s}")
                print("-" * 80)

            print(f"{run_count:>4d} {ts[-15:-1]:>12s} {amb:>5.1f} {speed:>5.0f} {solar_d:>7.0f} {solar_p:>7.0f} "
                  f"{driver_pmv:>8.3f}" if driver_pmv else f"{'N/A':>8s}",
                  end=" ")
            print(f"{passenger_pmv:>8.3f}" if passenger_pmv else f"{'N/A':>8s}", end=" ")
            print(f"{status:>6s}")

            append_param_log(
                args.param_log,
                run_index=run_count,
                timestamp=ts,
                signal_cache=signal_cache,
                pmv_input=pmv_input,
                pmv_output=pmv_output,
            )

            # Update shared API data
            api_data = build_api_data(pmv_output, run_count, amb_t_c=amb)
            with _api_lock:
                _latest_api_data = api_data

            if args.output_dir:
                out_path = Path(args.output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                out_file = out_path / f"pmv_output_{run_count:05d}.json"
                out_file.write_text(json.dumps(pmv_output, indent=2, ensure_ascii=False), encoding="utf-8")

    sock.close()
    print(f"[consumer] Stopped. Total PMV runs: {run_count}")


if __name__ == "__main__":
    main()
