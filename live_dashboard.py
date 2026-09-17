#!/usr/bin/env python3
"""
AIAC Live Dashboard — 实时可视化主副驾 PMV + 推荐模式输出。

用法:
    python live_dashboard.py [--port 7862] [--pmv-api http://localhost:7861/pmv]
                             [--face-api http://localhost:7860/stats]

访问: http://<ip>:7862
"""

import json
import os
import re
import sys
import time
import argparse
import threading
import subprocess
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# ---- 配置 ----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# 推荐日志可能在多个位置：start_all.sh 重定向到 .startup_logs，模块自身写到各自 logs/
_REC_LOG_CANDIDATES = [
    SCRIPT_DIR / ".startup_logs" / "recommendation.log",
    SCRIPT_DIR / "sls_recommendation_weather26_0829" / "sls_recommendation_weather26" / "logs" / "recommendation.log",
    SCRIPT_DIR / "sls_recommendation" / "logs" / "recommendation.log",
    SCRIPT_DIR / "sls_recommendation" / "src" / "logs" / "recommendation.log",
    SCRIPT_DIR / "sls_recommendation_weather26" / "logs" / "recommendation.log",
]

def _pick_log_path() -> str:
    """选择最近修改的日志文件。"""
    best = ""
    best_mtime = 0
    for p in _REC_LOG_CANDIDATES:
        try:
            mtime = p.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best = str(p)
        except OSError:
            pass
    return best

DEFAULT_REC_LOG = _pick_log_path()
DEFAULT_PMV_API = "http://localhost:7861/pmv"
DEFAULT_FACE_API = "http://localhost:7860/stats"
DEFAULT_WIND_WEIGHT_K = 0.5


def _resolve_wind_k_path(log_path: str = "") -> Path:
    """定位推荐服务读取的运行时K配置文件。"""
    if log_path:
        candidate = Path(log_path).resolve()
        for parent in candidate.parents:
            if (parent / "config.json").is_file():
                return parent / "data" / "occupant_policy_runtime.json"

    project_candidates = [
        SCRIPT_DIR / "sls_recommendation_weather26_0829" / "sls_recommendation_weather26",
        SCRIPT_DIR / "sls_recommendation_weather26",
        SCRIPT_DIR / "sls_recommendation",
    ]
    for project_dir in project_candidates:
        if (project_dir / "config.json").is_file():
            return project_dir / "data" / "occupant_policy_runtime.json"
    return project_candidates[0] / "data" / "occupant_policy_runtime.json"


def _clamp_wind_k(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _read_wind_k(path: Path, default: float = DEFAULT_WIND_WEIGHT_K) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _clamp_wind_k(payload["occupant_wind_weight_k"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return _clamp_wind_k(default)


def _write_wind_k(path: Path, value: float) -> float:
    """原子写入运行时K，避免推荐进程读到半个JSON文件。"""
    wind_k = _clamp_wind_k(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps({"occupant_wind_weight_k": wind_k}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)
    return wind_k

# 空调出风模式映射 (来自 CAN DBC)
AIR_MODE_MAP = {
    0: "无请求",
    1: "吹面",
    2: "吹面+吹脚",
    3: "吹脚",
    4: "吹脚+除霜",
    5: "除霜",
    6: "吹面+除霜",
    7: "吹面+吹脚+除霜",
}

# ---- 共享状态 ------------------------------------------------------------
_latest: Dict[str, Any] = {
    "pmv": None,
    "face": None,
    "recommendation": None,
    "log_entries": [],
    "updated": "",
}
_lock = threading.Lock()
_log_position = 0
_wind_k_path = _resolve_wind_k_path(DEFAULT_REC_LOG)
_wind_weight_k = _read_wind_k(_wind_k_path)


# ---- 数据采集 ------------------------------------------------------------

def fetch_json(url: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def parse_log_recommendation(text: str) -> Optional[dict]:
    """从一段日志文本中提取最新的推荐结果。"""
    result = {}

    # 基础模型推理结果 (mode1/mode3 / 旧版格式)
    m = re.search(
        r"\[基础模型推理\]\s+driver_temp=([\d.]+)°C,\s+passenger_temp=([\d.]+)°C,\s+wind_speed=(\d+),\s+air_mode=(\d+)",
        text,
    )
    if m:
        result["base_rf"] = {
            "driver_temp": float(m.group(1)),
            "passenger_temp": float(m.group(2)),
            "wind_speed": int(m.group(3)),
            "air_mode": int(m.group(4)),
        }
        result["stage"] = "base_rf"

    # 模式4联合推理 (weather26) —— 副驾温度固定使用基础RF（单人时按存在性同步）
    m = re.search(
        r"\[模式4联合推理\]\s+RF=([\d.]+)°C/([\d.]+)°C\s*\+\s*MLP\s*Δ=([+-][\d.]+)/([+-][\d.]+)\s*→\s*([\d.]+)°C/([\d.]+)°C",
        text,
    )
    if m:
        rf_drv = float(m.group(1))
        rf_pax = float(m.group(2))
        delta_drv = float(m.group(3))
        # mode4下副驾不应用MLP残差。即使页面读到修改前的
        # 历史日志，也不再将其标记为副驾个性化修正。
        delta_pax = 0.0
        final_drv = float(m.group(5))
        final_pax = float(m.group(6))
        # 填充 base_rf (RF 原始输出)
        result["base_rf"] = {
            "driver_temp": rf_drv,
            "passenger_temp": rf_pax,
            "wind_speed": 0,
            "air_mode": 0,
        }
        # 填充联合结果；副驾个性化残差固定为0
        result["mlp"] = {
            "driver_temp": final_drv,
            "driver_delta": delta_drv,
            "passenger_temp": final_pax,
            "passenger_delta": delta_pax,
            "wind_speed": 0,
            "air_mode": 0,
        }
        result["stage"] = "joint_mlp"

    # MLP 残差修正结果 (旧版格式，兼容)
    m = re.search(
        r"\[MLP残差修正\]\s+RF基础\+Δ:\s+主驾=([\d.]+)°C\s+\(Δ=([+-][\d.]+)°C\),\s+副驾=([\d.]+)°C\s+\(Δ=([+-][\d.]+)°C\),\s+风速=(\d+),\s+模式=(\d+)",
        text,
    )
    if m:
        result["mlp"] = {
            "driver_temp": float(m.group(1)),
            "driver_delta": float(m.group(2)),
            "passenger_temp": float(m.group(3)),
            "passenger_delta": float(m.group(4)),
            "wind_speed": int(m.group(5)),
            "air_mode": int(m.group(6)),
        }
        if "stage" not in result:
            result["stage"] = "mlp"

    # 偏好层约束日志 (weather26)
    m_drv_pref = re.search(
        r"\[偏好层\]\s+用户稳定主驾温度约束:\s+MLP=([\d.]+)°C\s*→\s*偏好=([\d.]+)°C",
        text,
    )
    m_pax_pref = re.search(
        r"\[偏好层\]\s+用户稳定副驾温度约束:\s+MLP=([\d.]+)°C\s*→\s*偏好=([\d.]+)°C",
        text,
    )
    if m_drv_pref or m_pax_pref:
        pref = result.get("preference_layer", {})
        if m_drv_pref:
            pref["driver_mlp"] = float(m_drv_pref.group(1))
            pref["driver_pref"] = float(m_drv_pref.group(2))
        if m_pax_pref:
            pref["passenger_mlp"] = float(m_pax_pref.group(1))
            pref["passenger_pref"] = float(m_pax_pref.group(2))
        result["preference_layer"] = pref

    # 发送结果
    sent_matches = list(re.finditer(
        r"发送空调推荐结果:\s+主驾温度=([\d.]+)°C,\s+副驾温度=([\d.]+)°C,\s+风量=(\d+),\s+模式=(\d+)",
        text,
    ))
    m = sent_matches[-1] if sent_matches else None
    if m:
        result["sent"] = {
            "driver_temp": float(m.group(1)),
            "passenger_temp": float(m.group(2)),
            "wind_speed": int(m.group(3)),
            "air_mode": int(m.group(4)),
        }

    # mode4成功写入CAN Socket后记录的权威快照。使用最后一条，
    # 并重建与该次下发同属一轮的RF/MLP展示数据，避免混入旧日志。
    mode4_joint_matches = list(re.finditer(
        r"\[模式4最终下发\]\s+stage=joint_mlp,\s*"
        r"driver_temp=([\d.]+),\s*passenger_temp=([\d.]+),\s*"
        r"wind_speed=(\d+),\s*air_mode=(\d+),\s*"
        r"base_driver_temp=([\d.]+),\s*base_passenger_temp=([\d.]+),\s*"
        r"driver_delta=([+-][\d.]+),\s*passenger_delta=([+-][\d.]+)"
        r"(?:,\s*base_wind_speed=(\d+),\s*base_air_mode=(\d+),\s*"
        r"wind_delta=([+-]\d+),\s*mode_class=(\d+),\s*joint_wind_speed=(\d+))?",
        text,
    ))
    mode4_base_matches = list(re.finditer(
        r"\[模式4最终下发\]\s+stage=base_rf,\s*"
        r"driver_temp=([\d.]+),\s*passenger_temp=([\d.]+),\s*"
        r"wind_speed=(\d+),\s*air_mode=(\d+)",
        text,
    ))
    mode4_joint = mode4_joint_matches[-1] if mode4_joint_matches else None
    mode4_base = mode4_base_matches[-1] if mode4_base_matches else None

    if mode4_joint and (not mode4_base or mode4_joint.start() > mode4_base.start()):
        sent = {
            "driver_temp": float(mode4_joint.group(1)),
            "passenger_temp": float(mode4_joint.group(2)),
            "wind_speed": int(mode4_joint.group(3)),
            "air_mode": int(mode4_joint.group(4)),
        }
        result["sent"] = sent
        result["base_rf"] = {
            "driver_temp": float(mode4_joint.group(5)),
            "passenger_temp": float(mode4_joint.group(6)),
            "wind_speed": int(mode4_joint.group(9)) if mode4_joint.group(9) else 0,
            "air_mode": int(mode4_joint.group(10)) if mode4_joint.group(10) else 0,
        }
        result["mlp"] = {
            **sent,
            "driver_delta": float(mode4_joint.group(7)),
            "passenger_delta": 0.0,
            "wind_delta": int(mode4_joint.group(11)) if mode4_joint.group(11) else None,
            "mode_class": int(mode4_joint.group(12)) if mode4_joint.group(12) else None,
            "joint_wind_speed": int(mode4_joint.group(13)) if mode4_joint.group(13) else None,
        }
        result["stage"] = "joint_mlp"
        result["run_mode"] = "mode4"
    elif mode4_base:
        sent = {
            "driver_temp": float(mode4_base.group(1)),
            "passenger_temp": float(mode4_base.group(2)),
            "wind_speed": int(mode4_base.group(3)),
            "air_mode": int(mode4_base.group(4)),
        }
        result["sent"] = sent
        result["base_rf"] = dict(sent)
        result.pop("mlp", None)
        result["stage"] = "base_rf"
        result["run_mode"] = "mode4"

    # 推理阶段 (通用)
    if "stage" not in result:
        m = re.search(r"stage=(\w+)", text)
        if m:
            result["stage"] = m.group(1)

    # 定时推理模式检测 (weather26)
    m = re.search(r"\[定时推理\].*?模式=(\w+)", text)
    if m:
        result["run_mode"] = m.group(1)

    # 风量平滑 (通用 + weather26 MLP格式)
    m = re.search(r"\[风量平滑\]\s+(\w+)发送:\s+(\d+)\s+→\s+(\d+)", text)
    if m:
        result["wind_smooth"] = {
            "stage": m.group(1),
            "raw": int(m.group(2)),
            "smoothed": int(m.group(3)),
        }

    # 主副驾共享风量合成详情。最终发送值可能继续受到原有风量平滑限制。
    wind_blend_matches = list(re.finditer(
        r"\[乘员风量融合\]\s+driver_wind=(\d+),\s*"
        r"passenger_wind=(\d+),\s*K=([\d.]+),\s*"
        r"blended=([\d.]+),\s*rounded=(\d+),\s*sent=(\d+),\s*"
        r"driver_present=(\d+),\s*passenger_present=(\d+)",
        text,
    ))
    wind_blend = wind_blend_matches[-1] if wind_blend_matches else None
    if wind_blend:
        result["occupant_wind_blend"] = {
            "driver_wind": int(wind_blend.group(1)),
            "passenger_wind": int(wind_blend.group(2)),
            "k": float(wind_blend.group(3)),
            "blended": float(wind_blend.group(4)),
            "rounded": int(wind_blend.group(5)),
            "sent": int(wind_blend.group(6)),
            "driver_present": bool(int(wind_blend.group(7))),
            "passenger_present": bool(int(wind_blend.group(8))),
        }

    # 状态机
    m = re.search(r"\[状态机\]\s+(.+)$", text, re.MULTILINE)
    if m:
        result["state_machine"] = m.group(1).strip()

    # 等待信号
    m = re.search(
        r"\[等待信号\]\s+CAN:\s+(\d+)/(\d+),\s+Face:\s+(\d+)/(\d+).*?状态:\s+(\S+)",
        text,
    )
    if m:
        result["signal_status"] = {
            "can_ready": int(m.group(1)),
            "can_total": int(m.group(2)),
            "face_ready": int(m.group(3)),
            "face_total": int(m.group(4)),
            "state": m.group(5),
        }

    return result if result else None


def tail_log(log_path: str, last_pos: int) -> tuple:
    """读取日志文件新增内容，返回 (新文本, 新位置)。"""
    try:
        if not os.path.exists(log_path):
            return "", last_pos
        size = os.path.getsize(log_path)
        if size < last_pos:
            last_pos = 0
        if size == last_pos:
            return "", last_pos
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(last_pos)
            new_text = f.read()
        return new_text, size
    except Exception:
        return "", last_pos


def extract_recent_context(full_text: str, pattern: str, context_lines: int = 3) -> Optional[str]:
    """从全文提取最后一次匹配及其前后上下文。"""
    lines = full_text.split("\n")
    matching_indices = [i for i, line in enumerate(lines) if re.search(pattern, line)]
    if not matching_indices:
        return None
    idx = matching_indices[-1]
    start = max(0, idx - context_lines)
    end = min(len(lines), idx + context_lines + 1)
    return "\n".join(lines[start:end])


def data_collector_loop(log_path: str, pmv_api: str, face_api: str) -> None:
    """后台线程：每秒采集 PMV、人脸、推荐日志数据。"""
    global _log_position

    # 如果传入的路径不存在，尝试其他候选位置
    if not log_path or not os.path.exists(log_path):
        log_path = _pick_log_path()
    if not log_path:
        # 没有可用的日志文件，仍然运行但只采集 API 数据
        log_path = ""

    # 初始化：读取最近日志以获取已有推荐数据
    if log_path and os.path.exists(log_path):
        file_size = os.path.getsize(log_path)
        # 读取最后 64KB 来抓取最近的推荐结果
        start_pos = max(0, file_size - 65536)
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(start_pos)
            init_text = f.read()
        rec = parse_log_recommendation(init_text)
        if rec and rec.get("sent"):
            with _lock:
                _latest["recommendation"] = rec
        _log_position = file_size

    while True:
        try:
            pmv = fetch_json(pmv_api)
            face = fetch_json(face_api)

            # 增量读取推荐日志
            new_text, new_pos = tail_log(log_path, _log_position)
            _log_position = new_pos

            rec = None
            if new_text:
                rec = parse_log_recommendation(new_text)

            with _lock:
                if pmv:
                    _latest["pmv"] = pmv
                if face:
                    _latest["face"] = face
                if rec:
                    if rec.get("sent"):
                        # 只发布已成功下发的完整快照；新推理未发送成功时，
                        # 页面继续显示上一条真实下发命令。
                        if (
                            "occupant_wind_blend" not in rec
                            and _latest["recommendation"] is not None
                            and "occupant_wind_blend" in _latest["recommendation"]
                        ):
                            # 融合日志紧邻最终下发日志，跨两次tail读取时保留
                            # 刚解析到的同轮融合详情。
                            rec["occupant_wind_blend"] = _latest["recommendation"][
                                "occupant_wind_blend"
                            ]
                        _latest["recommendation"] = rec
                    elif _latest["recommendation"] is not None:
                        # 状态诊断字段可独立刷新，但不覆盖已下发的控制值和阶段。
                        for key in ("state_machine", "signal_status", "occupant_wind_blend"):
                            if key in rec:
                                _latest["recommendation"][key] = rec[key]
                _latest["updated"] = datetime.now().strftime("%H:%M:%S")
        except Exception:
            pass

        time.sleep(1.0)


# ---- HTTP 服务 -----------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIAC 热舒适度实时监控</title>
<style>
:root {
  --bg: #0f1119;
  --card: #1a1d2e;
  --card2: #212438;
  --border: #2a2e42;
  --text: #e0e2f0;
  --dim: #7b7f9a;
  --green: #34d399;
  --yellow: #fbbf24;
  --orange: #f97316;
  --red: #ef4444;
  --blue: #60a5fa;
  --cyan: #22d3ee;
  --purple: #a78bfa;
  --driver: #3b82f6;
  --passenger: #f59e0b;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}
.header {
  background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
  border-bottom: 1px solid var(--border);
  padding: 12px 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 10;
  backdrop-filter: blur(10px);
}
.header h1 {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -.3px;
  background: linear-gradient(135deg, #e0e2f0, #a5b4fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.header .status-row {
  display: flex;
  gap: 16px;
  align-items: center;
  font-size: 13px;
}
.status-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 6px;
}
.status-dot.ok { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.warn { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
.status-dot.err { background: var(--red); box-shadow: 0 0 6px var(--red); }
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px 28px;
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  transition: border-color .3s;
  margin-bottom: 0;
}
.card:hover { border-color: #3d4260; }
.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 18px;
}
.card-header .section-icon {
  width: 28px; height: 28px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 700;
}
.section-icon.vision { background: rgba(34,211,238,.15); color: var(--cyan); }
.section-icon.pmv { background: rgba(16,185,129,.15); color: var(--green); }
.section-icon.rec { background: rgba(167,139,250,.15); color: var(--purple); }
.card-header h2 { font-size: 16px; font-weight: 600; }
.card-header .badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 20px;
  font-weight: 500;
}
.badge.present { background: rgba(52,211,153,.15); color: var(--green); }
.badge.absent { background: rgba(239,68,68,.15); color: var(--red); }

/* ---- 流程箭头 ---- */
.flow-arrow {
  display: flex;
  justify-content: center;
  padding: 8px 0;
}
.flow-arrow svg { opacity: .35; }

/* ---- 视觉 + 环境 顶层卡片 ---- */
.vision-env-grid {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 28px;
  align-items: start;
}
.cabin-svg-wrap {
  position: relative;
  background: var(--card2);
  border-radius: 14px;
  padding: 16px 12px 10px;
  text-align: center;
}
.cabin-svg-wrap .cabin-label {
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  color: var(--dim);
  letter-spacing: 1px;
}
.seat-info {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.seat-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 14px 16px;
  background: var(--card2);
  border-radius: 12px;
  border-left: 3px solid transparent;
  transition: border-color .4s;
}
.seat-row.driver-seat { border-left-color: var(--driver); }
.seat-row.passenger-seat { border-left-color: var(--passenger); }
.seat-row.empty-seat { opacity: .45; }
.seat-icon {
  width: 44px; height: 44px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 16px;
  flex-shrink: 0;
}
.seat-icon.driver { background: rgba(59,130,246,.2); color: var(--driver); }
.seat-icon.passenger { background: rgba(245,158,11,.2); color: var(--passenger); }
.seat-icon.empty { background: rgba(123,127,154,.1); color: var(--dim); }
.seat-detail { flex: 1; min-width: 0; }
.seat-detail .seat-title { font-size: 13px; font-weight: 600; margin-bottom: 2px; }
.seat-detail .seat-status {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 10px;
  display: inline-block;
  margin-bottom: 6px;
}
.seat-status.occupied { background: rgba(52,211,153,.15); color: var(--green); }
.seat-status.vacant { background: rgba(239,68,68,.15); color: var(--red); }
.seat-detail .seat-attrs {
  font-size: 12px;
  color: var(--dim);
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.seat-detail .seat-attrs span { white-space: nowrap; }
.env-metrics-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 18px;
}
.env-metric {
  background: var(--card2);
  border-radius: 12px;
  padding: 14px 16px;
  text-align: center;
}
.env-metric .em-value { font-size: 26px; font-weight: 700; }
.env-metric .em-label { font-size: 11px; color: var(--dim); margin-top: 4px; }
.env-metric.cabin .em-value { color: var(--cyan); }
.env-metric.ambient .em-value { color: var(--yellow); }
.env-metric.run .em-value { font-size: 18px; color: var(--text); }
.env-metric.state .em-value { font-size: 15px; color: var(--purple); }

/* ---- PMV 中间层 ---- */
.pmv-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.pmv-display {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 12px;
}
.pmv-circle {
  width: 100px; height: 100px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  border: 3px solid;
  transition: all .5s;
  flex-shrink: 0;
}
.pmv-circle .val { font-size: 30px; line-height: 1; }
.pmv-circle .label { font-size: 11px; opacity: .8; margin-top: 2px; }
.pmv-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  flex: 1;
}
.pmv-metric {
  background: var(--card2);
  border-radius: 10px;
  padding: 12px 14px;
}
.pmv-metric .m-label { font-size: 11px; color: var(--dim); margin-bottom: 4px; text-transform: uppercase; letter-spacing: .5px; }
.pmv-metric .m-value { font-size: 18px; font-weight: 600; }
.pmv-metric .m-sub { font-size: 11px; color: var(--dim); margin-top: 1px; }

/* ---- 推荐输出 底层 ---- */
.rec-output-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
.rec-item {
  background: var(--card2);
  border-radius: 12px;
  padding: 18px 16px;
  text-align: center;
}
.rec-item .r-label { font-size: 11px; color: var(--dim); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .5px; }
.rec-item .r-value { font-size: 24px; font-weight: 700; }
.rec-item .r-sub { font-size: 12px; color: var(--dim); margin-top: 2px; }
.stage-info {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}
.stage-chip {
  font-size: 12px;
  padding: 4px 12px;
  border-radius: 14px;
  font-weight: 500;
  background: rgba(167,139,250,.15);
  color: var(--purple);
}
.stage-chip.rf { background: rgba(96,165,250,.15); color: var(--blue); }
.stage-chip.mlp { background: rgba(167,139,250,.15); color: var(--purple); }
.wind-blend-panel {
  margin-top: 14px;
  padding: 14px 16px;
  border: 1px solid rgba(34,211,238,.25);
  border-radius: 12px;
  background: rgba(34,211,238,.06);
}
.wind-blend-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.wind-formula { color: var(--cyan); font-size: 14px; font-weight: 600; }
.wind-k-control { display: flex; align-items: center; gap: 10px; min-width: 280px; }
.wind-k-control input { flex: 1; accent-color: var(--cyan); }
.wind-k-value { min-width: 48px; color: var(--cyan); font-weight: 700; }
.wind-blend-detail { margin-top: 9px; color: var(--dim); font-size: 12px; }
.wind-k-status { margin-left: 8px; }
.footer {
  text-align: center;
  padding: 12px;
  color: var(--dim);
  font-size: 12px;
}

/* PMV color coding */
.pmv-cold3 { border-color: #1e40af; background: rgba(30,64,175,.2); color: #93c5fd; }
.pmv-cold2 { border-color: #2563eb; background: rgba(37,99,235,.2); color: #93c5fd; }
.pmv-cold1 { border-color: #06b6d4; background: rgba(6,182,212,.2); color: #67e8f9; }
.pmv-neutral { border-color: #10b981; background: rgba(16,185,129,.2); color: #6ee7b7; }
.pmv-warm1 { border-color: #f59e0b; background: rgba(245,158,11,.2); color: #fcd34d; }
.pmv-warm2 { border-color: #f97316; background: rgba(249,115,22,.2); color: #fdba74; }
.pmv-hot3 { border-color: #ef4444; background: rgba(239,68,68,.2); color: #fca5a5; }

@media (max-width: 900px) {
  .vision-env-grid { grid-template-columns: 1fr; }
  .pmv-grid { grid-template-columns: 1fr; }
  .rec-output-grid { grid-template-columns: repeat(2, 1fr); }
  .env-metrics-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<div class="header">
  <h1>座舱热舒适度 · 实时监控</h1>
  <div class="status-row" id="statusRow">
    <span><span class="status-dot ok" id="dotVision"></span>视觉</span>
    <span><span class="status-dot ok" id="dotPmv"></span>PMV</span>
    <span><span class="status-dot ok" id="dotRec"></span>推荐</span>
    <span style="color:var(--dim);font-size:12px;" id="updateTime">--</span>
  </div>
</div>

<div class="container">

  <!-- ====== 第一层: 视觉感知 + 环境状态 ====== -->
  <div class="card">
    <div class="card-header">
      <div class="section-icon vision">V</div>
      <h2>视觉感知 · 环境状态</h2>
    </div>
    <div class="vision-env-grid">
      <!-- 座舱座舱图 -->
      <div class="cabin-svg-wrap">
        <div class="cabin-label">座舱 Occupancy</div>
        <svg viewBox="0 0 320 200" width="100%" style="max-width:320px;">
          <!-- 车身轮廓 -->
          <rect x="20" y="10" width="280" height="180" rx="28" fill="none" stroke="#2a2e42" stroke-width="2"/>
          <rect x="40" y="70" width="240" height="90" rx="12" fill="none" stroke="#2a2e42" stroke-width="1" stroke-dasharray="6,3"/>
          <!-- 方向盘 -->
          <circle cx="90" cy="115" r="22" fill="none" stroke="#3d4260" stroke-width="2.5"/>
          <line x1="90" y1="93" x2="90" y2="137" stroke="#3d4260" stroke-width="2"/>
          <line x1="68" y1="115" x2="112" y2="115" stroke="#3d4260" stroke-width="2"/>
          <!-- 主驾座椅 -->
          <rect id="svgDriverSeat" x="60" y="72" width="60" height="44" rx="10" fill="rgba(59,130,246,.08)" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="4,2"/>
          <text id="svgDriverLabel" x="90" y="98" text-anchor="middle" fill="#7b7f9a" font-size="12">主驾</text>
          <text id="svgDriverStatus" x="90" y="112" text-anchor="middle" fill="#ef4444" font-size="9">无人</text>
          <!-- 副驾座椅 -->
          <rect id="svgPaxSeat" x="200" y="72" width="60" height="44" rx="10" fill="rgba(245,158,11,.08)" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,2"/>
          <text id="svgPaxLabel" x="230" y="98" text-anchor="middle" fill="#7b7f9a" font-size="12">副驾</text>
          <text id="svgPaxStatus" x="230" y="112" text-anchor="middle" fill="#ef4444" font-size="9">无人</text>
        </svg>
      </div>
      <!-- 乘员身份 -->
      <div class="seat-info">
        <div class="seat-row driver-seat empty-seat" id="driverSeatRow">
          <div class="seat-icon driver">D</div>
          <div class="seat-detail">
            <div class="seat-title">主驾 Driver</div>
            <span class="seat-status vacant" id="driverStatusChip">无人</span>
            <div class="seat-attrs" id="driverAttrsDetail">
              <span id="driverIdDetail">--</span>
              <span id="driverGenderAge">--</span>
              <span id="driverCloth">--</span>
              <span id="driverHeight">--</span>
              <span id="driverBmi">--</span>
            </div>
          </div>
        </div>
        <div class="seat-row passenger-seat empty-seat" id="passengerSeatRow">
          <div class="seat-icon passenger">P</div>
          <div class="seat-detail">
            <div class="seat-title">副驾 Passenger</div>
            <span class="seat-status vacant" id="passengerStatusChip">无人</span>
            <div class="seat-attrs" id="passengerAttrsDetail">
              <span id="passengerIdDetail">--</span>
              <span id="passengerGenderAge">--</span>
              <span id="passengerCloth">--</span>
              <span id="passengerHeight">--</span>
              <span id="passengerBmi">--</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <!-- 环境指标 -->
    <div class="env-metrics-row">
      <div class="env-metric cabin">
        <div class="em-value" id="cabinTemp">--°</div>
        <div class="em-label">座舱温度</div>
      </div>
      <div class="env-metric ambient">
        <div class="em-value" id="ambTemp">--°</div>
        <div class="em-label">车外温度</div>
      </div>
      <div class="env-metric run">
        <div class="em-value" id="pmvRunIndex">--</div>
        <div class="em-label">PMV 更新轮次</div>
      </div>
      <div class="env-metric state">
        <div class="em-value" id="inferenceStateTop">--</div>
        <div class="em-label">推理状态</div>
      </div>
    </div>
  </div>

  <!-- 流程箭头: 视觉+环境 → PMV -->
  <div class="flow-arrow">
    <svg width="24" height="28" viewBox="0 0 24 28">
      <line x1="12" y1="0" x2="12" y2="18" stroke="#7b7f9a" stroke-width="2"/>
      <polyline points="4,14 12,24 20,14" fill="none" stroke="#7b7f9a" stroke-width="2" stroke-linecap="round"/>
    </svg>
  </div>

  <!-- ====== 第二层: PMV 热舒适度 ====== -->
  <div class="card">
    <div class="card-header">
      <div class="section-icon pmv">P</div>
      <h2>PMV 热舒适度评估</h2>
    </div>
    <div class="pmv-grid">
      <!-- 主驾 PMV -->
      <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
          <span style="font-size:14px;font-weight:600;color:var(--driver);">主驾 Driver</span>
          <span class="badge absent" id="driverBadge">无人</span>
        </div>
        <div class="pmv-display">
          <div class="pmv-circle pmv-neutral" id="driverPmvCircle">
            <span class="val">--</span>
            <span class="label">PMV</span>
          </div>
          <div class="pmv-metrics">
            <div class="pmv-metric">
              <div class="m-label">PPD 不满意率</div>
              <div class="m-value" id="driverPpd">--%</div>
            </div>
            <div class="pmv-metric">
              <div class="m-label">头部温度</div>
              <div class="m-value" id="driverHead">--°C</div>
            </div>
            <div class="pmv-metric">
              <div class="m-label">脚部温度</div>
              <div class="m-value" id="driverFeet">--°C</div>
            </div>
            <div class="pmv-metric">
              <div class="m-label">PMV 值</div>
              <div class="m-value" id="driverPmvVal" style="font-size:15px;">--</div>
              <div class="m-sub" id="driverPmvLabel"></div>
            </div>
          </div>
        </div>
      </div>
      <!-- 副驾 PMV -->
      <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
          <span style="font-size:14px;font-weight:600;color:var(--passenger);">副驾 Passenger</span>
          <span class="badge absent" id="passengerBadge">无人</span>
        </div>
        <div class="pmv-display">
          <div class="pmv-circle pmv-neutral" id="passengerPmvCircle">
            <span class="val">--</span>
            <span class="label">PMV</span>
          </div>
          <div class="pmv-metrics">
            <div class="pmv-metric">
              <div class="m-label">PPD 不满意率</div>
              <div class="m-value" id="passengerPpd">--%</div>
            </div>
            <div class="pmv-metric">
              <div class="m-label">头部温度</div>
              <div class="m-value" id="passengerHead">--°C</div>
            </div>
            <div class="pmv-metric">
              <div class="m-label">脚部温度</div>
              <div class="m-value" id="passengerFeet">--°C</div>
            </div>
            <div class="pmv-metric">
              <div class="m-label">PMV 值</div>
              <div class="m-value" id="passengerPmvVal" style="font-size:15px;">--</div>
              <div class="m-sub" id="passengerPmvLabel"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- 流程箭头: PMV → 推荐输出 -->
  <div class="flow-arrow">
    <svg width="24" height="28" viewBox="0 0 24 28">
      <line x1="12" y1="0" x2="12" y2="18" stroke="#7b7f9a" stroke-width="2"/>
      <polyline points="4,14 12,24 20,14" fill="none" stroke="#7b7f9a" stroke-width="2" stroke-linecap="round"/>
    </svg>
  </div>

  <!-- ====== 第三层: 空调推荐输出 ====== -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-header">
      <div class="section-icon rec">R</div>
      <h2>空调推荐输出</h2>
    </div>
    <div class="rec-output-grid">
      <div class="rec-item">
        <div class="r-label">主驾设定温度</div>
        <div class="r-value" id="recDriverTemp" style="color:var(--driver);">--°</div>
        <div class="r-sub" id="recDriverDelta"></div>
      </div>
      <div class="rec-item">
        <div class="r-label">副驾设定温度</div>
        <div class="r-value" id="recPassengerTemp" style="color:var(--passenger);">--°</div>
        <div class="r-sub" id="recPassengerDelta"></div>
      </div>
      <div class="rec-item">
        <div class="r-label">风量档位</div>
        <div class="r-value" id="recWind">--</div>
        <div class="r-sub" id="recWindSmooth"></div>
      </div>
      <div class="rec-item">
        <div class="r-label">出风模式</div>
        <div class="r-value" id="recMode" style="font-size:17px;">--</div>
        <div class="r-sub" id="recModeDelta"></div>
      </div>
    </div>
    <div class="wind-blend-panel">
      <div class="wind-blend-head">
        <div class="wind-formula">最终风量 = 主驾风量 × K + 副驾风量 × (1 − K)</div>
        <label class="wind-k-control" for="windKSlider">
          <span>K</span>
          <input id="windKSlider" type="range" min="0" max="1" step="0.05" value="0.5">
          <span class="wind-k-value" id="windKValue">0.50</span>
        </label>
      </div>
      <div class="wind-blend-detail">
        <span id="windBlendDetail">等待风量融合数据...</span>
        <span class="wind-k-status" id="windKStatus"></span>
      </div>
    </div>
    <div class="stage-info" id="stageInfo">
      <span style="color:var(--dim);font-size:12px;">等待推荐数据...</span>
    </div>
  </div>

</div>

<div class="footer">AIAC Dashboard · 自动刷新 1s · <span id="footerTime">--</span></div>

<script>
const AIR_MODE_MAP = {0:"无请求",1:"吹面",2:"吹面+吹脚",3:"吹脚",4:"吹脚+除霜",5:"除霜",6:"吹面+除霜",7:"吹面+吹脚+除霜"};
const GENDER_MAP = {"0":"女","1":"男"};
const AGE_MAP = {"0":"12-17","1":"18-40","2":"41-59","3":"60-74","4":"75+"};
const CLOTH_MAP = {"0":"西装外套","1":"薄夹克","2":"长款大衣","3":"羽绒服","4":"长袖针织毛衣","5":"长袖衬衫","6":"短袖","7":"背心","8":"长袖","9":"连帽卫衣"};
let windKEditing = false;

function renderWindK(value) {
  const k = Math.max(0, Math.min(1, Number(value)));
  document.getElementById('windKSlider').value = k;
  document.getElementById('windKValue').textContent = k.toFixed(2);
}

async function saveWindK(value) {
  const status = document.getElementById('windKStatus');
  status.textContent = '保存中...';
  try {
    const resp = await fetch('/settings/wind-k', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({occupant_wind_weight_k: Number(value)})
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const payload = await resp.json();
    renderWindK(payload.occupant_wind_weight_k);
    status.textContent = '已生效';
  } catch (e) {
    status.textContent = '保存失败';
  } finally {
    windKEditing = false;
    setTimeout(() => { status.textContent = ''; }, 1500);
  }
}

// ---- 视觉识别防抖状态 ----
const AGE_DEBOUNCE_MS = 3000;   // faceId 变动后 3s 锁定年龄
const CLOTH_COOLDOWN_MS = 2000; // 衣着最小更新间隔 2s

const _debounce = {
    driver: { lastFaceId: null, faceIdChangeTime: 0, lockedAge: null, lastClothTime: 0, lastClothValue: null },
    passenger: { lastFaceId: null, faceIdChangeTime: 0, lockedAge: null, lastClothTime: 0, lastClothValue: null },
};

function pmvClass(pmv) {
  if (pmv === null || pmv === undefined) return 'pmv-neutral';
  if (pmv <= -2.5) return 'pmv-cold3';
  if (pmv <= -1.5) return 'pmv-cold2';
  if (pmv <= -0.5) return 'pmv-cold1';
  if (pmv < 0.5) return 'pmv-neutral';
  if (pmv < 1.5) return 'pmv-warm1';
  if (pmv < 2.5) return 'pmv-warm2';
  return 'pmv-hot3';
}

function pmvLabel(pmv) {
  if (pmv === null || pmv === undefined) return 'N/A';
  if (pmv <= -2.5) return '冷';
  if (pmv <= -1.5) return '凉';
  if (pmv <= -0.5) return '稍凉';
  if (pmv < 0.5) return '舒适';
  if (pmv < 1.5) return '稍暖';
  if (pmv < 2.5) return '暖';
  return '热';
}

function updatePmvCircle(el, pmv) {
  el.className = 'pmv-circle ' + pmvClass(pmv);
  el.querySelector('.val').textContent = pmv !== null && pmv !== undefined ? pmv.toFixed(2) : '--';
  el.querySelector('.label').textContent = 'PMV ' + pmvLabel(pmv);
}

function updateDot(el, ok) {
  el.className = 'status-dot ' + (ok ? 'ok' : 'err');
}

function updateSeatUI(side, personData, present) {
  const prefix = side === 'driver' ? 'driver' : 'passenger';
  const svgPrefix = side === 'driver' ? 'svgDriver' : 'svgPax';

  // 座舱 SVG
  const seatRect = document.getElementById(svgPrefix + 'Seat');
  const statusText = document.getElementById(svgPrefix + 'Status');
  if (present) {
    seatRect.setAttribute('fill', side === 'driver' ? 'rgba(59,130,246,.2)' : 'rgba(245,158,11,.2)');
    seatRect.setAttribute('stroke', side === 'driver' ? '#3b82f6' : '#f59e0b');
    seatRect.setAttribute('stroke-dasharray', 'none');
    statusText.textContent = '在座';
    statusText.setAttribute('fill', '#34d399');
  } else {
    seatRect.setAttribute('fill', side === 'driver' ? 'rgba(59,130,246,.08)' : 'rgba(245,158,11,.08)');
    seatRect.setAttribute('stroke', side === 'driver' ? '#3b82f6' : '#f59e0b');
    seatRect.setAttribute('stroke-dasharray', '4,2');
    statusText.textContent = '无人';
    statusText.setAttribute('fill', '#ef4444');
  }

  // 右侧详情
  const row = document.getElementById(prefix + 'SeatRow');
  const chip = document.getElementById(prefix + 'StatusChip');
  if (present) {
    row.classList.remove('empty-seat');
    chip.textContent = '在座';
    chip.className = 'seat-status occupied';
  } else {
    row.classList.add('empty-seat');
    chip.textContent = '无人';
    chip.className = 'seat-status vacant';
  }

  if (present && personData) {
    const db = _debounce[side];
    const now = Date.now();
    const faceId = personData.identity_id || null;

    // faceId 变动检测
    if (faceId !== db.lastFaceId) {
        db.lastFaceId = faceId;
        db.faceIdChangeTime = now;
        db.lockedAge = null;  // 重置年龄锁定
    }

    // 年龄防抖: faceId 变动 3s 后锁定，不再跳跃
    let displayAge;
    const ageRaw = AGE_MAP[personData.age] || (personData.age !== undefined ? personData.age : '--');
    if (db.lockedAge !== null) {
        displayAge = db.lockedAge;
    } else if (now - db.faceIdChangeTime >= AGE_DEBOUNCE_MS) {
        db.lockedAge = ageRaw;
        displayAge = db.lockedAge;
    } else {
        displayAge = ageRaw;
    }

    // 衣着防抖: 至少间隔 CLOTH_COOLDOWN_MS 才更新
    const clothKey = String(personData.cloth !== undefined ? personData.cloth : '');
    const clothRaw = CLOTH_MAP[clothKey] || (clothKey !== '' ? '衣着' + clothKey : '--');
    let displayCloth;
    if (db.lastClothValue === null || (now - db.lastClothTime >= CLOTH_COOLDOWN_MS && clothRaw !== db.lastClothValue)) {
        db.lastClothTime = now;
        db.lastClothValue = clothRaw;
        displayCloth = clothRaw;
    } else {
        displayCloth = db.lastClothValue;
    }

    // 性别 / BMI: 不加限制，直接显示最新值
    const idStr = faceId ? faceId.substring(0,12) + '...' : '--';
    const gender = GENDER_MAP[personData.gender] || (personData.gender !== undefined ? personData.gender : '--');

    document.getElementById(prefix + 'IdDetail').textContent = 'ID: ' + idStr;
    document.getElementById(prefix + 'GenderAge').textContent = gender + ' · ' + displayAge;
    document.getElementById(prefix + 'Cloth').textContent = displayCloth;
    document.getElementById(prefix + 'Height').textContent = personData.height !== undefined ? personData.height.toFixed(1) + 'cm' : '--';
    document.getElementById(prefix + 'Bmi').textContent = personData.bmi !== undefined ? 'BMI ' + personData.bmi.toFixed(1) : '--';
  } else {
    // 座位无人时重置防抖状态
    const db = _debounce[side];
    db.lastFaceId = null;
    db.faceIdChangeTime = 0;
    db.lockedAge = null;
    db.lastClothTime = 0;
    db.lastClothValue = null;

    document.getElementById(prefix + 'IdDetail').textContent = '--';
    document.getElementById(prefix + 'GenderAge').textContent = '--';
    document.getElementById(prefix + 'Cloth').textContent = '--';
    document.getElementById(prefix + 'Height').textContent = '--';
    document.getElementById(prefix + 'Bmi').textContent = '--';
  }
}

async function refresh() {
  try {
    const resp = await fetch('/data');
    const data = await resp.json();
    const pmv = data.pmv || {};
    const face = data.face || {};
    const rec = data.recommendation || {};
    const configuredK = data.settings?.occupant_wind_weight_k;
    if (!windKEditing && configuredK !== undefined) renderWindK(configuredK);

    // --- 状态指示 ---
    updateDot(document.getElementById('dotVision'), !!data.face && data.face.status === 'ok');
    updateDot(document.getElementById('dotPmv'), !!data.pmv && data.pmv.status === 'ok');
    updateDot(document.getElementById('dotRec'), !!data.recommendation);
    document.getElementById('updateTime').textContent = data.updated || '--';
    document.getElementById('footerTime').textContent = new Date().toLocaleTimeString('zh-CN');

    // --- 视觉: 座舱占位 + 乘员信息 ---
    const drvPresent = face.driver && face.driver.identity_id;
    const paxPresent = face.passenger && face.passenger.identity_id;
    updateSeatUI('driver', face.driver, !!drvPresent);
    updateSeatUI('passenger', face.passenger, !!paxPresent);

    // --- 环境 ---
    document.getElementById('cabinTemp').textContent = pmv.cabin_temp_c !== undefined ? pmv.cabin_temp_c.toFixed(1) + '°C' : '--°';
    document.getElementById('ambTemp').textContent = pmv.amb_temp_c !== undefined ? pmv.amb_temp_c.toFixed(1) + '°C' : '--°';
    document.getElementById('pmvRunIndex').textContent = pmv.run_index !== undefined ? '#' + pmv.run_index : '--';
    document.getElementById('inferenceStateTop').textContent = rec.state_machine || rec.signal_status?.state || '--';

    // --- PMV: 主驾 ---
    const drv = pmv.driver || {};
    document.getElementById('driverBadge').textContent = drvPresent ? '在座' : '无人';
    document.getElementById('driverBadge').className = 'badge ' + (drvPresent ? 'present' : 'absent');
    updatePmvCircle(document.getElementById('driverPmvCircle'), drv.pmv);
    document.getElementById('driverPpd').textContent = drv.ppd !== undefined ? drv.ppd.toFixed(1) + '%' : '--%';
    document.getElementById('driverHead').textContent = drv.head_temp_c !== undefined ? drv.head_temp_c.toFixed(1) + '°C' : '--°C';
    document.getElementById('driverFeet').textContent = drv.feet_temp_c !== undefined ? drv.feet_temp_c.toFixed(1) + '°C' : '--°C';
    document.getElementById('driverPmvVal').textContent = drv.pmv !== undefined ? drv.pmv.toFixed(2) : '--';
    document.getElementById('driverPmvLabel').textContent = pmvLabel(drv.pmv);

    // --- PMV: 副驾 ---
    const pax = pmv.passenger || {};
    document.getElementById('passengerBadge').textContent = paxPresent ? '在座' : '无人';
    document.getElementById('passengerBadge').className = 'badge ' + (paxPresent ? 'present' : 'absent');
    updatePmvCircle(document.getElementById('passengerPmvCircle'), pax.pmv);
    document.getElementById('passengerPpd').textContent = pax.ppd !== undefined ? pax.ppd.toFixed(1) + '%' : '--%';
    document.getElementById('passengerHead').textContent = pax.head_temp_c !== undefined ? pax.head_temp_c.toFixed(1) + '°C' : '--°C';
    document.getElementById('passengerFeet').textContent = pax.feet_temp_c !== undefined ? pax.feet_temp_c.toFixed(1) + '°C' : '--°C';
    document.getElementById('passengerPmvVal').textContent = pax.pmv !== undefined ? pax.pmv.toFixed(2) : '--';
    document.getElementById('passengerPmvLabel').textContent = pmvLabel(pax.pmv);

    // --- 推荐输出 ---
    // 控制卡片只显示成功写入CAN Socket的最终命令。
    // 推理失败、发送失败或Shadow模式均不用RF/MLP预测值冒充已下发值。
    const sentRec = rec.sent || {};
    document.getElementById('recDriverTemp').textContent = sentRec.driver_temp !== undefined ? sentRec.driver_temp.toFixed(1) + '°C' : '--°';
    document.getElementById('recPassengerTemp').textContent = sentRec.passenger_temp !== undefined ? sentRec.passenger_temp.toFixed(1) + '°C' : '--°';
    document.getElementById('recWind').textContent = sentRec.wind_speed !== undefined ? sentRec.wind_speed : '--';
    const mode = sentRec.air_mode !== undefined ? sentRec.air_mode : null;
    document.getElementById('recMode').textContent = mode !== null ? (AIR_MODE_MAP[mode] || ('模式' + mode)) : '--';

    const blend = rec.occupant_wind_blend;
    if (blend) {
      const occupancy = blend.driver_present && blend.passenger_present
        ? '主副驾均在'
        : (blend.driver_present ? '仅主驾在' : (blend.passenger_present ? '仅副驾在' : '无人'));
      let detail = occupancy + '：' + blend.driver_wind + ' × ' + blend.k.toFixed(2) +
                   ' + ' + blend.passenger_wind + ' × ' + (1 - blend.k).toFixed(2) +
                   ' = ' + blend.blended.toFixed(2) + '，取整 ' + blend.rounded;
      if (blend.sent !== blend.rounded) detail += '，平滑后下发 ' + blend.sent;
      document.getElementById('windBlendDetail').textContent = detail;
    } else {
      document.getElementById('windBlendDetail').textContent = '等待风量融合数据...';
    }

    // MLP delta 显示 (优先 weather26 格式，兼容旧格式)
    const mlpRec = rec.mlp || {};
    const prefRec = rec.preference_layer || {};
    let deltaInfo = '';

    // 模式4联合推理: RF + MLP Δ → 最终
    if (rec.stage === 'joint_mlp' && rec.base_rf && rec.mlp) {
      const dDrv = rec.mlp.driver_delta !== undefined ? rec.mlp.driver_delta : 0;
      deltaInfo = 'RF ' + rec.base_rf.driver_temp.toFixed(1) + '°/' + rec.base_rf.passenger_temp.toFixed(1) + '°' +
                  ' + 主驾MLP Δ' + (dDrv >= 0 ? '+' : '') + dDrv.toFixed(1) + ' / 副驾基础RF';
      document.getElementById('recDriverDelta').textContent = 'Δ' + (dDrv >= 0 ? '+' : '') + dDrv.toFixed(1) + '°C';
      document.getElementById('recPassengerDelta').textContent = '基础RF';

      if (rec.mlp.wind_delta !== undefined && rec.mlp.wind_delta !== null) {
        const dWind = rec.mlp.wind_delta;
        let windText = 'MLP Δ' + (dWind >= 0 ? '+' : '') + dWind + '档';
        if (rec.mlp.joint_wind_speed !== undefined && rec.mlp.joint_wind_speed !== null &&
            sentRec.wind_speed !== undefined && rec.mlp.joint_wind_speed !== sentRec.wind_speed) {
          windText += ' · 平滑' + rec.mlp.joint_wind_speed + '→' + sentRec.wind_speed;
        }
        document.getElementById('recWindSmooth').textContent = windText;
      } else {
        document.getElementById('recWindSmooth').textContent = '';
      }

      if (rec.mlp.mode_class !== undefined && rec.mlp.mode_class !== null) {
        const baseModeName = AIR_MODE_MAP[rec.base_rf.air_mode] || ('模式' + rec.base_rf.air_mode);
        const finalModeName = AIR_MODE_MAP[sentRec.air_mode] || ('模式' + sentRec.air_mode);
        document.getElementById('recModeDelta').textContent = rec.mlp.mode_class === 0
          ? 'MLP 保持基础RF（' + baseModeName + '）'
          : 'MLP ' + baseModeName + ' → ' + finalModeName;
      } else {
        document.getElementById('recModeDelta').textContent = '';
      }
    } else if (rec.mlp && rec.mlp.driver_delta !== undefined) {
      document.getElementById('recDriverDelta').textContent = 'Δ' + (rec.mlp.driver_delta >= 0 ? '+' : '') + rec.mlp.driver_delta.toFixed(1) + '°C';
      document.getElementById('recPassengerDelta').textContent = 'Δ' + (rec.mlp.passenger_delta >= 0 ? '+' : '') + rec.mlp.passenger_delta.toFixed(1) + '°C';
      document.getElementById('recModeDelta').textContent = '';
    } else {
      document.getElementById('recDriverDelta').textContent = '';
      document.getElementById('recPassengerDelta').textContent = '';
      document.getElementById('recModeDelta').textContent = '';
    }

    if (rec.stage !== 'joint_mlp' && rec.wind_smooth) {
      document.getElementById('recWindSmooth').textContent = '原始' + rec.wind_smooth.raw + '→' + rec.wind_smooth.smoothed;
    } else if (rec.stage !== 'joint_mlp') {
      document.getElementById('recWindSmooth').textContent = '';
    }

    // --- 阶段信息 ---
    const stageDiv = document.getElementById('stageInfo');
    let stageHTML = '';
    if (rec.stage) {
      const cls = rec.stage === 'base_rf' ? 'rf' : 'mlp';
      const labels = { base_rf: '基础RF', mlp: 'MLP修正', joint_mlp: '联合推理RF+MLP' };
      stageHTML += '<span class="stage-chip ' + cls + '">阶段: ' + (labels[rec.stage] || rec.stage) + '</span>';
    }
    if (rec.run_mode) {
      stageHTML += '<span class="stage-chip" style="background:rgba(34,211,238,.15);color:var(--cyan);">模式: ' + rec.run_mode + '</span>';
    }
    if (rec.state_machine) {
      stageHTML += '<span style="color:var(--dim);font-size:12px;">状态机: ' + rec.state_machine + '</span>';
    }
    if (rec.signal_status) {
      const ss = rec.signal_status;
      stageHTML += '<span style="color:var(--dim);font-size:12px;">信号: CAN ' + ss.can_ready + '/' + ss.can_total + ' Face ' + ss.face_ready + '/' + ss.face_total + '</span>';
    }
    if (rec.base_rf && !rec.mlp && rec.stage !== 'joint_mlp') {
      stageHTML += '<span class="stage-chip rf">基础RF</span>';
    }
    if (rec.mlp && rec.stage !== 'joint_mlp') {
      stageHTML += '<span class="stage-chip mlp">MLP偏好学习</span>';
    }
    if (rec.base_rf && rec.mlp && rec.stage === 'joint_mlp') {
      stageHTML += '<span class="stage-chip mlp">联合推理 (RF+MLP)</span>';
    }
    stageDiv.innerHTML = stageHTML || '<span style="color:var(--dim);font-size:12px;">等待推荐数据...</span>';

  } catch(e) {
    console.error('refresh error:', e);
  }
}

const windKSlider = document.getElementById('windKSlider');
windKSlider.addEventListener('input', (event) => {
  windKEditing = true;
  renderWindK(event.target.value);
});
windKSlider.addEventListener('change', (event) => saveWindK(event.target.value));

setInterval(refresh, 1000);
refresh();
</script>

</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    def log_message(self, format, *args):
        pass  # 静默日志

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/data":
            self._serve_data()
        elif self.path == "/health":
            self._serve_json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/settings/wind-k":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 4096:
                raise ValueError("invalid content length")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if "occupant_wind_weight_k" not in payload:
                raise ValueError("missing occupant_wind_weight_k")
            requested = float(payload["occupant_wind_weight_k"])
            if not 0.0 <= requested <= 1.0:
                raise ValueError("K must be between 0 and 1")
            global _wind_weight_k
            with _lock:
                _wind_weight_k = _write_wind_k(_wind_k_path, requested)
                response = {"occupant_wind_weight_k": _wind_weight_k}
            self._serve_json(response)
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            self._serve_json({"error": str(exc)}, status=400)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _serve_data(self):
        global _wind_weight_k
        with _lock:
            _wind_weight_k = _read_wind_k(_wind_k_path, _wind_weight_k)
            data = {
                "pmv": _latest.get("pmv"),
                "face": _latest.get("face"),
                "recommendation": _latest.get("recommendation"),
                "updated": _latest.get("updated"),
                "settings": {"occupant_wind_weight_k": _wind_weight_k},
            }
        self._serve_json(data)

    def _serve_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global _wind_k_path, _wind_weight_k
    parser = argparse.ArgumentParser(description="AIAC Live Dashboard")
    parser.add_argument("--port", type=int, default=7862, help="HTTP 服务端口 (默认 7862)")
    parser.add_argument("--pmv-api", default=DEFAULT_PMV_API, help="PMV API 地址")
    parser.add_argument("--face-api", default=DEFAULT_FACE_API, help="Face API 地址")
    parser.add_argument("--rec-log", default=str(DEFAULT_REC_LOG), help="推荐日志路径")
    parser.add_argument("--wind-k-file", default="", help="运行时风量权重K配置路径")
    args = parser.parse_args()
    log_path = args.rec_log
    pmv_api = args.pmv_api
    face_api = args.face_api
    _wind_k_path = Path(args.wind_k_file).resolve() if args.wind_k_file else _resolve_wind_k_path(log_path)
    _wind_weight_k = _read_wind_k(_wind_k_path, DEFAULT_WIND_WEIGHT_K)

    print(f"AIAC Live Dashboard")
    print(f"  PMV API : {pmv_api}")
    print(f"  Face API: {face_api}")
    print(f"  Rec Log : {log_path}")
    print(f"  Wind K  : {_wind_weight_k:.2f} ({_wind_k_path})")
    print(f"  访问地址: http://0.0.0.0:{args.port}")
    print()

    # 启动后台数据采集线程
    collector = threading.Thread(
        target=data_collector_loop, args=(log_path, pmv_api, face_api), daemon=True
    )
    collector.start()

    # 启动 HTTP 服务
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务...")
        server.shutdown()


if __name__ == "__main__":
    main()
