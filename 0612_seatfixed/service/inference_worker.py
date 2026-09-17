"""
service/inference_worker.py
============================
GPU 推理进程入口。不依赖 FastAPI，独立运行：
  - 加载所有模型
  - 运行 VehicleRecognizer 实时识别
  - 通过 IPC (共享内存) 发布帧和结果
  - 通过 IPC (Unix Socket) 接收 API 命令
  - 连接 CAN socket 实时获取座椅位置

启动方式：
  python -m service.inference_worker
"""

from __future__ import annotations

import os
import sys
import time
import signal
import json
import logging
import threading
import faulthandler
from pathlib import Path
from collections import deque as _deque
import numpy as np

# 确保项目根目录在 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 尽早启用 faulthandler，便于定位 SIGSEGV ──
_fault_log = str(PROJECT_ROOT / "logs" / "inference_worker.fault.log")
faulthandler.enable(file=open(_fault_log, "a"), all_threads=True)

# ── 限制 CPU 线程爆炸（每个库都可能自己开线程池）──
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

# 加载 .env（不依赖 shell）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ── 设备主开关 ──
_DEVICE = os.getenv("DEVICE", "cuda").lower()
_ORT_DEFAULT = "CUDAExecutionProvider" if _DEVICE == "cuda" else "CPUExecutionProvider"
_DET_CTX_DEFAULT = "0" if _DEVICE == "cuda" else "-1"
_GA_DEVICE_DEFAULT = "gpu" if _DEVICE == "cuda" else "cpu"

os.environ.setdefault("ORT_PROVIDERS", _ORT_DEFAULT)
os.environ.setdefault("DETECTOR_CTX_ID", _DET_CTX_DEFAULT)
os.environ.setdefault("GENDER_AGE_DEVICE", _GA_DEVICE_DEFAULT)
os.environ.setdefault("GENDER_AGE_ORT_PROVIDERS", _ORT_DEFAULT)

# ── ONNX Runtime 线程限制，减少 CUDA 资源争用 ──
os.environ.setdefault("ORT_THREAD_POOL_AFFINITY", "0,1,2,3,4,5,6,7")
os.environ.setdefault("ORT_DISABLE_THREADING", "0")

# 轻量导入（先不加载模型依赖）
from service.ipc import IPCServer, RESULT_BUFFER_SIZE, DEFAULT_FRAME_W, DEFAULT_FRAME_H

# ═══════════════════════════════════════════════════════════════════════
# 0. IPC 最先就绪（让 API 进程在模型加载期间即可连接）
# ═══════════════════════════════════════════════════════════════════════
_ipc_early = IPCServer(frame_w=DEFAULT_FRAME_W, frame_h=DEFAULT_FRAME_H)
_ipc_early.start_cmd_server(lambda c: {"error": "models loading"})

# ── 再加载重模块 ──
from core.gallery import Gallery, DEFAULT_SIM_THRESHOLD
from service.detector import FaceDetector
from service.recognizer import VehicleRecognizer
from service.cloth_detector import ClothDetector
from service.gender_age_recognizer import GenderAgeRecognizer
from service.can_client import CANSeatClient
from service.body_detector import BodyDetector

try:
    from service.height_estimator import HeightEstimator
except ImportError:
    HeightEstimator = None
try:
    from service.bmi_predictor import BMIPredictor
except ImportError:
    BMIPredictor = None

# ── 配置 ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
GALLERY_FILE = os.getenv("GALLERY_FILE", str(PROJECT_ROOT / "data" / "gallery.pkl"))
HEIGHT_MODELS_DIR = os.getenv("HEIGHT_MODELS_DIR", str(PROJECT_ROOT / "models" / "height"))
CAMERA_ID_RAW = os.getenv("CAMERA_ID", "0")
try:
    CAMERA_ID = int(CAMERA_ID_RAW)
except ValueError:
    CAMERA_ID = CAMERA_ID_RAW
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", str(DEFAULT_SIM_THRESHOLD)))
SMOOTH_WINDOW = int(os.getenv("SMOOTH_WINDOW", "50"))
POSITIONLESS = os.getenv("POSITIONLESS", "true").lower() not in ("0", "false", "no", "off")
AUTO_ENROLL = os.getenv("AUTO_ENROLL", "true").lower() not in ("0", "false", "no", "off")

# 衣物动态稳定配置
# 初次识别需要70%一致
_CLOTH_INIT_RATIO = float(
    os.getenv("CLOTH_INIT_RATIO", "0.7")
)

# 已有类别后，新的类别需要85%一致才切换
_CLOTH_CHANGE_RATIO = float(
    os.getenv("CLOTH_CHANGE_RATIO", "0.85")
)

# 投票窗口
_CLOTH_HISTORY_SIZE = int(
    os.getenv("CLOTH_HISTORY_SIZE", "125")
)

# ── 日志 ──
_LEVEL_MAP = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
log_level = _LEVEL_MAP.get(LOG_LEVEL, logging.INFO)
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "inference_worker.log", encoding="utf-8"),
    ],
)
logging.getLogger().handlers[0].setLevel(log_level)
for lib in ("insightface", "onnxruntime", "PIL", "matplotlib", "ultralytics"):
    logging.getLogger(lib).setLevel(logging.WARNING)

log = logging.getLogger("worker")

# ── 全局引用（供命令处理）──
_recognizer = None
_gallery = None
_height_est = None
_ipc: IPCServer | None = None
_running = True
_can_client = None  # 原代码缺少定义，补上


# ═══════════════════════════════════════════════════════════════════════
# 命令处理
# ═══════════════════════════════════════════════════════════════════════

def _handle_cmd(cmd: dict) -> dict:
    """处理来自 API 进程的命令。返回响应 dict。"""
    cmd_type = cmd.get("cmd", "")
    try:
        if cmd_type == "pause":
            if _recognizer:
                _recognizer.pause()
            return {"ok": True}

        elif cmd_type == "resume":
            if _recognizer:
                _recognizer.resume()
            return {"ok": True}

        elif cmd_type == "gallery_list":
            if _gallery:
                items = {}
                for identity_id, entry in _gallery.get_all().items():
                    items[identity_id] = {
                        "name": entry.name,
                        "is_registered": entry.is_registered,
                        "created_at": str(entry.created_at),
                        "metadata": dict(getattr(entry, "metadata", {}) or {}),
                    }
                return {"ok": True, "gallery": items}
            return {"error": "gallery not available"}

        elif cmd_type == "register":
            identity_id = cmd.get("identity_id", "")
            name = cmd.get("name", identity_id)
            metadata = cmd.get("metadata", {})
            emb_data = cmd.get("embedding")
            if not emb_data or not identity_id:
                return {"error": "identity_id and embedding required"}
            import numpy as np
            from core.gallery import normalize_emb
            emb = np.array(emb_data, dtype=np.float32)
            _gallery.register(identity_id, normalize_emb(emb), name=name, metadata=metadata)
            _gallery.save()
            return {"ok": True}

        elif cmd_type == "unregister":
            identity_id = cmd.get("identity_id", "")
            if identity_id and _gallery:
                _gallery.remove(identity_id)
                _gallery.save()
                return {"ok": True}
            return {"error": "identity_id required"}

        elif cmd_type == "clear_gallery":
            if _gallery:
                _gallery.clear()
                _gallery.save()
                return {"ok": True}
            return {"error": "gallery not available"}

        elif cmd_type == "set_seat_params":
            if _height_est:
                dx = float(cmd.get("dx", 0.18))
                dy = float(cmd.get("dy", 0.28))
                px = float(cmd.get("px", 0.27))
                py = float(cmd.get("py", 0.58))
                d_gen = float(cmd.get("d_gender", 1.0))
                p_gen = float(cmd.get("p_gender", 1.0))
                _height_est.set_seat_params(dx, dy, px, py, d_gender=d_gen, p_gender=p_gen)
                return {"ok": True}
            return {"error": "height estimator not available"}

        elif cmd_type == "swap":
            if _gallery:
                _gallery.swap_driver_passenger()
                _gallery.save()
                return {"ok": True}
            return {"error": "gallery not available"}

        elif cmd_type == "stats":
            if _recognizer and _gallery:
                cur = _recognizer.get_current()
                return {
                    "ok": True,
                    "stats": {
                        "gallery_count": len(_gallery.get_all()),
                        "gallery_file": GALLERY_FILE,
                        "total_detected": cur.get("total_detected", 0),
                        "total_known": cur.get("total_known", 0),
                        "running": bool(_recognizer._running),
                        "paused": bool(_recognizer._paused),
                    }
                }
            return {"error": "not ready"}

        elif cmd_type == "recognize_image":
            image_path = cmd.get("image_path", "")
            if not image_path or not os.path.isfile(image_path):
                return {"error": f"image not found: {image_path}"}
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "failed to decode image"}
            if _recognizer:
                result = _recognizer.recognize_image(img)
                return {"ok": True, **result}
            return {"error": "recognizer not available"}

        elif cmd_type == "process_video":
            video_path = cmd.get("video_path", "")
            output_path = cmd.get("output_path") or None
            skip_frames = int(cmd.get("skip_frames", 3))
            if not video_path or not os.path.isfile(video_path):
                return {"error": f"video not found: {video_path}"}
            if _recognizer:
                result = _recognizer.process_video(video_path, output_path=output_path, skip_frames=skip_frames)
                return {"ok": True, **result}
            return {"error": "recognizer not available"}

        elif cmd_type == "set_body_roi":
            left = cmd.get("left", [0.0, 0.05, 0.40, 0.80])
            right = cmd.get("right", [0.52, 0.05, 1.0, 0.80])
            from utils import set_body_rois
            set_body_rois(left, right)
            return {"ok": True}

        elif cmd_type == "start_recording":
            if _recognizer:
                _recognizer.start_recording()
                return {"ok": True, "recording": True}
            return {"error": "recognizer not available"}

        elif cmd_type == "stop_recording":
            if _recognizer:
                _recognizer.stop_recording()
                return {"ok": True, "recording": False}
            return {"error": "recognizer not available"}

        elif cmd_type == "recording_status":
            if _recognizer:
                return {
                    "ok": True,
                    "recording": _recognizer.is_recording,
                    "frames": _recognizer.recording_frames,
                    "max_frames": _recognizer._recording_max,
                }
            return {"error": "recognizer not available"}

        elif cmd_type == "health":
            return {"ok": True, "status": "running"}

        else:
            return {"error": f"unknown command: {cmd_type}"}

    except Exception as e:
        log.exception("Command handler error: %s", cmd_type)
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    global _recognizer, _gallery, _height_est, _ipc, _running, _can_client

    log.info("=== GPU 推理进程启动 ===")
    log.info("DEVICE=%s | ORT=%s | DETECTOR_CTX=%s | GA_DEVICE=%s",
             _DEVICE, _ORT_DEFAULT, _DET_CTX_DEFAULT, _GA_DEVICE_DEFAULT)

    # ── 1. IPC 复用模块级预创建（共享内存和 socket 已在导入阶段就绪）─
    global _ipc
    _ipc = _ipc_early
    # 用真正的 handler 重启 cmd server（模块级用的是占位 handler）
    _ipc.start_cmd_server(_handle_cmd)
    cmd_thread = threading.Thread(target=_ipc.accept_cmd_loop, daemon=True, name="CmdServer")
    cmd_thread.start()

    # ── 2. Gallery (先加载，识别器依赖) ─────────────────────────────────
    _gallery = Gallery(GALLERY_FILE, sim_threshold=SIM_THRESHOLD)
    log.info("Gallery loaded: %s (%d identities)", GALLERY_FILE, len(_gallery.get_all()))

    # ── 3. 并行加载核心模型；可选模型后台异步 ───────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _height_est = None
    _bmi_predictor = None
    _detector = None
    _cloth = None
    _ga = None
    _body_detector = None

    def _load_detector():
        d = FaceDetector()
        d.ensure()
        return "detector", d

    def _load_cloth():
        return "cloth", ClothDetector()

    def _load_gender_age():
        ga_device = os.getenv("GENDER_AGE_DEVICE") or None
        ga = GenderAgeRecognizer(device=ga_device, enabled=True)
        ga.preload_async()
        return "gender_age", ga

    def _load_bmi():
        if BMIPredictor is not None:
            bmi = BMIPredictor()
            bmi.ensure_loaded()
            return "bmi", bmi
        return "bmi", None

    def _load_body():
        bd = BodyDetector()
        bd.ensure()
        return "body", bd

    t0 = time.time()
    core_tasks = [_load_cloth, _load_body, _load_detector, _load_gender_age]
    with ThreadPoolExecutor(max_workers=len(core_tasks)) as pool:
        futures = {pool.submit(t): t for t in core_tasks}
        for fut in as_completed(futures):
            try:
                name, obj = fut.result()
                if name == "detector":
                    _detector = obj
                elif name == "cloth":
                    _cloth = obj
                elif name == "gender_age":
                    _ga = obj
                elif name == "body":
                    _body_detector = obj
                log.info("  %s loaded in %.1fs", name, time.time() - t0)
            except Exception as e:
                log.error("  %s failed: %s", futures[fut].__name__, e)
    log.info("Core models loaded in %.1fs", time.time() - t0)

    # ── 4. VehicleRecognizer ───────────────────────────────────────────
    _recognizer = VehicleRecognizer(
        gallery=_gallery,
        camera_id=CAMERA_ID,
        positionless=POSITIONLESS,
        sim_threshold=SIM_THRESHOLD,
        smooth_window=SMOOTH_WINDOW,
        update_centroid=True,
        auto_enroll=AUTO_ENROLL,
        detector=_detector,
        gender_age_recognizer=_ga,
        cloth_detector=_cloth,
        body_detector=_body_detector,
    )

    # ── 5. 后台加载可选模型（height / BMI，就绪后注入）─────────────────
    def _load_optional():
        global _can_client, _bmi_predictor
        # BMI
        if BMIPredictor is not None:
            try:
                bmi = BMIPredictor()
                bmi.ensure_loaded()
                _recognizer.bmi_predictor = bmi
                _bmi_predictor = bmi
                log.info("  bmi loaded (background), ready=%s", bmi.ready)
            except Exception as e:
                log.error("  bmi failed: %s", e)
        # Height + CAN
        if HeightEstimator is not None:
            try:
                _height_est = HeightEstimator(models_dir=HEIGHT_MODELS_DIR)
                dx = float(os.getenv("HEIGHT_SEAT_DX", "0.62"))
                dy = float(os.getenv("HEIGHT_SEAT_DY", "0.26"))
                px = float(os.getenv("HEIGHT_SEAT_PX", "0.41"))
                py = float(os.getenv("HEIGHT_SEAT_PY", "0.59"))
                _height_est.set_seat_params(dx, dy, px, py)
                _recognizer.height_estimator = _height_est
                log.info("  height loaded (background)")
                _can = CANSeatClient(_height_est)
                _can.start()
                _can_client = _can
            except Exception as e:
                log.error("  height failed: %s", e)

    threading.Thread(target=_load_optional, daemon=True, name="OptionalModels").start()

    # ── 5. 启动识别 ────────────────────────────────────────────────────
    _recognizer.start(blocking=False)
    if _recognizer._running:
        log.info("识别线程已启动 camera=%s", CAMERA_ID)
    else:
        log.warning("摄像头不可用，运行在仅图片识别模式")

    log.info("=== GPU 推理进程就绪 ===")

    # ── 维持一份最后有效结果，任何字段空了就兜底 ──
    _last_good: dict = {}

    # ── 衣物动态稳定机制 ──
    # 持续检测
    # 新类别连续稳定后自动切换
    _cloth_history = {
    "driver": _deque(maxlen=_CLOTH_HISTORY_SIZE),
    "passenger": _deque(maxlen=_CLOTH_HISTORY_SIZE)
}
    # 当前稳定输出类别
    _cloth_current = {
    "driver": None,
    "passenger": None
}
    # 最终输出
    _cloth_emit = {
    "driver": None,
    "passenger": None
}


    # ── BMI 平滑（滑动窗口中值滤波 + EMA + 稳定锁）──
    _bmi_ema = {"driver": None, "passenger": None}
    _bmi_window = {"driver": _deque(maxlen=10), "passenger": _deque(maxlen=10)}
    _bmi_stable_count = {"driver": 0, "passenger": 0}
    _bmi_locked_until = {"driver": 0.0, "passenger": 0.0}
    _bmi_last_ema = {"driver": None, "passenger": None}  # 用于检测变化幅度
    _BMI_ALPHA = 0.03        # EMA 系数（越小越平滑）
    _BMI_MAX_STEP = 0.3      # 单步最大变化（人体 BMI 不会瞬间突变）
    _BMI_STABLE_DELTA = 0.2  # 连续 N 次变化 < 此值视为稳定
    _BMI_STABLE_N = 5        # 连续稳定次数阈值
    _BMI_LOCK_SEC = 60       # 稳定后锁定秒数（期间不更新）

    # ── 6. 主循环：发布结果到共享内存 ──────────────────────────────────
    while _running:
        time.sleep(0.05)  # ~20Hz 发布频率

        if _recognizer:
            cur = _recognizer.get_current()
            height_data = _recognizer.get_height_status() if _recognizer else {}

            raw_cloth = cur.get("cloth", {}) or getattr(_recognizer, "_last_cloth_cache", {})

            # ── 衣物动态稳定投票 ──
            smoothed_cloth = {}
            
            for role in ("driver", "passenger"):
                val = raw_cloth.get(role)
                
                # 当前检测结果加入历史
                if val is not None and val != "":
                    label = val.get("label") if isinstance(val, dict) else val
                    if label:
                        _cloth_history[role].append(label)
                
                history = _cloth_history[role]
                
                if len(history) > 0:
                    from collections import Counter
                    
                    counts = Counter(history)
                    best_val, best_count = counts.most_common(1)[0]
                    ratio = best_count / len(history)
                    
                    # ==========================
                    # 第一次建立衣物类别
                    # ==========================
                    if _cloth_current[role] is None:
                        if ratio >= _CLOTH_INIT_RATIO:
                            _cloth_current[role] = best_val
                            log.info(
                                "衣物初始化 %s=%s ratio=%.2f",
                                role,
                                best_val,
                                ratio
                            )
                    
                    # ==========================
                    # 已经有类别
                    # ==========================
                    else:
                        # 新类别出现
                        if best_val != _cloth_current[role]:
                            if ratio >= _CLOTH_CHANGE_RATIO:
                                old = _cloth_current[role]
                                _cloth_current[role] = best_val
                                log.info(
                                    "衣物变化 %s: %s -> %s ratio=%.2f",
                                    role,
                                    old,
                                    best_val,
                                    ratio
                                )
                    
                    # 输出当前稳定类别
                    if _cloth_current[role] is not None:
                        _cloth_emit[role] = _cloth_current[role]
                        smoothed_cloth[role] = _cloth_current[role]
                
                else:
                    # 没有历史，输出原始
                    if val is not None:
                        smoothed_cloth[role] = val

            raw_bmi = getattr(_recognizer, "bmi_cache", {})
            # ── BMI 平滑（中值滤波 + EMA + 稳定锁）──
            smoothed_bmi = {}
            for role in ("driver", "passenger"):
                bmi_obj = raw_bmi.get(role) if raw_bmi else {}
                if bmi_obj and bmi_obj.get("face_detected"):
                    raw_val = float(bmi_obj.get("bmi", 0) or 0)
                    if raw_val > 0:
                        # 滑动窗口中值滤波：消除单帧极端异常值
                        _bmi_window[role].append(raw_val)
                        median_val = float(np.median(list(_bmi_window[role])))
                        if _bmi_ema[role] is None:
                            _bmi_ema[role] = median_val
                            _bmi_last_ema[role] = median_val
                        else:
                            now_ts = time.time()
                            # 稳定锁期间跳过更新
                            if now_ts >= _bmi_locked_until[role]:
                                delta = median_val - _bmi_ema[role]
                                if abs(delta) > _BMI_MAX_STEP:
                                    delta = _BMI_MAX_STEP * (1 if delta > 0 else -1)
                                _bmi_ema[role] = _bmi_ema[role] + delta * _BMI_ALPHA
                                # 检测是否趋于稳定
                                if _bmi_last_ema[role] is not None:
                                    change = abs(_bmi_ema[role] - _bmi_last_ema[role])
                                    if change < _BMI_STABLE_DELTA:
                                        _bmi_stable_count[role] += 1
                                    else:
                                        _bmi_stable_count[role] = 0
                                _bmi_last_ema[role] = _bmi_ema[role]
                                # 连续稳定 N 次 → 锁定 60 秒
                                if _bmi_stable_count[role] >= _BMI_STABLE_N:
                                    _bmi_locked_until[role] = now_ts + _BMI_LOCK_SEC
                                    _bmi_stable_count[role] = 0
                                    log.info("[BMI] %s locked for %ds (EMA=%.1f)",
                                             role, _BMI_LOCK_SEC, _bmi_ema[role])
                    # 仅当人脸当前存在时才输出（无人脸不残留旧值）
                    if _bmi_ema[role] is not None:
                        smoothed_bmi[role] = {
                            "face_detected": True,
                            "bmi": round(_bmi_ema[role], 1),
                            "class_id": -1,
                            "class_name": "smoothed",
                            "confidence": 0.0,
                        }
                else:
                    # 无人脸时清理：清空窗口、重置 EMA 和计数器
                    _bmi_window[role].clear()
                    _bmi_ema[role] = None
                    _bmi_last_ema[role] = None
                    _bmi_stable_count[role] = 0
                    _bmi_locked_until[role] = 0.0

            # 构建完整的 API 结果（包含所有 /stats /status-widget 需要的字段）
            api_result = {
                "faces": cur.get("faces", []),
                "cloth": smoothed_cloth if smoothed_cloth else raw_cloth,
                "height": height_data,
                "bmi_cache": smoothed_bmi if smoothed_bmi else raw_bmi,
                "seat_positions": getattr(_recognizer, "_seat_positions", {"left": False, "right": False}),
                "seat_states": cur.get("seat_states", {"left": "empty", "right": "empty"}),
                "face_lost": cur.get("face_lost", False),
                "gallery_count": len(_gallery.get_all()) if _gallery else 0,
                "frame_count": cur.get("frame_count", 0),
                "total_detected": cur.get("total_detected", 0),
                "total_known": cur.get("total_known", 0),
                "running": bool(_recognizer._running),
                "paused": bool(_recognizer._paused),
                "enroll_enabled": bool(_recognizer.auto_enroll),
                "enroll_min_frames": int(getattr(_recognizer, "auto_enroll_min_frames", 10)),
                "enroll_queue_depth": {
                    k: len(v) for k, v in (
                        getattr(_recognizer, "_enroll_queues", {}) or {}
                    ).items()
                },
                "recording": getattr(_recognizer, "is_recording", False),
                "recording_frames": getattr(_recognizer, "recording_frames", 0),
            }
            # ── 缓存有效结果，字段空了就用旧值兜底 ──
            if not api_result["face_lost"]:
                # 正常帧：更新缓存
                _last_good = {
                    "cloth": dict(api_result["cloth"]),
                    "height": dict(api_result["height"]) if api_result["height"] else {},
                    "bmi_cache": dict(api_result["bmi_cache"]),
                    "faces": [dict(f) for f in api_result["faces"]],
                    "total_detected": api_result["total_detected"],
                    "total_known": api_result["total_known"],
                }
            elif _last_good:
                # face_lost：用缓存兜底空字段
                for key in ("cloth", "height", "bmi_cache"):
                    if not api_result.get(key):
                        api_result[key] = _last_good.get(key, api_result[key])
                if not api_result.get("faces"):
                    api_result["faces"] = _last_good.get("faces", [])
                    api_result["total_detected"] = _last_good.get("total_detected", 0)
                    api_result["total_known"] = _last_good.get("total_known", 0)
            _ipc.write_result(api_result)

            annotated = getattr(_recognizer, "_last_annotated_frame", None)
            if annotated is not None:
                _ipc.write_frame(annotated.copy())

            raw_frame = getattr(_recognizer, "_last_raw_frame", None)
            if raw_frame is not None:
                _ipc.write_raw(raw_frame.copy())

    # ── 7. 清理 ────────────────────────────────────────────────────────
    log.info("Shutting down...")
    if _can_client:
        _can_client.stop()
    if _recognizer:
        _recognizer.stop()
    if _gallery:
        try:
            _gallery.save()
        except Exception:
            log.exception("Gallery save failed")
    _ipc.cleanup()
    log.info("GPU inference worker stopped")


def _signal_handler(signum, frame):
    global _running
    log.info("Received signal %s, shutting down...", signum)
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

if __name__ == "__main__":
    main()