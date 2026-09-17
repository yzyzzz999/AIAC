"""
service/recognizer.py
====================

支持以下运行模式：
  - 实时视频流模式（摄像头）
  - 视频文件模式（处理本地视频）
  - 图片批量模式（处理图片目录）
"""

from __future__ import annotations

import os
import uuid
import cv2
import time
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock, Event
from collections import Counter, deque
from typing import Optional, List, Dict, Any, Literal, Callable

from core.gallery import Gallery, DEFAULT_SIM_THRESHOLD, normalize_emb
from service.detector import FaceDetector, DetectedFace
from service.gender_age_recognizer import GenderAgeRecognizer
from service.cloth_detector import ClothDetector
import queue as _q
from utils import select_front_row_faces
DEFAULT_SMOOTH_WINDOW = 50       # 滑动窗口大小（用于平滑结果）
DEFAULT_FRAME_INTERVAL = 3      # 每隔几帧处理一次（降低GPU压力）
# FRONT_ROW_ROIS 及其参数统一从 utils.py import
log = logging.getLogger(__name__)


class VehicleRecognizer:
    """
    # 1. 初始化
    recognizer = VehicleRecognizer(
        gallery_file="gallery.pkl",
        camera_id=0,          # 或视频文件路径
        positionless=True,    # 位置无关模式
        update_centroid=True, # 增量更新 Gallery
    )

    # 2a. 实时模式：后台线程运行
    recognizer.start()
    while True:
        state = recognizer.get_current()
        print(state)

    # 2b. 单帧模式：处理单张图片
    result = recognizer.recognize_image("car.jpg")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 2c. 批量模式：处理视频文件
    recognizer.process_video("dashcam.mp4", output="result.mp4")

    # 3. 停止
    recognizer.stop()
    """

    def __init__(
        self,
        gallery_file: str | Path = "gallery.pkl",
        camera_id: int = 0,
        gallery: Optional[Gallery] = None,   # 共享 Gallery 实例
        positionless: bool = True,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        frame_interval: int = DEFAULT_FRAME_INTERVAL,
        update_centroid: bool = True,
        auto_enroll: bool = False,# 是否启用自动注册
        auto_enroll_min_frames: int = 10,# 自动注册最小帧数
        auto_enroll_sim: float = 0.35,# 自动注册相似度阈值
        enable_gender_age: bool | None = None,# 是否启用年龄/性别识别
        gender_age_device: Optional[str] = None,# 属性模型设备，None=自动
        on_recognized: Optional[Callable] = None,# 识别回调
        on_enrolled: Optional[Callable] = None,# 注册回调
        on_unknown: Optional[Callable] = None,# 未知回调
        seat_check_callback: Optional[Callable] = None,# 占位检测回调 (positions: {"left": bool, "right": bool}) -> None
        height_estimator = None,  # HeightEstimator 实例（可选）
        detector: Optional[FaceDetector] = None,        # 预加载的检测器
        gender_age_recognizer: Optional[GenderAgeRecognizer] = None,  # 预加载的性别年龄
        cloth_detector = None,   # 预加载的衣物检测器
        body_detector=None,   # BodyDetector 实例（可选）
    ):
        self.gallery = gallery if gallery is not None else Gallery(gallery_file=str(gallery_file), sim_threshold=sim_threshold)
        self.detector = detector if detector is not None else FaceDetector()
        self.camera_id = camera_id
        self.positionless = positionless
        self.sim_threshold = sim_threshold
        self.smooth_window = smooth_window
        self.frame_interval = frame_interval
        self.update_centroid = update_centroid
        self.auto_enroll = auto_enroll
        self.auto_enroll_min_frames = auto_enroll_min_frames
        self.auto_enroll_sim = auto_enroll_sim
        self.auto_enroll_max_yaw = float(os.getenv("AUTO_ENROLL_MAX_YAW", "70.0"))
        self.auto_enroll_max_pitch = float(os.getenv("AUTO_ENROLL_MAX_PITCH", "45.0"))
        if enable_gender_age is None:
            enable_gender_age = os.getenv("ENABLE_GENDER_AGE", "1").lower() not in ("0", "false", "no", "off")
        if gender_age_device is None:
            gender_age_device = os.getenv("GENDER_AGE_DEVICE") or None
        if gender_age_recognizer is not None:
            self.gender_age = gender_age_recognizer
        else:
            self.gender_age = GenderAgeRecognizer(device=gender_age_device, enabled=enable_gender_age)
        self.gender_age.preload_async()
        self.gender_age_interval = max(1, int(os.getenv("GENDER_AGE_INTERVAL", "30")))
        self._gender_age_cache: Dict[str, Dict[str, Any]] = {}
        self.attr_cache_by_seat = os.getenv(
            "ATTR_CACHE_BY_SEAT",
            "1",
        ).lower() not in ("0", "false", "no", "off")
        # 年龄稳定性参数
        self.age_stable_window = max(1, int(os.getenv("AGE_STABLE_WINDOW", "12")))
        self.age_stable_min_count = max(1, int(os.getenv("AGE_STABLE_MIN_COUNT", "8")))
        self.age_stable_min_ratio = max(0.0, min(1.0, float(os.getenv("AGE_STABLE_MIN_RATIO", "0.60"))))
        self.age_stable_min_score = max(0.0, float(os.getenv("AGE_STABLE_MIN_SCORE", "0.35")))
        # 性别稳定性参数（窗口更大、阈值更高）
        self.gender_stable_window = max(1, int(os.getenv("GENDER_STABLE_WINDOW", "20")))
        self.gender_stable_min_count = max(1, int(os.getenv("GENDER_STABLE_MIN_COUNT", "18")))
        self.gender_stable_min_ratio = max(0.0, min(1.0, float(os.getenv("GENDER_STABLE_MIN_RATIO", "0.85"))))
        # 稳定值刷新间隔：每隔 N 帧重新跑一次 FLIP 验证（默认 300 帧 ≈ 10 分钟）
        self.stable_refresh_interval = max(1, int(os.getenv("STABLE_REFRESH_INTERVAL", "300")))
        self.use_legacy_stable_attrs = os.getenv(
            "USE_LEGACY_STABLE_ATTRS",
            "0",
        ).lower() in ("1", "true", "yes", "on")
        self._age_stability_history: Dict[str, deque] = {}
        self._gender_stability_history: Dict[str, deque] = {}
        self._stable_attrs_cache: Dict[str, Dict[str, Any]] = {}
        self._stability_lock = Lock()
        self.seat_check_callback = seat_check_callback
        self.height_estimator = height_estimator  # 身高估计模块
        self.bmi_predictor = None  # BMI预测模块（由外部注入）
        self.bmi_cache = {"passenger": {"face_detected": False, "class_id": -1, "class_name": "no_face", "bmi": 0.0, "confidence": 0.0}, "driver": {"face_detected": False, "class_id": -1, "class_name": "no_face", "bmi": 0.0, "confidence": 0.0}}
        self._seat_positions = {"left": False, "right": False}

        self.body_detector = body_detector  # 人体检测器（YOLO，降级为 debug 可视化）
        self._body_det_enabled = os.getenv("BODY_DET_ENABLED", "true").lower() not in ("0", "false", "no", "off")
        self._body_det_debug = os.getenv("BODY_DET_DEBUG", "false").lower() in ("1", "true", "yes", "on")
        # ── 运动检测（帧差法）替掉 YOLO ──
        self._motion_enabled = os.getenv("MOTION_ENABLED", "true").lower() not in ("0", "false", "no", "off")
        self._motion_threshold = float(os.getenv("MOTION_THRESHOLD", "3.5"))
        self._motion_ema_alpha = float(os.getenv("MOTION_EMA_ALPHA", "0.3"))
        self._motion_ema: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self._motion_ema_lower: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self._prev_gray: Optional[np.ndarray] = None
        self._body_lost_frames_threshold = int(os.getenv("BODY_DET_BODY_LOST_FRAMES", "30"))
        # 人体检测信任超时：该侧多久没看到人脸就不再信人体检测（秒），防座椅误检
        self._body_face_timeout = float(os.getenv("BODY_DET_FACE_TIMEOUT", "20"))
        # 座位状态机: per-seat {state, held_result, body_lost_frames}
        self._seat_states: Dict[str, dict] = {
            "left":  {"state": "empty", "held": None, "body_lost_frames": 0},
            "right": {"state": "empty", "held": None, "body_lost_frames": 0},
        }
        self._last_face_seen: Dict[str, float] = {}  # side → timestamp
        self._last_body_boxes = []
        self._last_cloth_cache = {}  # 缓存最新衣服结果，bypass 时不丢

        self.cloth_detector = cloth_detector if cloth_detector is not None else ClothDetector()

        # 回调函数
        self.on_recognized = on_recognized
        self.on_enrolled = on_enrolled
        self.on_unknown = on_unknown

        # 当前状态（线程安全）
        self._current: Dict[str, Any] = {}
        self._current_lock = Lock()

        # 实时 embedding 缓冲（用于平滑）
        self._emb_history: Dict[str, deque] = {}
        self._emb_lock = Lock()

        # 自动注册缓冲
        self._enroll_queues: Dict[str, deque] = {}  # identity_id -> deque of embs
        self._position_identity: Dict[str, str] = {}  # position -> last known identity_id

        # 运行状态
        self._running = False
        self._paused = False  # 暂停标志（保留摄像头连接）
        self._thread: Optional[Thread] = None
        self._cap = None  # 持有 VideoCapture 以便暂停/继续

        # 共享标注帧队列（供 /preview 接口使用）
        self._annotated_queue: _q.Queue = _q.Queue(maxsize=2)
        self._last_annotated_frame: Optional[np.ndarray] = None
        self._last_annotated_lock = Lock()
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_raw_lock = Lock()

        # ── 录制 ──
        self._recorder: Optional[cv2.VideoWriter] = None
        self._recording_frames: int = 0
        self._recording_max: int = 7500
        self._last_frame_shape: tuple = (0, 0, 0)

        # ── 流水线并行 ─────────────────────────────────────────────────
        self._process_lock = Lock()  # 防止 recognize_image 与流水线并发操作共享状态
        self._pipeline_mode = False  # _loop 使用流水线，recognize_image 沿用串行
        self._frame_queue: _q.Queue = _q.Queue(maxsize=1)
        self._det_queue: _q.Queue = _q.Queue(maxsize=1)
        self._result_queue: _q.Queue = _q.Queue(maxsize=1)
        self._det_thread: Optional[Thread] = None
        self._rec_thread: Optional[Thread] = None


    def start(self, blocking: bool = False):
        """启动识别线程（仅初始化一次，暂停后可继续）。"""
        if self._thread is not None and self._thread.is_alive():
            return  # 线程已存在，不重复创建
        self.detector.ensure()
        self._running = True
        if blocking:
            self._loop()
        else:
            self._thread = Thread(target=self._loop, daemon=True, name="VehicleRecognizer")
            self._thread.start()
            log.info("识别线程已启动 camera=%s", self.camera_id)

    def pause(self):
        """暂停识别（保留摄像头连接）。清空流水线队列，立即停止推理。"""
        self._paused = True
        for q in (self._frame_queue, self._det_queue, self._result_queue):
            while True:
                try:
                    q.get_nowait()
                except _q.Empty:
                    break
        log.info("识别已暂停")

    def resume(self):
        """继续识别。"""
        self._paused = False
        log.info("识别已恢复")

    def stop(self):
        """完全停止识别线程（释放摄像头和流水线）。"""
        self._running = False
        self._paused = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        for t in (self._det_thread, self._rec_thread, self._dedup_thread):
            if t and t.is_alive():
                t.join(timeout=2)
        self._det_thread = None
        self._rec_thread = None
        self._dedup_thread = None
        self._dedup_running = False
        try:
            self.gallery.save()
        except Exception:
            log.exception("停止时保存 Gallery 失败")
        log.info("识别线程已停止")

    # ── 流水线工人方法 ─────────────────────────────────────────────────

    def _detection_worker(self):
        """Stage A：人脸检测 + 前排选择 + 衣着检测。
        无脸时如果 held 数据可用，直接产出结果跳过 recognition。"""
        while self._running:
            try:
                frame, frame_count = self._frame_queue.get(timeout=0.5)
            except _q.Empty:
                continue
            if self._paused:
                continue
            try:
                raw = self.detector.detect_frame(frame)
                H, W = frame.shape[:2]

                # 从原始检测结果判断每侧是否有脸（状态机用，不依赖后续处理）
                raw_face_present = {"left": False, "right": False}
                for face in raw:
                    bbox = getattr(face, "bbox", None)
                    if bbox is None:
                        continue
                    xc = (bbox[0] + bbox[2]) / 2
                    raw_face_present["right" if xc >= W / 2 else "left"] = True

                detected = self._select_front_row_faces(raw, W, H)

                # 无脸 + 有 held → 直接产出 held_result，跳过 recognition + cloth
                if not detected:
                    # 每秒输出一次 bypass 统计
                    if not hasattr(self, '_bypass_count'):
                        self._bypass_count = 0
                        self._bypass_last_log = 0
                    self._bypass_count += 1
                    if frame_count - self._bypass_last_log >= 25:
                        log.info("[Pipeline] bypass 跳过 recognition: %d次 (raw=%d faces, detected=0)",
                                 self._bypass_count, len(raw))
                        self._bypass_count = 0
                        self._bypass_last_log = frame_count
                    from datetime import datetime as _dt
                    held_faces = []
                    for side in ("left", "right"):
                        st = self._seat_states.get(side, {})
                        if st.get("held"):
                            h = dict(st["held"])
                            h["_held"] = True
                            held_faces.append(h)
                    if held_faces:
                        self._result_queue.put((frame, {
                            "faces": held_faces,
                            "frame_count": frame_count,
                            "timestamp": _dt.now().isoformat(),
                            "process_time_ms": 0,
                            "total_detected": len(held_faces),
                            "total_raw_detected": len(raw),
                            "total_known": sum(1 for f in held_faces if f.get("is_known")),
                            "height": None,
                            "raw_detected": raw,
                            "_raw_face_present": raw_face_present,
                            "face_lost": True,
                            "seat_states": {k: v["state"] for k, v in self._seat_states.items()},
                            "cloth": dict(self._last_cloth_cache),
                        }))
                        continue

                cloth = self.cloth_detector.detect(frame) or {}
                self._det_queue.put((frame, frame_count, detected, cloth, raw))
            except Exception:
                log.exception("[Pipeline] Detection worker failed, skipping frame")

    def _recognition_worker(self):
        """Stage B：识别匹配 + 性别年龄 + 身高 + BMI + 自动注册。"""
        while self._running:
            try:
                frame, frame_count, detected, cloth, raw = self._det_queue.get(timeout=0.5)
            except _q.Empty:
                continue
            if self._paused:
                continue
            try:
                with self._process_lock:
                    result = self._process_frame(frame, frame_count,
                                                 pre_detected=detected, pre_raw=raw)
                result["cloth"] = cloth
                # BMI 预测（复用已检测到的人脸），间隔 ≥ 3s 避免过频
                if self.bmi_predictor and self.bmi_predictor.ready:
                    now_bmi = time.time()
                    if now_bmi - getattr(self, "_last_bmi_time", 0) >= 3.0:
                        self._last_bmi_time = now_bmi
                        try:
                            pf, df = None, None
                            for f in result.get("faces", []):
                                bbox = f.get("bbox")
                                if not bbox or not all(v > 0 for v in bbox):
                                    continue
                                x1, y1, x2, y2 = map(int, bbox)
                                crop = frame[y1:y2, x1:x2]
                                if crop.size == 0:
                                    continue
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                from PIL import Image
                                face_pil = Image.fromarray(crop_rgb)
                                if f.get("position") == "left":
                                    pf = face_pil
                                elif f.get("position") == "right":
                                    df = face_pil
                            bmi_result = self.bmi_predictor.predict(frame, pf, df)
                            self.bmi_cache = bmi_result
                            log.info("[BMI] predict done: p=%s d=%s",
                                     bmi_result.get("passenger", {}).get("face_detected"),
                                     bmi_result.get("driver", {}).get("face_detected"))
                        except Exception:
                            log.exception("[BMI] predict failed")
                self._result_queue.put((frame, result))
            except Exception:
                log.exception("[Pipeline] Recognition worker failed, skipping frame")

    # ── 主循环 ─────────────────────────────────────────────────────────

    def _loop(self):
        """主循环：持续从摄像头读取帧并处理（流水线模式）。"""
        if self._cap is not None and self._cap.isOpened():
            pass  # 复用已有摄像头连接（resume 场景）
        elif isinstance(self.camera_id, (str, Path)):
            self._cap = cv2.VideoCapture(str(self.camera_id))
        else:
            self._cap = cv2.VideoCapture(self.camera_id)

        if not self._cap.isOpened():
            log.error("无法打开摄像头/视频: %s", self.camera_id)
            return

        # ── 启动流水线工作线程 ────────────────────────────────────────────
        self._det_thread = Thread(target=self._detection_worker, daemon=True, name="StageA-Det")
        self._rec_thread = Thread(target=self._recognition_worker, daemon=True, name="StageB-Rec")
        self._dedup_thread = Thread(target=self._dedup_loop, daemon=True, name="Dedup")
        self._dedup_running = True
        self._det_thread.start()
        self._rec_thread.start()
        self._dedup_thread.start()

        frame_count = 0
        last_processed_frame = -100  # 首帧立即处理
        SEAT_CHECK_INTERVAL = 5  # 占位检测固定间隔（帧）
        consecutive_fails = 0
        MAX_FAILS_BEFORE_RECONNECT = 30  # 3 秒 (30 × 100ms)
        last_result = {"faces": [], "frame_count": 0, "total_detected": 0, "total_known": 0}
        last_cloth = {}  # 保存上一次的服装检测结果
        last_frame_for_result = None  # 保存 last_result 对应的帧
        prev_frame_gray = None
        last_processed_time = 0.0
        MOTION_THRESHOLD = 5.0  # 帧差均值阈值
        FORCE_REFRESH_SEC = 1.0  # 静止超过此时间强制刷新

        while self._running:
            ok, frame = self._cap.read()
            self._last_frame_shape = frame.shape
            if not ok:
                consecutive_fails += 1
                if consecutive_fails >= MAX_FAILS_BEFORE_RECONNECT:
                    log.warning("摄像头读取连续失败 %s 次，尝试重连...", consecutive_fails)
                    self._cap.release()
                    time.sleep(1.0)
                    if isinstance(self.camera_id, (str, Path)):
                        self._cap = cv2.VideoCapture(str(self.camera_id))
                    else:
                        self._cap = cv2.VideoCapture(self.camera_id)
                    if self._cap.isOpened():
                        log.info("摄像头重连成功")
                        consecutive_fails = 0
                    else:
                        log.error("摄像头重连失败，50 秒后重试")
                        consecutive_fails = MAX_FAILS_BEFORE_RECONNECT + 500  # 延迟下次重试
                time.sleep(0.1)
                continue
            consecutive_fails = 0

            frame_count += 1

            # ── 录制原始帧（最原始，不 resize）──
            if self._recorder is not None:
                try:
                    self._recorder.write(frame)
                    self._recording_frames += 1
                    if self._recording_frames >= self._recording_max:
                        self.stop_recording()
                except Exception:
                    log.exception("录制写入失败")

            # ── 存储原始帧供 /raw 流（不参与管线，仅一次 resize）──
            try:
                raw = cv2.resize(frame, (800, 600))
                with self._last_raw_lock:
                    self._last_raw_frame = raw
            except Exception:
                pass

            # ── 运动检测：画面无变化时跳过推理 ──────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 身体检测 + 状态机（每帧必跑，不参与帧跳过）──
            body_present = {"left": False, "right": False}
            self._last_body_boxes = []

            if self._motion_enabled and prev_frame_gray is not None:
                try:
                    from utils import FRONT_ROW_BODY_ROIS
                    diff = cv2.absdiff(gray, prev_frame_gray)
                    H_m, W_m = gray.shape
                    for slot, roi in FRONT_ROW_BODY_ROIS.items():
                        rx1 = int(roi[0] * W_m)
                        ry1 = int(roi[1] * H_m)
                        rx2 = int(roi[2] * W_m)
                        ry2 = int(roi[3] * H_m)
                        if rx2 <= rx1 or ry2 <= ry1:
                            continue

                        # 整区运动能量
                        energy = float(np.mean(diff[ry1:ry2, rx1:rx2]))
                        prev_ema = self._motion_ema.get(slot, energy)
                        self._motion_ema[slot] = (
                            self._motion_ema_alpha * energy
                            + (1.0 - self._motion_ema_alpha) * prev_ema
                        )

                        # 下半区运动能量（防后排大动作穿透）
                        mid_y = (ry1 + ry2) // 2
                        energy_lower = float(np.mean(diff[mid_y:ry2, rx1:rx2]))
                        prev_ema_lower = self._motion_ema_lower.get(slot, energy_lower)
                        self._motion_ema_lower[slot] = (
                            self._motion_ema_alpha * energy_lower
                            + (1.0 - self._motion_ema_alpha) * prev_ema_lower
                        )

                        # 同时满足：整区超阈值 + 下半区也超阈值×0.5（前排特征）
                        if (self._motion_ema[slot] >= self._motion_threshold
                                and self._motion_ema_lower[slot] >= self._motion_threshold * 0.5):
                            body_present[slot] = True

                    if self._body_det_debug and frame_count % 25 == 0:
                        log.info("[Motion] frame=%d body=%s "
                                 "L(ema=%.2f lo=%.2f) R(ema=%.2f lo=%.2f)",
                                 frame_count, body_present,
                                 self._motion_ema.get("left", 0),
                                 self._motion_ema_lower.get("left", 0),
                                 self._motion_ema.get("right", 0),
                                 self._motion_ema_lower.get("right", 0))
                except Exception:
                    log.warning("Motion detection failed", exc_info=True)

            # ── YOLO 兜底：运动低但真人静止时接管 ──
            _yolo_bodies = []
            if self.body_detector is not None and self._body_det_enabled:
                _yolo_interval = 10
                if not hasattr(self, '_yolo_cache_counter'):
                    self._yolo_cache_counter = _yolo_interval  # 首帧立即跑
                    self._yolo_cache_bodies = []
                self._yolo_cache_counter += 1
                if self._yolo_cache_counter >= _yolo_interval:
                    self._yolo_cache_counter = 0
                    try:
                        self._yolo_cache_bodies = self.body_detector.detect(frame)
                    except Exception:
                        self._yolo_cache_bodies = []
                _yolo_bodies = self._yolo_cache_bodies

                if self._body_det_debug:
                    self._last_body_boxes = _yolo_bodies

                # YOLO 覆盖：纠正运动检测的跨侧污染（同一人不能同时在两侧）
                from utils import select_front_row_bodies
                H_y, W_y = frame.shape[:2]
                yolo_result = select_front_row_bodies(_yolo_bodies, W_y, H_y)
                yolo_sides = [s for s in ("left", "right") if yolo_result.get(s)]
                for side in ("left", "right"):
                    last_seen = self._last_face_seen.get(side, 0)
                    body_trusted = last_seen and (time.time() - last_seen) < self._body_face_timeout
                    if yolo_result.get(side) and body_trusted:
                        body_present[side] = True
                        if self._body_det_debug and frame_count % 25 == 0:
                            log.info("[YoloFallback] %s 接管: motion低但yolo检测到", side)
                    elif not yolo_result.get(side) and body_trusted and body_present[side]:
                        # YOLO 说没人在这一侧但运动检测说有 → 覆盖运动检测
                        body_present[side] = False
                        if self._body_det_debug and frame_count % 25 == 0:
                            log.info("[YoloOverride] %s 清除: motion检测到但yolo未检测到", side)

            # 直接使用 detection worker 计算的原始检测结果做状态机判断
            face_present = dict(last_result.get("_raw_face_present",
                                {"left": False, "right": False}))

            # 诊断日志：每秒打印状态机输入
            if self._body_det_debug and frame_count % 25 == 0:
                log.info("[SeatInput] face=%s body=%s states=%s",
                         face_present, body_present,
                         {s: self._seat_states[s]["state"] for s in ("left", "right")})

            # 记录状态转换前的旧状态，用于检测 face_lost 首次触发
            old_states = {s: self._seat_states[s]["state"] for s in ("left", "right")}
            self._update_seat_states(face_present, body_present)
            self._seat_positions = {
                "left": self._seat_states["left"]["state"] != "empty",
                "right": self._seat_states["right"]["state"] != "empty",
            }

            # ── face_lost 首次触发：立即发布 held 结果，清空 pipeline 队列 ──
            just_entered_face_lost = False
            for side in ("left", "right"):
                if (old_states[side] != "face_lost"
                        and self._seat_states[side]["state"] == "face_lost"):
                    just_entered_face_lost = True
                    break
            if just_entered_face_lost:
                held_faces = []
                for side in ("left", "right"):
                    st = self._seat_states[side]
                    if st.get("held"):
                        h = dict(st["held"])
                        h["_held"] = True
                        held_faces.append(h)
                last_result = {
                    "faces": held_faces,
                    "frame_count": frame_count,
                    "total_detected": len(held_faces),
                    "total_known": sum(1 for f in held_faces if f.get("is_known")),
                    "_raw_face_present": {"left": False, "right": False},
                    "face_lost": True,
                    "seat_states": {k: v["state"] for k, v in self._seat_states.items()},
                    "cloth": dict(self._last_cloth_cache),
                }
                # 清空全部 pipeline 队列（frame/det/result），不让旧帧干扰
                for q in (self._frame_queue, self._det_queue, self._result_queue):
                    while True:
                        try:
                            q.get_nowait()
                        except _q.Empty:
                            break
                # 立即更新 _current + last_cloth，不等 pipeline
                last_cloth = dict(self._last_cloth_cache)
                with self._current_lock:
                    self._current = dict(last_result)

            # ── 进入 empty：立即清除该侧的人脸框和 API 数据 ──
            for side in ("left", "right"):
                if (old_states[side] != "empty"
                        and self._seat_states[side]["state"] == "empty"):
                    pos = "left" if side == "left" else "right"
                    last_result["faces"] = [f for f in last_result.get("faces", [])
                                            if f.get("position") != pos]
                    with self._current_lock:
                        cur_faces = self._current.get("faces", [])
                        self._current["faces"] = [f for f in cur_faces
                                                  if f.get("position") != pos]
                        self._current["seat_states"] = {k: v["state"] for k, v in self._seat_states.items()}
                    log.info("[StateMachine] %s 数据已清除", side)

            # ── 消费已完成的结果 ──
            while True:
                try:
                    rframe, rresult = self._result_queue.get_nowait()
                    # 始终更新 _raw_face_present（即使 face_lost），否则状态机看不到真实的人脸检测结果
                    if "_raw_face_present" in rresult:
                        last_result["_raw_face_present"] = rresult["_raw_face_present"]
                    if not rresult.get("face_lost"):
                        last_result = rresult
                    else:
                        # face_lost 时也更新 faces，避免预览框冻结在旧位置
                        last_result["faces"] = rresult.get("faces", [])
                    last_cloth = rresult.get("cloth", {})
                    if not last_cloth:
                        last_cloth = dict(self._last_cloth_cache)
                    last_frame_for_result = rframe
                except _q.Empty:
                    break
            with self._current_lock:
                # 有实时人脸才更新 _current，否则保护 held 结果不被空数据覆盖
                live_faces = [f for f in last_result.get("faces", []) if not f.get("_held")]
                if live_faces:
                    self._current = last_result.copy()
                    self._current["cloth"] = last_cloth if last_cloth else self._last_cloth_cache
                elif not self._current.get("face_lost"):
                    self._current = last_result.copy()
                    self._current["cloth"] = last_cloth if last_cloth else self._last_cloth_cache

            if last_cloth:
                self._last_cloth_cache = last_cloth

            # 暂停时
            if self._paused:
                preview = cv2.resize(frame, (1280, 720))
                cv2.putText(preview, "PAUSED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
                with self._last_annotated_lock:
                    self._last_annotated_frame = preview
                continue

            # ── 帧跳过：静止空场景跳过 pipeline 推理（身体检测+状态机已跑完）──
            if prev_frame_gray is not None:
                motion = float(np.mean(cv2.absdiff(gray, prev_frame_gray)))
                now = time.time()
                no_faces = len(last_result.get("faces", [])) == 0
                if no_faces and motion < MOTION_THRESHOLD and (now - last_processed_time) < FORCE_REFRESH_SEC:
                    prev_frame_gray = gray
                    # 消费已完成结果（保持 face_present 更新）
                    while True:
                        try:
                            rframe, rresult = self._result_queue.get_nowait()
                            if "_raw_face_present" in rresult:
                                last_result["_raw_face_present"] = rresult["_raw_face_present"]
                            last_cloth = rresult.get("cloth", {}) or self._last_cloth_cache
                        except _q.Empty:
                            break
                    self._draw_preview(frame, frame_count, last_result)
                    continue
            prev_frame_gray = gray

            # ── 提交帧到 pipeline（detection worker 无脸时会自动 bypass recognition）──
            interval = self._adaptive_interval()
            is_processed_frame = frame_count - last_processed_frame >= interval
            # 有 body 但超过 3 帧未提交 → 强制提交，平衡延迟和负载
            body_detected_any = body_present.get("left", False) or body_present.get("right", False)
            if body_detected_any and frame_count - last_processed_frame >= 3:
                is_processed_frame = True
            if is_processed_frame:
                last_processed_frame = frame_count
                last_processed_time = time.time()
                self._frame_queue.put((frame.copy(), frame_count))

            self._draw_preview(frame, frame_count, last_result)

        if self._cap:
            self._cap.release()
            self._cap = None

    def _update_seat_states(self, face_present: dict, body_present: dict):
        """
        更新座位状态机。由 _loop() 每帧调用。

        核心规则：
        - 有人脸 → confirmed，计数器清零
        - 没人脸 + 有体 → face_lost，计数器清零（信任运动检测，不限时）
        - 没人脸 + 无体 → 计数器累加 → 到阈值 → empty (~0.6s)
        - 从未见过脸 → 有体也不能从 empty 创建状态（防冷启动误检）
        """
        now = time.time()
        for side in ("left", "right"):
            st = self._seat_states[side]
            prev_state = st["state"]
            has_face = face_present.get(side, False)
            has_body = body_present.get(side, False)

            if has_face:
                self._last_face_seen[side] = now

            # 人脸出现 → 立即确认，清零
            if has_face:
                if prev_state != "confirmed":
                    log.info("[StateMachine] %s: %s → confirmed (face)", side, prev_state)
                st["state"] = "confirmed"
                st["body_lost_frames"] = 0
                continue

            # ── 以下 has_face = False ──
            last_seen = self._last_face_seen.get(side, 0)
            body_trusted = (now - last_seen) < self._body_face_timeout

            # 从未见过人脸 → 人体不可信，保持 empty
            if prev_state == "empty" and not body_trusted:
                continue

            # 有体（运动检测可信）→ 维持 face_lost，不限时
            if has_body:
                if prev_state != "face_lost":
                    log.info("[StateMachine] %s: %s → face_lost", side, prev_state)
                st["state"] = "face_lost"
                st["body_lost_frames"] = 0
                continue

            # ── 以下 has_body = False ──
            st["body_lost_frames"] += 1

            if st["body_lost_frames"] >= self._body_lost_frames_threshold:
                old = st["state"]
                st["state"] = "empty"
                st["held"] = None
                st["body_lost_frames"] = 0
                log.info("[StateMachine] %s: %s → empty (%d frames)",
                         side, old, self._body_lost_frames_threshold)

    def _adaptive_interval(self) -> int:
        """根据场景状态动态返回完整流水线的处理间隔（帧数）。

        基于 25fps 摄像头（40ms/帧）和典型帧 85ms / 重帧 130ms 的耗时计算：
        - interval=2: 窗口 80ms，典型帧落后 5ms（可接受，短期使用）
        - interval=3: 窗口 120ms，余量 35ms（已验证安全）
        - interval=10: 窗口 400ms，余量 315ms（低频维护）
        """
        for queue in self._enroll_queues.values():
            if 0 < len(queue) < self.auto_enroll_min_frames:
                return 3  # 有待注册队列时适度提速，保持稳定
        if not any(self._seat_positions.values()):
            return 30
        with self._current_lock:
            faces = self._current.get("faces", [])
        if faces:
            all_known_stable = all(
                f.get("is_known") and f.get("is_stable")
                for f in faces
            )
            if all_known_stable:
                return 10
        return self.frame_interval

    def _select_front_row_faces(
        self,
        faces: List[DetectedFace],
        image_w: int,
        image_h: int,
    ) -> List[DetectedFace]:
        """委托 utils.select_front_row_faces, 适配 DetectedFace → dict 再映射结果。"""
        if not faces:
            return []

        face_dicts = [
            {"bbox": [float(v) for v in f.bbox], "score": float(f.score), "_detected_face": f}
            for f in faces
        ]

        selected_dicts = select_front_row_faces(face_dicts, image_w, image_h)

        selected = []
        for d in selected_dicts:
            face = d.get("_detected_face")
            if face is None:
                continue
            slot = d.get("_front_row_slot", "")
            setattr(face, "front_row_position", slot)
            selected.append(face)

        return sorted(selected, key=lambda f: f.xc)

    def _draw_preview(self, frame: np.ndarray, frame_count: int, result: Dict):
        """绘制标注帧。先缩到 1280×720 再绘制，大幅降低拷贝和绘制开销。"""
        annotated = self.draw_recognition(frame, result)
        info = f"Faces: {result.get('total_detected', 0)}  Known: {result.get('total_known', 0)}  Frame: {frame_count}"
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # ── Body detection debug 叠加 ──
        if self._body_det_enabled and self._body_det_debug and self.body_detector is not None:
            try:
                from utils import FRONT_ROW_BODY_ROIS, _scale_roi
                H_a, W_a = annotated.shape[:2]
                # 绘制 body ROI 矩形
                roi_colors = {"left": (255, 0, 0), "right": (0, 255, 0)}
                for slot_name, roi in FRONT_ROW_BODY_ROIS.items():
                    rx1, ry1, rx2, ry2 = _scale_roi(roi, W_a, H_a)
                    cv2.rectangle(annotated,
                                  (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                                  roi_colors.get(slot_name, (255, 255, 255)), 2)
                    cv2.putText(annotated, f"body_{slot_name}",
                                (int(rx1) + 4, int(ry1) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                roi_colors.get(slot_name, (255, 255, 255)), 1)
                # 使用 _last_body_boxes（与状态机同源），缩放到 annotated 尺寸
                H_orig, W_orig = frame.shape[:2]
                scale_x = W_a / W_orig
                scale_y = H_a / H_orig
                for b in self._last_body_boxes:
                    bx1, by1, bx2, by2 = b["bbox"]
                    bx1 = int(bx1 * scale_x); by1 = int(by1 * scale_y)
                    bx2 = int(bx2 * scale_x); by2 = int(by2 * scale_y)
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    cv2.putText(annotated, f"body {b['score']:.2f}",
                                (bx1, by1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                # 状态文字
                y_offset = H_a - 20
                for side in ("left", "right"):
                    st = self._seat_states.get(side, {})
                    state_text = f"{side}: {st.get('state', '?')}"
                    color = (0, 255, 0) if st.get("state") != "empty" else (0, 0, 255)
                    cv2.putText(annotated, state_text, (10 if side == "left" else W_a // 2, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception:
                pass
        with self._last_annotated_lock:
            self._last_annotated_frame = annotated

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_count: int = 0,
        pre_detected=None,
        pre_raw=None,
    ) -> Dict[str, Any]:
        """
        处理单帧。支持流水线模式：传入 pre_detected/pre_raw 跳过检测阶段。

        Returns:
            {
                "faces": [{
                    "bbox": [...],
                    "position": "left" | "right" | "visual_only",
                    "is_known": bool,
                    "identity_id": str | None,
                    "similarity": float,
                    "confidence": float,
                    "name": str | None,
                    "is_new": bool,
                    "is_stable": bool,         # 滑动窗口是否稳定
                }, ...],
                "frame_count": int,
                "timestamp": str,
                "total_detected": int,
                "total_known": int,
            }
        """
        t0 = time.time()
        # ── 判断当前帧是否有检测到的人脸 ──
        any_face_detected = pre_detected is not None and len(pre_detected) > 0
        if not any_face_detected and pre_raw is not None:
            any_face_detected = len(pre_raw) > 0

        # ── face_lost 跳过: 当前帧无实时人脸 + 有 held 数据可兜底 → 直接返回 ──
        if not any_face_detected:
            held_sides = [
                s for s in ("left", "right")
                if self._seat_states[s].get("held") is not None
            ]
            if held_sides:
                # 计算原始检测的 face_present
                _raw_fp = {"left": False, "right": False}
                raw_list = pre_raw if pre_raw is not None else []
                for face in raw_list:
                    bbox = getattr(face, "bbox", None)
                    if bbox is None:
                        continue
                    xc = (bbox[0] + bbox[2]) / 2
                    _raw_fp["right" if xc >= frame.shape[1] / 2 else "left"] = True

                held_result = {
                    "faces": [],
                    "frame_count": frame_count,
                    "timestamp": datetime.now().isoformat(),
                    "process_time_ms": 0,
                    "total_detected": 0,
                    "total_raw_detected": 0,
                    "total_known": 0,
                    "height": None,
                    "raw_detected": raw_list,
                    "_raw_face_present": _raw_fp,
                    "face_lost": True,
                    "seat_states": {k: v["state"] for k, v in self._seat_states.items()},
                }
                for side in held_sides:
                    st = self._seat_states[side]
                    held_copy = dict(st["held"])
                    held_copy["_held"] = True
                    held_result["faces"].append(held_copy)
                    held_result["total_detected"] += 1
                    if st["held"].get("is_known"):
                        held_result["total_known"] += 1
                return held_result
        H, W = frame.shape[:2]
        if pre_detected is not None:
            detected = pre_detected
            raw_detected = pre_raw if pre_raw is not None else []
        else:
            raw_detected = self.detector.detect_frame(frame)
            detected = self._select_front_row_faces(raw_detected, W, H)

        face_results = []
        embeddings_to_match = []
        emb_epoch = frame_count // self.smooth_window

        for face in detected:
            position_raw = getattr(face, "front_row_position", None) or ("left" if face.xc < W / 2 else "right")

            # ── 多帧平滑 ────────────────────────────────────────────────
            emb_key = f"{position_raw}_{emb_epoch}_{frame_count % self.smooth_window}"
            with self._emb_lock:
                if emb_key not in self._emb_history:
                    self._emb_history[emb_key] = deque(maxlen=self.smooth_window)
                    # 清理同位置旧 epoch 的 key（防止跨 epoch 碰撞和泄漏）
                    stale_prefix = f"{position_raw}_"
                    for old_key in list(self._emb_history.keys()):
                        if old_key.startswith(stale_prefix) and old_key != emb_key:
                            del self._emb_history[old_key]
                self._emb_history[emb_key].append(face.emb.copy())

                history = list(self._emb_history[emb_key])
                is_stable = len(history) >= self.smooth_window

            # 识别：滑动窗口内做加权融合
            if is_stable and len(history) > 1:
                # 指数加权，最新帧权重最高
                weights = np.exp(np.linspace(0, 1, len(history)))
                weights /= weights.sum()
                fused = np.sum(np.stack(history) * weights[:, None], axis=0)
                fused = normalize_emb(fused)
                embeddings_to_match.append(fused)
            else:
                embeddings_to_match.append(face.emb)

        # ── 批量识别（一次 matmul 完成所有比对，极端侧脸 None 不参与）──────────
        embs_real = [e for e in embeddings_to_match if e is not None]
        real_indices = [i for i, e in enumerate(embeddings_to_match) if e is not None]
        matches_raw = self.gallery.batch_recognize(embs_real, self.sim_threshold) if embs_real else []
        # 重建完整 matches 列表，极端侧脸用空 match
        matches = []
        real_iter = iter(matches_raw)
        for i in range(len(detected)):
            if i in real_indices:
                matches.append(next(real_iter))
            else:
                matches.append({"is_known": False, "identity_id": None, "similarity": -1.0, "confidence": 0.0, "name": None})

        for idx, face in enumerate(detected):
            position_raw = getattr(face, "front_row_position", None) or ("left" if face.xc < W / 2 else "right")

            # ── 极端侧脸（yaw 可靠且 ≥ 60°）：跳过匹配，但继续走 auto_enroll ──

            emb_key = f"{position_raw}_{emb_epoch}_{frame_count % self.smooth_window}"
            with self._emb_lock:
                history = list(self._emb_history.get(emb_key, []))
                is_stable = len(history) >= self.smooth_window

            match = matches[idx] if idx < len(matches) else {"is_known": False, "identity_id": None, "similarity": -1.0, "confidence": 0.0, "name": None}

            face_result = {
                "bbox": [float(v) for v in face.bbox],
                "position": position_raw,   # 仅用于可视化，不做位置约束
                "is_known": match["is_known"],
                "identity_id": match["identity_id"],
                "similarity": match["similarity"],
                "confidence": match["confidence"],
                "name": match.get("name"),
                "is_new": not match["is_known"],
                "detection_score": face.score,
                "is_stable": is_stable,
                "_emb": embeddings_to_match[idx] if idx < len(embeddings_to_match) else None,
            }
            face_results.append(face_result)

            # ── 头部姿态（所有脸都计算，输出到 API）─────────────────────
            _yaw, _pitch, _roll = self._get_face_pose(getattr(face, "kps", None))
            face_result["yaw"] = round(_yaw, 2) if _yaw >= 0 else None
            face_result["pitch"] = round(_pitch, 2) if _yaw >= 0 else None
            face_result["roll"] = round(_roll, 2) if _yaw >= 0 else None
            yaw_raw = abs(_yaw) if _yaw >= 0 else (0.0 if getattr(face, "kps", None) is not None else 999.0)
            pitch_raw = abs(_pitch) if _yaw >= 0 else (0.0 if getattr(face, "kps", None) is not None else 999.0)

            # ── 增量更新 ────────────────────────────────────────────────
            if self.update_centroid and match["is_known"] and match["identity_id"]:
                self.gallery.update_if_confident(
                    match["identity_id"], face.emb, save=False
                )

            # ── SIM < 0.3：弱匹配，用缓存身份替换 ──
            if match["is_known"] and match.get("similarity", -1.0) < 0.30:
                st = self._seat_states.get(position_raw, {})
                held = st.get("held") if st else None
                if held and held.get("identity_id"):
                    # 用缓存身份（之前确认过的），显示为沿袭
                    face_result["identity_id"] = held["identity_id"]
                    face_result["similarity"] = match["similarity"]
                    face_result["is_new"] = False
                    face_result["_inherited"] = True
                    match["is_known"] = False  # 走下面的沿袭/注册逻辑
                    log.debug("弱匹配: sim=%.3f pos=%s → 用缓存身份 %s",
                              match["similarity"], position_raw, held["identity_id"][:8])
                else:
                    match["is_known"] = False  # 没缓存，当未知
                    log.debug("弱匹配: sim=%.3f pos=%s → 无缓存，当未知",
                              match["similarity"], position_raw)

            # ── 回调 & 自动注册（不依赖 GA 结果）─────────────────────────
            if match["is_known"]:
                self._position_identity[position_raw] = match["identity_id"]
                if self.on_recognized:
                    self.on_recognized(match["identity_id"], match["similarity"], match["confidence"])
            else:
                prev_id = self._position_identity.get(position_raw)

                # 身份沿袭：有历史身份就沿用，不限角度，保证不显示"未知"
                if prev_id:
                    # 身份在 gallery 中不存在（被清空等）→ 不沿用
                    held_entry = self.gallery.get(prev_id)
                    if held_entry is None:
                        prev_id = None
                        self._position_identity.pop(position_raw, None)
                    else:
                        # 检查 embedding 相似度，排除换人情况
                        sim = float(np.dot(normalize_emb(face.emb.copy()), held_entry.embedding))
                        if sim < 0.15:
                            prev_id = None  # 换人了，不用旧身份
                    if prev_id:
                        face_result["is_known"] = True
                        face_result["identity_id"] = prev_id
                        face_result["is_new"] = False
                        face_result["_inherited"] = True  # 标记沿袭，跳过属性识别
                        log.debug("沿用上一身份: yaw=%.0f° pitch=%.0f° pos=%s id=%s",
                                  yaw_raw, pitch_raw, position_raw, prev_id)

                # 极端角度（侧脸/低头/抬头）且无历史身份 → 拦截注册
                angle_blocked = (yaw_raw >= self.auto_enroll_max_yaw
                                 or pitch_raw >= self.auto_enroll_max_pitch)
                if angle_blocked:
                    if not prev_id:
                        log.debug("跳过极端角度注册: yaw=%.0f° pitch=%.0f° pos=%s",
                                  yaw_raw, pitch_raw, position_raw)
                else:
                    # held 存在 + body 未断 → 旧人，不注册新身份
                    st = self._seat_states.get(position_raw, {})
                    held_id = (st.get("held") or {}).get("identity_id")
                    if held_id and not self.gallery.get(held_id):
                        # 身份已被删除（gallery 清空等），清除 held
                        st["held"] = None
                        held_id = None
                    if held_id and st.get("state") != "empty":
                        log.debug("跳过自动注册: 该位置已有身份 %s", st["held"].get("identity_id", "?")[:8])
                    else:
                        kps = getattr(face, "kps", None)
                        det_score = getattr(face, "score", 0.0)
                        if self.on_unknown:
                            self.on_unknown(face.emb, position_raw)
                        if self.auto_enroll:
                            if kps is None:
                                log.info("[AutoEnroll] %s 跳过: 无关键点 det_score=%.3f", position_raw, det_score)
                            elif det_score <= 0.5:
                                log.info("[AutoEnroll] %s 跳过: det_score=%.3f ≤0.5", position_raw, det_score)
                            else:
                                self._auto_enroll(face.emb, position_raw, yaw_raw, pitch_raw)

        # ── 同一ID去重：ID不能同时出现在左右两侧 ──
        id_positions: Dict[str, list] = {}
        for i, fr in enumerate(face_results):
            iid = fr.get("identity_id")
            if iid:
                id_positions.setdefault(iid, []).append(i)
        for iid, indices in id_positions.items():
            if len(indices) >= 2:
                # 保留相似度高的，另一边强制走auto_enroll创建新ID
                indices.sort(key=lambda i: face_results[i].get("similarity", 0.0), reverse=True)
                for idx in indices[1:]:
                    pos = face_results[idx].get("position", "")
                    log.warning("[DedupID] %s 同一ID %s 出现在两侧，%s侧清除并强制创建新ID",
                                indices, iid[:12], pos)
                    face_results[idx]["identity_id"] = None
                    face_results[idx]["is_known"] = False
                    face_results[idx]["name"] = None
                    face_results[idx]["_inherited"] = False
                    # 清除缓存，让auto_enroll能创建新ID
                    self._position_identity.pop(pos, None)
                    st = self._seat_states.get(pos)
                    if st:
                        st["held"] = None

        # ── GA 预测并行（沿袭帧跳过，直接用 held 属性）────
        ga_faces = [(i, fr) for i, fr in enumerate(face_results) if not fr.get("_inherited")]
        if ga_faces:
            if len(ga_faces) > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {i: pool.submit(self._predict_gender_age_cached, frame, fr, frame_count)
                               for i, fr in ga_faces}
                    for i, future in futures.items():
                        face_results[i].update(future.result())
            else:
                i, fr = ga_faces[0]
                face_results[i].update(
                    self._predict_gender_age_cached(frame, fr, frame_count)
                )
        # 沿袭帧从 held 填充属性
        for i, fr in enumerate(face_results):
            if fr.get("_inherited"):
                side = fr.get("position", "")
                st = self._seat_states.get(side, {})
                held = st.get("held") or {}
                for key in ("gender", "gender_score", "gender_stable", "gender_stable_source",
                            "gender_stable_samples", "age_group", "age_score", "age_stable",
                            "age_stable_source", "age_stable_samples",
                            "attribute_enabled", "attribute_backend", "attribute_error"):
                    if key in held:
                        fr[key] = held[key]

        if self.update_centroid:
            # 每隔约1秒（按frame_interval）才保存一次，减少磁盘冲突
            if not hasattr(self, "_save_counter"):
                self._save_counter = 0
            self._save_counter += 1
            if self._save_counter >= 10:
                self._request_gallery_save()
                self._save_counter = 0

        total_known = sum(1 for f in face_results if f["is_known"])

        # ── 身高估计（每10个处理帧跑一次 ≈ 1秒间隔）───────────────
        height_result = None
        if self.height_estimator is not None and self.height_estimator.ready:
            if not hasattr(self, "_height_counter"):
                self._height_counter = 0
            self._height_counter += 1

            # 从人脸识别结果提取性别，同步给身高估计
            d_gen, p_gen = 1.0, 1.0
            d_gender_found, p_gender_found = False, False
            for f in face_results:
                g_score = float(f.get("gender_score", 0.5))
                if f.get("position") == "right":
                    d_gen = g_score
                    d_gender_found = True
                elif f.get("position") == "left":
                    p_gen = g_score
                    p_gender_found = True
            if not d_gender_found and face_results and not hasattr(self, "_warned_d_gen"):
                log.warning("身高估计: 主驾性别未识别，默认使用 male")
                self._warned_d_gen = True
            if not p_gender_found and face_results and not hasattr(self, "_warned_p_gen"):
                log.warning("身高估计: 副驾性别未识别，默认使用 male")
                self._warned_p_gen = True
            self.height_estimator.set_gender(d_gen, p_gen)

            if self._height_counter >= 10:
                self._height_counter = 0
                try:
                    height_result = self.height_estimator.predict(frame)
                except Exception as e:
                    log.debug(f"[VehicleRecognizer] height estimation failed: {e}")

            # ── 身高稳定后写入 Gallery（只写一次）────────────────────
            if height_result and self.height_estimator._can_received:
                for f in face_results:
                    if not f.get("identity_id") or f.get("_held"):
                        continue
                    role = "driver" if f.get("position") == "right" else "passenger"
                    h_info = height_result.get(role, {})
                    if h_info.get("stable") and h_info.get("ema"):
                        meta = self.gallery.get_metadata(f["identity_id"])
                        if "height" not in meta:
                            self.gallery.update_metadata(
                                f["identity_id"],
                                {"height": round(h_info["ema"], 1)},
                                save=False,
                            )
                            self._request_gallery_save()
                            log.info("[HeightStable] 身高写入 gallery: id=%s role=%s height=%.1f",
                                     f["identity_id"], role, h_info["ema"])

        # ── 检测到脸但未识别：同一座位 body 持续在 → 大概率同一人，用 held ──
        # 仅用极低 embedding 阈值（0.15）排除快速换人：两个不同人 sim < 0.15
        for f in face_results:
            side = f.get("position", "")
            if side and not f.get("identity_id"):
                st = self._seat_states.get(side, {})
                held = st.get("held") if st else None
                if held and held.get("identity_id") and st.get("state") != "empty":
                    emb = f.get("_emb")
                    if emb is not None:
                        held_entry = self.gallery.get(held["identity_id"])
                        if held_entry is not None:
                            sim = float(np.dot(normalize_emb(emb.copy()), held_entry.embedding))
                            if sim < 0.15:  # 完全不同的人 → 不用 held
                                continue
                    f["identity_id"] = held["identity_id"]
                    f["is_known"] = True
                    f["name"] = held.get("name")
                    f["_identity_from_held"] = True

        # ── 更新 held 结果 ──
        for side in ("left", "right"):
            st = self._seat_states[side]
            if st["state"] == "confirmed":
                for f in face_results:
                    if f.get("position") == side and f.get("identity_id"):
                        # 检查是否换人
                        old_held = st.get("held") or {}
                        old_id = old_held.get("identity_id")
                        new_id = f.get("identity_id")
                        if old_id and new_id and old_id != new_id:
                            log.info("[SeatState] %s 换人: %s → %s (即时切换)", side, old_id, new_id)
                        held_copy = dict(f)
                        held_copy.pop("_emb", None)  # numpy array, 不能 JSON 序列化
                        st["held"] = held_copy
                        break

        # ── 清理内部字段（_emb 是 numpy array，不能 JSON 序列化）──
        for f in face_results:
            f.pop("_emb", None)

        # ── 为 face_lost 侧注入 held 人脸（确保 API 有数据输出）──
        any_face_lost = False
        for side in ("left", "right"):
            st = self._seat_states[side]
            if st["state"] == "face_lost" and st.get("held"):
                already_in = any(f.get("position") == side for f in face_results)
                if not already_in:
                    held_copy = dict(st["held"])
                    held_copy["_held"] = True
                    face_results.append(held_copy)
                any_face_lost = True

        return {
            "faces": face_results,
            "frame_count": frame_count,
            "timestamp": datetime.now().isoformat(),
            "process_time_ms": round((time.time() - t0) * 1000, 1),
            "total_detected": len(face_results),
            "total_raw_detected": len(raw_detected),
            "total_known": total_known,
            "height": height_result,
            "raw_detected": raw_detected,
            "_raw_face_present": {
                "left": any(f.get("position") == "left" and not f.get("_held") for f in face_results),
                "right": any(f.get("position") == "right" and not f.get("_held") for f in face_results),
            },
            "face_lost": any_face_lost,
            "seat_states": {k: v["state"] for k, v in self._seat_states.items()},
        }

    def get_height_status(self) -> Dict[str, Any]:
        """获取身高估计状态。"""
        if self.height_estimator is None:
            return {"driver": None, "passenger": None, "ready": False}
        return self.height_estimator.get_status()

    # ── 相机标定 + 3D 人脸模型（solvePnP 用）────────────────────────
    _CAM_MATRIX = np.array(
        [[998.287744, 0.0, 1279.77179],
         [0.0, 1000.45038, 730.399691],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    _CAM_DIST = np.array(
        [[-0.11169482, -0.02339049, -0.0007184, 0.0006266, 0.01236564]],
        dtype=np.float64)
    _MODEL_3D = np.array([  # InsightFace 5点顺序: 左眼 右眼 鼻 左嘴 右嘴
        [-150.0, -170.0, -135.0],   # 左眼
        [150.0, -170.0, -135.0],    # 右眼
        [0.0, 0.0, 0.0],            # 鼻尖
        [-150.0, 150.0, -125.0],    # 左嘴角
        [150.0, 150.0, -125.0],     # 右嘴角
    ], dtype=np.float64)

    @classmethod
    def _get_face_pose(cls, kps) -> tuple:
        """cv2.solvePnP 计算头部姿态角（度）。
        返回 (yaw, pitch, roll):
          - yaw:   左右转头，正=右转，负=左转
          - pitch: 抬头/低头，正=低头，负=抬头
          - roll:  左右歪头，正=右歪，负=左歪
        无关键点时返回 (90, 0, 0)。
        """
        if kps is None:
            return (90.0, 0.0, 0.0)
        pts = np.asarray(kps, dtype=np.float64)
        if pts.shape != (5, 2):
            return (90.0, 0.0, 0.0)
        try:
            ok, rvec, _ = cv2.solvePnP(
                cls._MODEL_3D, pts, cls._CAM_MATRIX, cls._CAM_DIST,
                flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                return (-1.0, 0.0, 0.0)
            R, _ = cv2.Rodrigues(rvec)
            # solvePnP 有两个解，R[2,2]<0 表示法向量指向相机后方（翻转解），取反
            if R[2, 2] < 0:
                rvec = -rvec
                R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
            pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            roll = float(np.degrees(np.arctan2(-R[1, 0], R[0, 0])))
            return (yaw, pitch, roll)
        except Exception:
            return (-1.0, 0.0, 0.0)

    def _auto_enroll(self, emb, position: str, yaw: float = 0.0, pitch: float = 0.0):
        """
        自动注册逻辑：
          1. 极端角度（yaw/pitch 过大）不入队
          2. 入队后用运行质心查 gallery（阈值 0.30，比注册阈值更严）
          3. 该位置曾有熟人 → 阈值压到 0.25，极难误建
          4. 攒满 10 帧且始终低于注册阈值 → 新人，修剪均值注册
          5. 注册后全量比对去重
        """
        import numpy as _np

        if yaw >= self.auto_enroll_max_yaw or pitch >= self.auto_enroll_max_pitch:
            return

        queue_key = position
        if queue_key not in self._enroll_queues:
            self._enroll_queues[queue_key] = deque(maxlen=self.auto_enroll_min_frames * 3)

        queue = self._enroll_queues[queue_key]
        norm_emb = normalize_emb(emb.copy())
        queue.append(norm_emb)
        log.info("[AutoEnroll] %s 队列: %d/%d 帧 (yaw=%.0f° pitch=%.0f°)",
                 position, len(queue), self.auto_enroll_min_frames, yaw, pitch)

        embs = list(queue)
        centroid = _np.mean(embs, axis=0)

        # ── 运行质心查 gallery（攒够 3 帧后，阈值 0.30，旧位置 0.25）──
        block_threshold = 0.25 if self._position_identity.get(position) else 0.30
        if len(embs) >= 3:
            norm_centroid = normalize_emb(centroid)
            for identity_id, entry in self.gallery.get_all().items():
                sim = float(_np.dot(norm_centroid, entry.embedding))
                if sim > block_threshold:
                    queue.clear()
                    log.debug("[AutoEnroll] 质心命中已有身份 %s (sim=%.4f, thresh=%.2f)，清队列",
                             identity_id[:8], sim, block_threshold)
                    return

        # ── 跨队列去重 ────────────────────────────────────────────
        for other_key, other_queue in self._enroll_queues.items():
            if other_key == queue_key or len(other_queue) < 3:
                continue
            other_centroid = _np.mean(list(other_queue), axis=0)
            if _np.dot(normalize_emb(centroid), normalize_emb(other_centroid)) > 0.30:
                queue.clear()
                return

        if len(embs) < self.auto_enroll_min_frames:
            return

        # ── 修剪：去掉离质心最远的 2 帧 ──────────────────────────
        if len(embs) >= self.auto_enroll_min_frames + 2:
            dists = [_np.dot(e, centroid) for e in embs]
            keep_idx = _np.argsort(dists)[2:]
            embs = [embs[i] for i in keep_idx]
            centroid = _np.mean(embs, axis=0)

        # ── 簇内方差过滤 ─────────────────────────────────────────
        norms = [_np.dot(e, centroid) for e in embs]
        if _np.std(norms) > 0.08:
            queue.clear()
            log.debug("[AutoEnroll] 簇内方差过大，清除缓冲: std=%.4f", _np.std(norms))
            return

        # ── 最终比对（用注册阈值 0.35）──
        norm_centroid = normalize_emb(centroid)
        for identity_id, entry in self.gallery.get_all().items():
            if _np.dot(norm_centroid, entry.embedding) > self.auto_enroll_sim:
                queue.clear()
                return

        for _ in range(100):
            new_id = uuid.uuid4().hex[:16]
            if self.gallery.get(new_id) is None:
                break
        identity_id = new_id
        self.gallery.register(
            identity_id, norm_centroid,
            name=identity_id,
            is_registered=False,
            metadata={"position": position, "enroll_frames": len(embs)},
            save=False,
        )
        self._request_gallery_save()
        if self.on_enrolled:
            self.on_enrolled(identity_id, identity_id)
        queue.clear()

        # ── 注册后全量去重：扫所有 ID 对，删掉重复的新 ID ──────────
        self._dedup_gallery(new_id=identity_id)

    def _dedup_loop(self):
        """后台线程：每 30 秒全量去重一次。"""
        while getattr(self, "_dedup_running", True):
            time.sleep(30)
            if not getattr(self, "_dedup_running", True):
                break
            try:
                self._dedup_gallery()
            except Exception:
                log.debug("[Dedup] 巡检异常", exc_info=True)

    def _dedup_gallery(self, new_id: str = ""):
        """全量比对去重：相似度 > 0.35 的一对，删除较新的那个。"""
        import numpy as _np
        items = list(self.gallery.get_all().items())
        if len(items) < 2:
            return
        removed = set()
        for i in range(len(items)):
            id_i, entry_i = items[i]
            if id_i in removed:
                continue
            for j in range(i + 1, len(items)):
                id_j, entry_j = items[j]
                if id_j in removed:
                    continue
                sim = float(_np.dot(entry_i.embedding, entry_j.embedding))
                if sim > self.auto_enroll_sim:
                    if entry_i.created_at > entry_j.created_at:
                        dup, keep = id_i, id_j
                    else:
                        dup, keep = id_j, id_i
                    self.gallery.unregister(dup, save=False)
                    removed.add(dup)
                    log.info("[Dedup] 重复身份已删除 %s (sim=%.4f, keep=%s)",
                             dup[:12], sim, keep[:12])
        if removed:
            self._request_gallery_save()

    def _request_gallery_save(self):
        """请求后台保存 Gallery，避免同步写磁盘导致视频线程卡顿。"""
        save_async = getattr(self.gallery, "save_async", None)
        if callable(save_async):
            save_async()
            return
        async_save = getattr(self.gallery, "_async_save", None)
        if callable(async_save):
            async_save()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def recognize_image(
        self,
        image: np.ndarray | str | Path,
        return_image: bool = False,
    ) -> Dict[str, Any]:
        """
        识别单张图片

        Args:
            image: cv2 图片 或 图片路径
            return_image: 是否返回带标注的图片

        Returns:
            {
                "faces": [...],
                "total_detected": int,
                "total_known": int,
                "image_base64": str (可选),
            }
        """
        if not self._running:
            self.detector.ensure()

        if isinstance(image, (str, Path)) and not Path(image).is_file():
            raise FileNotFoundError(f"图片文件不存在: {image}")

        with self._process_lock:
            result = self._process_frame(
                cv2.imread(str(image)) if isinstance(image, (str, Path)) else image,
                frame_count=0,
            )

        if return_image:
            img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
            annotated = self.draw_recognition(img, result)
            import base64
            _, buf = cv2.imencode(".jpg", annotated)
            result["image_base64"] = base64.b64encode(buf).decode("utf-8")

        return result

    def recognize_images_batch(
        self,
        image_paths: List[str | Path],
    ) -> List[Dict[str, Any]]:
        """批量识别多张图片。"""
        if not self._running:
            self.detector.ensure()
        return [self.recognize_image(p) for p in image_paths]

    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        skip_frames: int = 1,
        show: bool = False,
    ) -> Dict[str, Any]:
        """
        处理视频文件，逐帧识别。

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（带标注）
            skip_frames: 每隔几帧处理一帧
            show: 是否实时显示
        """
        if not self._running:
            self.detector.ensure()

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps / skip_frames, (W, H))

        frame_count = 0
        results = []
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            with self._process_lock:
                result = self._process_frame(frame, frame_count)

            # 绘制
            if writer or show:
                annotated = self.draw_recognition(frame, result)
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Recognition", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            results.append(result)

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        elapsed = time.time() - t0
        total_detected = sum(r["total_detected"] for r in results)
        total_known = sum(r["total_known"] for r in results)

        return {
            "video_path": str(video_path),
            "output_path": str(output_path) if output_path else None,
            "total_frames": frame_count,
            "processed_frames": len(results),
            "total_detected": total_detected,
            "total_known": total_known,
            "elapsed_seconds": round(elapsed, 1),
            "fps": round(len(results) / elapsed, 1),
        }

    def get_current(self) -> Dict[str, Any]:
        """获取当前识别结果快照（线程安全）。"""
        with self._current_lock:
            return dict(self._current)

    def get_gallery(self) -> Gallery:
        """获取 Gallery 实例。"""
        return self.gallery

    def _predict_gender_age_cached(
        self,
        frame: np.ndarray,
        face_result: Dict[str, Any],
        frame_count: int,
    ) -> Dict[str, Any]:
        """性别/年龄预测，带预测间隔缓存和稳定性持久化。

        - 每帧优先用缓存（间隔内不重复推理）
        - 达到稳定条件后写入 Gallery，后续用稳定值输出
        - 稳定值定期刷新验证，不一致时更新（不锁死）
        """
        identity_id = face_result.get("identity_id")
        position = face_result.get("position") or ""
        cache_key = f"pos:{position}" if self.attr_cache_by_seat else (f"id:{identity_id}" if identity_id else f"pos:{position}")
        cached = self._gender_age_cache.get(cache_key)
        stable = self._get_stable_attrs(identity_id)

        # 有稳定值且在刷新间隔内 → 直接用，不跑 FLIP（不依赖 cache 是否存在）
        if stable:
            stable_frame = int(stable.get("stable_frame", 0))
            if frame_count - stable_frame < self.stable_refresh_interval:
                return dict(stable)

        # 缓存未过期（30帧）→ 直接返回，有稳定值则合并
        if cached and frame_count - int(cached.get("frame_count", -10**9)) < self.gender_age_interval:
            return self._merge_stable(dict(cached.get("attrs", {})), stable)

        # 跑 FLIP 预测
        attrs = self.gender_age.predict_face(frame, face_result["bbox"])
        attrs = self._guard_realtime_attrs(attrs)

        # 已知身份 → 累计稳定性
        if identity_id and face_result.get("is_known"):
            stable_age = self._update_age_stability(identity_id, attrs)
            stable_gender = self._update_gender_stability(identity_id, attrs)
            # 合并最新的稳定值
            merged_stable = {}
            if stable_gender:
                merged_stable.update(stable_gender)
            if stable_age:
                merged_stable.update(stable_age)
            if merged_stable:
                stable = merged_stable

        attrs = self._merge_stable(attrs, stable)

        if attrs.get("gender") or attrs.get("age_group") or attrs.get("attribute_error"):
            self._gender_age_cache[cache_key] = {
                "frame_count": frame_count,
                "attrs": dict(attrs),
            }
        elif cached:
            return self._merge_stable(dict(cached.get("attrs", {})), stable)
        return attrs

    @staticmethod
    def _guard_realtime_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Realtime guard: avoid unsupported old-age spikes in cabin webcam runs."""
        if not attrs or attrs.get("age_group") != "75+":
            return attrs
        probs = attrs.get("age_probs") or {}
        candidates = {
            label: float(score)
            for label, score in probs.items()
            if label and label != "75+"
        }
        if not candidates:
            return attrs
        fallback_age, fallback_score = max(candidates.items(), key=lambda item: item[1])
        guarded = dict(attrs)
        guarded["age_group"] = fallback_age
        guarded["age_score"] = fallback_score
        guarded["age_guarded_from"] = "75+"
        return guarded

    def _get_stable_attrs(self, identity_id: Optional[str]) -> Dict[str, Any]:
        """从内存缓存或 Gallery metadata 读取已稳定的性别/年龄。"""
        if not identity_id:
            return {}
        with self._stability_lock:
            cached = self._stable_attrs_cache.get(identity_id)
            if cached:
                return dict(cached)

        get_metadata = getattr(self.gallery, "get_metadata", None)
        if callable(get_metadata):
            metadata = get_metadata(identity_id)
        else:
            entry = self.gallery.get(identity_id)
            metadata = dict(getattr(entry, "metadata", {}) or {}) if entry else {}

        result = {}
        stable_age = metadata.get("fixed_age_group")
        stable_age_source = "fixed"
        if not stable_age and self.use_legacy_stable_attrs:
            stable_age = metadata.get("stable_age_group")
            stable_age_source = "stable"
        if stable_age:
            result.update({
                "age_group": stable_age,
                "age_score": float(metadata.get(f"{stable_age_source}_age_score", 1.0) or 1.0),
                "age_stable": True,
                "age_stable_source": metadata.get(f"{stable_age_source}_age_source", "gallery"),
                "age_stable_samples": int(metadata.get(f"{stable_age_source}_age_samples", 0) or 0),
            })
        stable_gender = metadata.get("fixed_gender")
        stable_gender_source = "fixed"
        if not stable_gender and self.use_legacy_stable_attrs:
            stable_gender = metadata.get("stable_gender")
            stable_gender_source = "stable"
        if stable_gender:
            result.update({
                "gender": stable_gender,
                "gender_score": float(metadata.get(f"{stable_gender_source}_gender_score", 0.0) or 0.0),
                "gender_stable": True,
                "gender_stable_source": metadata.get(f"{stable_gender_source}_gender_source", "gallery"),
                "gender_stable_samples": int(metadata.get(f"{stable_gender_source}_gender_samples", 0) or 0),
            })
        if result:
            result["stable_frame"] = int(metadata.get("stable_frame", 0) or 0)
            with self._stability_lock:
                self._stable_attrs_cache[identity_id] = dict(result)
        return result

    @staticmethod
    def _merge_stable(attrs: Dict[str, Any], stable: Dict[str, Any]) -> Dict[str, Any]:
        """用稳定值覆盖对应的预测字段。"""
        if not stable:
            return attrs
        merged = dict(attrs or {})
        merged.update(stable)
        merged["attribute_enabled"] = merged.get("attribute_enabled", True)
        return merged

    def _update_age_stability(self, identity_id: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """累计年龄预测，达到稳定后持久化到 Gallery（可被后续刷新更新）。"""
        age_group = str(attrs.get("age_group") or "").strip()
        age_score = float(attrs.get("age_score") or 0.0)
        if not age_group or age_score < self.age_stable_min_score:
            return {}

        with self._stability_lock:
            history = self._age_stability_history.setdefault(
                identity_id,
                deque(maxlen=self.age_stable_window),
            )
            history.append((age_group, age_score))
            snapshot = list(history)

        counts = Counter(age for age, _score in snapshot)
        best_age, best_count = counts.most_common(1)[0]
        ratio = best_count / max(1, len(snapshot))
        if best_count < self.age_stable_min_count or ratio < self.age_stable_min_ratio:
            return {}

        scores = [score for age, score in snapshot if age == best_age]
        result = {
            "age_group": best_age,
            "age_score": round(float(np.mean(scores)), 4),
            "age_stable": True,
            "age_stable_source": "stable",
            "age_stable_samples": best_count,
        }
        # 内存缓存
        with self._stability_lock:
            stable = self._stable_attrs_cache.setdefault(identity_id, {})
            stable.update(result)
        # 持久化到 Gallery
        metadata = {
            "fixed_age_group": best_age,
            "fixed_age_score": result["age_score"],
            "fixed_age_samples": best_count,
            "fixed_age_window": len(snapshot),
            "fixed_age_ratio": round(ratio, 4),
            "fixed_age_source": "stable",
            "fixed_age_at": datetime.now().isoformat(),
            "stable_age_group": best_age,
            "stable_age_score": result["age_score"],
            "stable_age_samples": best_count,
            "stable_age_window": len(snapshot),
            "stable_age_ratio": round(ratio, 4),
            "stable_age_source": "stable",
            "stable_age_at": datetime.now().isoformat(),
        }
        update_metadata = getattr(self.gallery, "update_metadata", None)
        if callable(update_metadata):
            update_metadata(identity_id, metadata, save=False)
            self._request_gallery_save()
        log.info(
            "[AgeStable] 年龄稳定: id=%s age=%s count=%s/%s ratio=%.3f",
            identity_id, best_age, best_count, len(snapshot), ratio,
        )
        with self._stability_lock:
            self._age_stability_history.pop(identity_id, None)
        return result

    def _update_gender_stability(self, identity_id: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """累计性别预测，达到稳定后持久化到 Gallery（可被后续刷新更新）。"""
        gender = str(attrs.get("gender") or "").strip()
        gender_score = float(attrs.get("gender_score") or 0.0)
        if not gender or gender_score < 0.5:
            return {}

        with self._stability_lock:
            history = self._gender_stability_history.setdefault(
                identity_id,
                deque(maxlen=self.gender_stable_window),
            )
            history.append((gender, gender_score))
            snapshot = list(history)

        counts = Counter(g for g, _score in snapshot)
        best_gender, best_count = counts.most_common(1)[0]
        ratio = best_count / max(1, len(snapshot))
        if best_count < self.gender_stable_min_count or ratio < self.gender_stable_min_ratio:
            return {}

        scores = [score for g, score in snapshot if g == best_gender]
        result = {
            "gender": best_gender,
            "gender_score": round(float(np.mean(scores)), 4),
            "gender_stable": True,
            "gender_stable_source": "stable",
            "gender_stable_samples": best_count,
        }
        # 内存缓存
        with self._stability_lock:
            stable = self._stable_attrs_cache.setdefault(identity_id, {})
            stable.update(result)
        # 持久化到 Gallery
        metadata = {
            "fixed_gender": best_gender,
            "fixed_gender_score": result["gender_score"],
            "fixed_gender_samples": best_count,
            "fixed_gender_window": len(snapshot),
            "fixed_gender_ratio": round(ratio, 4),
            "fixed_gender_source": "stable",
            "fixed_gender_at": datetime.now().isoformat(),
            "stable_gender": best_gender,
            "stable_gender_score": result["gender_score"],
            "stable_gender_samples": best_count,
            "stable_gender_window": len(snapshot),
            "stable_gender_ratio": round(ratio, 4),
            "stable_gender_source": "stable",
            "stable_gender_at": datetime.now().isoformat(),
        }
        update_metadata = getattr(self.gallery, "update_metadata", None)
        if callable(update_metadata):
            update_metadata(identity_id, metadata, save=False)
            self._request_gallery_save()
        log.info(
            "[GenderStable] 性别稳定: id=%s gender=%s count=%s/%s ratio=%.3f",
            identity_id, best_gender, best_count, len(snapshot), ratio,
        )
        with self._stability_lock:
            self._gender_stability_history.pop(identity_id, None)
        return result

    def draw_recognition(self, frame: np.ndarray, result: Dict[str, Any],
                          target_w: int = 1280, target_h: int = 720) -> np.ndarray:
        """绘制身份、相似度、年龄和性别标注。先缩到目标分辨率再绘制。"""
        H, W = frame.shape[:2]
        scale_x = target_w / W
        scale_y = target_h / H
        annotated = cv2.resize(frame, (target_w, target_h))
        for face in result.get("faces", []):
            x1 = int(face["bbox"][0] * scale_x)
            y1 = int(face["bbox"][1] * scale_y)
            x2 = int(face["bbox"][2] * scale_x)
            y2 = int(face["bbox"][3] * scale_y)
            color = (0, 255, 0) if face.get("is_known") else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            identity_id = face.get("identity_id") or ""
            label = face.get("name") or identity_id or ""
            sim = float(face.get("similarity", 0.0) or 0.0)
            attr_label = self.gender_age.format_attr(face)
            lines = [f"ID: {label}", f"SIM: {sim:.2f}"]
            if attr_label:
                lines.append(attr_label)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            padding = 4
            line_gap = 2
            widths, heights, baselines = [], [], []
            for ln in lines:
                (w, h), b = cv2.getTextSize(ln, font, font_scale, thickness)
                widths.append(w); heights.append(h); baselines.append(b)
            tw = max(widths)
            th = sum(heights) + line_gap * (len(lines) - 1)
            baseline = baselines[-1]

            text_x = x1
            text_y = max(y1 - 6, th + baseline + padding * 2)
            bg_top = text_y - th - baseline - padding
            bg_bottom = text_y + baseline + padding
            bg_right = min(annotated.shape[1] - 1, text_x + tw + padding * 2)
            cv2.rectangle(annotated, (text_x, bg_top), (bg_right, bg_bottom), color, -1)
            y = text_y - th - baseline + heights[0] + baselines[0]
            for i, ln in enumerate(lines):
                cv2.putText(annotated, ln, (text_x + padding, y),
                            font, font_scale, (0, 0, 0), thickness)
                if i < len(lines) - 1:
                    y += heights[i + 1] + line_gap
        cloth_data = result.get("cloth", {}) 
        if cloth_data:  
            cloth_color = (255, 255, 0)  # 青色，区别于人脸框的绿色和橙色   
            driver_cloth = cloth_data.get("driver")
            if driver_cloth and driver_cloth.get("bbox"):
                x1 = int(driver_cloth["bbox"][0] * scale_x)
                y1 = int(driver_cloth["bbox"][1] * scale_y)
                x2 = int(driver_cloth["bbox"][2] * scale_x)
                y2 = int(driver_cloth["bbox"][3] * scale_y)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), cloth_color, 2)
            passenger_cloth = cloth_data.get("passenger")      
            if passenger_cloth and passenger_cloth.get("bbox"):
                x1 = int(passenger_cloth["bbox"][0] * scale_x)
                y1 = int(passenger_cloth["bbox"][1] * scale_y)
                x2 = int(passenger_cloth["bbox"][2] * scale_x)
                y2 = int(passenger_cloth["bbox"][3] * scale_y)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), cloth_color, 2)
        return annotated

    def summary(self) -> str:
        """返回摘要信息。"""
        return self.gallery.summary()

    # ── 录制 ──────────────────────────────────────────────────────

    def start_recording(self, output_dir: str = "recordings"):
        """开始录制原始摄像头帧到 MP4 文件，同时保存座椅参数 JSON。"""
        if self._recorder is not None:
            log.warning("录制已经开始，忽略重复 start_recording")
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir, f"{ts}.mp4")
            h, w, _ = self._last_frame_shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 25.0
            if fps <= 0:
                fps = 25.0
            self._recorder = cv2.VideoWriter(path, fourcc, fps, (w, h))
            self._recording_frames = 0
            log.info("录制开始: %s (%dx%d @ %.1ffps, max=%d)", path, w, h, fps, self._recording_max)

            # 保存座椅参数 JSON（同名不同后缀）
            seat_path = os.path.join(output_dir, f"{ts}.json")
            he = self.height_estimator
            seat_data = {
                "can_received": he._can_received if he else False,
                "seat": dict(he._seat) if he else {},
                "recorded_at": datetime.now().isoformat(),
            }
            with open(seat_path, "w", encoding="utf-8") as f:
                json.dump(seat_data, f, ensure_ascii=False, indent=2)
            log.info("座椅参数已保存: %s", seat_path)
        except Exception as e:
            log.exception("start_recording 失败")
            raise

    def stop_recording(self):
        """停止录制并释放资源。"""
        if self._recorder is None:
            return
        try:
            self._recorder.release()
        except Exception:
            log.exception("stop_recording release 失败")
        self._recorder = None
        log.info("录制停止: 共 %d 帧", self._recording_frames)

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    @property
    def recording_frames(self) -> int:
        return self._recording_frames
