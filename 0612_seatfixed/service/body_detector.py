"""
service/body_detector.py
========================
YOLOv8n-pose 人体检测器，用于座位占位兜底。
当人脸检测失败但人体可见时，保持座位"有人"状态。

通过关键点置信度过滤误检（如座椅），并利用肩部中点判定左右位置。
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from utils import select_front_row_bodies

log = logging.getLogger("body_detector")

DEFAULT_MODEL_PATH = os.getenv(
    "BODY_DET_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "body" / "yolov8n-pose.pt"),
)
DEFAULT_MIN_CONFIDENCE = float(os.getenv("BODY_DET_MIN_CONFIDENCE", "0.5"))

# COCO 17点关键点索引
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
# 肩部关键点 — 真人必须有肩，座椅头枕不会有
SHOULDER_KPS: List[int] = [KP["left_shoulder"], KP["right_shoulder"]]
MIN_SHOULDER_KP_COUNT: int = 1
MIN_SHOULDER_KP_CONF: float = 0.35


class BodyDetector:
    """轻量人体检测器，用 YOLOv8n-pose 检测上半身存在。"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._model = None
        self._loaded = False

    def ensure(self):
        if self._loaded:
            return
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        import torch
        if torch.cuda.is_available():
            self._model.to("cuda")
        self._loaded = True
        log.info("BodyDetector loaded: %s", self.model_path)

    @property
    def ready(self) -> bool:
        return self._loaded

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测帧中的人体，利用肩部关键点严格过滤误检（座椅等）。

        核心逻辑：真人一定有肩部关键点，座椅头枕不会有。无关键点数据
        时宁可漏检也不误报。

        Returns:
            [{"bbox": [x1,y1,x2,y2], "score": float, "xc": float, "yc": float,
              "shoulder_mid": (float,float)|None}, ...]
        """
        if not self._loaded:
            self.ensure()

        results = self._model(frame, verbose=False)
        bodies = []
        if not results or not results[0].boxes:
            return bodies

        kps = results[0].keypoints
        if kps is None or not hasattr(kps, "conf") or kps.conf is None:
            # 没有关键点数据 → 无法验证是真人，全部丢弃
            return bodies

        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            if cls != 0:
                continue
            conf = float(box.conf[0])
            if conf < self.min_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            try:
                kp_conf = kps.conf[i].cpu().numpy()  # shape (17,)
                kp_xy = kps.xy[i].cpu().numpy()      # shape (17, 2)

                # 必须检测到肩部关键点（真人标志）
                shoulder_count = int(sum(
                    1 for idx in SHOULDER_KPS if kp_conf[idx] > MIN_SHOULDER_KP_CONF
                ))
                if shoulder_count < MIN_SHOULDER_KP_COUNT:
                    log.info("[BodyDet] 丢弃无肩关键点: conf=%.3f shoulder_kps=%d",
                             conf, shoulder_count)
                    continue

                # 肩部中点：只用置信度达标的那一侧
                left_ok = kp_conf[KP["left_shoulder"]] > MIN_SHOULDER_KP_CONF
                right_ok = kp_conf[KP["right_shoulder"]] > MIN_SHOULDER_KP_CONF
                if left_ok and right_ok:
                    sx = float((kp_xy[KP["left_shoulder"]][0] + kp_xy[KP["right_shoulder"]][0]) / 2)
                    sy = float((kp_xy[KP["left_shoulder"]][1] + kp_xy[KP["right_shoulder"]][1]) / 2)
                elif left_ok:
                    sx = float(kp_xy[KP["left_shoulder"]][0])
                    sy = float(kp_xy[KP["left_shoulder"]][1])
                else:
                    sx = float(kp_xy[KP["right_shoulder"]][0])
                    sy = float(kp_xy[KP["right_shoulder"]][1])
                shoulder_mid = (sx, sy)

            except Exception:
                # 关键点提取异常 → 无法验证，丢弃
                continue

            xc = shoulder_mid[0]
            yc = shoulder_mid[1]

            bodies.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": conf,
                "xc": xc,
                "yc": yc,
                "shoulder_mid": shoulder_mid,
                "kps_xy": kp_xy.tolist(),      # 17点坐标 [[x,y]*17]
                "kps_conf": kp_conf.tolist(),  # 17点置信度 [conf*17]
            })

        return bodies

    def check_seats(self, frame: np.ndarray) -> Dict[str, bool]:
        """检测前排左右座位是否有人体。"""
        bodies = self.detect(frame)
        h, w = frame.shape[:2]
        return select_front_row_bodies(bodies, w, h)
