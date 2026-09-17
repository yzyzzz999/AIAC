# gender_age_recognizer.py
# -*- coding: utf-8 -*-
"""
年龄 / 性别属性识别适配器。

用服务端人脸检测 bbox 裁剪人脸，通过 gender_age_backends 模块
（默认 ONNX 后端）做属性预测。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from service.gender_age_backends import create_gender_age_backend


PROJECT_ROOT = Path(__file__).resolve().parent
ATTR_CODE_DIR = PROJECT_ROOT
DEFAULT_MODEL_DIR = PROJECT_ROOT / "FLIP-base-32"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def crop_face_pil(image_bgr: np.ndarray, bbox: Any, expand_ratio: float = 0.15) -> Image.Image:
    """从 BGR 原图按 bbox 裁剪人脸，输出 RGB PIL.Image。"""
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        raise ValueError(f"Invalid bbox: {bbox}")

    ex = int(bw * expand_ratio)
    ey = int(bh * expand_ratio)

    x1 = max(0, x1 - ex)
    y1 = max(0, y1 - ey)
    x2 = min(w, x2 + ex)
    y2 = min(h, y2 + ey)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Expanded bbox out of bounds: {[x1, y1, x2, y2]}")

    crop_bgr = image_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        raise ValueError(f"Empty crop for bbox: {bbox}")

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


class GenderAgeRecognizer:
    """懒加载 FLIP 属性模型，给根目录识别流程调用。"""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        enabled: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.backend = (backend or os.getenv("GENDER_AGE_BACKEND") or "onnx").strip().lower()
        self.enabled = enabled
        self._predictor = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()
        self._loading = False

    @property
    def ready(self) -> bool:
        return self.enabled and self._predictor is not None and self._load_error is None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def ensure(self, blocking: bool = True):
        """首次预测时加载模型，避免拖慢服务启动。"""
        if not self.enabled or self._predictor is not None or self._load_error is not None:
            return

        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            return
        try:
            if self._predictor is not None or self._load_error is not None:
                return
            self._loading = True
            try:
                self._predictor = create_gender_age_backend(
                    backend=self.backend,
                    model_dir=str(self.model_dir),
                    device=self.device,
                )
                actual_device = getattr(self._predictor, "device", self.device or "auto")
                actual_backend = getattr(self._predictor, "backend_name", self.backend)
                print(
                    f"[GenderAgeRecognizer] 属性模型已加载: {self.model_dir} "
                    f"| backend={actual_backend} | device={actual_device}"
                )
            except Exception as exc:
                self._load_error = str(exc)
                print(f"[GenderAgeRecognizer] 属性模型加载失败: {self._load_error}")
            finally:
                self._loading = False
        finally:
            self._lock.release()

    def preload_async(self):
        """后台预热属性模型，避免第一帧预测阻塞视频线程。"""
        if not self.enabled or self._predictor is not None or self._load_error is not None:
            return
        threading.Thread(target=self.ensure, daemon=True).start()

    def predict_face(self, image_bgr: np.ndarray, bbox: Any) -> Dict[str, Any]:
        """对单个人脸 bbox 预测性别和年龄段。"""
        if not self.enabled:
            return {"attribute_enabled": False}

        self.ensure(blocking=False)
        if self._predictor is None:
            return {
                "attribute_enabled": False,
                "attribute_error": self._load_error or ("属性模型加载中" if self._loading else "属性模型未加载"),
            }

        try:
            face_pil = crop_face_pil(image_bgr, bbox)
            attr_result = self._predictor.predict_all(face_pil)
            return {
                "attribute_enabled": True,
                "attribute_backend": self.backend,
                "gender": str(_safe_get(attr_result, "gender", "label", default="")),
                "gender_score": _safe_float(_safe_get(attr_result, "gender", "score", default=0.0)),
                "gender_probs": _safe_get(attr_result, "gender", "probs", default={}),
                "age_group": str(_safe_get(attr_result, "age", "label", default="")),
                "age_score": _safe_float(_safe_get(attr_result, "age", "score", default=0.0)),
                "age_probs": _safe_get(attr_result, "age", "probs", default={}),
            }
        except Exception as exc:
            return {
                "attribute_enabled": False,
                "attribute_error": str(exc),
            }

    @staticmethod
    def format_attr(face: Dict[str, Any]) -> str:
        """用于 OpenCV 标注的短文本。"""
        gender = face.get("gender") or ""
        age_group = face.get("age_group") or ""
        parts = [p for p in (gender, age_group) if p]
        return " ".join(parts)
