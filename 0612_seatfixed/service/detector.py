"""
service/detector.py
==================
InsightFace 人脸检测与特征提取
支持摄像头视频流和图片文件两种输入

所有可配置参数通过环境变量控制：
  DETECTOR_CTX_ID      InsightFace 设备：-1=CPU, 0=GPU0, 1=GPU1 (默认 0)
  DETECTOR_SIZE        检测输入尺寸 (默认 640)
  DETECTOR_MIN_SCORE   最低检测阈值 (默认 0)
"""

from __future__ import annotations

import os
import time
import sys
import logging
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from insightface.app import FaceAnalysis
from core.gallery import normalize_emb

MODEL_NAME = "buffalo_l"
DEFAULT_DET_SIZE = int(os.getenv("DETECTOR_SIZE", "640"))
DEFAULT_CTX_ID = int(os.getenv("DETECTOR_CTX_ID", "0"))
DEFAULT_MIN_DET_SCORE = float(os.getenv("DETECTOR_MIN_SCORE", "0"))

log = logging.getLogger("detector")


@dataclass
class DetectedFace:
    """
    单个人脸检测结果。

    emb : 归一化后的 512 维 embedding
    bbox: 人脸框 (x1, y1, x2, y2)
    kps : 5个关键点 (10维) 或 None
    score: 检测置信度
    xc  : 人脸中心 x 坐标
    yc  : 人脸中心 y 坐标
    """
    emb: np.ndarray
    bbox: tuple
    kps: Optional[np.ndarray]
    score: float
    xc: float
    yc: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [float(v) for v in self.bbox],
            "kps": self.kps.tolist() if self.kps is not None else None,
            "score": float(self.score),
            "xc": float(self.xc),
            "yc": float(self.yc),
        }


class FaceDetector:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        det_size: int = DEFAULT_DET_SIZE,
        ctx_id: int = DEFAULT_CTX_ID,
        min_det_score: float = DEFAULT_MIN_DET_SCORE,
        providers: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        if isinstance(det_size, int):
            det_size = (det_size, det_size)
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.min_det_score = float(min_det_score)
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self._app: Optional[FaceAnalysis] = None
        self._model_loaded: bool = False
        self._load_lock = threading.Lock()

    def ensure(self):
        """加载模型（线程安全）。"""
        if self._model_loaded:
            return
        with self._load_lock:
            if self._model_loaded:
                return
            t0 = time.time()
            os.environ.setdefault("ORTProviders", ",".join(self.providers))
            self._app = FaceAnalysis(name=self.model_name, allowed_modules=["detection", "recognition"])
            self._app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self._model_loaded = True
            log.info("模型已加载: %s ctx_id=%s det_size=%s (%.1fs)", self.model_name, self.ctx_id, self.det_size, time.time()-t0)

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            self.ensure()
        return self._app

    def detect_image(
        self,
        image: np.ndarray | str | Path,
        filter_dashboard: bool = True,
    ) -> List[DetectedFace]:
        """
        检测图片中所有人脸。

        Args:
            image: cv2 图片数组 (BGR) 或图片路径
            filter_dashboard: 是否过滤仪表盘区域

        Returns:
            List[DetectedFace]，按 xc 从左到右排序
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            return []

        H, W = image.shape[:2]
        faces = self.app.get(image)

        results = []
        half_h = H // 2
        for face in faces:
            det_score = float(face.det_score)
            if det_score < self.min_det_score:
                continue

            xc = (face.bbox[0] + face.bbox[2]) / 2
            yc = (face.bbox[1] + face.bbox[3]) / 2

            # 过滤仪表盘区域（方向盘等干扰）
            if filter_dashboard and yc < half_h and W * (1/3) <= xc <= W * (2/3):
                continue

            emb = normalize_emb(face.embedding)
            results.append(DetectedFace(
                emb=emb,
                bbox=tuple(face.bbox),
                kps=face.kps,
                score=det_score,
                xc=xc,
                yc=yc,
            ))

        # 按 xc 从左到右排序
        results.sort(key=lambda f: f.xc)
        return results

    def detect_faces_dicts(
        self,
        image_bgr: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        兼容旧 main.py 的接口：返回 list[dict] 格式。

        每张脸返回：
            {
                "bbox": [x1, y1, x2, y2],
                "kps": [[x,y], ...] or None,
                "score": float,
                "det_score": float,
                "gender": int,
                "age": int,
                "embedding": np.ndarray or None,
            }
        """
        if image_bgr is None:
            return []

        if len(image_bgr.shape) < 2:
            return []

        h, w = image_bgr.shape[:2]

        raw_faces = self.app.get(image_bgr)

        results = []
        for face in raw_faces:
            det_score = float(face.det_score)
            if det_score < self.min_det_score:
                continue

            bbox = face.bbox
            if bbox is None:
                continue

            x1 = max(0, min(int(round(bbox[0])), w - 1))
            y1 = max(0, min(int(round(bbox[1])), h - 1))
            x2 = max(0, min(int(round(bbox[2])), w - 1))
            y2 = max(0, min(int(round(bbox[3])), h - 1))
            bbox_clipped = [x1, y1, x2, y2]

            kps = getattr(face, "kps", None)
            if kps is not None:
                try:
                    kps = np.asarray(kps, dtype=np.float32).tolist()
                except Exception:
                    kps = None

            gender = -1
            age = -1
            try:
                gender = int(getattr(face, "gender", -1) or -1)
            except Exception:
                pass
            try:
                age = int(getattr(face, "age", -1) or -1)
            except Exception:
                pass

            embedding = getattr(face, "embedding", None)
            if embedding is not None:
                try:
                    embedding = np.asarray(embedding, dtype=np.float32)
                except Exception:
                    embedding = None

            results.append({
                "bbox": bbox_clipped,
                "kps": kps,
                "score": det_score,
                "det_score": det_score,
                "gender": gender,
                "age": age,
                "embedding": embedding,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def detect_frame(
        self,
        frame: np.ndarray,
        filter_dashboard: bool = True,
    ) -> List[DetectedFace]:
        """检测视频帧（与 detect_image 相同接口）。"""
        return self.detect_image(frame, filter_dashboard=filter_dashboard)


    @staticmethod
    def normalize_emb(emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def draw_faces(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        labels: Optional[List[str]] = None,
        show_unknown: bool = True,
    ) -> np.ndarray:
        """
        在图片上绘制人脸框和标签。

        Args:
            image: cv2 图片
            faces: 检测到的人脸列表
            labels: 每个人的标签列表（与 faces 一一对应）
            show_unknown: 是否标注未知身份
        """
        display = image.copy()
        POS_COLORS = {"left": (255, 140, 0), "right": (30, 144, 255)}
        COLOR_MATCHED = (76, 175, 80)
        COLOR_UNMATCHED = (244, 67, 54)

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            W = image.shape[1]
            position = "left" if face.xc < W / 2 else "right"
            pos_color = POS_COLORS[position]

            if labels and i < len(labels):
                label = labels[i]
                is_known = bool(label)
                color = COLOR_MATCHED if is_known else COLOR_UNMATCHED
                text = f"{position.upper()} | {label} ({face.score:.2f})"
            else:
                color = pos_color
                text = f"{position.upper()} | score={face.score:.2f}"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, text, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return display

    def to_dict(self, faces: List[DetectedFace]) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in faces]

