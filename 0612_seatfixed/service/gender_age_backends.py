# gender_age_backends.py
# -*- coding: utf-8 -*-
"""
性别/年龄属性识别后端。

默认使用 ONNXRuntime 后端，减少线上运行时对 torch 的依赖。
torch/transformers 版本保留为开发和对照测试后端。TensorRT 后端先保留统一入口，
方便后续在目标 NVIDIA 设备上接入 engine。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
from PIL import Image


GENDER_LABELS = ["male", "female"]
AGE_LABELS = ["12-17", "18-40", "41-59", "60-74", "75+"]
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _ensure_pil_image(face_pil: Image.Image) -> Image.Image:
    if face_pil is None:
        raise ValueError("face_pil is None")
    if not isinstance(face_pil, Image.Image):
        raise TypeError(f"face_pil must be PIL.Image.Image, got {type(face_pil)}")
    if face_pil.mode != "RGB":
        face_pil = face_pil.convert("RGB")
    w, h = face_pil.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid face image size: {(w, h)}")
    return face_pil


def _format_result(probs: Iterable[float], labels: list[str]) -> Dict[str, Any]:
    probs = [float(v) for v in probs]
    if len(probs) != len(labels):
        raise ValueError(f"Length mismatch: len(probs)={len(probs)}, len(labels)={len(labels)}")
    best_idx = int(np.argmax(probs))
    return {
        "label": labels[best_idx],
        "score": float(probs[best_idx]),
        "probs": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def preprocess_clip_image(face_pil: Image.Image, image_size: int = CLIP_IMAGE_SIZE) -> np.ndarray:
    """CLIP 图像预处理，输出 NCHW float32。"""
    face_pil = _ensure_pil_image(face_pil)
    w, h = face_pil.size
    scale = image_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = face_pil.resize((new_w, new_h), resampling)

    left = max(0, (new_w - image_size) // 2)
    top = max(0, (new_h - image_size) // 2)
    image = image.crop((left, top, left + image_size, top + image_size))

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[None, ...].astype(np.float32)


def compute_quality_features(face_pil: Image.Image, modality: str = "rgb") -> np.ndarray:
    """Quality and modality features used by the 2026-08-04 model."""
    face_pil = _ensure_pil_image(face_pil)
    rgb = np.asarray(face_pil.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    mean = float(gray.mean() / 255.0)
    std = float(gray.std() / 255.0)
    p05 = float(np.percentile(gray, 5) / 255.0)
    p50 = float(np.percentile(gray, 50) / 255.0)
    p95 = float(np.percentile(gray, 95) / 255.0)
    contrast = float(p95 - p05)
    blur = float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0, 10.0) / 10.0)
    dark_ratio = float((gray < 45).mean())
    bright_ratio = float((gray > 220).mean())
    r_mean = float(rgb[:, :, 0].mean() / 255.0)
    g_mean = float(rgb[:, :, 1].mean() / 255.0)
    b_mean = float(rgb[:, :, 2].mean() / 255.0)
    is_rgb = 1.0 if str(modality).lower() == "rgb" else 0.0
    is_nir = 1.0 if str(modality).lower() == "nir" else 0.0
    low_light = 1.0 if (mean < 0.33 or dark_ratio > 0.35 or is_nir > 0.5) else 0.0
    return np.asarray([
        mean,
        std,
        p05,
        p50,
        p95,
        contrast,
        blur,
        dark_ratio,
        bright_ratio,
        r_mean,
        g_mean,
        b_mean,
        r_mean - g_mean,
        b_mean - g_mean,
        float(w / max(h, 1)),
        is_rgb,
        is_nir,
        low_light,
    ], dtype=np.float32)


class TorchGenderAgeBackend:
    """当前线上使用的 torch/transformers FLIP 后端。"""

    backend_name = "torch"

    def __init__(self, model_dir: str | Path, device: Optional[str] = None):
        model_dir = Path(model_dir)
        flip_attr_path = Path(__file__).resolve().parent / "flip_attr.py"
        if not flip_attr_path.is_file():
            raise FileNotFoundError(f"找不到属性识别文件: {flip_attr_path}")

        spec = importlib.util.spec_from_file_location("aiac_flip_attr", flip_attr_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载属性识别模块: {flip_attr_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.predictor = module.FLIPAttrPredictor(model_dir=str(model_dir), device=device)
        self.device = getattr(self.predictor, "device", device or "auto")

    def predict_all(self, face_pil: Image.Image) -> Dict[str, Any]:
        return self.predictor.predict_all(face_pil)


class OnnxGenderAgeBackend:
    """ONNX 后端：只跑图像 encoder，文本特征离线预计算。"""

    backend_name = "onnx"

    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
        onnx_model: Optional[str | Path] = None,
        text_features: Optional[str | Path] = None,
        providers: Optional[list[str]] = None,
    ):
        model_dir = Path(model_dir)
        self.onnx_model = Path(
            onnx_model
            or os.getenv("GENDER_AGE_ONNX_MODEL", "")
            or model_dir / "gender_age_image_encoder.onnx"
        )
        self.text_features_path = Path(
            text_features
            or os.getenv("GENDER_AGE_TEXT_FEATURES", "")
            or model_dir / "gender_age_text_features.npz"
        )
        if not self.onnx_model.is_file():
            raise FileNotFoundError(f"找不到性别年龄 ONNX 模型: {self.onnx_model}")
        if not self.text_features_path.is_file():
            raise FileNotFoundError(f"找不到性别年龄文本特征: {self.text_features_path}")

        import onnxruntime as ort

        if providers is None:
            env_providers = os.getenv("GENDER_AGE_ORT_PROVIDERS") or os.getenv("ORT_PROVIDERS") or ""
            providers = [p.strip() for p in env_providers.split(",") if p.strip()]
        if not providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.onnx_model), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = ",".join(self.session.get_providers())

        data = np.load(self.text_features_path, allow_pickle=False)
        self.gender_features = _l2_normalize(data["gender_features"].astype(np.float32), axis=1)
        self.age_features = _l2_normalize(data["age_features"].astype(np.float32), axis=1)
        self.logit_scale = float(data["logit_scale"][0]) if "logit_scale" in data else 100.0

    def encode_image_features(self, face_pil: Image.Image) -> np.ndarray:
        pixel_values = preprocess_clip_image(face_pil)
        image_features = self.session.run(None, {self.input_name: pixel_values})[0]
        return _l2_normalize(np.asarray(image_features, dtype=np.float32), axis=1)[0]

    def _predict_probs(self, face_pil: Image.Image, text_features: np.ndarray) -> np.ndarray:
        image_features = self.encode_image_features(face_pil)[None, :]
        logits = self.logit_scale * (image_features @ text_features.T)
        return _softmax(logits[0])

    def predict_all(self, face_pil: Image.Image) -> Dict[str, Any]:
        gender_probs = self._predict_probs(face_pil, self.gender_features)
        age_probs = self._predict_probs(face_pil, self.age_features)
        return {
            "gender": _format_result(gender_probs, GENDER_LABELS),
            "age": _format_result(age_probs, AGE_LABELS),
        }


class FlipMlpGenderAgeBackend(OnnxGenderAgeBackend):
    """FLIP image encoder + supervised MLP heads trained on in-car data."""

    backend_name = "flip_mlp"

    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
        onnx_model: Optional[str | Path] = None,
        text_features: Optional[str | Path] = None,
        providers: Optional[list[str]] = None,
        mlp_model: Optional[str | Path] = None,
    ):
        super().__init__(
            model_dir=model_dir,
            device=device,
            onnx_model=onnx_model,
            text_features=text_features,
            providers=providers,
        )
        model_dir = Path(model_dir)
        self.mlp_model_path = Path(
            mlp_model
            or os.getenv("GENDER_AGE_MLP_MODEL", "")
            or model_dir / "gender_age_ordinal_adaptive_v4.joblib"
        )
        if not self.mlp_model_path.is_file():
            raise FileNotFoundError(
                f"找不到车内 MLP 属性模型: {self.mlp_model_path}。"
                "请先运行 train_flip_mlp_heads.py 训练。"
            )
        import joblib
        self.mlp_model = joblib.load(self.mlp_model_path)
        self.gender_model = self.mlp_model["gender_model"]
        self.feature_mode = str(self.mlp_model.get("feature_mode", "flip_only"))
        self.age_strategy = str(self.mlp_model.get("age_strategy", "multiclass"))
        self.age_model = self.mlp_model.get("age_model")
        self.age_binary_model = self.mlp_model.get("age_binary_model")
        self.age_non1840_model = self.mlp_model.get("age_non1840_model")
        self.age_ordinal_models = list(self.mlp_model.get("age_ordinal_models", []))
        self.age_boundary_models = dict(self.mlp_model.get("age_boundary_models", {}))
        self.reliability_gates = dict(self.mlp_model.get("reliability_gates", {}))
        self.selection = dict(self.mlp_model.get("selection", {}))
        self.gender_labels = list(self.mlp_model.get("gender_labels", GENDER_LABELS))
        self.age_labels = list(self.mlp_model.get("age_labels", AGE_LABELS))
        self.age_binary_labels = list(self.mlp_model.get("age_binary_labels", ["18-40", "non_18-40"]))
        self.device = f"{self.device}+mlp"

    def _make_supervised_features(self, face_pil: Image.Image, modality: str = "rgb") -> np.ndarray:
        image_features = self.encode_image_features(face_pil)
        if self.feature_mode == "flip_quality_v1":
            return np.concatenate([image_features, compute_quality_features(face_pil, modality=modality)], axis=0).astype(np.float32)
        return image_features

    @staticmethod
    def _predict_model(model, features: np.ndarray, labels: list[str]) -> Dict[str, Any]:
        x = features.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            probs_arr = model.predict_proba(x)[0]
            classes = [str(c) for c in getattr(model, "classes_", labels)]
            probs = {label: 0.0 for label in labels}
            for cls, prob in zip(classes, probs_arr):
                if cls in probs:
                    probs[cls] = float(prob)
            best_label = max(probs.items(), key=lambda item: item[1])[0]
            return {
                "label": best_label,
                "score": float(probs[best_label]),
                "probs": probs,
            }

        label = str(model.predict(x)[0])
        return {
            "label": label,
            "score": 1.0,
            "probs": {item: 1.0 if item == label else 0.0 for item in labels},
        }

    def _predict_two_stage_age(self, features: np.ndarray) -> Dict[str, Any]:
        if self.age_binary_model is None or self.age_non1840_model is None:
            return self._predict_model(self.age_model, features, self.age_labels)

        x = features.reshape(1, -1)
        binary_probs_arr = self.age_binary_model.predict_proba(x)[0]
        binary_classes = [str(c) for c in getattr(self.age_binary_model, "classes_", self.age_binary_labels)]
        binary_probs = {label: 0.0 for label in self.age_binary_labels}
        for cls, prob in zip(binary_classes, binary_probs_arr):
            if cls in binary_probs:
                binary_probs[cls] = float(prob)

        non_probs_arr = self.age_non1840_model.predict_proba(x)[0]
        non_classes = [str(c) for c in getattr(self.age_non1840_model, "classes_", self.age_labels)]
        non_probs = {label: 0.0 for label in self.age_labels}
        for cls, prob in zip(non_classes, non_probs_arr):
            if cls in non_probs:
                non_probs[cls] = float(prob)

        probs = {label: 0.0 for label in self.age_labels}
        probs["18-40"] = float(binary_probs.get("18-40", 0.0))
        other_mass = float(binary_probs.get("non_18-40", 0.0))
        for label in self.age_labels:
            if label != "18-40":
                probs[label] = other_mass * float(non_probs.get(label, 0.0))

        total = sum(probs.values())
        if total > 0:
            probs = {label: float(value / total) for label, value in probs.items()}
        label = max(probs.items(), key=lambda item: item[1])[0]
        return {
            "label": label,
            "score": float(probs[label]),
            "probs": probs,
        }

    def _predict_ordinal_age(self, features: np.ndarray) -> Dict[str, Any]:
        greater = np.zeros(max(len(self.age_labels) - 1, 0), dtype=np.float64)
        for entry in self.age_ordinal_models:
            idx = int(entry["threshold_index"])
            model = entry.get("model")
            if model is None:
                greater[idx] = float(entry.get("default_gt_probability", 0.0))
            else:
                result = self._predict_model(model, features, ["le", "gt"])
                greater[idx] = float(result["probs"].get("gt", 0.0))
        if len(greater):
            greater = np.minimum.accumulate(greater)

        values = np.zeros(len(self.age_labels), dtype=np.float64)
        if len(values) == 1:
            values[0] = 1.0
        elif len(values) > 1:
            values[0] = 1.0 - greater[0]
            for idx in range(1, len(values) - 1):
                values[idx] = greater[idx - 1] - greater[idx]
            values[-1] = greater[-1]
        values = np.clip(values, 0.0, 1.0)
        if float(values.sum()) > 0:
            values /= float(values.sum())

        blend = float(self.selection.get("boundary_blend", 0.0))
        min_mass = float(self.selection.get("boundary_min_pair_mass", 0.12))
        label_index = {label: idx for idx, label in enumerate(self.age_labels)}
        for key, model in self.age_boundary_models.items():
            lower, upper = key.split("|", 1)
            if lower not in label_index or upper not in label_index:
                continue
            lower_idx, upper_idx = label_index[lower], label_index[upper]
            pair_mass = float(values[lower_idx] + values[upper_idx])
            if pair_mass < min_mass or blend <= 0:
                continue
            specialist = self._predict_model(model, features, [lower, upper])
            specialist_lower = float(specialist["probs"].get(lower, 0.5))
            original_lower = float(values[lower_idx] / pair_mass) if pair_mass > 0 else 0.5
            lower_ratio = (1.0 - blend) * original_lower + blend * specialist_lower
            values[lower_idx] = pair_mass * lower_ratio
            values[upper_idx] = pair_mass * (1.0 - lower_ratio)

        probs = {label: float(values[idx]) for idx, label in enumerate(self.age_labels)}
        label = max(probs.items(), key=lambda item: item[1])[0]
        return {"label": label, "score": probs[label], "probs": probs}

    @staticmethod
    def _reliability_features(quality: np.ndarray, probs: Dict[str, float]) -> np.ndarray:
        values = sorted((float(value) for value in probs.values()), reverse=True)
        confidence = values[0] if values else 0.0
        margin = values[0] - values[1] if len(values) > 1 else confidence
        entropy = -sum(value * np.log(max(value, 1e-12)) for value in values)
        entropy /= max(np.log(max(len(values), 2)), 1e-12)
        return np.asarray([[
            confidence,
            margin,
            float(entropy),
            float(quality[0]),
            float(quality[1]),
            float(quality[5]),
            float(quality[6]),
            float(quality[7]),
            float(quality[8]),
            float(quality[14]),
            float(quality[15]),
            float(quality[16]),
            float(quality[17]),
        ]], dtype=np.float32)

    def _predict_reliability(
        self, kind: str, quality: np.ndarray, result: Dict[str, Any]
    ) -> float:
        gate = self.reliability_gates.get(kind)
        confidence = float(result.get("score", 0.0))
        if gate is None or self.selection.get("fusion_mode") != "learned":
            return float(np.clip(0.20 + 0.80 * confidence, 0.05, 1.0))
        classes = list(gate[-1].classes_)
        if 1 not in classes:
            return float(np.clip(0.20 + 0.80 * confidence, 0.05, 1.0))
        correct_idx = classes.index(1)
        features = self._reliability_features(quality, result["probs"])
        return float(np.clip(gate.predict_proba(features)[0, correct_idx], 0.05, 1.0))

    def predict_all(self, face_pil: Image.Image) -> Dict[str, Any]:
        modality = os.getenv("GENDER_AGE_MODALITY", "rgb")
        quality = compute_quality_features(face_pil, modality=modality)
        image_features = self.encode_image_features(face_pil)
        features = (
            np.concatenate([image_features, quality], axis=0).astype(np.float32)
            if self.feature_mode == "flip_quality_v1"
            else image_features
        )
        if self.age_strategy.startswith("ordinal"):
            age_result = self._predict_ordinal_age(features)
        elif self.age_strategy.startswith("two_stage"):
            age_result = self._predict_two_stage_age(features)
        else:
            age_result = self._predict_model(self.age_model, features, self.age_labels)
        gender_result = self._predict_model(self.gender_model, features, self.gender_labels)
        gender_result["reliability"] = self._predict_reliability(
            "gender", quality, gender_result
        )
        age_result["reliability"] = self._predict_reliability("age", quality, age_result)
        return {
            "gender": gender_result,
            "age": age_result,
        }


class TensorRTGenderAgeBackend:
    """TensorRT 后端占位：部署到 NVIDIA 设备后接入 engine。"""

    backend_name = "tensorrt"

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TensorRT 后端入口已预留，但当前工程尚未包含 TensorRT engine runner。"
            "请先使用 ONNX 后端验证精度，再在目标设备上将 ONNX 转为 TensorRT engine。"
        )


def create_gender_age_backend(
    backend: str,
    model_dir: str | Path,
    device: Optional[str] = None,
):
    backend = (backend or "onnx").strip().lower()
    if backend in ("torch", "pytorch", "flip"):
        return TorchGenderAgeBackend(model_dir=model_dir, device=device)
    if backend in ("onnx", "ort", "onnxruntime"):
        return OnnxGenderAgeBackend(model_dir=model_dir, device=device)
    if backend in ("flip_mlp", "onnx_mlp", "mlp"):
        return FlipMlpGenderAgeBackend(model_dir=model_dir, device=device)
    if backend in ("trt", "tensorrt"):
        return TensorRTGenderAgeBackend(model_dir=model_dir, device=device)
    raise ValueError(f"不支持的性别年龄后端: {backend}")
