#!/usr/bin/env python3
"""weather26 随机森林模型加载与推理。"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger_config import get_logger
from src.utils.sklearn_compat_loader import load_model_compat

logger = get_logger(__name__)


@dataclass
class ModelPredictionResult:
    driver_temp: float
    passenger_temp: float
    wind_speed: str
    air_mode: str
    inference_time_ms: float = 0.0
    model_type: str = "rf_seres_weather26"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "driver_temp": self.driver_temp,
            "passenger_temp": self.passenger_temp,
            "wind_speed": self.wind_speed,
            "air_mode": self.air_mode,
            "inference_time_ms": self.inference_time_ms,
            "model_type": self.model_type,
        }


class BaseModelLoader:
    def __init__(self, model_dir: Optional[str] = None, feature_config: Optional[str] = None):
        project_root = Path(__file__).resolve().parents[3]
        self.model_dir = Path(model_dir or project_root / "models" / "weather26")
        if not self.model_dir.is_absolute():
            self.model_dir = project_root / self.model_dir
        self.feature_config_path = Path(feature_config or self.model_dir / "manifest.json")
        if not self.feature_config_path.is_absolute():
            self.feature_config_path = project_root / self.feature_config_path
        self._feature_columns: List[str] = []
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_time = 0.0

    def load_all(self) -> bool:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return self._loaded

    def get_error(self) -> Optional[str]:
        return self._load_error

    def get_feature_columns(self) -> List[str]:
        return self._feature_columns.copy()

    def get_model_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, features_df: pd.DataFrame) -> ModelPredictionResult:
        raise NotImplementedError


class RandomForestModelLoader(BaseModelLoader):
    """仅加载 rf_seres_weather26_delivery 三模型交付包。"""

    def __init__(self, model_dir: Optional[str] = None,
                 feature_config: Optional[str] = None,
                 package_mode: str = "weather26"):
        if package_mode != "weather26":
            raise ValueError(f"仅支持 weather26 模型包，收到: {package_mode}")
        super().__init__(model_dir, feature_config)
        self.package_mode = "weather26"
        self._manifest: Optional[Dict[str, Any]] = None
        self._assets: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_all(self) -> bool:
        if self._loaded:
            return True
        started = time.time()
        try:
            manifest_path = self.model_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"weather26 manifest不存在: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("package_format") != "rf_seres_weather26_delivery":
                raise ValueError("模型包格式必须为 rf_seres_weather26_delivery")
            features = list(manifest.get("external_feature_order", []))
            if len(features) != 26 or len(set(features)) != 26:
                raise ValueError("weather26外部输入必须是26个不重复字段")

            assets: Dict[str, Dict[str, Any]] = {}
            for task in ("temperature", "wind", "mode"):
                spec = manifest["tasks"][task]
                model_path = self.model_dir / spec["model_file"]
                expected_sha = spec.get("model_sha256")
                if expected_sha and self._sha256(model_path) != expected_sha:
                    raise ValueError(f"{task}模型SHA-256不一致: {model_path.name}")
                internal_features = list(spec["internal_features"])
                model = load_model_compat(model_path)
                n_features = getattr(model, "n_features_in_", None)
                if n_features is not None and int(n_features) != len(internal_features):
                    raise ValueError(f"{task}模型维度与manifest不一致")
                assets[task] = {"features": internal_features, "model": model}
            classes = np.asarray(assets["mode"]["model"].classes_, dtype=int)
            if not np.array_equal(classes, np.arange(1, 8)):
                raise ValueError("模式森林classes_必须完整包含1至7")

            self._manifest = manifest
            self._assets = assets
            self._feature_columns = features
            self._loaded = True
            self._load_time = time.time() - started
            logger.info("weather26随机森林加载完成: 26维输入、三模型")
            return True
        except Exception as exc:
            self._load_error = f"weather26模型加载失败: {exc}"
            logger.error(self._load_error, exc_info=True)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "error": self._load_error,
            "load_time_sec": self._load_time,
            "model_dir": str(self.model_dir),
            "package_mode": "weather26",
            "package_format": self._manifest.get("package_format") if self._manifest else None,
            "feature_count": len(self._feature_columns),
            "models": {
                task: {
                    "loaded": asset.get("model") is not None,
                    "type": type(asset["model"]).__name__,
                    "feature_count": len(asset["features"]),
                }
                for task, asset in self._assets.items()
            },
        }

    def predict(self, features_df: pd.DataFrame) -> ModelPredictionResult:
        if not self._loaded:
            raise RuntimeError("weather26模型未加载")
        missing = [c for c in self._feature_columns if c not in features_df.columns]
        if missing:
            raise ValueError(f"输入缺少weather26特征列: {missing}")
        numeric = features_df[self._feature_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError("weather26输入包含缺失、非数值或非有限值")
        for col in ("主驾PMV", "副驾PMV"):
            numeric[col] = numeric[col].clip(-5.0, 5.0)
        if not np.isin(numeric["天气状况"].to_numpy(float), [1.0, 2.0, 3.0, 4.0]).all():
            raise ValueError("weather26天气状况必须是1、2、3或4")

        started = time.time()
        temp_asset = self._assets["temperature"]
        raw_temp = np.asarray(temp_asset["model"].predict(
            numeric[temp_asset["features"]].to_numpy(float)), dtype=float)
        if raw_temp.ndim != 2 or raw_temp.shape[1] != 2:
            raise ValueError("温度森林未输出主驾和副驾两列")

        wind_frame = numeric.copy()
        cabin = numeric["内温（整车）"].to_numpy(float)
        wind_frame["内部热负荷差"] = 0.5 * (
            np.abs(cabin - raw_temp[:, 0]) + np.abs(cabin - raw_temp[:, 1]))
        wind_asset = self._assets["wind"]
        raw_wind = np.asarray(wind_asset["model"].predict(
            wind_frame[wind_asset["features"]].to_numpy(float)), dtype=float)
        mode_asset = self._assets["mode"]
        raw_mode = np.asarray(mode_asset["model"].predict(
            numeric[mode_asset["features"]].to_numpy(float)), dtype=int)
        if not np.isin(raw_mode, np.arange(1, 8)).all():
            raise ValueError("模式森林输出超出1至7")

        temperature = np.clip(np.rint(raw_temp * 2.0) / 2.0, 16.0, 31.0)
        return ModelPredictionResult(
            driver_temp=float(temperature[0, 0]),
            passenger_temp=float(temperature[0, 1]),
            wind_speed=str(int(np.clip(np.rint(raw_wind)[0], 1, 9))),
            air_mode=str(int(raw_mode[0])),
            inference_time_ms=(time.time() - started) * 1000,
        )


def load_models(model_dir: Optional[str] = None,
                package_mode: str = "weather26") -> RandomForestModelLoader:
    loader = RandomForestModelLoader(model_dir=model_dir, package_mode=package_mode)
    loader.load_all()
    return loader
