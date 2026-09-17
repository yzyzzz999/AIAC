#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Inference Engine
==============================================
推理引擎模块，负责weather26特征构建和基础模型推理。

职责边界:
- 接收原始信号并构建模型特征
- 调用基础模型进行推理
- 管理滑动窗口缓存
- 提供统一的推理结果格式

接口:
- InferenceEngine: 主推理引擎
- InferenceResult: 推理结果数据结构
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.features.feature_builder import SinglePointFeatureBuilder, WindowFeatureBuilder
from src.core.features.sliding_window_cache import SlidingWindowCache
from src.core.models.base_model_loader import BaseModelLoader
from src.utils.constants import (
    ACTIVE_CAN_SIGNAL_NAMES,
    CAN_SIGNAL_NAMES,
    FACE_SIGNAL_NAMES,
    FEATURE_PREFIXES,
)
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# ============================================================
# 推理结果
# ============================================================
@dataclass
class InferenceResult:
    """统一推理结果"""
    timestamp: float
    driver_temp: Optional[float] = None
    passenger_temp: Optional[float] = None
    wind_speed: Optional[str] = None
    air_mode: Optional[str] = None
    ready: bool = False
    missing: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    mode: str = "single"  # "single" or "window"
    demo_mode: bool = False
    default_signals: List[str] = field(default_factory=list)
    base_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "driver_temp": self.driver_temp,
            "passenger_temp": self.passenger_temp,
            "wind_speed": self.wind_speed,
            "air_mode": self.air_mode,
            "ready": self.ready,
            "missing": self.missing,
            "inference_time_ms": self.inference_time_ms,
            "mode": self.mode,
            "demo_mode": self.demo_mode,
            "default_signals": self.default_signals,
            "base_result": self.base_result,
            "error": self.error,
        }

    def format_output(self) -> str:
        """格式化输出字符串"""
        if not self.ready:
            return f"[未就绪] 缺失信号: {', '.join(self.missing)}"
        return (
            f"driver_temp={self.driver_temp:.1f}°C, "
            f"passenger_temp={self.passenger_temp:.1f}°C, "
            f"wind_speed={self.wind_speed}, "
            f"air_mode={self.air_mode}, "
            f"耗时={self.inference_time_ms:.1f}ms"
        )


# ============================================================
# 推理引擎
# ============================================================
class InferenceEngine:
    """weather26基础推理引擎。"""

    def __init__(
        self,
        model_loader: Optional[BaseModelLoader] = None,
        use_sliding_window: bool = True,
        window_seconds: float = 5.0,
    ):
        self.model_loader = model_loader
        self.use_sliding_window = use_sliding_window
        self.window_seconds = window_seconds

        # 滑动窗口缓存
        self._window_cache: Optional[SlidingWindowCache] = None
        if use_sliding_window:
            self._window_cache = SlidingWindowCache(window_seconds=window_seconds)

        # 特征构建器
        self._single_builder = SinglePointFeatureBuilder()
        self._window_builder: Optional[WindowFeatureBuilder] = None
        if self._window_cache is not None:
            self._window_builder = WindowFeatureBuilder(self._window_cache)

        # 统计
        self._inference_count = 0
        self._error_count = 0

    @property
    def window_cache(self) -> Optional[SlidingWindowCache]:
        """公开只读窗口缓存，供主控制器持续写入实车采样。"""
        return self._window_cache

    # ---------------------------
    # 信号输入
    # ---------------------------
    def add_can_samples(self, samples: Dict[str, Any]) -> None:
        """添加CAN信号采样点到滑动窗口"""
        if self._window_cache is not None:
            self._window_cache.add_samples(samples)

    def add_can_sample(self, signal_name: str, value: Any) -> None:
        """添加单个CAN信号采样点"""
        if self._window_cache is not None:
            self._window_cache.add_sample(signal_name, value)

    def set_person_signals(self, signals: Dict[str, Any]) -> None:
        """设置人员信号"""
        if self._window_cache is not None:
            self._window_cache.set_person_signals(signals)

    # ---------------------------
    # 信号就绪检查
    # ---------------------------
    def check_signals_ready_single(
        self,
        can_features: Dict[str, Any],
        face_features: Dict[str, Any],
        weather_features: Optional[Dict[str, Any]] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
        demo_mode: bool = False,
    ) -> tuple[bool, List[str]]:
        """检查单点模式信号是否就绪"""
        feature_columns = self.model_loader.get_feature_columns() if self.model_loader else []
        return self._check_required_features_ready(
            feature_columns=feature_columns,
            sources=(can_features, face_features, weather_features or {}, pmv_features or {}),
            demo_mode=demo_mode,
        )

    def _check_required_features_ready(
        self,
        feature_columns: List[str],
        sources: tuple[Dict[str, Any], ...],
        demo_mode: bool = False,
    ) -> tuple[bool, List[str]]:
        """按当前模型特征列检查输入是否存在且为有效数值。"""
        values: Dict[str, Any] = {}
        for source in sources:
            values.update(source)

        missing: List[str] = []
        for feature in feature_columns:
            raw_name = feature
            for prefix in FEATURE_PREFIXES:
                if feature.startswith(prefix):
                    raw_name = feature.replace(prefix, "", 1)
                    break
            value = values.get(raw_name)
            if value is None:
                missing.append(raw_name)
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                missing.append(raw_name)
                continue
            if not np.isfinite(numeric):
                missing.append(raw_name)

        missing = list(dict.fromkeys(missing))
        if demo_mode:
            return True, missing
        return len(missing) == 0, missing

    def check_window_ready(
        self,
        min_samples: int = 2,
        demo_mode: bool = False,
        face_features: Optional[Dict[str, Any]] = None,
        weather_features: Optional[Dict[str, Any]] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, List[str]]:
        """检查滑动窗口是否就绪"""
        if self._window_cache is None:
            return False, ["滑动窗口未初始化"]

        can_ready, can_missing = self._window_cache.is_window_ready(
            ACTIVE_CAN_SIGNAL_NAMES, min_samples=min_samples
        )

        feature_columns = self.model_loader.get_feature_columns() if self.model_loader else []
        current_values: Dict[str, Any] = {}
        if self._window_cache is not None:
            for name in CAN_SIGNAL_NAMES:
                stats = self._window_cache.get_window_stats(name)
                if stats:
                    current_values[name] = stats.last
            current_values.update(self._window_cache.get_person_signals())
        if weather_features:
            current_values.update(weather_features)
        if pmv_features:
            current_values.update(pmv_features)

        _, feature_missing = self._check_required_features_ready(
            feature_columns=feature_columns,
            sources=(current_values,),
            demo_mode=demo_mode,
        )

        missing = list(dict.fromkeys(can_missing + feature_missing))
        return len(missing) == 0, missing

    # ---------------------------
    # 推理入口
    # ---------------------------
    def infer(
        self,
        can_features: Optional[Dict[str, Any]] = None,
        face_features: Optional[Dict[str, Any]] = None,
        weather_features: Optional[Dict[str, Any]] = None,
        demo_mode: bool = False,
        default_signals: Optional[List[str]] = None,
        presence: Optional[Dict[str, bool]] = None,
        pmv_info: Optional[str] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """
        统一推理入口

        如果启用了滑动窗口模式，优先使用窗口模式；
        否则使用单点退化模式。

        Args:
            presence: 人员存在状态 {"driver_exists": bool, "passenger_exists": bool}
                用于日志打印时过滤不存在的驾驶员特征。None表示不过滤。
            pmv_info: PMV/PPD 信息字符串，用于日志展示。None或空字符串表示不打印。
        """
        if self.use_sliding_window and self._window_cache is not None:
            if can_features:
                self.add_can_samples(can_features)
            if face_features is not None:
                self.set_person_signals(face_features)
            return self._infer_window(
                demo_mode=demo_mode,
                default_signals=default_signals,
                weather_features=weather_features,
                presence=presence,
                pmv_info=pmv_info,
                pmv_features=pmv_features,
            )
        else:
            return self._infer_single(
                can_features or {},
                face_features or {},
                weather_features=weather_features,
                demo_mode=demo_mode,
                default_signals=default_signals,
                presence=presence,
                pmv_info=pmv_info,
                pmv_features=pmv_features,
            )

    def _infer_single(
        self,
        can_features: Dict[str, Any],
        face_features: Dict[str, Any],
        weather_features: Optional[Dict[str, Any]] = None,
        demo_mode: bool = False,
        default_signals: Optional[List[str]] = None,
        presence: Optional[Dict[str, bool]] = None,
        pmv_info: Optional[str] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """单点模式推理"""
        ready, missing = self.check_signals_ready_single(
            can_features,
            face_features,
            weather_features=weather_features,
            pmv_features=pmv_features,
            demo_mode=demo_mode,
        )

        if not ready:
            return InferenceResult(
                timestamp=__import__("time").time(),
                ready=False,
                missing=missing,
                mode="single",
                demo_mode=demo_mode,
            )

        return self._do_inference(
            lambda: self._single_builder.build_features(
                self.model_loader.get_feature_columns(),
                can_features=can_features,
                face_features=face_features,
                weather_features=weather_features,
                pmv_features=pmv_features,
                fill_defaults=demo_mode,
            ),
            demo_mode=demo_mode,
            default_signals=default_signals or missing,
            presence=presence,
            pmv_info=pmv_info,
        )

    def _infer_window(
        self,
        min_samples: int = 2,
        demo_mode: bool = False,
        default_signals: Optional[List[str]] = None,
        weather_features: Optional[Dict[str, Any]] = None,
        presence: Optional[Dict[str, bool]] = None,
        pmv_info: Optional[str] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """滑动窗口模式推理"""
        if self._window_builder is None:
            return InferenceResult(
                timestamp=__import__("time").time(),
                ready=False,
                missing=["窗口特征构建器未初始化"],
                mode="window",
                demo_mode=demo_mode,
            )

        # 从窗口缓存获取当前人员信号，用于就绪检查
        face_features = self._window_cache.get_person_signals() if self._window_cache else None
        ready, missing = self.check_window_ready(
            min_samples,
            demo_mode,
            face_features=face_features,
            weather_features=weather_features,
            pmv_features=pmv_features,
        )

        if not ready and not demo_mode:
            return InferenceResult(
                timestamp=__import__("time").time(),
                ready=False,
                missing=missing,
                mode="window",
                demo_mode=demo_mode,
            )

        return self._do_inference(
            lambda: self._window_builder.build_features(
                self.model_loader.get_feature_columns(),
                fill_defaults=demo_mode,
                weather_features=weather_features,
                pmv_features=pmv_features,
            ),
            demo_mode=demo_mode,
            default_signals=default_signals or missing,
            presence=presence,
            pmv_info=pmv_info,
        )

    # ---------------------------
    # 内部推理逻辑
    # ---------------------------
    def _do_inference(
        self,
        feature_builder,
        demo_mode: bool = False,
        default_signals: Optional[List[str]] = None,
        presence: Optional[Dict[str, bool]] = None,
        pmv_info: Optional[str] = None,
    ) -> InferenceResult:
        """执行实际的模型推理

        Args:
            presence: 人员存在状态 {"driver_exists": bool, "passenger_exists": bool}
                用于日志打印时过滤不存在的驾驶员特征。None表示不过滤。
            pmv_info: PMV/PPD 信息字符串，用于日志展示。None或空字符串表示不打印。
        """
        import time

        if not self.model_loader or not self.model_loader.is_loaded():
            return InferenceResult(
                timestamp=time.time(),
                ready=False,
                missing=["模型未加载"],
                error="模型未加载",
                mode="window" if self.use_sliding_window else "single",
            )

        start_time = time.time()
        try:
            features_df = feature_builder()
            if features_df is None:
                return InferenceResult(
                    timestamp=time.time(),
                    ready=False,
                    missing=["特征构建失败"],
                    error="特征构建失败",
                    mode="window" if self.use_sliding_window else "single",
                )

            # 根据人员存在状态过滤日志打印的特征
            # 实际推理仍使用完整特征（主驾/副驾互拷），仅日志展示时过滤
            display_df = self._filter_features_for_display(features_df, presence)
            log_msg = f"模型输入特征:\n{display_df.T.to_string()}"
            # 附加 PMV/PPD 信息（根据人员存在状态已在外部格式化）
            if pmv_info:
                log_msg += f"\n{pmv_info}"
            logger.info(log_msg)

            # 基础模型推理
            base_prediction = self.model_loader.predict(features_df)
            base_result = base_prediction.to_dict()

            inference_time = (time.time() - start_time) * 1000
            self._inference_count += 1

            return InferenceResult(
                timestamp=time.time(),
                driver_temp=base_result["driver_temp"],
                passenger_temp=base_result["passenger_temp"],
                wind_speed=base_result["wind_speed"],
                air_mode=base_result["air_mode"],
                ready=True,
                inference_time_ms=inference_time,
                mode="window" if self.use_sliding_window else "single",
                demo_mode=demo_mode,
                default_signals=default_signals or [],
                base_result=base_result,
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"推理失败: {e}", exc_info=True)
            return InferenceResult(
                timestamp=time.time(),
                ready=False,
                missing=[f"推理异常: {str(e)}"],
                error=str(e),
                mode="window" if self.use_sliding_window else "single",
            )

    # ---------------------------
    # 状态查询
    # ---------------------------
    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        return {
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "use_sliding_window": self.use_sliding_window,
            "window_seconds": self.window_seconds,
        }

    def get_window_stats(self, signal_name: str) -> Optional[Dict[str, Any]]:
        """获取窗口中某个信号的统计信息"""
        if self._window_cache is None:
            return None
        stats = self._window_cache.get_window_stats(signal_name)
        if stats is None:
            return None
        return {
            "count": stats.count,
            "mean": stats.mean,
            "last": stats.last,
            "delta": stats.delta,
        }

    def _filter_features_for_display(
        self,
        features_df: Any,
        presence: Optional[Dict[str, bool]] = None,
    ) -> Any:
        """根据人员存在状态过滤用于日志展示的特征列

        实际推理仍使用完整特征（主驾/副驾互拷保证模型输入完整），
        此方法仅用于日志打印时过滤掉不存在的驾驶员特征，使日志更清晰。

        Args:
            features_df: 完整特征DataFrame
            presence: 人员存在状态 {"driver_exists": bool, "passenger_exists": bool}
                None表示不过滤（打印全部特征）

        Returns:
            过滤后的DataFrame（仅用于展示）
        """
        if presence is None:
            return features_df

        driver_exists = presence.get("driver_exists", True)
        passenger_exists = presence.get("passenger_exists", True)

        # 主副驾都存在，无需过滤
        if driver_exists and passenger_exists:
            return features_df

        # 根据存在状态过滤列
        cols_to_drop = []
        for col in features_df.columns:
            if not driver_exists and col.startswith("主驾"):
                cols_to_drop.append(col)
            if not passenger_exists and col.startswith("副驾"):
                cols_to_drop.append(col)

        if not cols_to_drop:
            return features_df

        return features_df.drop(columns=cols_to_drop)
