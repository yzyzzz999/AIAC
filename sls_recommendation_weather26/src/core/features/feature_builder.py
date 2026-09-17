#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Feature Builder
=============================================
特征构建模块，将原始信号转换为模型所需的特征格式。

职责边界:
- 从原始信号构建 mean_5s__ / last_5s__ / delta_5s__ 特征
- 支持单点退化模式（兼容旧逻辑）和滑动窗口模式
- 处理信号缺失和默认值填充
- 处理混合类型信号（数值和字符串）

接口:
- FeatureBuilder: 特征构建器基类
- SinglePointFeatureBuilder: 单点模式特征构建器
- WindowFeatureBuilder: 滑动窗口模式特征构建器
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.constants import (
    CAN_SIGNAL_NAMES,
    FACE_SIGNAL_NAMES,
    FEATURE_DEFAULTS,
    FEATURE_PREFIXES,
    PMV_SIGNAL_NAMES,
    WEATHER_SIGNAL_NAMES,
)
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# ============================================================
# 特征构建器基类
# ============================================================
class FeatureBuilder:
    """特征构建器基类"""

    def build_features(
        self,
        feature_columns: List[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        构建模型特征DataFrame

        Args:
            feature_columns: 特征列名列表
            **kwargs: 子类特定的输入参数

        Returns:
            包含所有特征列的DataFrame（单行）
        """
        raise NotImplementedError

    def _fill_missing_signals(
        self,
        signals: Dict[str, Any],
        required_names: List[str],
        defaults: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """填充缺失信号"""
        filled = signals.copy()
        default_map = defaults or FEATURE_DEFAULTS
        for name in required_names:
            if name not in filled or filled[name] is None:
                if name in default_map:
                    filled[name] = default_map[name]
                else:
                    filled[name] = 0.0
        return filled


# ============================================================
# 单点模式特征构建器
# ============================================================
class SinglePointFeatureBuilder(FeatureBuilder):
    """
    单点退化模式特征构建器
    mean=last=当前值, delta=0
    用于兼容旧逻辑或信号频率不足的场景
    """

    def build_features(
        self,
        feature_columns: List[str],
        can_features: Dict[str, Any],
        face_features: Dict[str, Any],
        weather_features: Optional[Dict[str, Any]] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
        fill_defaults: bool = True,
    ) -> pd.DataFrame:
        """
        构建单点退化特征

        Args:
            feature_columns: 特征列名列表
            can_features: CAN信号字典
            face_features: 人脸信号字典
            weather_features: 天气信号字典（可选）
            fill_defaults: 是否用默认值填充缺失信号
        """
        all_features = {}

        if fill_defaults:
            can_features = self._fill_missing_signals(
                can_features, CAN_SIGNAL_NAMES, FEATURE_DEFAULTS
            )
            face_features = self._fill_missing_signals(
                face_features, FACE_SIGNAL_NAMES, FEATURE_DEFAULTS
            )
            if weather_features:
                weather_features = self._fill_missing_signals(
                    weather_features, WEATHER_SIGNAL_NAMES, FEATURE_DEFAULTS
                )
            if pmv_features:
                pmv_features = self._fill_missing_signals(
                    pmv_features, PMV_SIGNAL_NAMES, FEATURE_DEFAULTS
                )

        all_features.update(can_features)
        all_features.update(face_features)
        if weather_features:
            all_features.update(weather_features)
        if pmv_features:
            all_features.update(pmv_features)

        row = self._build_feature_row(feature_columns, all_features)
        return pd.DataFrame([row])

    def _build_feature_row(
        self,
        feature_columns: List[str],
        all_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建单行特征字典"""
        row: Dict[str, Any] = {}
        for feature in feature_columns:
            if feature.startswith("mean_5s__"):
                raw_col = feature.replace("mean_5s__", "", 1)
                val = all_features.get(raw_col, 0.0)
                row[feature] = val if isinstance(val, (int, float)) else 0.0
            elif feature.startswith("last_5s__"):
                raw_col = feature.replace("last_5s__", "", 1)
                row[feature] = all_features.get(raw_col, 0.0)
            elif feature.startswith("delta_5s__"):
                raw_col = feature.replace("delta_5s__", "", 1)
                val = all_features.get(raw_col, 0.0)
                row[feature] = val if isinstance(val, (int, float)) else 0.0
            else:
                row[feature] = all_features.get(feature, 0.0)
        return row


# ============================================================
# 滑动窗口模式特征构建器
# ============================================================
class WindowFeatureBuilder(FeatureBuilder):
    """
    滑动窗口模式特征构建器
    使用真实的5秒窗口计算mean/last/delta
    """

    def __init__(self, window_cache: "SlidingWindowCache"):
        from src.core.features.sliding_window_cache import SlidingWindowCache
        self.window_cache = window_cache

    def build_features(
        self,
        feature_columns: List[str],
        fill_defaults: bool = True,
        weather_features: Optional[Dict[str, Any]] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        构建滑动窗口特征

        Args:
            feature_columns: 特征列名列表
            fill_defaults: 是否用默认值填充缺失信号
            weather_features: 天气特征字典（风向、气压等）
        """
        features = self.window_cache.build_model_features(
            can_signal_names=CAN_SIGNAL_NAMES,
            person_signal_names=FACE_SIGNAL_NAMES,
            feature_columns=feature_columns,
            weather_features=weather_features,
            pmv_features=pmv_features,
        )

        if fill_defaults:
            features = self._fill_missing_features(features, feature_columns)

        return pd.DataFrame([features])

    def _fill_missing_features(
        self,
        features: Dict[str, Any],
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        """填充缺失特征值"""
        filled = features.copy()
        for feature in feature_columns:
            if feature not in filled or filled[feature] is None:
                if feature.startswith("mean_5s__") or feature.startswith("delta_5s__"):
                    filled[feature] = 0.0
                elif feature.startswith("last_5s__"):
                    raw_col = feature.replace("last_5s__", "", 1)
                    filled[feature] = FEATURE_DEFAULTS.get(raw_col, 0.0)
                else:
                    filled[feature] = 0.0
        return filled


# ============================================================
# 便捷函数
# ============================================================
def raw_columns_from_feature_columns(feature_columns: List[str]) -> List[str]:
    """从特征列名提取原始信号列名"""
    raw_columns: List[str] = []
    for feature in feature_columns:
        raw_col = feature
        for prefix in FEATURE_PREFIXES:
            if feature.startswith(prefix):
                raw_col = feature.replace(prefix, "", 1)
                break
        if raw_col not in raw_columns:
            raw_columns.append(raw_col)
    return raw_columns


def validate_raw_input(raw_df: pd.DataFrame, required_columns: List[str]) -> None:
    """验证输入DataFrame是否包含所需列"""
    missing = [col for col in required_columns if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required raw input columns: {missing}")
