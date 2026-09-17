#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Sliding Window Cache
==================================================
滑动窗口缓存模块，维护高频信号的时间序列，计算窗口统计特征。

职责边界:
- 收集高频CAN信号，维护5秒滑动窗口
- 自动计算窗口内的 mean / last / delta 统计量
- 支持混合类型信号（数值和字符串）
- 提供实时特征输出，供推理引擎使用

接口:
- SlidingWindowCache: 主窗口缓存类
- WindowStats: 窗口统计结果数据结构
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# ============================================================
# 窗口统计结果
# ============================================================
@dataclass
class WindowStats:
    """单个信号在窗口内的统计结果"""
    count: int = 0
    mean: float = 0.0
    last: float = 0.0
    delta: float = 0.0
    first: float = 0.0
    min: float = 0.0
    max: float = 0.0


# ============================================================
# 滑动窗口缓存
# ============================================================
class SlidingWindowCache:
    """5秒滑动窗口缓存，维护多个信号的时间序列"""

    def __init__(self, window_seconds: float = 5.0, max_samples: int = 500):
        """
        Args:
            window_seconds: 窗口长度（秒）
            max_samples: 最大采样点数（防止内存无限增长）
        """
        self.window_seconds = window_seconds
        self.max_samples = max_samples

        # {信号名: deque([(timestamp, value), ...])}
        self._buffers: Dict[str, deque] = {}
        self._lock = threading.Lock()

        # 人员信号（不随时间变化，直接存储当前值）
        self._person_signals: Dict[str, Any] = {}

    # ---------------------------
    # 数据写入
    # ---------------------------
    def add_sample(self, signal_name: str, value: Any, timestamp: Optional[float] = None) -> None:
        """添加一个采样点到窗口（支持数值和字符串）"""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            if signal_name not in self._buffers:
                self._buffers[signal_name] = deque(maxlen=self.max_samples)
            self._buffers[signal_name].append((timestamp, value))
            self._cleanup_old(signal_name)

    def add_samples(self, samples: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        """批量添加多个采样点（支持数值和字符串混合）"""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            for signal_name, value in samples.items():
                if signal_name not in self._buffers:
                    self._buffers[signal_name] = deque(maxlen=self.max_samples)
                self._buffers[signal_name].append((timestamp, value))
                self._cleanup_old(signal_name)

    def set_person_signal(self, signal_name: str, value: Any) -> None:
        """设置人员信号（不随时间变化，直接存储，支持字符串）"""
        with self._lock:
            self._person_signals[signal_name] = value

    def set_person_signals(self, signals: Dict[str, Any]) -> None:
        """批量设置人员信号（支持混合类型），空字典表示清空"""
        with self._lock:
            self._person_signals = signals.copy()

    def _cleanup_old(self, signal_name: str) -> None:
        """清理窗口外的过期数据"""
        buffer = self._buffers[signal_name]
        cutoff = time.time() - self.window_seconds
        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()

    # ---------------------------
    # 窗口统计
    # ---------------------------
    def get_window_stats(self, signal_name: str) -> Optional[WindowStats]:
        """获取单个信号在窗口内的统计量"""
        with self._lock:
            buffer = self._buffers.get(signal_name)
            if not buffer:
                return None

            # 提取数值型数据
            numeric_values = []
            for _, value in buffer:
                if isinstance(value, (int, float)) and not np.isnan(value):
                    numeric_values.append(float(value))

            if not numeric_values:
                return None

            values = np.array(numeric_values, dtype=np.float32)
            return WindowStats(
                count=len(values),
                mean=float(np.mean(values)),
                last=float(values[-1]),
                delta=float(values[-1] - values[0]),
                first=float(values[0]),
                min=float(np.min(values)),
                max=float(np.max(values)),
            )

    def get_person_signals(self) -> Dict[str, Any]:
        """获取当前人员信号"""
        with self._lock:
            return self._person_signals.copy()

    def get_person_signal(self, signal_name: str) -> Any:
        """获取单个人员信号"""
        with self._lock:
            return self._person_signals.get(signal_name)

    # ---------------------------
    # 窗口就绪检查
    # ---------------------------
    def is_window_ready(
        self,
        signal_names: List[str],
        min_samples: int = 2,
    ) -> Tuple[bool, List[str]]:
        """
        检查指定信号列表是否都已收集足够数据

        Returns:
            (bool, list): (是否全部就绪, 未就绪信号列表)
        """
        missing = []
        with self._lock:
            for name in signal_names:
                buffer = self._buffers.get(name)
                if not buffer:
                    missing.append(name)
                    continue

                # 统计数值型样本数
                numeric_count = sum(
                    1 for _, value in buffer
                    if isinstance(value, (int, float)) and not np.isnan(value)
                )
                if numeric_count < min_samples:
                    missing.append(name)

        return len(missing) == 0, missing

    # ---------------------------
    # 特征构建
    # ---------------------------
    def build_model_features(
        self,
        can_signal_names: List[str],
        person_signal_names: List[str],
        feature_columns: List[str],
        weather_features: Optional[Dict[str, Any]] = None,
        pmv_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        构建模型所需的特征字典

        对CAN信号计算 mean_5s__ / last_5s__ / delta_5s__
        对人员信号直接使用 last_5s__ 前缀透传
        对天气信号直接使用传入值
        """
        features: Dict[str, Any] = {}

        # 获取所有信号的当前值（用于 last_5s__）
        all_current_values: Dict[str, Any] = {}

        for name in can_signal_names:
            stats = self.get_window_stats(name)
            if stats:
                all_current_values[name] = stats.last
            else:
                all_current_values[name] = 0.0

        for name in person_signal_names:
            val = self.get_person_signal(name)
            if val is not None:
                all_current_values[name] = val
            else:
                all_current_values[name] = 0.0 if name not in ["主驾性别", "副驾性别", "主驾衣着", "副驾衣着"] else "未知"

        # 添加天气信号
        if weather_features:
            for name, value in weather_features.items():
                all_current_values[name] = value

        if pmv_features:
            for name, value in pmv_features.items():
                all_current_values[name] = value

        # 构建特征
        for feature in feature_columns:
            if feature.startswith("mean_5s__"):
                raw_col = feature.replace("mean_5s__", "", 1)
                stats = self.get_window_stats(raw_col)
                if stats:
                    features[feature] = stats.mean
                else:
                    # 字符串类型信号 mean 设为 0.0
                    val = all_current_values.get(raw_col, 0.0)
                    features[feature] = val if isinstance(val, (int, float)) else 0.0
            elif feature.startswith("last_5s__"):
                raw_col = feature.replace("last_5s__", "", 1)
                features[feature] = all_current_values.get(raw_col, 0.0)
            elif feature.startswith("delta_5s__"):
                raw_col = feature.replace("delta_5s__", "", 1)
                stats = self.get_window_stats(raw_col)
                if stats:
                    features[feature] = stats.delta
                else:
                    val = all_current_values.get(raw_col, 0.0)
                    features[feature] = val if isinstance(val, (int, float)) else 0.0
            else:
                features[feature] = all_current_values.get(feature, 0.0)

        return features

    # ---------------------------
    # 状态查询
    # ---------------------------
    def get_buffer_counts(self) -> Dict[str, int]:
        """获取各信号缓冲区中的样本数"""
        with self._lock:
            return {
                name: len(buffer)
                for name, buffer in self._buffers.items()
            }

    def get_window_info(self) -> Dict[str, Any]:
        """获取窗口整体信息"""
        with self._lock:
            return {
                "window_seconds": self.window_seconds,
                "buffer_count": len(self._buffers),
                "person_signal_count": len(self._person_signals),
                "buffer_samples": {name: len(buf) for name, buf in self._buffers.items()},
            }

    def clear(self) -> None:
        """清空所有窗口数据"""
        with self._lock:
            self._buffers.clear()
            self._person_signals.clear()
