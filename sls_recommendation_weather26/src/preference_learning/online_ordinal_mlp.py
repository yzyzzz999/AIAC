#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定义在线偏好学习使用的四头有序分类 MLP。"""

from __future__ import annotations

import numpy as np

from src.preference_learning.ordinal_multihead_mlp import (
    OrdinalMLPConfig,
    OrdinalMultiHeadClassifier,
)


# 新偏好数据的主驾温度残差达到 +4.5℃，因此温度空间扩展到正负 6℃。
TEMPERATURE_DELTAS = np.arange(-6.0, 6.01, 0.5)
# 低温偏好段存在"随机森林 6 档、人工 3 档"，风量需要覆盖 -3 档。
WIND_DELTAS = np.arange(-3.0, 4.0, 1.0)
MODE_CLASSES = np.arange(0.0, 8.0, 1.0)


class OnlineOrdinalMultiHeadClassifier(OrdinalMultiHeadClassifier):
    """共享主干，同时输出主驾温度、副驾温度、风量和模式修正。"""

    head_names = (
        "driver_temperature",
        "passenger_temperature",
        "wind",
        "mode",
    )
    head_values = (
        TEMPERATURE_DELTAS,
        TEMPERATURE_DELTAS,
        WIND_DELTAS,
        MODE_CLASSES,
    )
    ordinal_heads = (True, True, True, False)


__all__ = [
    "MODE_CLASSES",
    "OrdinalMLPConfig",
    "OnlineOrdinalMultiHeadClassifier",
    "TEMPERATURE_DELTAS",
    "WIND_DELTAS",
]
