#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sklearn模型兼容性加载器
========================

解决sklearn版本不兼容导致的决策树模型加载失败问题。

问题背景:
- sklearn 1.3.0 在决策树节点结构中新增了 `missing_go_to_left` 字段
- 用sklearn 1.2.2 训练的模型没有该字段，无法在 1.3.0+ 环境加载
- 错误信息: "node array from the pickle has an incompatible dtype"

解决方案:
- 通过monkey-patch sklearn.tree._tree._check_node_ndarray 函数
- 在校验时自动将旧版节点数组升级为新版结构
- 添加 missing_go_to_left 字段（默认值0）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.logger_config import get_logger

logger = get_logger(__name__)

# sklearn 1.2.2 的决策树节点dtype字段（缺少 missing_go_to_left）
_OLD_NODE_FIELDS = {
    "left_child",
    "right_child",
    "feature",
    "threshold",
    "impurity",
    "n_node_samples",
    "weighted_n_node_samples",
}

# 升级后的新dtype（与sklearn 1.3+一致）
_NEW_NODE_DTYPE = np.dtype([
    ("left_child", "<i8"),
    ("right_child", "<i8"),
    ("feature", "<i8"),
    ("threshold", "<f8"),
    ("impurity", "<f8"),
    ("n_node_samples", "<i8"),
    ("weighted_n_node_samples", "<f8"),
    ("missing_go_to_left", "u1"),
])

_patch_applied = False
_original_check_node_ndarray = None


def _is_old_tree_node_dtype(dtype: np.dtype) -> bool:
    """判断dtype是否为旧版（1.2.2）决策树节点结构"""
    if dtype.names is None:
        return False
    return set(dtype.names) == _OLD_NODE_FIELDS


def _upgrade_tree_node_array(arr: np.ndarray) -> np.ndarray:
    """将旧版决策树节点数组升级为新版结构，添加 missing_go_to_left 字段"""
    if not _is_old_tree_node_dtype(arr.dtype):
        return arr

    new_arr = np.zeros(arr.shape, dtype=_NEW_NODE_DTYPE)
    for field in _OLD_NODE_FIELDS:
        new_arr[field] = arr[field]
    # missing_go_to_left 默认为0（缺失值走左子树，与sklearn默认行为一致）

    logger.debug(
        f"决策树节点dtype已升级: {arr.dtype.itemsize}B -> {_NEW_NODE_DTYPE.itemsize}B, "
        f"节点数={len(arr)}"
    )
    return new_arr


def _patched_check_node_ndarray(*args, **kwargs):
    """
    补丁版 _check_node_ndarray，自动升级旧版节点数组。

    sklearn在Tree.__setstate__中调用此函数校验节点数组dtype，
    我们在调用前先检查并升级旧版数组。
    """
    # 参数可能是 (node_ndarray, expected_dtype) 或其他形式
    # 不同版本签名可能不同，做兼容处理
    if args:
        node_array = args[0]
        if isinstance(node_array, np.ndarray) and _is_old_tree_node_dtype(node_array.dtype):
            # 升级数组
            upgraded = _upgrade_tree_node_array(node_array)
            # 替换参数
            args = (upgraded,) + args[1:]

    return _original_check_node_ndarray(*args, **kwargs)


def _apply_patch():
    """应用sklearn兼容性补丁"""
    global _patch_applied, _original_check_node_ndarray

    if _patch_applied:
        return

    try:
        from sklearn.tree import _tree as _tree_module

        _original_check_node_ndarray = _tree_module._check_node_ndarray
        _tree_module._check_node_ndarray = _patched_check_node_ndarray
        _patch_applied = True
        logger.debug("已应用sklearn决策树节点dtype兼容性补丁")
    except ImportError:
        logger.warning("无法导入sklearn.tree._tree，兼容性补丁未应用")


# sklearn 1.3+ 新增的决策树属性，旧模型可能缺失
# 格式: 属性名 -> 默认值
_TREE_MISSING_ATTRS = {
    "monotonic_cst": None,
    "_sklearn_version": None,
}

# sklearn 1.3+ 新增的森林属性
_FOREST_MISSING_ATTRS = {
    "monotonic_cst": None,
    "_sklearn_version": None,
}


def _patch_missing_attrs(obj: Any) -> None:
    """
    补全sklearn版本升级后新增的缺失属性。

    sklearn 1.3+ 在决策树和随机森林中新增了一些属性，
    旧版模型反序列化后这些属性不存在，导致推理时AttributeError。
    """
    from sklearn.tree import BaseDecisionTree

    # 处理决策树
    if isinstance(obj, BaseDecisionTree):
        for attr, default in _TREE_MISSING_ATTRS.items():
            if not hasattr(obj, attr):
                setattr(obj, attr, default)

    # 处理随机森林（通过检测estimators_属性）
    if hasattr(obj, "estimators_") and hasattr(obj, "n_estimators"):
        for attr, default in _FOREST_MISSING_ATTRS.items():
            if not hasattr(obj, attr):
                setattr(obj, attr, default)
        # 同时修复森林中的每棵树
        for estimator in getattr(obj, "estimators_", []):
            _patch_missing_attrs(estimator)


def load_model_compat(path: str | Path) -> Any:
    """
    兼容性加载sklearn模型。

    自动处理sklearn 1.2.2 训练的决策树模型在 1.3.0+ 环境的加载问题。

    Args:
        path: 模型文件路径

    Returns:
        加载的模型对象
    """
    _apply_patch()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    try:
        model = joblib.load(path)
        # 补全版本升级后缺失的属性
        _patch_missing_attrs(model)
        return model
    except ValueError as e:
        if "incompatible dtype" not in str(e):
            raise
        logger.warning(
            f"检测到sklearn版本不兼容（{path.name}），但补丁未生效: {e}"
        )
        raise
