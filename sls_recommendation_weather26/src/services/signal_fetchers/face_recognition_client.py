#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Face Recognition Client
=====================================================
人脸识别API客户端模块，从HTTP API获取人员信息信号。

基于最新 /stats API 文档更新：
- driver/passenger 字段: identity_id, gender, age, cloth, height, bmi
- 无人时返回 null
- 各属性就绪时间不同，就绪的字段用当前值、未就绪的沿用上次有效值
- 换人时(identity_id变化)立即清空旧数据

职责边界:
- 连接人脸识别服务HTTP API
- 获取主驾/副驾人员信息
- 支持后台自动刷新
- 提供信号就绪状态查询
- 支持演示模式（缺失信号使用默认值）

接口:
- FaceRecognitionClient: 人脸识别客户端
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from src.utils.config_manager import get_config
from src.utils.constants import FACE_SIGNAL_NAMES, FEATURE_DEFAULTS
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# BMI 连续值 -> 模型 BMI 类别编码 映射
# 模型编码定义：0偏瘦, 1正常, 2超重, 3肥胖
# 阈值参考中国成人体重判定标准：
#   <18.5 偏瘦, 18.5-24 正常, 24-28 超重, >=28 肥胖
BMI_BINS: List[Tuple[float, int]] = [
    (18.5, 0),   # BMI < 18.5 -> 偏瘦
    (24.0, 1),   # 18.5 <= BMI < 24 -> 正常
    (28.0, 2),   # 24 <= BMI < 28 -> 超重
]
BMI_OBESE_CODE: int = 3  # BMI >= 28 -> 肥胖

# 注意：age 和 cloth 字段模型训练时已使用与 API 一致的等级编码，
# 因此直接透传 API 返回值，无需做映射转换。


# ============================================================
# 人脸识别客户端
# ============================================================
class FaceRecognitionClient:
    """人脸识别服务HTTP API客户端"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        cfg = get_config()
        api_cfg = cfg.get_section("face_api")

        self.base_url = base_url or api_cfg.get("base_url", "http://localhost:7860")
        self.timeout = timeout or api_cfg.get("timeout", 2.0)
        self.refresh_interval = api_cfg.get("refresh_interval", 1.0)

        # 数据存储
        self._driver_info: Dict[str, Any] = {}
        self._passenger_info: Dict[str, Any] = {}
        self._last_driver_id: Optional[str] = None
        self._last_passenger_id: Optional[str] = None
        self._last_update_time = 0.0
        self._lock = threading.Lock()

        self._refresh_running = False
        self._refresh_thread: Optional[threading.Thread] = None

    # ---------------------------
    # HTTP请求
    # ---------------------------
    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """发送HTTP请求并返回JSON响应"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(
                method, url,
                timeout=self.timeout,
                headers={"accept": "application/json"},
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning(f"API连接失败: {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"API请求超时: {url}")
            return None
        except Exception as e:
            logger.warning(f"API请求异常: {url} - {e}")
            return None

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """获取识别状态与主副驾信息"""
        return self._request("GET", "/stats")

    # ---------------------------
    # 数据刷新
    # ---------------------------
    def refresh(self) -> bool:
        """手动刷新人员信息，处理换人逻辑"""
        data = self.get_stats()
        if data is None:
            # API 连接失败，清空缓存数据，避免 stale 状态
            with self._lock:
                self._driver_info = {}
                self._passenger_info = {}
                self._last_driver_id = None
                self._last_passenger_id = None
            return False

        with self._lock:
            new_driver = data.get("driver")
            new_passenger = data.get("passenger")
            logger.info(f"人脸API原始数据: driver={new_driver}, passenger={new_passenger}")

            # 处理主驾：换人时清空，否则合并
            if new_driver is None:
                self._driver_info = {}
                self._last_driver_id = None
            else:
                new_driver_id = new_driver.get("identity_id")
                if new_driver_id != self._last_driver_id:
                    # 换人，清空旧数据，使用新数据
                    self._driver_info = self._filter_valid_fields(new_driver)
                    self._last_driver_id = new_driver_id
                    logger.info(f"主驾换人: {new_driver_id}")
                else:
                    # 同一人，合并新就绪的字段
                    self._driver_info = self._merge_fields(self._driver_info, new_driver)

            # 处理副驾：换人时清空，否则合并
            if new_passenger is None:
                self._passenger_info = {}
                self._last_passenger_id = None
            else:
                new_passenger_id = new_passenger.get("identity_id")
                if new_passenger_id != self._last_passenger_id:
                    # 换人，清空旧数据，使用新数据
                    self._passenger_info = self._filter_valid_fields(new_passenger)
                    self._last_passenger_id = new_passenger_id
                    logger.info(f"副驾换人: {new_passenger_id}")
                else:
                    # 同一人，合并新就绪的字段
                    self._passenger_info = self._merge_fields(self._passenger_info, new_passenger)

            self._last_update_time = time.time()

        return True

    @staticmethod
    def _filter_valid_fields(info: Dict[str, Any]) -> Dict[str, Any]:
        """过滤出非null的有效字段"""
        return {k: v for k, v in info.items() if v is not None and k != "identity_id"}

    @staticmethod
    def _merge_fields(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """合并字段：新数据中非null的覆盖旧数据，null的保留旧数据"""
        merged = old.copy()
        for k, v in new.items():
            if v is not None and k != "identity_id":
                merged[k] = v
        return merged

    def start_auto_refresh(self, interval: Optional[float] = None) -> None:
        """启动后台自动刷新"""
        interval = interval or self.refresh_interval
        self._refresh_running = True

        def _refresh_loop():
            while self._refresh_running:
                self.refresh()
                time.sleep(interval)

        self._refresh_thread = threading.Thread(target=_refresh_loop, daemon=True)
        self._refresh_thread.start()
        logger.info(f"人脸识别自动刷新已启动，间隔 {interval}s")

    def stop_auto_refresh(self) -> None:
        """停止后台自动刷新"""
        self._refresh_running = False
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=2.0)
        logger.info("人脸识别自动刷新已停止")

    # ---------------------------
    # 驾驶员存在性判断
    # ---------------------------
    def get_driver_presence(self) -> Dict[str, bool]:
        """基于 identity_id 判断主驾/副驾是否存在
        
        Returns:
            {"driver_exists": bool, "passenger_exists": bool}
        """
        with self._lock:
            return {
                "driver_exists": self._last_driver_id is not None,
                "passenger_exists": self._last_passenger_id is not None,
            }

    def get_display_features_and_missing(self) -> Tuple[Dict[str, Any], List[str], Dict[str, bool]]:
        """获取用于展示的特征和缺失信号（仅包含实际存在的驾驶员）
        
        Returns:
            (features, missing_signals, presence_status)
            - features: 实际存在驾驶员的特征字典
            - missing_signals: 实际存在驾驶员中缺失的信号名列表
            - presence_status: {"driver_exists": bool, "passenger_exists": bool}
        """
        presence = self.get_driver_presence()
        driver_exists = presence["driver_exists"]
        passenger_exists = presence["passenger_exists"]

        features = {}
        missing = []
        driver_signal_names = [name for name in FACE_SIGNAL_NAMES if name.startswith("主驾")]
        passenger_signal_names = [name for name in FACE_SIGNAL_NAMES if name.startswith("副驾")]

        try:
            with self._lock:
                driver_info = self._driver_info.copy()
                passenger_info = self._passenger_info.copy()

            if driver_exists:
                driver_features = self._extract_features(driver_info, "主驾")
                features.update(driver_features)
                for name in driver_signal_names:
                    if name not in driver_features:
                        missing.append(name)

            if passenger_exists:
                passenger_features = self._extract_features(passenger_info, "副驾")
                features.update(passenger_features)
                for name in passenger_signal_names:
                    if name not in passenger_features:
                        missing.append(name)

        except Exception as e:
            logger.error(f"特征数据获取异常: {e}", exc_info=True)
            missing = []
            if driver_exists:
                missing.extend(driver_signal_names)
            if passenger_exists:
                missing.extend(passenger_signal_names)

        return features, missing, presence

    # ---------------------------
    # 信号提取
    # ---------------------------
    def _extract_features(self, info: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """从人员信息字典提取特征，适配新API字段名"""
        features = {}
        if info is None:
            return features
        try:
            bmi_api = info.get("bmi")
            bmi_model = self._convert_bmi_code(bmi_api)

            mapping = {
                f"{prefix}年龄": info.get("age"),       # API 返回年龄等级，与模型训练编码一致，直接透传
                f"{prefix}性别": info.get("gender"),
                f"{prefix}衣着": info.get("cloth"),     # API 返回衣着等级，与模型训练编码一致，直接透传
                f"{prefix}BMI": bmi_model,              # API 返回 BMI 连续值，已转换为模型类别编码
                f"{prefix}身高/cm": info.get("height"),
            }
            for key, value in mapping.items():
                if value is not None:
                    features[key] = value
        except Exception as e:
            logger.warning(f"提取{prefix}特征时异常: {e}")
        return features

    @staticmethod
    def _convert_bmi_code(bmi_api: Optional[Any]) -> Optional[int]:
        """将 API 返回的 BMI 连续值转换为模型输入类别编码"""
        if bmi_api is None:
            return None
        try:
            bmi = float(bmi_api)
        except (ValueError, TypeError):
            logger.warning(f"未知的 BMI 值类型: {bmi_api}，将使用原值")
            return bmi_api

        for threshold, code in BMI_BINS:
            if bmi < threshold:
                return code
        return BMI_OBESE_CODE

    def get_all_features(self, demo_mode: bool = False, for_inference: bool = False) -> Dict[str, Any]:
        """获取所有人员特征
        
        Args:
            demo_mode: 演示模式，缺失信号使用默认值
            for_inference: 是否为推理使用。如果为True，当主驾/副驾有一方缺失时，
                          用另一方信息复制填充，保证模型推理所需特征完整。
        """
        with self._lock:
            driver_info = self._driver_info
            passenger_info = self._passenger_info

            driver_features = self._extract_features(driver_info, "主驾")
            passenger_features = self._extract_features(passenger_info, "副驾")

        # 推理模式：主驾/副驾互拷逻辑
        if for_inference:
            has_driver = bool(driver_features)
            has_passenger = bool(passenger_features)

            if has_driver and not has_passenger:
                # 只有主驾，复制给副驾用于推理
                passenger_features = {
                    k.replace("主驾", "副驾"): v for k, v in driver_features.items()
                }
            elif has_passenger and not has_driver:
                # 只有副驾，复制给主驾用于推理
                driver_features = {
                    k.replace("副驾", "主驾"): v for k, v in passenger_features.items()
                }

            # 互拷后仍有缺失（如一方存在但部分字段为None），用对方特征或默认值补齐
            features = {**driver_features, **passenger_features}
            for name in FACE_SIGNAL_NAMES:
                if name in features and features[name] is not None:
                    continue
                # 尝试从对方获取
                other = name.replace("主驾", "副驾") if name.startswith("主驾") else name.replace("副驾", "主驾")
                if other in features and features[other] is not None:
                    features[name] = features[other]
                else:
                    features[name] = FEATURE_DEFAULTS.get(name, 0.0)
            return features

        if demo_mode:
            for name in FACE_SIGNAL_NAMES:
                if name not in features or features[name] is None:
                    features[name] = FEATURE_DEFAULTS.get(name, 0.0)

        return features

    def get_ready_features(self, demo_mode: bool = False, for_inference: bool = False) -> Tuple[List[str], List[str]]:
        """获取已就绪和缺失的人员特征"""
        all_features = self.get_all_features(demo_mode=demo_mode, for_inference=for_inference)
        ready = []
        missing = []

        for name in FACE_SIGNAL_NAMES:
            val = all_features.get(name)
            if val is not None and not (isinstance(val, (int, float)) and np.isnan(val)):
                ready.append(name)
            else:
                missing.append(name)

        return ready, missing

    def is_data_fresh(self, timeout: float = 5.0) -> bool:
        """检查数据是否在有效期内"""
        with self._lock:
            return (time.time() - self._last_update_time) < timeout

    def get_identity_info(self) -> Dict[str, Any]:
        """获取当前主副驾身份ID信息"""
        with self._lock:
            return {
                "driver_id": self._last_driver_id,
                "passenger_id": self._last_passenger_id,
            }
