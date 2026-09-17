#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - PMV Client
========================================
PMV（预测平均投票）API 客户端模块，从 HTTP API 获取主副驾的热舒适度指标。

接口示例: GET http://localhost:7861/pmv
响应格式:
{
    "driver": {"pmv": 1.151, "ppd": 35.4, "head_temp_c": 29.61, "feet_temp_c": 29.69},
    "passenger": {"pmv": 1.236, "ppd": 38.7, "head_temp_c": 29.99, "feet_temp_c": 29.69},
    "cabin_temp_c": 29.75,
    "amb_temp_c": 27.4,
    "timestamp": "2026-06-19T08:43:01.711013Z",
    "status": "ok",
    "run_index": 12
}

职责边界:
- 连接 PMV 服务 HTTP API
- 获取主驾/副驾 PMV/PPD 指标
- 支持后台自动刷新
- 提供用于日志展示的格式化字符串

接口:
- PmvClient: PMV 客户端
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import requests

from src.utils.config_manager import get_config
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# PMV 数值 -> 中文热感觉描述映射
# pmv > 2.5          热
# 1.5 < pmv <= 2.5   暖
# 0.5 < pmv <= 1.5   稍暖
# -0.5 < pmv <= 0.5  舒适
# -1.5 < pmv <= -0.5 稍凉
# -2.5 < pmv <= -1.5 凉
# pmv <= -2.5        冷
def describe_pmv(pmv: Optional[float]) -> str:
    """根据 PMV 数值返回中文热感觉描述"""
    if pmv is None:
        return "未知"
    try:
        v = float(pmv)
    except (ValueError, TypeError):
        return "未知"
    if v > 2.5:
        return "热"
    if v > 1.5:
        return "暖"
    if v > 0.5:
        return "稍暖"
    if v > -0.5:
        return "舒适"
    if v > -1.5:
        return "稍凉"
    if v > -2.5:
        return "凉"
    return "冷"


# ============================================================
# PMV 客户端
# ============================================================
class PmvClient:
    """PMV 服务 HTTP API 客户端"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        cfg = get_config()
        pmv_cfg = cfg.get_section("pmv_api")

        self.base_url = base_url or pmv_cfg.get("base_url", "http://localhost:7861")
        self.timeout = timeout or pmv_cfg.get("timeout", 2.0)
        self.refresh_interval = pmv_cfg.get("refresh_interval", 1.0)

        # 数据缓存
        self._driver_pmv: Optional[float] = None
        self._driver_ppd: Optional[float] = None
        self._passenger_pmv: Optional[float] = None
        self._passenger_ppd: Optional[float] = None
        self._last_update_time: float = 0.0
        self._lock = threading.Lock()

        # 自动刷新线程
        self._refresh_running = False
        self._refresh_thread: Optional[threading.Thread] = None

    # ---------------------------
    # HTTP 请求
    # ---------------------------
    def _request(self, endpoint: str = "/pmv") -> Optional[Dict[str, Any]]:
        """发送 HTTP GET 请求并返回 JSON 响应"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning(f"PMV API 连接失败: {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"PMV API 请求超时: {url}")
            return None
        except Exception as e:
            logger.warning(f"PMV API 请求异常: {url} - {e}")
            return None

    # ---------------------------
    # 数据刷新
    # ---------------------------
    def refresh(self) -> bool:
        """手动刷新 PMV 数据"""
        data = self._request("/pmv")
        if data is None:
            return False

        try:
            with self._lock:
                driver = data.get("driver")
                passenger = data.get("passenger")

                if driver and isinstance(driver, dict):
                    self._driver_pmv = self._safe_float(driver.get("pmv"))
                    self._driver_ppd = self._safe_float(driver.get("ppd"))
                else:
                    # driver 为 None 表示主驾无人，清空缓存
                    self._driver_pmv = None
                    self._driver_ppd = None

                if passenger and isinstance(passenger, dict):
                    self._passenger_pmv = self._safe_float(passenger.get("pmv"))
                    self._passenger_ppd = self._safe_float(passenger.get("ppd"))
                else:
                    # passenger 为 None 表示副驾无人，清空缓存
                    self._passenger_pmv = None
                    self._passenger_ppd = None

                self._last_update_time = time.time()
            return True
        except Exception as e:
            logger.warning(f"PMV 数据解析异常: {e}")
            return False

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """安全转换为 float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # ---------------------------
    # 自动刷新
    # ---------------------------
    def start_auto_refresh(self, interval: Optional[float] = None) -> None:
        """启动后台自动刷新线程"""
        if self._refresh_running:
            return
        if interval is not None:
            self.refresh_interval = interval
        self._refresh_running = True
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name="pmv-refresh"
        )
        self._refresh_thread.start()
        logger.info(f"PMV API 自动刷新已启动 (间隔 {self.refresh_interval}s)")

    def stop_auto_refresh(self) -> None:
        """停止后台自动刷新"""
        self._refresh_running = False
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=2.0)
        self._refresh_thread = None

    def _refresh_loop(self) -> None:
        """后台刷新循环"""
        while self._refresh_running:
            try:
                self.refresh()
            except Exception as e:
                logger.warning(f"PMV 自动刷新异常: {e}")
            time.sleep(self.refresh_interval)

    # ---------------------------
    # 数据查询
    # ---------------------------
    def get_driver_pmv(self) -> Optional[float]:
        """获取主驾 PMV 值"""
        with self._lock:
            return self._driver_pmv

    def get_driver_ppd(self) -> Optional[float]:
        """获取主驾 PPD 值"""
        with self._lock:
            return self._driver_ppd

    def get_passenger_pmv(self) -> Optional[float]:
        """获取副驾 PMV 值"""
        with self._lock:
            return self._passenger_pmv

    def get_passenger_ppd(self) -> Optional[float]:
        """获取副驾 PPD 值"""
        with self._lock:
            return self._passenger_ppd

    def get_features(self, demo_mode: bool = False) -> Dict[str, float]:
        """获取基础RF需要的PMV特征。"""
        with self._lock:
            driver_pmv = self._driver_pmv
            passenger_pmv = self._passenger_pmv

        features: Dict[str, float] = {}
        if driver_pmv is not None:
            features["主驾PMV"] = float(driver_pmv)
        elif demo_mode:
            features["主驾PMV"] = 0.0

        if passenger_pmv is not None:
            features["副驾PMV"] = float(passenger_pmv)
        elif demo_mode:
            features["副驾PMV"] = 0.0

        return features

    def is_data_fresh(self, max_age: float = 10.0) -> bool:
        """检查缓存数据是否新鲜"""
        with self._lock:
            if self._last_update_time == 0.0:
                return False
            return (time.time() - self._last_update_time) <= max_age

    # ---------------------------
    # 日志展示
    # ---------------------------
    def format_for_log(
        self,
        driver_exists: bool,
        passenger_exists: bool,
    ) -> str:
        """格式化 PMV/PPD 信息用于日志展示

        Args:
            driver_exists: 主驾是否存在
            passenger_exists: 副驾是否存在

        Returns:
            格式化字符串，如:
                "主驾 PMV=1.15(稍暖), PPD=35.4% | 副驾 PMV=1.24(暖), PPD=38.7%"
            若无人则返回空字符串
        """
        parts = []
        with self._lock:
            if driver_exists:
                pmv_desc = describe_pmv(self._driver_pmv)
                pmv_str = f"{self._driver_pmv:.2f}" if self._driver_pmv is not None else "N/A"
                ppd_str = f"{self._driver_ppd:.1f}" if self._driver_ppd is not None else "N/A"
                parts.append(f"主驾 PMV={pmv_str}({pmv_desc}), PPD={ppd_str}%")

            if passenger_exists:
                pmv_desc = describe_pmv(self._passenger_pmv)
                pmv_str = f"{self._passenger_pmv:.2f}" if self._passenger_pmv is not None else "N/A"
                ppd_str = f"{self._passenger_ppd:.1f}" if self._passenger_ppd is not None else "N/A"
                parts.append(f"副驾 PMV={pmv_str}({pmv_desc}), PPD={ppd_str}%")

        if not parts:
            return ""
        return " | ".join(parts)
