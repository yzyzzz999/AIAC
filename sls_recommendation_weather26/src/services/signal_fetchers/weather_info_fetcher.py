#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Weather Info Fetcher
==================================================
天气信息在线获取模块，从公开API获取风向和气压数据。

职责边界:
- 从在线天气API获取实时风向和气压数据
- 支持多种天气数据源（OpenWeatherMap、和风天气等）
- 提供本地缓存和降级机制
- 支持演示模式（使用默认值）

接口:
- WeatherInfoFetcher: 天气信息获取器
- WeatherData: 天气数据结构
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.utils.config_manager import get_config
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# ============================================================
# 天气数据结构
# ============================================================
@dataclass
class WeatherData:
    """天气数据"""
    wind_direction: float = 0.0  # 风向，度
    pressure: float = 1013.25    # 气压，hPa
    wind_speed: float = 0.0      # 风速，m/s
    temperature: float = 25.0   # 气温，°C
    humidity: float = 50.0      # 湿度，%
    weather_condition: int = 2  # 天气状况编码: 1=多云, 2=晴, 3=小雨, 4=阴
    timestamp: float = 0.0
    source: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "风向": self.wind_direction,
            "气压(hPa)": self.pressure,
            "风速": self.wind_speed,
            "气温": self.temperature,
            "湿度": self.humidity,
            "天气状况": self.weather_condition,
        }


# ============================================================
# 天气信息获取器
# ============================================================
class WeatherInfoFetcher:
    """
    天气信息在线获取器
    
    支持的数据源:
    - OpenWeatherMap (需要API Key)
    - 和风天气 (需要API Key)
    - 演示模式 (使用固定默认值)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        city: Optional[str] = None,
        refresh_interval: float = 300.0,  # 5分钟刷新
    ):
        cfg = get_config()
        weather_cfg = cfg.get_section("weather")

        self.provider = provider or weather_cfg.get("provider", "demo")
        self.api_key = api_key or weather_cfg.get("api_key", "")
        self.lat = lat or weather_cfg.get("lat", 39.9042)  # 默认北京
        self.lon = lon or weather_cfg.get("lon", 116.4074)
        self.city = city or weather_cfg.get("city", "北京")
        self.refresh_interval = refresh_interval or weather_cfg.get("refresh_interval", 300.0)
        self.timeout = weather_cfg.get("timeout", 10.0)

        self._current_data = WeatherData()
        self._last_update = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---------------------------
    # 数据获取
    # ---------------------------
    def fetch(self) -> WeatherData:
        """获取最新天气数据"""
        if self.provider == "openweathermap":
            return self._fetch_openweathermap()
        elif self.provider == "qweather":
            return self._fetch_qweather()
        else:
            return self._fetch_demo()

    def _fetch_openweathermap(self) -> WeatherData:
        """从OpenWeatherMap获取"""
        if not self.api_key:
            logger.warning("OpenWeatherMap API Key未配置，使用默认值")
            return self._fetch_demo()

        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units=metric"
            )
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            wind = data.get("wind", {})
            main = data.get("main", {})
            weather_list = data.get("weather", [{}])
            weather_id = weather_list[0].get("id", 800) if weather_list else 800

            weather = WeatherData(
                wind_direction=wind.get("deg", 0.0),
                pressure=main.get("pressure", 1013.25),
                wind_speed=wind.get("speed", 0.0),
                temperature=main.get("temp", 25.0),
                humidity=main.get("humidity", 50.0),
                weather_condition=self._map_weather_id(weather_id),
                timestamp=time.time(),
                source="openweathermap",
            )

            with self._lock:
                self._current_data = weather
                self._last_update = time.time()

            logger.info(f"天气数据更新: 气压={weather.pressure}hPa, 风向={weather.wind_direction}°")
            return weather

        except Exception as e:
            logger.error(f"OpenWeatherMap获取失败: {e}")
            return self._fetch_demo()

    def _fetch_qweather(self) -> WeatherData:
        """从和风天气获取"""
        if not self.api_key:
            logger.warning("和风天气 API Key未配置，使用默认值")
            return self._fetch_demo()

        try:
            # 和风天气API v7
            url = (
                f"https://devapi.qweather.com/v7/weather/now"
                f"?location={self.lon},{self.lat}&key={self.api_key}"
            )
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            now = data.get("now", {})

            # 和风天气icon映射: 100/150=晴, 101-104/151-154=多云, 其他=雨/阴
            icon_code = int(now.get("icon", "100"))
            if icon_code in (100, 150):
                wc = 2  # 晴
            elif icon_code in range(101, 105) or icon_code in range(151, 155):
                wc = 1  # 多云
            elif icon_code in range(104, 105) or icon_code in range(154, 155):
                wc = 4  # 阴
            else:
                wc = 3  # 小雨

            weather = WeatherData(
                wind_direction=float(now.get("wind360", 0)),
                pressure=float(now.get("pressure", 1013)),
                wind_speed=float(now.get("windSpeed", 0)),
                temperature=float(now.get("temp", 25)),
                humidity=float(now.get("humidity", 50)),
                weather_condition=wc,
                timestamp=time.time(),
                source="qweather",
            )

            with self._lock:
                self._current_data = weather
                self._last_update = time.time()

            logger.info(f"天气数据更新: 气压={weather.pressure}hPa, 风向={weather.wind_direction}°")
            return weather

        except Exception as e:
            logger.error(f"和风天气获取失败: {e}")
            return self._fetch_demo()

    @staticmethod
    def _map_weather_id(weather_id: int) -> int:
        """将OpenWeatherMap天气代码映射为weather26编码: 1=多云, 2=晴, 3=小雨, 4=阴"""
        if weather_id == 800:
            return 2  # 晴
        if weather_id in (801, 802):
            return 1  # 多云
        if weather_id in (803, 804):
            return 4  # 阴
        return 3  # 小雨（2xx雷暴/3xx毛毛雨/5xx雨/6xx雪/7xx雾霾统一归入）

    def _fetch_demo(self) -> WeatherData:
        """演示模式：使用默认值"""
        weather = WeatherData(
            wind_direction=7.0,
            pressure=1007.0,
            wind_speed=0.0,
            temperature=25.0,
            humidity=50.0,
            weather_condition=2,  # 默认晴
            timestamp=time.time(),
            source="demo",
        )
        with self._lock:
            self._current_data = weather
            self._last_update = time.time()
        return weather

    # ---------------------------
    # 自动刷新
    # ---------------------------
    def start_auto_refresh(self) -> None:
        """启动后台自动刷新"""
        self._running = True

        def _refresh_loop():
            while self._running:
                self.fetch()
                time.sleep(self.refresh_interval)

        self._thread = threading.Thread(target=_refresh_loop, daemon=True)
        self._thread.start()
        logger.info(f"天气自动刷新已启动，间隔 {self.refresh_interval}s，数据源: {self.provider}")

    def stop_auto_refresh(self) -> None:
        """停止后台自动刷新"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("天气自动刷新已停止")

    # ---------------------------
    # 数据查询
    # ---------------------------
    def get_current(self) -> WeatherData:
        """获取当前天气数据"""
        with self._lock:
            return self._current_data

    def get_features(self) -> Dict[str, float]:
        """获取风向、气压和天气状况特征"""
        data = self.get_current()
        return {
            "风向": data.wind_direction,
            "气压(hPa)": data.pressure,
            "天气状况": float(data.weather_condition),
        }

    def is_fresh(self, timeout: float = 600.0) -> bool:
        """检查数据是否在有效期内"""
        with self._lock:
            return (time.time() - self._last_update) < timeout
