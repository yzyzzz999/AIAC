#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Configuration Management
====================================================
统一配置管理模块，支持从JSON配置文件加载和运行时动态覆盖。

职责边界:
- 加载和管理所有系统配置参数
- 支持点号路径访问配置项（如 "system.log_level"）
- 提供默认值回退机制
- 支持命令行参数覆盖

接口:
- ConfigManager: 主配置管理类
- get_config(): 获取全局配置实例（单例）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 配置管理器
# ============================================================
class ConfigManager:
    """配置管理器，负责加载和管理所有配置参数"""

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """从JSON文件加载配置"""
        if not self._config_path.exists():
            logger.warning(f"配置文件不存在: {self._config_path}，使用默认配置")
            self._config = self._default_config()
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
            logger.info(f"配置加载成功: {self._config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON解析失败: {e}")
            self._config = self._default_config()
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            self._config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """返回 mode4 + weather26 的最小安全默认配置。"""
        return {
            "system": {
                "log_level": "INFO",
                "log_file": "logs/recommendation.log",
                "inference_interval": 2.0,
                "ai_off_inference_interval": 5.0,
                "main_loop_tick": 1.0,
                "use_sliding_window": True,
                "window_seconds": 5.0,
                "demo_mode": False,
                "wind_max_delta_per_step": 1,
                "wind_min": 2,
            },
            "model": {
                "type": "random_forest",
                "model_dir": "models/weather26",
                "feature_config": "models/weather26/manifest.json",
                "temp_min": 16.0,
                "temp_max": 31.0,
                "temp_step": 0.5,
            },
            "preference_learning": {
                "user_mlp_dir": "models/user_mlps",
                "pref_layer_model_dir": "models/preference_layer",
                "base_recommendation_overrides": {},
                "occupant_wind_weight_k": 0.5,
                "force_zero_mlp_residuals": True,
                "pref_layer_collection_duration": 300,
                "pref_layer_min_samples": 30,
                "pref_layer_takeover_confirm_seconds": 5.0,
                "face_id_confirm_seconds": 1.5,
                "face_id_new_user_confirm_seconds": 4.0,
                "face_id_vote_window_seconds": 15.0,
                "face_id_vote_min_samples": 10,
                "face_id_vote_ratio": 0.8,
                "face_id_switch_cooldown_seconds": 20.0,
                "command_ack_timeout": 8.0,
                "validation_ratio": 0.2,
            },
            "can": {
                "socket_path": "/home/data/AIAC/can_service/sock/can0_bus.sock",
                "can_ids": ["0x18F", "0x35B", "0x338", "0x33A", "0x3B9", "0x371", "0x33F"],
                "heartbeat_interval": 5.0,
                "heartbeat_timeout": 15.0,
                "retry_interval": 5.0,
                "signal_timeout": 5.0,
                "client_id": "recommendation_client",
            },
            "face_api": {
                "base_url": "http://localhost:7860",
                "timeout": 2.0,
                "refresh_interval": 1.0,
            },
            "pmv_api": {"base_url": "http://localhost:7861", "timeout": 2.0, "refresh_interval": 1.0},
            "weather": {"provider": "demo", "refresh_interval": 300.0, "timeout": 10.0},
            "result": {
                "enabled": True,
                "socket_path": "/home/data/AIAC/can_service/sock/can0_bus.sock",
                "can_id": "0x387",
                "send_count": 5,
                "send_interval_ms": 20,
            },
            "signals": {"feature_defaults": {}},
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        使用点号路径获取配置值，如 'system.log_level'
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """
        使用点号路径设置配置值
        """
        keys = key.split(".")
        target = self._config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置的一个完整section"""
        return self._config.get(section, {})

    def get_all(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return self._config.copy()

    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()


# ============================================================
# 全局配置实例（单例）
# ============================================================
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """获取全局配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = ConfigManager(config_path)
    return _config_instance


def reset_config() -> None:
    """重置全局配置实例"""
    global _config_instance
    _config_instance = None
