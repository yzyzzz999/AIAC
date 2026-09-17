#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC Recommendation System - Logger Configuration
================================================
统一日志配置模块，支持文件日志和控制台日志同时输出。

职责边界:
- 配置日志格式和输出目标
- 支持不同日志级别
- 提供全局logger获取接口

接口:
- setup_logging(): 配置日志系统
- get_logger(): 获取命名logger实例
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


# ============================================================
# 日志配置
# ============================================================
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """
    配置日志系统

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径，None则不写入文件
        console: 是否输出到控制台
        format_string: 自定义日志格式
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handlers: list[logging.Handler] = []

    # 文件日志
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)

    # 控制台日志
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(console_handler)

    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    logging.info(f"日志系统已启动: level={level}, file={log_file}, console={console}")


def get_logger(name: str) -> logging.Logger:
    """获取命名logger实例"""
    return logging.getLogger(name)
