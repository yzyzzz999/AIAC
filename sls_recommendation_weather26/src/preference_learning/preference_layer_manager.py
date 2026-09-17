#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""偏好层生命周期管理器。

状态机:
    RF_ONLY ──(人工接管检测)──> COLLECTING ──(20分钟满)──> TRAINING ──(完成)──> JOINT_ACTIVE
    JOINT_ACTIVE ──(再次接管)──> COLLECTING (重新开始)

职责:
- 检测人工接管 (CAN反馈 != RF推荐 且持续确认)
- 接管期间抑制RF输出，透传人工控制信号
- 累积训练样本
- 20分钟后触发车端训练
- 训练完成后加载MLP，切换到联合推理模式
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger_config import get_logger

logger = get_logger(__name__)


class PreferenceState(Enum):
    RF_ONLY = "rf_only"
    COLLECTING = "collecting"
    TRAINING = "training"
    JOINT_ACTIVE = "joint_active"


@dataclass
class CollectedSample:
    """一条训练样本: 26维特征 + 4个人工动作 + 4个RF基础推荐。"""
    timestamp: float
    features: Dict[str, float]         # 26维
    human_actions: Dict[str, float]    # 主驾温度显示,副驾温度显示,UI界面_风速,空调出风模式请求
    base_predictions: Dict[str, float] # base_driver_temperature,base_passenger_temperature,base_wind,base_mode


@dataclass
class PreferenceLayerStatus:
    state: PreferenceState = PreferenceState.RF_ONLY
    state_name: str = "rf_only"
    collecting_since: float = 0.0
    sample_count: int = 0
    mlp_loaded: bool = False
    mlp_user_id: str = ""
    last_training_report: dict = field(default_factory=dict)
    human_takeover_active: bool = False
    takeover_pending: bool = False


class PreferenceLayerManager:
    """偏好层管理器，接管/训练/联合推理的完整生命周期。"""

    COLLECTION_DURATION = 20 * 60       # 20分钟
    MIN_TRAINING_SAMPLES = 30           # 最少样本数（约20分钟/5秒采样=240条）
    SAMPLE_INTERVAL = 5.0               # 5秒一条
    TAKEOVER_CONFIRM_SECONDS = 5.0      # 人工接管确认窗口：差异持续5秒才确认
    TEMP_TOLERANCE = 0.5

    ACTION_COLUMNS = [
        "主驾温度显示", "副驾温度显示", "UI界面_风速", "空调出风模式请求",
    ]
    BASE_COLUMNS = [
        "base_driver_temperature", "base_passenger_temperature",
        "base_wind", "base_mode",
    ]

    def __init__(
        self,
        model_loader,
        output_dir: Path | str,
        feature_columns: Optional[list] = None,
        collection_duration: float = 1200.0,
        min_training_samples: int = 30,
        takeover_confirm_seconds: float = 5.0,
        identity_confirm_seconds: float = 1.5,
        new_user_confirm_seconds: float = 4.0,
        identity_vote_window_seconds: float = 15.0,
        identity_vote_min_samples: int = 10,
        identity_vote_ratio: float = 0.8,
        identity_switch_cooldown_seconds: float = 20.0,
        command_ack_timeout: float = 8.0,
        user_mlp_base_dir: str = "models/user_mlps",
        force_zero_mlp_residuals: bool = False,
    ):
        self._model_loader = model_loader
        self._output_dir = Path(output_dir)
        self._feature_columns = feature_columns or model_loader.get_feature_columns()

        # 可配置参数
        self.COLLECTION_DURATION = collection_duration
        self.MIN_TRAINING_SAMPLES = min_training_samples
        self.TAKEOVER_CONFIRM_SECONDS = takeover_confirm_seconds
        self.IDENTITY_CONFIRM_SECONDS = identity_confirm_seconds
        self.NEW_USER_CONFIRM_SECONDS = new_user_confirm_seconds
        self.IDENTITY_VOTE_WINDOW_SECONDS = max(1.0, identity_vote_window_seconds)
        self.IDENTITY_VOTE_MIN_SAMPLES = max(2, identity_vote_min_samples)
        self.IDENTITY_VOTE_RATIO = min(1.0, max(0.5, identity_vote_ratio))
        self.IDENTITY_SWITCH_COOLDOWN_SECONDS = max(0.0, identity_switch_cooldown_seconds)
        self.COMMAND_ACK_TIMEOUT = command_ack_timeout
        self.FORCE_ZERO_MLP_RESIDUALS = bool(force_zero_mlp_residuals)
        # 2026-08-03 P2: ack挂起期间差异持续超过该秒数, 直接按人工接管处理。
        # 实测(2026-08-03 11:27)用户在被识别后立刻调温, ack永远等不到匹配,
        # 确认门永久关闭导致AI每5s无限顶回人工设定。回读回显实测~2s,
        # 阈值取4s兼顾执行器延迟与用户体验(早于下一次5s调度发送)。
        self.ACK_MISMATCH_FALLBACK_SECONDS = 4.0

        # 用户MLP存储根目录
        self._user_mlp_base_dir = Path(user_mlp_base_dir)
        if not self._user_mlp_base_dir.is_absolute():
            from pathlib import Path as _Path
            self._user_mlp_base_dir = _Path(__file__).resolve().parents[2] / user_mlp_base_dir

        self._state = PreferenceState.RF_ONLY
        self._lock = threading.Lock()
        self._samples: List[CollectedSample] = []
        self._collection_start: float = 0.0
        self._last_sample_time: float = 0.0
        self._takeover_start: float = 0.0
        self._takeover_confirmed: bool = False
        self._post_train_cooldown: float = 0.0  # 训练后冷却期，避免误触发接管
        self._last_rf_result: Optional[Dict[str, float]] = None
        self._last_human_actions: Optional[Dict[str, float]] = None
        self._mlp_package: Optional[dict] = None
        self._mlp: Any = None
        self._mlp_feature_order: list = []
        self._mlp_loaded: bool = False
        self._last_training_report: dict = {}
        self._current_user_id: Optional[str] = None
        self._pending_user_id: Optional[str] = None
        self._pending_user_since: float = 0.0
        self._identity_votes: deque[tuple[float, str]] = deque()
        self._last_identity_switch_time: float = 0.0
        self._expected_command: Optional[Dict[str, float]] = None
        self._expected_command_time: float = 0.0
        self._expected_command_acknowledged: bool = False
        self._command_ack_timeout_logged: bool = False
        self._preference_profile: dict = {}
        self._last_ai_state: str = "on"
        self._external_session_exists: bool = False
        self._external_collection_active: bool = False
        self._external_collection_elapsed: float = 0.0
        self._external_collection_segment_start: float = 0.0
        self._external_resume_state: PreferenceState = PreferenceState.RF_ONLY
        self._external_training_triggered: bool = False
        self._external_off_cycle_consumed: bool = False

    # ============================================================
    # 每帧调用：检测接管 + 收集样本
    # ============================================================
    def update(
        self,
        features: Dict[str, Any],
        rf_result: Optional[Dict[str, float]],
        can_feedback: Optional[Dict[str, float]],
        driver_id: Optional[str] = None,
        passenger_id: Optional[str] = None,
        ai_state: str = "on",
    ) -> Dict[str, Any]:
        """主循环每帧调用。

        Returns:
            dict with keys:
              - suppress_rf: bool, RF输出是否被抑制
              - use_human_control: bool, 是否使用人工控制值
              - human_actions: dict or None, 人工控制值
              - state: str, 当前状态
              - trigger_training: bool, 是否触发训练
              - takeover_pending: bool, 是否接管确认中
              - user_changed: bool, 本帧是否发生了用户切换
        """
        with self._lock:
            self._last_rf_result = rf_result
            ai_state = ai_state if ai_state in ("on", "off") else "on"
            if ai_state == "on":
                self._pause_external_collection()
                if self._last_ai_state == "off":
                    self._external_off_cycle_consumed = False
            self._last_ai_state = ai_state

            # Face API会短暂输出None或错误ID。只有同一候选ID持续稳定后才切换，
            # 并且主驾确认不再依赖副驾是否稳定。
            user_changed, identity_pending = self._update_user_identity(driver_id)
            if user_changed:
                return {
                    "suppress_rf": False,
                    "use_human_control": False,
                    "human_actions": None,
                    "state": self._state.value,
                    "trigger_training": False,
                    "takeover_pending": False,
                    "identity_pending": False,
                    "user_changed": True,
                    "external_control": ai_state == "off",
                }
            if identity_pending:
                return {
                    "suppress_rf": True,
                    "use_human_control": False,
                    "human_actions": None,
                    "state": self._state.value,
                    "trigger_training": False,
                    "takeover_pending": False,
                    "identity_pending": True,
                    "user_changed": False,
                    "external_control": ai_state == "off",
                }

            if ai_state == "off":
                response = self._handle_external_control(features, rf_result, can_feedback)
                response.setdefault("user_changed", False)
                response.setdefault("identity_pending", False)
                response["external_control"] = True
                return response

            if self._state == PreferenceState.RF_ONLY:
                response = self._handle_rf_only(features, rf_result, can_feedback)

            elif self._state == PreferenceState.COLLECTING:
                response = self._handle_collecting(features, rf_result, can_feedback)

            elif self._state == PreferenceState.TRAINING:
                response = {"suppress_rf": False, "use_human_control": False,
                            "human_actions": None, "state": "training", "trigger_training": False}

            elif self._state == PreferenceState.JOINT_ACTIVE:
                response = self._handle_joint_active(features, rf_result, can_feedback)
            else:
                response = self._default_response()

            response.setdefault("user_changed", False)
            response.setdefault("identity_pending", False)
            response.setdefault("external_control", False)
            return response

    def _preferred_inference_state(self) -> PreferenceState:
        return PreferenceState.JOINT_ACTIVE if self._mlp_loaded else PreferenceState.RF_ONLY

    def _external_elapsed(self, now: Optional[float] = None) -> float:
        elapsed = self._external_collection_elapsed
        if self._external_collection_active:
            elapsed += (now or time.time()) - self._external_collection_segment_start
        return elapsed

    def _pause_external_collection(self) -> None:
        if (not self._external_collection_active and
                not self._external_training_triggered):
            return
        if self._external_training_triggered:
            if self._state == PreferenceState.TRAINING:
                self._state = self._external_resume_state
                self._clear_command_baseline()
                logger.info("[偏好层] AI恢复on，训练在后台继续，立即恢复AI控制")
            return
        now = time.time()
        self._external_collection_elapsed = self._external_elapsed(now)
        self._external_collection_segment_start = 0.0
        self._external_collection_active = False
        if self._state == PreferenceState.COLLECTING:
            self._state = self._external_resume_state
        self._clear_command_baseline()
        logger.info(
            f"[偏好层] AI恢复on，暂停CDC采集: "
            f"累计{self._external_collection_elapsed:.1f}秒/{len(self._samples)}条"
        )

    def _handle_external_control(self, features, rf_result, cdc_actions) -> dict:
        """AI off显式接管：立即采集CDC动作，不走AI命令差异检测。"""
        if self._external_off_cycle_consumed:
            return {
                "suppress_rf": True,
                "use_human_control": cdc_actions is not None,
                "human_actions": cdc_actions,
                "state": self._state.value,
                "trigger_training": False,
            }

        if not self._external_session_exists:
            # 不混合此前由AC差异触发的旧采集会话。
            if self._state == PreferenceState.COLLECTING:
                self._samples.clear()
                self._last_human_actions = None
            if self._state == PreferenceState.TRAINING:
                self._external_off_cycle_consumed = True
                return {
                    "suppress_rf": True, "use_human_control": cdc_actions is not None,
                    "human_actions": cdc_actions, "state": "training",
                    "trigger_training": False,
                }
            self._external_resume_state = (
                self._state if self._state in (PreferenceState.RF_ONLY, PreferenceState.JOINT_ACTIVE)
                else self._preferred_inference_state()
            )
            self._external_session_exists = True
            self._external_collection_elapsed = 0.0
            self._external_training_triggered = False
            self._collection_start = time.time()
            self._last_sample_time = 0.0

        if not self._external_collection_active and not self._external_training_triggered:
            self._external_collection_active = True
            self._external_collection_segment_start = time.time()
            self._state = PreferenceState.COLLECTING
            self._clear_command_baseline()
            logger.info(
                f"[偏好层] AI off → CDC人工接管，"
                f"恢复累计{self._external_collection_elapsed:.1f}秒/{len(self._samples)}条"
            )

        if self._external_training_triggered:
            return {
                "suppress_rf": True, "use_human_control": cdc_actions is not None,
                "human_actions": cdc_actions, "state": self._state.value,
                "trigger_training": False,
            }

        response = self._handle_collecting(features, rf_result, cdc_actions)
        response["external_control"] = True
        return response

    def _clear_external_session(self, clear_samples: bool = True) -> None:
        self._external_session_exists = False
        self._external_collection_active = False
        self._external_collection_elapsed = 0.0
        self._external_collection_segment_start = 0.0
        self._external_training_triggered = False
        self._external_resume_state = self._preferred_inference_state()
        if clear_samples:
            self._samples.clear()
            self._last_human_actions = None
            self._last_sample_time = 0.0

    def _update_user_identity(self, driver_id: Optional[str]) -> tuple[bool, bool]:
        """对主驾FaceID做滚动多数票和切换滞回。

        首次上车仍使用连续帧快速确认；已有用户之间的切换必须在时间窗口内
        占明显多数，避免同一张脸被短暂识别成其他ID时来回加载MLP。
        None和空ID只视为暂时丢帧，不立即卸载当前模型。
        """
        now = time.time()
        driver_id = str(driver_id).strip() if driver_id else None

        if driver_id:
            self._identity_votes.append((now, driver_id))
        cutoff = now - self.IDENTITY_VOTE_WINDOW_SECONDS
        while self._identity_votes and self._identity_votes[0][0] < cutoff:
            self._identity_votes.popleft()

        if driver_id is None:
            self._pending_user_id = None
            self._pending_user_since = 0.0
            return False, False

        # 进程刚启动、尚未绑定用户时保留原有快速确认，避免首位用户等待整个投票窗口。
        if self._current_user_id is None:
            if driver_id != self._pending_user_id:
                self._pending_user_id = driver_id
                self._pending_user_since = now
                logger.info(f"[偏好层] FaceID初始候选: {driver_id}，等待稳定确认")
                return False, True

            has_user_model = (self._user_mlp_base_dir / driver_id / "mlp_model.joblib").exists()
            confirm_seconds = (
                self.IDENTITY_CONFIRM_SECONDS if has_user_model else self.NEW_USER_CONFIRM_SECONDS
            )
            if now - self._pending_user_since < confirm_seconds:
                return False, True

            self._pending_user_id = None
            self._pending_user_since = 0.0
            self._identity_votes.clear()
            self._identity_votes.append((now, driver_id))
            self._last_identity_switch_time = now
            self._on_user_change(driver_id)
            return True, False

        if driver_id == self._current_user_id:
            self._pending_user_id = None
            self._pending_user_since = 0.0
            return False, False

        # 刚切换完的一小段时间内锁定当前用户，吸收Face API的瞬时抖动。
        if now - self._last_identity_switch_time < self.IDENTITY_SWITCH_COOLDOWN_SECONDS:
            self._pending_user_id = None
            self._pending_user_since = 0.0
            return False, False

        if len(self._identity_votes) < self.IDENTITY_VOTE_MIN_SAMPLES:
            self._pending_user_id = None
            self._pending_user_since = 0.0
            return False, False

        counts = Counter(candidate for _, candidate in self._identity_votes)
        candidate_id, candidate_count = counts.most_common(1)[0]
        vote_ratio = candidate_count / len(self._identity_votes)
        if candidate_id == self._current_user_id or vote_ratio < self.IDENTITY_VOTE_RATIO:
            self._pending_user_id = None
            self._pending_user_since = 0.0
            return False, False

        # 多数票成立后再保留原有连续确认，防止刚跨过阈值的一帧立即触发切换。
        driver_id = candidate_id
        if driver_id != self._pending_user_id:
            self._pending_user_id = driver_id
            self._pending_user_since = now
            logger.info(
                f"[偏好层] FaceID多数候选: {driver_id} "
                f"({candidate_count}/{len(self._identity_votes)}, {vote_ratio:.0%})，等待稳定确认"
            )
            return False, True

        # 已训练用户优先快速切换；陌生ID多观察几秒，避免识别抖动把现有模型卸载。
        has_user_model = (self._user_mlp_base_dir / driver_id / "mlp_model.joblib").exists()
        confirm_seconds = (
            self.IDENTITY_CONFIRM_SECONDS if has_user_model else self.NEW_USER_CONFIRM_SECONDS
        )
        if now - self._pending_user_since < confirm_seconds:
            return False, True

        self._pending_user_id = None
        self._pending_user_since = 0.0
        self._identity_votes.clear()
        self._identity_votes.append((now, driver_id))
        self._last_identity_switch_time = now
        self._on_user_change(driver_id)
        return True, False

    def note_command_sent(self, command: Dict[str, float], timestamp: Optional[float] = None) -> None:
        """登记实际成功写入Socket的最终命令；CAN回读确认前禁止触发人工接管。"""
        with self._lock:
            self._expected_command = {
                "driver_temp": float(command["driver_temp"]),
                "passenger_temp": float(command["passenger_temp"]),
                "wind_speed": float(command["wind_speed"]),
                "air_mode": float(command["air_mode"]),
            }
            self._expected_command_time = timestamp or time.time()
            self._expected_command_acknowledged = False
            self._command_ack_timeout_logged = False
            self._takeover_start = 0.0
            self._takeover_confirmed = False

    @property
    def state(self) -> PreferenceState:
        """当前状态（只读，供主循环采集期采样路由使用，2026-08-03 P4）。"""
        return self._state

    def _clear_command_baseline(self) -> None:
        """清空接管检测的命令基线（2026-08-03 P3）。

        训练完成/状态切换后，接管前最后一次AI命令(如19.5°C/风量6)若继续作为
        基线，恢复推理时会立刻与人工设定(24.0/风量3)产生phantom不一致，导致
        每次训练完秒级二次接管(2026-08-03 12:06车端实测复现)。清空后接管检测
        保持关闭，直到本阶段第一条命令真实发送(note_command_sent)并被CAN确认。
        """
        self._expected_command = None
        self._expected_command_time = 0.0
        self._expected_command_acknowledged = False
        self._command_ack_timeout_logged = False

    # ----------------------------------------------------------
    # 状态处理
    # ----------------------------------------------------------
    def _handle_rf_only(self, features, rf_result, can_feedback) -> dict:
        if rf_result is None or can_feedback is None:
            return self._default_response()

        # 2026-08-03 P3: 同JOINT_ACTIVE，仅在真实发送过命令后才检测接管，
        # 防止训练失败回退RF_ONLY时旧命令被误当基线。
        if self._expected_command is not None and self._detect_takeover(rf_result, can_feedback):
            if self._takeover_start == 0.0:
                self._takeover_start = time.time()
                logger.info("[偏好层] 检测到CAN反馈与RF推荐不一致，开始接管确认计时")
                return {"suppress_rf": True, "use_human_control": False,
                        "human_actions": None, "state": "rf_only",
                        "trigger_training": False, "takeover_pending": True}

            if time.time() - self._takeover_start >= self.TAKEOVER_CONFIRM_SECONDS:
                self._takeover_confirmed = True
                self._state = PreferenceState.COLLECTING
                self._collection_start = time.time()
                self._samples.clear()
                self._last_human_actions = can_feedback.copy()
                logger.info(
                    f"[偏好层] RF_ONLY → COLLECTING: 人工接管已确认 "
                    f"(持续{time.time() - self._takeover_start:.1f}秒), "
                    f"人工动作={can_feedback}"
                )
                return {
                    "suppress_rf": True,
                    "use_human_control": True,
                    "human_actions": can_feedback,
                    "state": "collecting",
                    "trigger_training": False,
                }
            return {"suppress_rf": True, "use_human_control": False,
                    "human_actions": None, "state": "rf_only",
                    "trigger_training": False, "takeover_pending": True}
        else:
            self._takeover_start = 0.0
            return self._default_response()

    def _handle_collecting(self, features, rf_result, can_feedback) -> dict:
        elapsed = (
            self._external_elapsed()
            if getattr(self, "_external_session_exists", False)
            else time.time() - self._collection_start
        )

        # CAN反馈断连时沿用上次有效值
        if can_feedback and all(v is not None for v in can_feedback.values()):
            self._last_human_actions = can_feedback
        effective_actions = self._last_human_actions or can_feedback

        # 每5秒生成一条训练样本
        now = time.time()
        if now - self._last_sample_time >= self.SAMPLE_INTERVAL and rf_result is not None and effective_actions:
            sample = self._make_sample(features, rf_result, effective_actions)
            if sample is not None:
                self._samples.append(sample)
                self._last_sample_time = now

        # 20分钟满 → 触发训练
        if elapsed >= self.COLLECTION_DURATION and len(self._samples) >= self.MIN_TRAINING_SAMPLES:
            logger.info(
                f"[偏好层] COLLECTING → TRAINING: 已收集{len(self._samples)}条样本 "
                f"({elapsed:.0f}秒)"
            )
            self._state = PreferenceState.TRAINING
            if getattr(self, "_external_session_exists", False):
                self._external_collection_elapsed = elapsed
                self._external_collection_segment_start = 0.0
                self._external_collection_active = False
                self._external_training_triggered = True
                self._external_off_cycle_consumed = True
            return {
                "suppress_rf": True,
                "use_human_control": True,
                "human_actions": self._last_human_actions,
                "state": "collecting",
                "trigger_training": True,
            }

        return {
            "suppress_rf": True,
            "use_human_control": True,
            "human_actions": effective_actions,
            "state": "collecting",
            "trigger_training": False,
        }

    def _handle_joint_active(self, features, rf_result, can_feedback) -> dict:
        # 联合推理期间仍检测接管，用户再次调节则重新收集→训练
        if rf_result is None or can_feedback is None:
            return {"suppress_rf": False, "use_human_control": False,
                    "human_actions": None, "state": "joint_active", "trigger_training": False}

        # 2026-08-03 P2: 移除训练后/切换后60秒冷却早退。
        # 原逻辑冷却期内完全不做接管检测, 但AI仍每5s发送, 用户在冷却窗内
        # 调温会被无限顶回(2026-08-03 11:27实测复现)。接管误判已由
        # ack门控+4s兜底覆盖, 冷却窗冗余且有害。

        # 2026-08-03 P3: 仅在本阶段真实发送过命令(_expected_command非空)后才
        # 检测接管; 否则调用方传入的rf_result可能是接管前的旧命令，会被误当
        # 待确认基线，造成训练后秒级phantom接管(12:06实测)。
        if self._expected_command is not None and self._detect_takeover(rf_result, can_feedback):
            if self._takeover_start == 0.0:
                self._takeover_start = time.time()
                logger.info("[偏好层] JOINT_ACTIVE期间检测到CAN不一致，开始接管确认计时")
                return {"suppress_rf": True, "use_human_control": False,
                        "human_actions": None, "state": "joint_active",
                        "trigger_training": False, "takeover_pending": True}

            if time.time() - self._takeover_start >= self.TAKEOVER_CONFIRM_SECONDS:
                self._takeover_confirmed = True
                self._state = PreferenceState.COLLECTING
                self._collection_start = time.time()
                self._samples.clear()
                self._last_human_actions = can_feedback.copy()
                logger.info(
                    f"[偏好层] JOINT_ACTIVE → COLLECTING: 二次接管已确认 "
                    f"(持续{time.time() - self._takeover_start:.1f}秒), "
                    f"人工动作={can_feedback}"
                )
                return {
                    "suppress_rf": True,
                    "use_human_control": True,
                    "human_actions": can_feedback,
                    "state": "collecting",
                    "trigger_training": False,
                }
            return {"suppress_rf": True, "use_human_control": False,
                    "human_actions": None, "state": "joint_active",
                    "trigger_training": False, "takeover_pending": True}
        else:
            self._takeover_start = 0.0
            return {"suppress_rf": False, "use_human_control": False,
                    "human_actions": None, "state": "joint_active", "trigger_training": False}

    # ----------------------------------------------------------
    # 训练
    # ----------------------------------------------------------
    def get_training_dataframe(self) -> pd.DataFrame:
        """返回用于训练的DataFrame（26特征 + 4人工动作列）。"""
        with self._lock:
            if not self._samples:
                raise ValueError("没有训练样本")
            rows = []
            for s in self._samples:
                row = {**s.features, **s.human_actions}
                rows.append(row)
            return pd.DataFrame(rows)

    def on_training_done(self, report: dict) -> None:
        """训练完成后调用，加载新MLP或回退。

        训练结果已由主控制器按用户路径保存（models/user_mlps/{user_id}/），
        此处从对应路径加载模型。
        仅当训练用户与当前用户一致时才切换状态，避免用户切换导致状态错乱。
        """
        with self._lock:
            was_external_training = getattr(self, "_external_training_triggered", False)
            self._last_training_report = report
            trained_user_id = report.get("user_id") or self._current_user_id

            # 用户已切换 → 训练数据属于旧用户，仅保存模型，不改变当前状态
            if trained_user_id and trained_user_id != self._current_user_id:
                logger.info(
                    f"[偏好层] 训练完成但用户已切换 "
                    f"({trained_user_id} → {self._current_user_id}), "
                    f"模型已保存到 {trained_user_id} 目录，不切换状态"
                )
                if was_external_training:
                    self._clear_external_session(clear_samples=True)
                return

            if report.get("success"):
                # 有用户ID → 从用户路径加载；否则从通用路径加载
                if trained_user_id:
                    loaded = self._load_mlp_for_user(trained_user_id)
                else:
                    loaded = self._load_mlp()
                if loaded:
                    self._state = PreferenceState.JOINT_ACTIVE
                    self._takeover_start = 0.0
                    self._clear_command_baseline()  # P3: 清除过期命令基线，防训练后秒级phantom接管
                    self._post_train_cooldown = time.time() + 60.0  # 训练后60秒冷却，等AC充分响应MLP指令
                    logger.info(
                        f"[偏好层] TRAINING → JOINT_ACTIVE: "
                        f"base_score={report.get('base_score', '?')} → "
                        f"updated_score={report.get('updated_score', '?')}"
                    )
                else:
                    self._state = PreferenceState.RF_ONLY
                    self._takeover_start = 0.0
                    self._clear_command_baseline()  # P3: 回退RF_ONLY同样清除过期基线
                    logger.warning(
                        f"[偏好层] 训练完成但MLP加载失败，回退到 RF_ONLY"
                    )
            else:
                self._state = PreferenceState.RF_ONLY
                self._takeover_start = 0.0
                self._clear_command_baseline()  # P3: 回退RF_ONLY同样清除过期基线
                logger.warning(
                    f"[偏好层] 训练未通过验证，回退到 RF_ONLY: {report.get('reason', '?')}"
                )
            if was_external_training:
                self._clear_external_session(clear_samples=True)
                self._external_off_cycle_consumed = self._last_ai_state == "off"

    # ----------------------------------------------------------
    # MLP联合推理
    # ----------------------------------------------------------
    def apply_mlp_residual(self, features: Dict[str, Any], rf_result: Dict[str, float]) -> Optional[Dict[str, float]]:
        """在RF结果上叠加MLP残差，返回最终动作。

        force_zero_mlp_residuals启用时，四头输出统一保持基础推荐。
        否则个性化仅作用于主驾温度，风量和模式沿用MLP联合控制策略。
        无MLP时返回None。
        """
        if not self._mlp_loaded or self._mlp is None:
            return None

        if self.FORCE_ZERO_MLP_RESIDUALS:
            return {
                "driver_temp": float(rf_result["driver_temp"]),
                "passenger_temp": float(rf_result["passenger_temp"]),
                "wind_speed": float(rf_result["wind_speed"]),
                "air_mode": float(rf_result["air_mode"]),
                "driver_delta": 0.0,
                "passenger_delta": 0.0,
                "wind_delta": 0,
                "mode_class": 0,
            }

        try:
            feature_order = [*self._feature_columns, *self.BASE_COLUMNS]
            input_vec = []
            for col in feature_order:
                if col == "base_driver_temperature":
                    input_vec.append(float(rf_result["driver_temp"]))
                elif col == "base_passenger_temperature":
                    input_vec.append(float(rf_result["passenger_temp"]))
                elif col == "base_wind":
                    input_vec.append(float(rf_result["wind_speed"]))
                elif col == "base_mode":
                    input_vec.append(float(rf_result["air_mode"]))
                else:
                    input_vec.append(float(features.get(col, 0.0) or 0.0))

            x = np.array([input_vec], dtype=float)
            probs = self._mlp.predict_proba(x)

            from src.preference_learning.online_ordinal_mlp import TEMPERATURE_DELTAS, WIND_DELTAS

            def _ordered_class(prob, base, deltas, lo, hi):
                b = np.atleast_1d(np.asarray(base, dtype=float))
                legal = (b[:, None] + deltas[None, :] >= lo) & (b[:, None] + deltas[None, :] <= hi)
                raw_classes = prob.argmax(axis=1)
                adj = prob.copy()
                adj[~legal] = 0.0
                total = adj.sum(axis=1, keepdims=True)
                empty = total[:, 0] <= 0
                if empty.any():
                    adj[empty] = legal[empty].astype(float)
                    total = adj.sum(axis=1, keepdims=True)
                adj /= np.where(total > 0, total, 1.0)
                expected = np.sum(adj * deltas[None, :], axis=1)
                dist = np.abs(expected[:, None] - deltas[None, :])
                dist[~legal] = np.inf
                decoded = dist.argmin(axis=1)
                for row_index, raw_class in enumerate(raw_classes):
                    if legal[row_index, raw_class]:
                        continue
                    legal_classes = np.flatnonzero(legal[row_index])
                    decoded[row_index] = legal_classes[
                        np.abs(deltas[legal_classes] - deltas[raw_class]).argmin()
                    ]
                return deltas[decoded]

            driver_delta = float(_ordered_class(probs[0],
                float(rf_result["driver_temp"]), TEMPERATURE_DELTAS, 16.0, 31.0)[0])
            # 副驾不应用个性化残差。保留MLP的四头结构以兼容现有模型文件，
            # 但推理输出固定采用基础RF副驾温度。
            passenger_delta = 0.0
            wind_delta = int(_ordered_class(probs[2],
                float(rf_result["wind_speed"]), WIND_DELTAS, 1.0, 9.0)[0])
            mode_class = int(probs[3].argmax(axis=1)[0])

            final_driver = np.clip(float(rf_result["driver_temp"]) + driver_delta, 16.0, 31.0)
            final_passenger = float(rf_result["passenger_temp"])
            final_wind = int(np.clip(int(rf_result["wind_speed"]) + wind_delta, 1, 9))
            final_mode = int(rf_result["air_mode"]) if mode_class == 0 else mode_class

            # preference_profile 仅用于记录/诊断最近一次人工偏好，不能覆盖
            # 实时RF+MLP结果。否则同一用户在不同工况下的动态偏好会被固定值压平。

            return {
                "driver_temp": round(final_driver * 2) / 2,
                "passenger_temp": final_passenger,
                "wind_speed": float(final_wind),
                "air_mode": float(final_mode),
                "driver_delta": driver_delta,
                "passenger_delta": passenger_delta,
                "wind_delta": wind_delta,
                "mode_class": mode_class,
            }
        except Exception as e:
            logger.error(f"[偏好层] MLP推理失败: {e}", exc_info=True)
            return None

    def _load_mlp(self) -> bool:
        """兼容旧接口：从通用输出目录加载MLP（非用户绑定）。"""
        mlp_path = self._output_dir / "mlp_model.joblib"
        if not mlp_path.exists():
            logger.warning(f"[偏好层] MLP模型不存在: {mlp_path}")
            self._mlp_loaded = False
            return False
        try:
            import joblib
            package = joblib.load(mlp_path)
            self._mlp_package = package
            self._mlp = package["model"]
            self._mlp_feature_order = list(package.get("feature_order", []))
            self._preference_profile = dict(package.get("preference_profile", {}))
            self._mlp_loaded = True
            logger.info(f"[偏好层] MLP模型已加载(通用路径): {mlp_path}")
            return True
        except Exception as e:
            logger.error(f"[偏好层] MLP模型加载失败: {e}", exc_info=True)
            self._mlp_loaded = False
            return False

    # ----------------------------------------------------------
    # 接管检测
    # ----------------------------------------------------------
    def _detect_takeover(self, rf_result: Optional[Dict[str, float]],
                         can_feedback: Optional[Dict[str, float]]) -> bool:
        if rf_result is None or can_feedback is None:
            return False

        # 只对已成功发送且被CAN反馈确认过的命令做接管判断。
        # Socket发送失败、CAN服务重启或执行器尚未落稳，都不能解释为人工操作。
        if self._expected_command is None:
            # 兼容直接调用管理器的离线测试/集成方：传入的rf_result视为待确认命令。
            # 车端主循环在发送失败时传None，因此不会绕过发送成功门控。
            self._expected_command = {
                "driver_temp": float(rf_result["driver_temp"]),
                "passenger_temp": float(rf_result["passenger_temp"]),
                "wind_speed": float(rf_result["wind_speed"]),
                "air_mode": float(rf_result["air_mode"]),
            }
            self._expected_command_time = time.time()
            self._expected_command_acknowledged = False
            self._command_ack_timeout_logged = False

        rf_result = self._expected_command

        model_driver = float(rf_result.get("driver_temp", 0))
        model_passenger = float(rf_result.get("passenger_temp", 0))
        model_wind = int(float(rf_result.get("wind_speed", 0) or 0))
        model_mode = int(float(rf_result.get("air_mode", 0) or 0))

        can_driver = float(can_feedback.get("driver_temp", 0))
        can_passenger = float(can_feedback.get("passenger_temp", 0))
        can_wind = int(float(can_feedback.get("wind_speed", 0) or 0))
        can_mode = int(float(can_feedback.get("air_mode", 0) or 0))

        # CAN反馈异常值保护：超出物理/协议范围视为信号未就绪。
        if not (16.0 <= can_driver <= 31.0 and 16.0 <= can_passenger <= 31.0):
            return False
        if not (1 <= can_wind <= 9 and 1 <= can_mode <= 7):
            return False

        mismatches = []
        if abs(model_driver - can_driver) > self.TEMP_TOLERANCE:
            mismatches.append("driver_temp")
        if abs(model_passenger - can_passenger) > self.TEMP_TOLERANCE:
            mismatches.append("passenger_temp")
        if model_wind != can_wind:
            mismatches.append("wind_speed")
        if model_mode != can_mode:
            mismatches.append("air_mode")

        if not mismatches:
            if not self._expected_command_acknowledged:
                logger.info("[偏好层] CAN已确认最近一次AI命令，开放人工接管检测")
            self._expected_command_acknowledged = True
            self._command_ack_timeout_logged = False
            return False

        if not self._expected_command_acknowledged:
            elapsed = time.time() - self._expected_command_time
            if elapsed >= self.ACK_MISMATCH_FALLBACK_SECONDS:
                # 2026-08-03 P2: ack超时且差异持续 → 判定为人工接管。
                # 命令已发出4s+, 执行器早已落地; 反馈仍不一致说明用户已改动,
                # 确认门继续关闭会导致AI无限顶回人工设定(当天实测复现)。
                logger.warning(
                    f"[偏好层] AI命令{elapsed:.1f}秒未获CAN回读确认且差异持续={mismatches}, "
                    f"按人工接管处理(P2兜底)"
                )
                self._command_ack_timeout_logged = True
                return True
            return False

        logger.info(f"[偏好层] 已确认AI命令后出现CAN变化: {mismatches}")
        return True

    # ----------------------------------------------------------
    # 用户切换 & 按faceID加载MLP
    # ----------------------------------------------------------
    def _on_user_change(self, user_id: str) -> None:
        """用户身份变化时的处理：加载该用户已有MLP或重置为RF_ONLY。"""
        old_id = self._current_user_id
        self._current_user_id = user_id
        self._expected_command = None
        self._expected_command_time = 0.0
        self._expected_command_acknowledged = False
        self._command_ack_timeout_logged = False
        self._preference_profile = {}
        self._clear_external_session(clear_samples=True)
        self._external_off_cycle_consumed = False
        logger.info(f"[偏好层] 用户切换: {old_id} → {user_id}")

        if user_id and self._load_mlp_for_user(user_id):
            self._state = PreferenceState.JOINT_ACTIVE
            self._takeover_start = 0.0
            self._takeover_confirmed = False
            self._samples.clear()
            self._post_train_cooldown = time.time() + 60.0  # 用户切换后60秒冷却，等AC充分响应
            logger.info(
                f"[偏好层] 加载用户 {user_id} 已有MLP → JOINT_ACTIVE"
            )
        else:
            self._state = PreferenceState.RF_ONLY
            self._takeover_start = 0.0
            self._takeover_confirmed = False
            self._samples.clear()
            self._last_rf_result = None
            self._last_human_actions = None
            self._mlp = None
            self._mlp_package = None
            self._mlp_feature_order = []
            self._mlp_loaded = False
            logger.info(
                f"[偏好层] 用户 {user_id} 无已有MLP → RF_ONLY (等待人工接管)"
            )

    def _load_mlp_for_user(self, user_id: str) -> bool:
        """加载指定用户的MLP模型 (models/user_mlps/{user_id}/mlp_model.joblib)。"""
        if not user_id:
            return False
        mlp_path = self._user_mlp_base_dir / user_id / "mlp_model.joblib"
        if not mlp_path.exists():
            return False
        try:
            import joblib
            package = joblib.load(mlp_path)
            self._mlp_package = package
            self._mlp = package["model"]
            self._mlp_feature_order = list(package.get("feature_order", []))
            self._preference_profile = dict(package.get("preference_profile", {}))
            profile_path = mlp_path.with_name("preference_profile.json")
            if profile_path.exists():
                try:
                    self._preference_profile.update(
                        json.loads(profile_path.read_text(encoding="utf-8"))
                    )
                except Exception as e:
                    logger.warning(f"[偏好层] 用户偏好摘要读取失败: {e}")
            self._mlp_loaded = True
            logger.info(
                f"[偏好层] 已加载用户 {user_id} 的MLP: {mlp_path}, "
                f"特征维度={len(self._mlp_feature_order)}"
            )
            return True
        except Exception as e:
            logger.error(f"[偏好层] 加载用户 {user_id} MLP失败: {e}", exc_info=True)
            self._mlp_loaded = False
            return False

    def get_current_user_id(self) -> Optional[str]:
        """获取当前绑定的用户ID。"""
        return self._current_user_id

    # ----------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------
    def _make_sample(self, features, rf_result, can_feedback) -> Optional[CollectedSample]:
        try:
            feat = {}
            for col in self._feature_columns:
                feat[col] = float(features.get(col, 0.0) or 0.0)
            if not feat or len(feat) < 10:
                logger.warning(f"[偏好层] 样本特征异常: features有{len(features)}个key, "
                               f"feature_columns={len(self._feature_columns)}个, "
                               f"features keys示例: {list(features.keys())[:5]}")
            actions = {
                "主驾温度显示": float(can_feedback.get("driver_temp", 16.0)),
                "副驾温度显示": float(can_feedback.get("passenger_temp", 16.0)),
                "UI界面_风速": float(can_feedback.get("wind_speed", 3.0)),
                "空调出风模式请求": float(can_feedback.get("air_mode", 1.0)),
            }
            base = {
                "base_driver_temperature": float(rf_result.get("driver_temp", 0)),
                "base_passenger_temperature": float(rf_result.get("passenger_temp", 0)),
                "base_wind": float(rf_result.get("wind_speed", 0)),
                "base_mode": float(rf_result.get("air_mode", 0)),
            }
            return CollectedSample(
                timestamp=time.time(),
                features=feat,
                human_actions=actions,
                base_predictions=base,
            )
        except Exception as e:
            logger.debug(f"[偏好层] 样本构造异常: {e}")
            return None

    @staticmethod
    def _default_response():
        return {
            "suppress_rf": False,
            "use_human_control": False,
            "human_actions": None,
            "state": "rf_only",
            "trigger_training": False,
            "takeover_pending": False,
            "identity_pending": False,
            "user_changed": False,
            "external_control": False,
        }

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------
    def get_status(self) -> PreferenceLayerStatus:
        with self._lock:
            return PreferenceLayerStatus(
                state=self._state,
                state_name=self._state.value,
                collecting_since=self._collection_start,
                sample_count=len(self._samples),
                mlp_loaded=self._mlp_loaded,
                last_training_report=self._last_training_report,
                human_takeover_active=(self._state in (PreferenceState.COLLECTING,)),
                takeover_pending=(self._takeover_start > 0 and not self._takeover_confirmed),
            )

    def set_feature_columns(self, columns: list) -> None:
        """延迟设置特征列（模型加载完成后调用）。

        启动时根据当前绑定的用户ID加载其MLP模型。
        若已有MLP则进入JOINT_ACTIVE，否则保持RF_ONLY。
        """
        self._feature_columns = list(columns)
        logger.info(f"[偏好层] 特征列已设置: {len(self._feature_columns)}维")
        if self._current_user_id and self._load_mlp_for_user(self._current_user_id):
            self._state = PreferenceState.JOINT_ACTIVE
            logger.info(
                f"[偏好层] 启动时加载用户 {self._current_user_id} 已有MLP → JOINT_ACTIVE"
            )

    def reset(self) -> None:
        with self._lock:
            self._state = PreferenceState.RF_ONLY
            self._samples.clear()
            self._collection_start = 0.0
            self._last_sample_time = 0.0
            self._takeover_start = 0.0
            self._takeover_confirmed = False
            self._last_rf_result = None
            self._last_human_actions = None
            self._pending_user_id = None
            self._pending_user_since = 0.0
            self._expected_command = None
            self._expected_command_time = 0.0
            self._expected_command_acknowledged = False
            self._command_ack_timeout_logged = False
            self._last_ai_state = "on"
            self._clear_external_session(clear_samples=True)
            self._external_off_cycle_consumed = False
            logger.info("[偏好层] 已重置")
