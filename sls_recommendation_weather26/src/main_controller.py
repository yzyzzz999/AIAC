#!/usr/bin/env python3
"""车载空调推荐系统主控制器：仅支持 mode4 + weather26。"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 保留 ``python src/main_controller.py`` 的兼容入口；包内导入统一从 src 开始。
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.inference_engine import InferenceEngine
from src.core.models.base_model_loader import RandomForestModelLoader
from src.preference_learning.preference_layer_manager import PreferenceLayerManager, PreferenceState
from src.services.result_sender.result_sender import ResultSender
from src.services.signal_fetchers.can_signal_fetcher import CanSignalFetcher
from src.services.signal_fetchers.face_recognition_client import FaceRecognitionClient
from src.services.signal_fetchers.pmv_client import PmvClient
from src.services.signal_fetchers.weather_info_fetcher import WeatherInfoFetcher
from src.utils.config_manager import get_config
from src.utils.logger_config import get_logger, setup_logging
from src.utils.occupant_recommendation_policy import (
    compose_occupant_recommendation,
    load_runtime_wind_weight,
)

logger = get_logger(__name__)


class MainController:
    """协调信号、weather26推理、偏好层训练与CAN下发。"""

    def __init__(self, demo_mode: bool = False, config_path: Optional[str] = None,
                 run_mode: str = "mode4", shadow_mode: bool = False,
                 model_package_mode: str = "weather26"):
        if run_mode != "mode4":
            raise ValueError(f"仅支持 run_mode=mode4，收到: {run_mode}")
        if model_package_mode != "weather26":
            raise ValueError(f"仅支持 model_package_mode=weather26，收到: {model_package_mode}")
        self.config = get_config(config_path)
        self.demo_mode = demo_mode or self.config.get("system.demo_mode", False)
        self.run_mode = "mode4"
        self.model_package_mode = "weather26"
        self._shadow_mode = bool(shadow_mode)

        self.can_fetcher = CanSignalFetcher()
        self.face_client = FaceRecognitionClient()
        self.weather_fetcher = WeatherInfoFetcher()
        self.pmv_client = PmvClient()
        model_dir = os.path.join(_PROJECT_ROOT, "models", "weather26")
        self.model_loader = RandomForestModelLoader(
            model_dir=model_dir,
            feature_config=os.path.join(model_dir, "manifest.json"),
            package_mode="weather26",
        )
        self.inference_engine = InferenceEngine(
            model_loader=self.model_loader,
            use_sliding_window=bool(self.config.get("system.use_sliding_window", True)),
            window_seconds=float(self.config.get("system.window_seconds", 5.0)),
        )
        pref_dir = self.config.get("preference_learning.pref_layer_model_dir", "models/preference_layer")
        self._preference_layer = PreferenceLayerManager(
            model_loader=self.model_loader,
            output_dir=os.path.join(_PROJECT_ROOT, pref_dir),
            collection_duration=float(self.config.get("preference_learning.pref_layer_collection_duration", 300)),
            min_training_samples=int(self.config.get("preference_learning.pref_layer_min_samples", 30)),
            takeover_confirm_seconds=float(self.config.get("preference_learning.pref_layer_takeover_confirm_seconds", 5.0)),
            identity_confirm_seconds=float(self.config.get("preference_learning.face_id_confirm_seconds", 1.5)),
            new_user_confirm_seconds=float(self.config.get("preference_learning.face_id_new_user_confirm_seconds", 4.0)),
            identity_vote_window_seconds=float(self.config.get("preference_learning.face_id_vote_window_seconds", 15.0)),
            identity_vote_min_samples=int(self.config.get("preference_learning.face_id_vote_min_samples", 10)),
            identity_vote_ratio=float(self.config.get("preference_learning.face_id_vote_ratio", 0.8)),
            identity_switch_cooldown_seconds=float(self.config.get("preference_learning.face_id_switch_cooldown_seconds", 20.0)),
            command_ack_timeout=float(self.config.get("preference_learning.command_ack_timeout", 8.0)),
            force_zero_mlp_residuals=bool(self.config.get("preference_learning.force_zero_mlp_residuals", True)),
        )
        self.result_sender = ResultSender()

        self.running = False
        self._main_loop_thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._inference_count = 0
        self._result_send_count = 0
        self._last_inference_time = 0.0
        self._last_base_inference_time = 0.0
        self._last_pref_inference_time = 0.0
        self._last_sampling_infer_time = 0.0
        self._ai_on_inference_interval = float(
            self.config.get("system.inference_interval", 2.0))
        self._ai_off_inference_interval = float(
            self.config.get("system.ai_off_inference_interval", 5.0))
        self._last_ai_off_carrier_time = 0.0
        self._last_observed_ai_state = "on"
        self._last_driver_id: Optional[str] = None
        self._last_passenger_id: Optional[str] = None
        self._last_applied_result: Optional[dict] = None
        self._last_rf_base_result: Optional[dict] = None
        self._last_command_result: Optional[dict] = None
        self._last_command_time = 0.0
        self._last_occupant_policy: Dict[str, object] = {}
        self._last_occupant_signature = None
        self._occupant_wind_runtime_path = Path(_PROJECT_ROOT) / "data" / "occupant_policy_runtime.json"
        self._last_sent_wind_speed: Optional[int] = None
        self._wind_max_delta = int(self.config.get("system.wind_max_delta_per_step", 1))
        self._wind_min = int(self.config.get("system.wind_min", 2))
        self._can_link_was_connected = True
        self._can_link_down_time = 0.0
        self._can_link_up_time = 0.0
        self._can_recover_guard_seconds = 10.0
        self._send_temp_jump_tolerance = 2.0
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start(self) -> None:
        logger.info("系统启动: mode4 + weather26")
        if not self.model_loader.load_all():
            raise RuntimeError(self.model_loader.get_error() or "weather26模型加载失败")
        self._preference_layer.set_feature_columns(self.model_loader.get_feature_columns())
        self.can_fetcher.start()
        time.sleep(1.0)
        self.face_client.start_auto_refresh(interval=self.config.get("face_api.refresh_interval", 1.0))
        self.pmv_client.start_auto_refresh(interval=self.config.get("pmv_api.refresh_interval", 1.0))
        self.weather_fetcher.start_auto_refresh()
        if self.config.get("result.enabled", True) and not self._shadow_mode:
            self.result_sender.connect()
        elif self._shadow_mode:
            logger.info("Shadow模式：只推理，不下发控制帧")
        self.running = True
        self._start_time = time.time()
        self._main_loop_thread = threading.Thread(target=self._main_loop, daemon=True, name="mode4-main-loop")
        self._main_loop_thread.start()
        # start_all.sh 以该固定日志作为推荐服务的就绪标记，请保持文本兼容。
        logger.info("=" * 60)
        logger.info("系统启动完成，开始运行")
        logger.info("=" * 60)

    def stop(self) -> None:
        self.running = False
        self.can_fetcher.stop()
        self.face_client.stop_auto_refresh()
        self.pmv_client.stop_auto_refresh()
        self.weather_fetcher.stop_auto_refresh()
        self.result_sender.disconnect()
        if self._main_loop_thread and self._main_loop_thread.is_alive():
            self._main_loop_thread.join(timeout=2.0)
        logger.info("系统已停止")

    def _signal_handler(self, signum, frame) -> None:
        logger.info(f"收到信号 {signum}，开始退出")
        self.stop()
        raise SystemExit(0)

    def _main_loop(self) -> None:
        loop_tick = float(self.config.get("system.main_loop_tick", 1.0))
        while self.running:
            try:
                self._update_can_link_state()
                now = time.time()
                ai_state = self.can_fetcher.get_ai_state()
                ac_feedback = self._get_current_hvac_from_can()
                if ai_state != self._last_observed_ai_state:
                    if ai_state == "off":
                        self._last_ai_off_carrier_time = 0.0
                    self._last_observed_ai_state = ai_state
                if ai_state == "off":
                    self._check_ai_off_carrier(now, ac_feedback)

                presence = self.face_client.get_driver_presence()
                if not (presence.get("driver_exists") or presence.get("passenger_exists")):
                    self._last_driver_id = self._last_passenger_id = None
                    time.sleep(loop_tick)
                    continue
                identity = self.face_client.get_identity_info()
                driver_id, passenger_id = identity.get("driver_id"), identity.get("passenger_id")
                if self.inference_engine.window_cache is not None:
                    can = self.can_fetcher.get_all_features(
                        demo_mode=self.demo_mode, weather_features=self.weather_fetcher.get_features())
                    if can:
                        self.inference_engine.add_can_samples(can)
                    face = self.face_client.get_all_features(demo_mode=self.demo_mode, for_inference=True)
                    if face:
                        self.inference_engine.set_person_signals(face)

                expected = self._last_command_result
                if (ai_state == "off" and
                        self._preference_layer.state == PreferenceState.COLLECTING and
                        time.time() - self._last_sampling_infer_time >= 5.0):
                    sampled = self._run_rf_inference_for_sampling()
                    self._last_sampling_infer_time = time.time()
                    if sampled is not None:
                        expected = sampled
                control_feedback = (
                    self.can_fetcher.get_cdc_hvac_actions(ac_feedback)
                    if ai_state == "off" else ac_feedback
                )
                pref = self._preference_layer.update(
                    features=self._get_current_features_dict(), rf_result=expected,
                    can_feedback=control_feedback, driver_id=driver_id,
                    passenger_id=passenger_id, ai_state=ai_state)
                if pref.get("user_changed"):
                    self._last_driver_id = self._preference_layer.get_current_user_id()
                    self._last_passenger_id = passenger_id
                    self._last_command_result = None
                    self._last_command_time = 0.0
                    self._last_sent_wind_speed = None
                    self._last_base_inference_time = self._last_pref_inference_time = 0.0
                elif not pref.get("identity_pending"):
                    self._last_driver_id = self._preference_layer.get_current_user_id() or driver_id
                    self._last_passenger_id = passenger_id
                if pref.get("trigger_training"):
                    samples = self._preference_layer.get_training_dataframe()
                    user_id = self._preference_layer.get_current_user_id() or "default"
                    threading.Thread(target=self._run_preference_training,
                                     args=(samples, user_id), daemon=True,
                                     name="pref-layer-train").start()
                if pref.get("use_human_control") and pref.get("human_actions"):
                    human = pref["human_actions"]
                    self._last_sent_wind_speed = int(float(human.get("wind_speed") or 3))
                    self._last_command_result = None
                    logger.info(f"[偏好层] 人工控制采集中: {human}")
                if ai_state == "off":
                    time.sleep(loop_tick)
                    continue
                if (pref.get("state") in ("collecting", "training") or
                        pref.get("takeover_pending") or pref.get("identity_pending")):
                    time.sleep(loop_tick)
                    continue
                self._check_scheduled_inference(time.time())
                time.sleep(loop_tick)
            except Exception as exc:
                logger.error(f"主循环异常: {exc}", exc_info=True)
                time.sleep(loop_tick)

    def _collect_inference_inputs(self):
        weather = self.weather_fetcher.get_features()
        return (
            self.can_fetcher.get_all_features(demo_mode=self.demo_mode, weather_features=weather),
            self.face_client.get_all_features(demo_mode=self.demo_mode, for_inference=True),
            weather,
            self.pmv_client.get_features(demo_mode=self.demo_mode),
            self.face_client.get_driver_presence(),
        )

    def _infer_rf(self):
        can, face, weather, pmv, presence = self._collect_inference_inputs()
        result = self.inference_engine.infer(
            can_features=can, face_features=face, weather_features=weather,
            pmv_features=pmv, demo_mode=self.demo_mode, presence=presence,
            pmv_info=self.pmv_client.format_for_log(
                bool(presence.get("driver_exists")), bool(presence.get("passenger_exists"))))
        return result, presence

    def _run_base_inference(self) -> bool:
        result, presence = self._infer_rf()
        if not result.ready:
            logger.info(f"[基础模型推理] 未就绪: {result.missing[:10]}")
            return False
        base = self._apply_occupant_recommendation_policy({
            "driver_temp": result.driver_temp, "passenger_temp": result.passenger_temp,
            "wind_speed": result.wind_speed, "air_mode": result.air_mode}, presence)
        base = self._apply_mode4_temperature_policy(base, base, presence)
        self._last_applied_result = self._last_rf_base_result = dict(base)
        self._inference_count += 1
        self._last_inference_time = time.time()
        return self._send_mode4_result(base, "base_rf")

    def _send_mode4_result(self, result: dict, stage: str) -> bool:
        # Shadow模式也保留最新推理快照，但只有真实发送成功才建立接管基线。
        self._last_applied_result = dict(result)
        should_send = self.config.get("result.enabled", True) and not self._shadow_mode
        if should_send and not self._can_link_healthy_for_send():
            logger.warning("[链路门控] CAN断开或反馈过期，本次结果不发送")
            should_send = False
        if not should_send:
            return True
        output = dict(result)
        output["wind_speed"] = str(self._smooth_wind_speed(output["wind_speed"]))
        output["air_mode"] = str(int(float(output["air_mode"])))
        if self._send_jump_suspect(output):
            logger.warning(f"[链路门控] 恢复窗口温度跳变超限，丢弃: {output}")
            return False
        if not self.result_sender.send_from_result_dict(output):
            return False
        self._result_send_count += 1
        self._last_sent_wind_speed = int(output["wind_speed"])
        self._log_occupant_wind_blend(self._last_sent_wind_speed)
        actual = {"driver_temp": float(output["driver_temp"]),
                  "passenger_temp": float(output["passenger_temp"]),
                  "wind_speed": float(output["wind_speed"]),
                  "air_mode": float(output["air_mode"])}
        self._last_applied_result = self._last_command_result = actual
        self._last_command_time = time.time()
        self._preference_layer.note_command_sent(actual, self._last_command_time)
        logger.info(f"[模式4最终下发] stage={stage}, {actual}")
        return True

    def _check_scheduled_inference(self, now: float) -> None:
        status = self._preference_layer.get_status()
        if status.takeover_pending or status.human_takeover_active:
            return
        inference_interval = self._current_inference_interval()
        if status.state_name == "joint_active":
            if now - self._last_pref_inference_time >= inference_interval:
                self._run_joint_inference_mode4()
                self._last_pref_inference_time = now
        elif now - self._last_base_inference_time >= inference_interval:
            self._run_base_inference()
            self._last_base_inference_time = now

    def _current_inference_interval(self) -> float:
        """AI开启时每2秒下发，关闭时每5秒下发。"""
        if self.can_fetcher.get_ai_state() == "off":
            return self._ai_off_inference_interval
        return self._ai_on_inference_interval

    def _check_ai_off_carrier(
        self, now: float, ac_feedback: Optional[dict]
    ) -> None:
        """AI关闭时每5秒发送载体帧，由can_service改写为后台控制值。"""
        if now - self._last_ai_off_carrier_time < self._ai_off_inference_interval:
            return
        self._last_ai_off_carrier_time = now
        if self._shadow_mode or not self.config.get("result.enabled", True):
            return
        if not self._can_link_healthy_for_send() or ac_feedback is None:
            logger.warning("[AI off载体] CAN断开或反馈过期，本次不发送")
            return
        carrier = {
            "driver_temp": float(ac_feedback["driver_temp"]),
            "passenger_temp": float(ac_feedback["passenger_temp"]),
            "wind_speed": float(ac_feedback["wind_speed"]),
            "air_mode": float(ac_feedback["air_mode"]),
        }
        if self.result_sender.send_from_result_dict(carrier):
            self._result_send_count += 1
            logger.info(f"[AI off载体] 已发送（不登记AI命令基线）: {carrier}")

    def _update_can_link_state(self) -> None:
        connected, now = bool(getattr(self.can_fetcher, "connected", False)), time.time()
        if self._can_link_was_connected and not connected:
            self._can_link_down_time = now
        elif not self._can_link_was_connected and connected:
            self._can_link_up_time = now
        self._can_link_was_connected = connected

    def _can_link_healthy_for_send(self) -> bool:
        if not bool(getattr(self.can_fetcher, "connected", False)):
            return False
        try:
            return all(self.can_fetcher.is_signal_fresh(name) for name in (
                "AC_DriverTempC", "AC_PassengerTempC", "AC_FBlowSpeedLevel", "AC_FModeAdjustSts"))
        except Exception:
            return False

    def _send_jump_suspect(self, result: dict) -> bool:
        if self._last_command_result is None or self._can_link_up_time <= 0 or self._can_link_down_time <= 0:
            return False
        if self._can_link_down_time > self._can_link_up_time:
            return False
        if time.time() - self._can_link_up_time > self._can_recover_guard_seconds:
            return False
        for key in ("driver_temp", "passenger_temp"):
            try:
                if abs(float(result[key]) - float(self._last_command_result[key])) > self._send_temp_jump_tolerance:
                    return True
            except (KeyError, TypeError, ValueError):
                return False
        return False

    def _get_current_hvac_from_can(self) -> Optional[dict]:
        mapping = {"AC_DriverTempC": "driver_temp", "AC_PassengerTempC": "passenger_temp",
                   "AC_FBlowSpeedLevel": "wind_speed", "AC_FModeAdjustSts": "air_mode"}
        feedback = {}
        try:
            for signal_name, key in mapping.items():
                if not self.can_fetcher.is_signal_fresh(signal_name):
                    return None
                value = self.can_fetcher.get_signal(signal_name)
                if value is None:
                    return None
                feedback[key] = float(value)
            return feedback
        except Exception:
            return None

    def _apply_mode4_temperature_policy(self, result: dict, rf_result: dict,
                                        presence: Optional[dict] = None) -> dict:
        result = dict(result)
        presence = presence or self.face_client.get_driver_presence()
        driver, passenger = bool(presence.get("driver_exists")), bool(presence.get("passenger_exists"))
        if passenger:
            result["passenger_temp"] = float(rf_result["passenger_temp"])
            result["passenger_delta"] = 0.0
        if driver and not passenger:
            result["passenger_temp"] = float(result["driver_temp"])
            result["passenger_delta"] = 0.0
        elif passenger and not driver:
            result["driver_temp"] = float(result["passenger_temp"])
            result["driver_delta"] = 0.0
        return result

    def _apply_occupant_recommendation_policy(self, result: dict,
                                               presence: Optional[dict] = None) -> dict:
        presence = presence or self.face_client.get_driver_presence()
        driver_id = self._preference_layer.get_current_user_id() if presence.get("driver_exists") else None
        passenger_id = self._last_passenger_id if presence.get("passenger_exists") else None
        default_k = float(self.config.get("preference_learning.occupant_wind_weight_k", 0.5))
        composed, metadata = compose_occupant_recommendation(
            result, driver_id=driver_id, passenger_id=passenger_id,
            driver_exists=bool(presence.get("driver_exists")),
            passenger_exists=bool(presence.get("passenger_exists")),
            overrides=self.config.get("preference_learning.base_recommendation_overrides", {}) or {},
            wind_weight_k=load_runtime_wind_weight(self._occupant_wind_runtime_path, default_k))
        metadata.update(driver_id=driver_id, passenger_id=passenger_id)
        signature = (bool(presence.get("driver_exists")), bool(presence.get("passenger_exists")),
                     driver_id, passenger_id)
        if self._last_occupant_signature not in (None, signature):
            self._last_sent_wind_speed = None
        self._last_occupant_signature, self._last_occupant_policy = signature, metadata
        return composed

    def _log_occupant_wind_blend(self, sent_wind: int) -> None:
        d = self._last_occupant_policy
        if d:
            logger.info(f"[乘员风量融合] driver_wind={int(d['driver_wind'])}, "
                        f"passenger_wind={int(d['passenger_wind'])}, K={float(d['wind_weight_k']):.2f}, "
                        f"blended={float(d['blended_wind']):.2f}, sent={sent_wind}")

    def _smooth_wind_speed(self, wind_speed: Any) -> int:
        try:
            value = int(float(wind_speed))
        except (TypeError, ValueError):
            value = self._wind_min
        if self._last_sent_wind_speed is not None:
            delta = value - self._last_sent_wind_speed
            if abs(delta) > self._wind_max_delta:
                value = self._last_sent_wind_speed + (self._wind_max_delta if delta > 0 else -self._wind_max_delta)
        return max(self._wind_min, min(9, value))

    @staticmethod
    def _mode4_should_apply_mlp(presence: dict) -> bool:
        return bool(presence.get("driver_exists", False))

    @staticmethod
    def _apply_mode4_mlp_mode_policy(joint: dict, base: dict) -> dict:
        adjusted = dict(joint)
        if int(adjusted.get("mode_class", 0)) == 0:
            adjusted["air_mode"] = float(base["air_mode"])
        return adjusted

    def _get_current_features_dict(self) -> dict:
        weather = self.weather_fetcher.get_features()
        features = {}
        features.update(self.can_fetcher.get_all_features(demo_mode=self.demo_mode, weather_features=weather))
        features.update(self.face_client.get_all_features(demo_mode=self.demo_mode, for_inference=True))
        features.update(weather or {})
        features.update(self.pmv_client.get_features(demo_mode=self.demo_mode) or {})
        return {col: float(features[col]) if features.get(col) is not None else 0.0
                for col in self.model_loader.get_feature_columns()}

    def _run_preference_training(self, samples_df=None, user_id: Optional[str] = None) -> None:
        try:
            samples_df = samples_df if samples_df is not None else self._preference_layer.get_training_dataframe()
            user_id = user_id or self._preference_layer.get_current_user_id() or "default"
            from preference_learning.online_trainer_v26 import Weather26OnlineTrainer
            trainer = Weather26OnlineTrainer(
                model_loader=self.model_loader,
                output_dir=os.path.join(_PROJECT_ROOT, "models", "user_mlps", user_id),
                min_samples=int(self.config.get("preference_learning.pref_layer_min_samples", 30)),
                validation_ratio=float(self.config.get("preference_learning.validation_ratio", 0.2)))
            self._preference_layer.on_training_done(trainer.train(samples_df, user_id=user_id))
        except Exception as exc:
            logger.error(f"[偏好层训练] 异常: {exc}", exc_info=True)
            self._preference_layer.on_training_done(
                {"success": False, "reason": str(exc), "user_id": user_id})

    def _run_rf_inference_for_sampling(self) -> Optional[dict]:
        if not self._can_link_healthy_for_send():
            return None
        result, presence = self._infer_rf()
        if not result.ready:
            return None
        return self._apply_occupant_recommendation_policy({
            "driver_temp": result.driver_temp, "passenger_temp": result.passenger_temp,
            "wind_speed": result.wind_speed, "air_mode": result.air_mode}, presence)

    def _run_joint_inference_mode4(self) -> bool:
        result, presence = self._infer_rf()
        if not result.ready:
            return False
        base = self._apply_occupant_recommendation_policy({
            "driver_temp": result.driver_temp, "passenger_temp": result.passenger_temp,
            "wind_speed": result.wind_speed, "air_mode": result.air_mode}, presence)
        joint = None
        if self._mode4_should_apply_mlp(presence):
            joint = self._preference_layer.apply_mlp_residual(self._get_current_features_dict(), base)
        mlp_applied = joint is not None
        if joint is None:
            joint = dict(base)
            joint.update(driver_delta=0.0, passenger_delta=0.0, wind_delta=0, mode_class=0)
        else:
            joint = self._apply_mode4_mlp_mode_policy(joint, base)
        joint = self._apply_mode4_temperature_policy(joint, base, presence)
        self._last_rf_base_result = dict(base)
        self._inference_count += 1
        self._last_inference_time = time.time()
        return self._send_mode4_result(joint, "joint_mlp" if mlp_applied else "base_rf")

    def get_status(self) -> dict:
        pref = self._preference_layer.get_status()
        return {"running": self.running, "run_mode": "mode4",
                "model_package_mode": "weather26", "demo_mode": self.demo_mode,
                "shadow_mode": self._shadow_mode,
                "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
                "inference_count": self._inference_count,
                "result_send_count": self._result_send_count,
                "model": self.model_loader.get_model_info(),
                "inference_engine": self.inference_engine.get_stats(),
                "preference_layer": {"state": pref.state_name, "sample_count": pref.sample_count,
                                     "mlp_loaded": pref.mlp_loaded,
                                     "human_takeover_active": pref.human_takeover_active,
                                     "takeover_pending": pref.takeover_pending}}


def run(demo_mode: bool = False, config_path: Optional[str] = None,
        run_mode: str = "mode4", shadow_mode: bool = False,
        model_package_mode: str = "weather26") -> None:
    cfg = get_config(config_path)
    setup_logging(level=cfg.get("system.log_level", "INFO"), log_file=cfg.get("system.log_file", None))
    controller = MainController(demo_mode=demo_mode, config_path=config_path,
                                run_mode=run_mode, shadow_mode=shadow_mode,
                                model_package_mode=model_package_mode)
    controller.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="车载空调智能推荐系统")
    parser.add_argument("--demo", action="store_true", help="启用演示模式")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--run-mode", choices=["mode4"], default="mode4", help="运行模式（仅支持mode4）")
    parser.add_argument("--shadow-mode", action="store_true", help="只推理，不下发控制帧")
    parser.add_argument("--model-package-mode", choices=["weather26"], default="weather26",
                        help="基础模型包（仅支持weather26）")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(demo_mode=args.demo, config_path=args.config, run_mode=args.run_mode,
        shadow_mode=args.shadow_mode, model_package_mode=args.model_package_mode)
