from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch

import src.main_controller as main_controller_module
from src.main_controller import MainController
from src.preference_learning.preference_layer_manager import (
    PreferenceLayerManager,
    PreferenceState,
)
from src.services.signal_fetchers.can_signal_fetcher import CanSignalFetcher


class _FakeCanFetcher:
    def __init__(self, ai_state):
        self.ai_state = ai_state

    def get_ai_state(self):
        return self.ai_state


def _controller(ai_state, preference_state="rf_only"):
    controller = MainController.__new__(MainController)
    controller.can_fetcher = _FakeCanFetcher(ai_state)
    controller._ai_on_inference_interval = 2.0
    controller._ai_off_inference_interval = 5.0
    controller._last_base_inference_time = 10.0
    controller._last_pref_inference_time = 10.0
    controller._preference_layer = SimpleNamespace(
        get_status=lambda: SimpleNamespace(
            takeover_pending=False,
            human_takeover_active=False,
            state_name=preference_state,
        )
    )
    return controller


def test_start_emits_start_all_readiness_marker_after_main_loop_starts():
    controller = MainController.__new__(MainController)
    controller.model_loader = SimpleNamespace(
        load_all=lambda: True,
        get_feature_columns=lambda: [],
    )
    controller._preference_layer = SimpleNamespace(set_feature_columns=lambda columns: None)
    controller.can_fetcher = SimpleNamespace(start=lambda: None)
    controller.face_client = SimpleNamespace(start_auto_refresh=lambda interval: None)
    controller.pmv_client = SimpleNamespace(start_auto_refresh=lambda interval: None)
    controller.weather_fetcher = SimpleNamespace(start_auto_refresh=lambda: None)
    controller.result_sender = SimpleNamespace(connect=lambda: None)
    controller.config = SimpleNamespace(get=lambda key, default=None: default)
    controller._shadow_mode = False
    controller.running = False
    controller._start_time = 0.0
    controller._main_loop_thread = None

    messages = []
    with patch.object(main_controller_module.time, "sleep"), \
            patch.object(main_controller_module.threading, "Thread") as thread_cls, \
            patch.object(main_controller_module.logger, "info",
                         side_effect=lambda message: messages.append(message)):
        controller.start()

    thread_cls.return_value.start.assert_called_once_with()
    assert "系统启动完成，开始运行" in messages


def test_can_fetcher_subscribes_to_and_processes_ai_state():
    fetcher = CanSignalFetcher(can_ids=["0x18F"])
    assert "0x387" in fetcher.can_ids
    assert "0xF001" in fetcher.can_ids
    assert fetcher.get_ai_state() == "on"

    fetcher._process_message({
        "type": "ai_state", "timestamp": 123.0, "data": {"ai": "off"}
    })

    assert fetcher.get_ai_state() == "off"
    assert "ai" not in fetcher._signals


def test_cdc_actions_latch_valid_requests_and_ignore_no_request():
    fetcher = CanSignalFetcher(can_ids=["0x18F"])
    fetcher._process_message({
        "type": "ai_state", "timestamp": 1.0, "data": {"ai": "off"}
    })
    fetcher._process_message({
        "type": "data",
        "timestamp": 2.0,
        "data": {
            "CDC_DriverTempCSet": "25.5",
            "CDC_PassengerTempCSet": 31,
            "CDC_FHvacBlowLvSet": 6,
            "CDC_FHvacModeSet": 1,
        },
        "raw": {
            "CDC_DriverTempCSet": 19,
            "CDC_PassengerTempCSet": 31,
            "CDC_FHvacBlowLvSet": 6,
            "CDC_FHvacModeSet": 1,
        },
    })
    fallback = {
        "driver_temp": 22.0, "passenger_temp": 24.0,
        "wind_speed": 4.0, "air_mode": 2.0,
    }
    assert fetcher.get_cdc_hvac_actions(fallback) == {
        "driver_temp": 25.5, "passenger_temp": 24.0,
        "wind_speed": 6.0, "air_mode": 1.0,
    }

    fetcher._process_message({
        "type": "data", "timestamp": 3.0,
        "data": {
            "CDC_DriverTempCSet": 31, "CDC_PassengerTempCSet": 31,
            "CDC_FHvacBlowLvSet": 0, "CDC_FHvacModeSet": 0,
        },
        "raw": {
            "CDC_DriverTempCSet": 31, "CDC_PassengerTempCSet": 31,
            "CDC_FHvacBlowLvSet": 0, "CDC_FHvacModeSet": 0,
        },
    })
    assert fetcher.get_cdc_hvac_actions(fallback)["driver_temp"] == 25.5
    assert fetcher.get_cdc_hvac_actions(fallback)["wind_speed"] == 6.0

    fetcher._process_message({
        "type": "data", "timestamp": 4.0,
        "data": {
            "CDC_DriverTempCSet": 0, "CDC_PassengerTempCSet": "31.0",
            "CDC_FHvacBlowLvSet": 10, "CDC_FHvacModeSet": 7,
        },
        "raw": {
            "CDC_DriverTempCSet": 0, "CDC_PassengerTempCSet": 30,
            "CDC_FHvacBlowLvSet": 10, "CDC_FHvacModeSet": 7,
        },
    })
    assert fetcher.get_cdc_hvac_actions(fallback) == {
        "driver_temp": 16.0, "passenger_temp": 31.0,
        "wind_speed": 9.0, "air_mode": 7.0,
    }


class _DummyModelLoader:
    @staticmethod
    def get_feature_columns():
        return [f"f{i}" for i in range(26)]


def _preference_manager(tmp_path: Path):
    manager = PreferenceLayerManager(
        model_loader=_DummyModelLoader(),
        output_dir=tmp_path,
        user_mlp_base_dir=str(tmp_path / "users"),
        collection_duration=300.0,
        min_training_samples=30,
    )
    manager._current_user_id = "driver-a"
    return manager


def test_ai_off_collects_immediately_and_on_pauses_without_losing_samples():
    with tempfile.TemporaryDirectory() as directory:
        manager = _preference_manager(Path(directory))
        features = {f"f{i}": float(i) for i in range(26)}
        rf = {"driver_temp": 22.0, "passenger_temp": 24.0,
              "wind_speed": 4.0, "air_mode": 2.0}
        cdc = {"driver_temp": 25.5, "passenger_temp": 24.0,
               "wind_speed": 6.0, "air_mode": 1.0}

        result = manager.update(features, rf, cdc, driver_id="driver-a", ai_state="off")
        assert result["external_control"] is True
        assert manager.state == PreferenceState.COLLECTING
        assert len(manager._samples) == 1
        manager._external_collection_segment_start -= 10.0

        manager.update(features, rf, rf, driver_id="driver-a", ai_state="on")
        paused_elapsed = manager._external_collection_elapsed
        assert manager.state == PreferenceState.RF_ONLY
        assert paused_elapsed >= 10.0
        assert len(manager._samples) == 1

        manager.update(features, rf, cdc, driver_id="driver-a", ai_state="off")
        assert manager.state == PreferenceState.COLLECTING
        assert manager._external_collection_elapsed == paused_elapsed
        assert len(manager._samples) == 1


def test_external_collection_triggers_training_once_per_off_cycle():
    with tempfile.TemporaryDirectory() as directory:
        manager = _preference_manager(Path(directory))
        manager.COLLECTION_DURATION = 5.0
        manager.MIN_TRAINING_SAMPLES = 2
        manager.SAMPLE_INTERVAL = 0.0
        features = {f"f{i}": float(i) for i in range(26)}
        rf = {"driver_temp": 22.0, "passenger_temp": 24.0,
              "wind_speed": 4.0, "air_mode": 2.0}
        cdc = {"driver_temp": 25.5, "passenger_temp": 24.0,
               "wind_speed": 6.0, "air_mode": 1.0}

        manager.update(features, rf, cdc, driver_id="driver-a", ai_state="off")
        manager._external_collection_segment_start -= 6.0
        result = manager.update(features, rf, cdc, driver_id="driver-a", ai_state="off")
        assert result["trigger_training"] is True
        assert manager.state == PreferenceState.TRAINING

        repeated = manager.update(features, rf, cdc, driver_id="driver-a", ai_state="off")
        assert repeated["trigger_training"] is False

        manager.update(features, rf, rf, driver_id="driver-a", ai_state="on")
        assert manager.state == PreferenceState.RF_ONLY
        assert manager._external_training_triggered is True


def test_user_change_discards_paused_external_samples():
    with tempfile.TemporaryDirectory() as directory:
        manager = _preference_manager(Path(directory))
        features = {f"f{i}": float(i) for i in range(26)}
        rf = {"driver_temp": 22.0, "passenger_temp": 24.0,
              "wind_speed": 4.0, "air_mode": 2.0}
        cdc = {"driver_temp": 25.5, "passenger_temp": 24.0,
               "wind_speed": 6.0, "air_mode": 1.0}
        manager.update(features, rf, cdc, driver_id="driver-a", ai_state="off")
        manager.update(features, rf, rf, driver_id="driver-a", ai_state="on")
        assert manager._samples

        manager._on_user_change("driver-b")

        assert manager._samples == []
        assert manager._external_session_exists is False


def test_ai_off_carrier_uses_five_seconds_and_does_not_touch_command_baseline():
    controller = MainController.__new__(MainController)
    controller._last_ai_off_carrier_time = 0.0
    controller._ai_off_inference_interval = 5.0
    controller._shadow_mode = False
    controller._result_send_count = 0
    controller._last_command_result = {"driver_temp": 19.0}
    controller.config = SimpleNamespace(get=lambda key, default=None: True)
    controller._can_link_healthy_for_send = lambda: True
    sent = []
    controller.result_sender = SimpleNamespace(
        send_from_result_dict=lambda payload: sent.append(payload) or True
    )
    feedback = {"driver_temp": 22.0, "passenger_temp": 24.0,
                "wind_speed": 4.0, "air_mode": 2.0}

    controller._check_ai_off_carrier(10.0, feedback)
    controller._check_ai_off_carrier(14.9, feedback)
    controller._check_ai_off_carrier(15.0, feedback)

    assert len(sent) == 2
    assert controller._result_send_count == 2
    assert controller._last_command_result == {"driver_temp": 19.0}


def test_ai_on_uses_two_second_base_interval():
    controller = _controller("on")
    calls = []
    controller._run_base_inference = lambda: calls.append("base")

    controller._check_scheduled_inference(11.9)
    assert calls == []
    controller._check_scheduled_inference(12.0)
    assert calls == ["base"]


def test_ai_off_uses_five_second_joint_interval():
    controller = _controller("off", preference_state="joint_active")
    calls = []
    controller._run_joint_inference_mode4 = lambda: calls.append("joint")

    controller._check_scheduled_inference(14.9)
    assert calls == []
    controller._check_scheduled_inference(15.0)
    assert calls == ["joint"]
