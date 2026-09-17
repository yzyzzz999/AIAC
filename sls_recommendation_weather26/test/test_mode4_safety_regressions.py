from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.preference_learning.preference_layer_manager import PreferenceLayerManager
from src.preference_learning.online_trainer_v26 import Weather26OnlineTrainer


class _ModelLoader:
    def get_feature_columns(self):
        return []


class _FixedMLP:
    def predict_proba(self, _x):
        driver = np.zeros((1, 25))
        passenger = np.zeros((1, 25))
        wind = np.zeros((1, 7))
        mode = np.zeros((1, 8))
        driver[0, 12] = 1.0       # 0.0 C residual
        passenger[0, 12] = 1.0
        wind[0, 6] = 1.0          # +3 levels without the stable-profile guard
        mode[0, 0] = 1.0
        return [driver, passenger, wind, mode]


class _OutOfRangeColdMLP:
    def predict_proba(self, _x):
        driver = np.zeros((1, 25))
        passenger = np.zeros((1, 25))
        wind = np.zeros((1, 7))
        mode = np.zeros((1, 8))
        driver[0, 4] = 1.0       # -4.0 C: illegal for a 19.5 C base
        passenger[0, 12] = 1.0
        wind[0, 3] = 1.0
        mode[0, 0] = 1.0
        return [driver, passenger, wind, mode]


class _PerRowModelLoader:
    def get_feature_columns(self):
        return ["feature"]

    def predict(self, features):
        value = float(features.iloc[0]["feature"])
        return SimpleNamespace(
            driver_temp=value,
            passenger_temp=value + 0.5,
            wind_speed=int(value),
            air_mode=2,
        )


def _manager(tmp_path: Path) -> PreferenceLayerManager:
    return PreferenceLayerManager(
        model_loader=_ModelLoader(),
        output_dir=tmp_path / "generic",
        user_mlp_base_dir=str(tmp_path / "users"),
        identity_confirm_seconds=0.0,
        new_user_confirm_seconds=0.0,
        command_ack_timeout=0.0,
    )


def test_takeover_requires_can_acknowledgement(tmp_path):
    manager = _manager(tmp_path)
    command = {
        "driver_temp": 20.0,
        "passenger_temp": 20.0,
        "wind_speed": 3.0,
        "air_mode": 2.0,
    }
    manager.note_command_sent(command)

    # A stale CAN value after send is not a human action.
    stale = {**command, "wind_speed": 5.0}
    assert manager._detect_takeover(command, stale) is False

    # Once the command has appeared on CAN, a later change is a takeover.
    assert manager._detect_takeover(command, command) is False
    assert manager._detect_takeover(command, stale) is True


def test_known_face_id_switch_is_debounced_then_reported(tmp_path):
    manager = _manager(tmp_path)
    user_id = "trained-user"
    model_path = tmp_path / "users" / user_id / "mlp_model.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    manager._load_mlp_for_user = lambda candidate: candidate == user_id

    first = manager.update({}, None, None, driver_id=user_id, passenger_id="noisy-seat")
    assert first["identity_pending"] is True
    second = manager.update({}, None, None, driver_id=user_id, passenger_id=None)
    assert second["user_changed"] is True
    assert manager.get_current_user_id() == user_id


def test_noisy_face_ids_do_not_replace_current_user(tmp_path):
    manager = _manager(tmp_path)
    manager._current_user_id = "current-user"
    manager.IDENTITY_VOTE_MIN_SAMPLES = 4
    manager.IDENTITY_VOTE_RATIO = 0.75
    manager.IDENTITY_SWITCH_COOLDOWN_SECONDS = 0.0

    for noisy_id in ["alias-a", "alias-b"] * 5:
        result = manager.update({}, None, None, driver_id=noisy_id)
        assert result["user_changed"] is False
        assert result["identity_pending"] is False

    assert manager.get_current_user_id() == "current-user"


def test_stable_majority_can_replace_current_user(tmp_path):
    manager = _manager(tmp_path)
    manager._current_user_id = "current-user"
    manager.IDENTITY_VOTE_MIN_SAMPLES = 4
    manager.IDENTITY_VOTE_RATIO = 0.75
    manager.IDENTITY_SWITCH_COOLDOWN_SECONDS = 0.0

    for _ in range(4):
        result = manager.update({}, None, None, driver_id="new-user")
    assert result["identity_pending"] is True

    result = manager.update({}, None, None, driver_id="new-user")
    assert result["user_changed"] is True
    assert manager.get_current_user_id() == "new-user"


def test_preference_profile_does_not_override_joint_wind(tmp_path):
    manager = _manager(tmp_path)
    manager._mlp = _FixedMLP()
    manager._mlp_loaded = True
    manager._preference_profile = {
        "preferred_wind": 2,
        "wind_confidence": 0.9,
        "wind_locked": True,
    }

    result = manager.apply_mlp_residual({}, {
        "driver_temp": 20.0,
        "passenger_temp": 20.0,
        "wind_speed": 3.0,
        "air_mode": 2.0,
    })
    assert result is not None
    assert result["wind_speed"] == 6.0


def test_preference_profile_does_not_override_joint_temperature(tmp_path):
    manager = _manager(tmp_path)
    manager._mlp = _FixedMLP()
    manager._mlp_loaded = True
    manager._preference_profile = {
        "preferred_driver_temperature": 16.5,
        "driver_temperature_confidence": 1.0,
        "driver_temperature_locked": True,
    }

    result = manager.apply_mlp_residual({}, {
        "driver_temp": 20.0,
        "passenger_temp": 20.0,
        "wind_speed": 3.0,
        "air_mode": 2.0,
    })
    assert result is not None
    assert result["driver_temp"] == 20.0
    assert result["driver_delta"] == 0.0


def test_profile_does_not_override_legal_boundary_projection(tmp_path):
    manager = _manager(tmp_path)
    manager._mlp = _OutOfRangeColdMLP()
    manager._mlp_loaded = True
    manager._preference_profile = {
        "preferred_driver_temperature": 16.5,
        "driver_temperature_confidence": 1.0,
        "driver_temperature_locked": True,
    }

    result = manager.apply_mlp_residual({}, {
        "driver_temp": 19.5,
        "passenger_temp": 20.0,
        "wind_speed": 3.0,
        "air_mode": 2.0,
    })
    assert result is not None
    assert result["driver_temp"] == 16.0
    assert result["driver_delta"] == -3.5


def test_illegal_cold_delta_projects_to_nearest_legal_temperature(tmp_path):
    manager = _manager(tmp_path)
    manager._mlp = _OutOfRangeColdMLP()
    manager._mlp_loaded = True

    result = manager.apply_mlp_residual({}, {
        "driver_temp": 19.5,
        "passenger_temp": 20.0,
        "wind_speed": 3.0,
        "air_mode": 2.0,
    })
    assert result is not None
    assert result["driver_delta"] == -3.5
    assert result["driver_temp"] == 16.0


def test_profile_uses_dominant_human_wind_setting():
    samples = pd.DataFrame({"UI界面_风速": [2] * 50 + [4] * 10})
    profile = Weather26OnlineTrainer._build_preference_profile(samples)
    assert profile["preferred_wind"] == 2
    assert profile["wind_locked"] is True


def test_profile_uses_dominant_human_temperature_settings():
    samples = pd.DataFrame({
        "主驾温度显示": [16.5] * 50 + [17.0] * 10,
        "副驾温度显示": [18.0] * 60,
    })
    profile = Weather26OnlineTrainer._build_preference_profile(samples)
    assert profile["preferred_driver_temperature"] == 16.5
    assert profile["driver_temperature_locked"] is True
    assert profile["preferred_passenger_temperature"] == 18.0
    assert profile["passenger_temperature_locked"] is True


def test_base_predictions_are_generated_for_every_training_row(tmp_path):
    trainer = Weather26OnlineTrainer(_PerRowModelLoader(), tmp_path)
    frame = pd.DataFrame({"feature": [3.0, 4.0, 5.0]})
    result = trainer._add_base_predictions(frame, ["feature"])
    assert result["base_driver_temperature"].tolist() == [3.0, 4.0, 5.0]
    assert result["base_passenger_temperature"].tolist() == [3.5, 4.5, 5.5]
    assert result["base_wind"].tolist() == [3, 4, 5]
