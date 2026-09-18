import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from run_vehicle_pmv import run


def _assert_pmv(output, driver, passenger):
    assert output["status"] == "ok"
    assert output["pmv"]["pmv_driver"] == pytest.approx(driver, abs=1e-12)
    assert output["pmv"]["pmv_passenger"] == pytest.approx(passenger, abs=1e-12)


def test_bypass_mode_characterizes_direct_inputs():
    output = run(
        {
            "amb_t_c": 24.0,
            "rh_percent": 50.0,
            "bypass_models": True,
            "use_next_state": False,
            "driver_air_temp_override_c": 26.5,
            "passenger_air_temp_override_c": 25.5,
            "driver_mrt_override_c": 24.0,
            "passenger_mrt_override_c": 24.0,
            "driver_air_speed_override_m_s": 0.3,
            "passenger_air_speed_override_m_s": 0.2,
        }
    )

    _assert_pmv(output, -0.636144815361309, -0.7191819933215703)
    diagnostics = output["diagnostics"]
    assert diagnostics["bypass_models"] is True
    assert diagnostics["driver_air_temp_source"] == "bypass_override"
    assert diagnostics["passenger_air_temp_source"] == "bypass_override"
    assert diagnostics["driver_air_temp_c"] == 26.5
    assert diagnostics["passenger_air_temp_c"] == 25.5
    assert diagnostics["driver_air_speed_m_s"] == 0.3
    assert diagnostics["passenger_air_speed_m_s"] == 0.2


def test_bypass_mode_characterizes_missing_input_fallbacks():
    output = run(
        {
            "amb_t_c": 24.0,
            "bypass_models": True,
            "use_next_state": False,
        }
    )

    _assert_pmv(output, -0.350780256550428, -0.350780256550428)
    diagnostics = output["diagnostics"]
    assert diagnostics["driver_air_temp_c"] == 25.0
    assert diagnostics["passenger_air_temp_c"] == 25.0
    assert diagnostics["driver_mrt_c"] == 25.0
    assert diagnostics["passenger_mrt_c"] == 25.0
    assert diagnostics["driver_air_speed_m_s"] == 0.1
    assert diagnostics["passenger_air_speed_m_s"] == 0.1


def test_model_mode_characterizes_measured_head_temperature_override():
    output = run(
        {
            "amb_t_c": 24.0,
            "rh_percent": 50.0,
            "use_next_state": False,
            "measured_head_air_temp_c": {
                "driver_left": 26.0,
                "driver_right": 28.0,
                "passenger_left": 25.0,
                "passenger_right": 27.0,
            },
        }
    )

    _assert_pmv(output, -0.05690108910242876, -0.26260657868114406)
    diagnostics = output["diagnostics"]
    assert diagnostics["driver_air_temp_source"] == "measured_head_points"
    assert diagnostics["passenger_air_temp_source"] == "measured_head_points"
    assert diagnostics["driver_air_temp_c"] == 27.0
    assert diagnostics["passenger_air_temp_c"] == 26.0
    assert diagnostics["driver_mrt_c"] == 24.0
    assert diagnostics["passenger_mrt_c"] == 24.0


def test_non_finite_bypass_overrides_use_existing_fallbacks():
    output = run(
        {
            "amb_t_c": 24.0,
            "bypass_models": True,
            "driver_air_temp_override_c": math.nan,
            "passenger_mrt_override_c": math.inf,
        }
    )

    diagnostics = output["diagnostics"]
    assert diagnostics["driver_air_temp_c"] == 25.0
    assert diagnostics["passenger_mrt_c"] == 25.0


def test_missing_image_seats_preserve_unoccupied_validity():
    output = run(
        {
            "amb_t_c": 24.0,
            "bypass_models": True,
            "image_inputs": {
                "driver": {"occupied": False},
                "passenger": {"occupied": False},
            },
            "feature_flags": {"enable_image_occupancy": True},
        }
    )

    assert output["comfort_flags"] == {
        "driver_comfortable": False,
        "passenger_comfortable": False,
    }


def test_per_seat_met_and_clo_remain_independent():
    output = run(
        {
            "amb_t_c": 24.0,
            "bypass_models": True,
            "image_inputs": {
                "driver": {
                    "occupied": True,
                    "activity_met": 1.4,
                    "clothing_clo": 0.8,
                },
                "passenger": {
                    "occupied": True,
                    "activity_met": 1.0,
                    "clothing_clo": 0.4,
                },
            },
        }
    )

    diagnostics = output["diagnostics"]
    assert diagnostics["met_driver"] == 1.4
    assert diagnostics["clo_driver"] == 0.8
    assert diagnostics["met_passenger"] == 1.0
    assert diagnostics["clo_passenger"] == 0.4
    assert output["pmv"]["pmv_driver"] != output["pmv"]["pmv_passenger"]
