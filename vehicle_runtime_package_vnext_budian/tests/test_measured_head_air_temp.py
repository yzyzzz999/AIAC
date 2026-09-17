import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pmv_socket_consumer import SUBSCRIBE_CAN_IDS, build_pmv_input
from run_vehicle_pmv import run


def test_socket_consumer_builds_measured_head_air_temp_block():
    pmv_input = build_pmv_input(
        {
            "VIU_AmbT": 24.0,
            "TA_FdHeadTempLe": 26.0,
            "TA_FdHeadTempRi": 28.0,
            "TA_FpHeadTempLe": 25.0,
            "TA_FpHeadTempRi": 27.0,
        }
    )

    assert pmv_input["measured_head_air_temp_c"] == {
        "driver_left": 26.0,
        "driver_right": 28.0,
        "passenger_left": 25.0,
        "passenger_right": 27.0,
    }


def test_socket_consumer_subscribes_to_can_send_2_head_points():
    assert 0x100 in SUBSCRIBE_CAN_IDS


def test_runtime_uses_measured_head_points_for_pmv_air_temp_only():
    out = run(
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

    diag = out["diagnostics"]
    assert diag["driver_air_temp_source"] == "measured_head_points"
    assert diag["passenger_air_temp_source"] == "measured_head_points"
    assert diag["driver_air_temp_c"] == 27.0
    assert diag["passenger_air_temp_c"] == 26.0
    assert diag["driver_mrt_source"] == "model_state"
    assert diag["passenger_mrt_source"] == "model_state"
    assert diag["forbidden_input_detected"] is False


if __name__ == "__main__":
    test_socket_consumer_builds_measured_head_air_temp_block()
    test_socket_consumer_subscribes_to_can_send_2_head_points()
    test_runtime_uses_measured_head_points_for_pmv_air_temp_only()
