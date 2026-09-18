import json
import logging
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pmv_socket_consumer
from hvac_sim.pipeline_diagnostics import PeriodicPmvObserver, PmvDiagnosticSnapshot
from hvac_sim.runtime_adapter import build_runtime_pipeline_inputs
from hvac_sim.runtime_input import RuntimeSignalInput
from hvac_sim.runtime_parameter_mode import (
    CHTD_PARAM_MODE_DEFAULT,
    CHTD_PARAM_MODE_PHASE3,
    CHTD_PARAM_MODE_SAFE_PREVIEW,
    resolve_chtd_param_mode,
)
from pmv_service.api import LatestApiData, build_api_data, start_api_server
from pmv_service import image_inputs
from pmv_service.input_builder import build_pmv_input
from pmv_service.param_trace import append_param_log
from pmv_service import runner
from pmv_service.runner import PmvServiceConfig, correct_previous_state
from pmv_service.socket_client import parse_data_message


def test_partial_and_non_finite_head_measurements_keep_one_sided_fallback():
    output = build_pmv_input(
        {
            "TA_FdHeadTempLe": math.nan,
            "TA_FdHeadTempRi": 27.5,
            "TA_FpHeadTempLe": math.inf,
        }
    )

    assert output["driver_air_temp_override_c"] == 27.5
    assert "passenger_air_temp_override_c" not in output


def test_incomplete_or_zero_airflow_does_not_create_speed_override():
    incomplete = build_pmv_input({"AC_Forward_AirFlowTarget": 200.0})
    invalid_damper = build_pmv_input(
        {
            "AC_Forward_AirFlowTarget": 200.0,
            "AC_BLOW_FaceVentilaPosn": math.nan,
            "AC_FrantFootVentPosn": 2.1,
            "AC_DefrostVentilaPosn": 4.6,
        }
    )
    zero = build_pmv_input(
        {
            "AC_Forward_AirFlowTarget": 0.0,
            "AC_BLOW_FaceVentilaPosn": 4.7,
            "AC_FrantFootVentPosn": 2.1,
            "AC_DefrostVentilaPosn": 4.6,
        }
    )

    assert "driver_air_speed_override_m_s" not in incomplete
    assert "driver_air_speed_override_m_s" not in invalid_damper
    assert "driver_air_speed_override_m_s" not in zero


def test_api_contract_contains_legacy_and_current_seat_fields():
    response = build_api_data(
        {
            "status": "ok",
            "pmv": {
                "pmv_driver": 0.1,
                "ppd_driver": 5.2,
                "pmv_passenger": 0.2,
                "ppd_passenger": 5.8,
            },
            "model_state": {
                "driver_feet_temp_c": 24.5,
                "passenger_feet_temp_c": 24.7,
            },
            "diagnostics": {
                "driver_air_temp_c": 25.0,
                "passenger_air_temp_c": 25.2,
                "driver_mrt_c": 24.0,
                "passenger_mrt_c": 24.1,
                "rh_percent": 50.0,
                "driver_air_speed_m_s": 0.3,
                "passenger_air_speed_m_s": 0.4,
            },
        },
        run_index=7,
        amb_t_c=30.0,
    )

    expected_fields = {
        "pmv",
        "ppd",
        "head_temp_c",
        "feet_temp_c",
        "mrt_c",
        "rh_percent",
        "air_speed_m_s",
    }
    assert set(response["driver"]) == expected_fields
    assert set(response["passenger"]) == expected_fields
    assert response["driver"]["feet_temp_c"] == 24.5
    assert response["passenger"]["feet_temp_c"] == 24.7
    assert response["cabin_temp_c"] == 25.0
    assert response["run_index"] == 7

    schema = json.loads(
        (ROOT / "config/schemas/pmv_http_api_schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(response, schema)


def test_http_handler_serves_snapshot_and_rejects_unknown_path():
    state = LatestApiData()
    state.update({"status": "ok", "run_index": 3})
    server = start_api_server(0, state)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/pmv") as response:
            assert json.load(response) == {"status": "ok", "run_index": 3}
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/missing")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("unknown API path should return 404")
    finally:
        server.shutdown()
        server.server_close()


def test_parameter_trace_keeps_human_readable_contract(tmp_path):
    output = tmp_path / "trace.log"
    append_param_log(
        str(output),
        run_index=2,
        timestamp="2026-09-17T00:00:00Z",
        signal_cache={"VIU_AmbT": 30.0},
        pmv_input={"amb_t_c": 30.0},
        pmv_output={
            "status": "ok",
            "pmv": {"pmv_driver": 0.1, "ppd_driver": 5.0},
            "diagnostics": {"driver_air_temp_c": 25.0},
        },
    )
    text = output.read_text(encoding="utf-8")
    assert "运行序号 run_index: 2" in text
    assert "主驾 PMV: 0.1" in text
    assert "driver_air_temp_c (主驾 PMV 空气温度" in text
    assert "VIU_AmbT: 30.0" in text


def test_periodic_observer_uses_logging(caplog):
    observer = PeriodicPmvObserver(interval=20)
    snapshot = PmvDiagnosticSnapshot(
        passenger_pmv=0.1,
        passenger_ppd=5.0,
        passenger_air_temp_c=25.0,
        passenger_air_temp_source="model_state",
        passenger_mrt_c=24.0,
        passenger_air_speed_m_s=0.2,
        relative_humidity_pct=50.0,
        passenger_met=1.0,
        passenger_clo=0.5,
        passenger_head_raw_c=25.0,
        passenger_cabin_raw_c=24.0,
        passenger_window_raw_c=36.0,
        passenger_roof_raw_c=24.0,
        passenger_window_mrt_c=24.0,
        passenger_roof_mrt_c=24.0,
        driver_head_raw_c=25.0,
        driver_feet_raw_c=24.0,
        passenger_feet_raw_c=24.0,
        driver_pmv=0.0,
        mrt_hot_surface_threshold_c=35.0,
    )
    with caplog.at_level(logging.INFO, logger="hvac_sim.pmv.diagnostics"):
        observer.observe(snapshot)
    assert "PMV diagnostic #1" in caplog.text


def test_image_api_failure_keeps_existing_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(image_inputs.requests, "get", fail)
    assert pmv_socket_consumer.fetch_image_stats(timeout=0.01) is None


def test_socket_disconnect_classification():
    assert pmv_socket_consumer.is_socket_disconnect_error(BrokenPipeError())
    assert not pmv_socket_consumer.is_socket_disconnect_error(ValueError())


def test_cli_log_level_is_configurable():
    args = pmv_socket_consumer.build_arg_parser().parse_args(
        ["--log-level", "DEBUG"]
    )
    assert args.log_level == "DEBUG"


def test_runtime_input_normalizes_aliases_and_partial_measurements():
    signal = RuntimeSignalInput.from_mapping(
        {
            "amb_t_c": "24.5",
            "raw_ambient_temp_c": "nan",
            "rh": "45",
            "veh_spd": "88.5",
            "measured_head_air_temp_c": {
                "driver_left": "nan",
                "driver_right": "27.5",
                "passenger_left": None,
                "passenger_right": "inf",
            },
            "tma": {"FrntFdvTma": "18.0", "bad": "not-a-number"},
            "flows": {"driver_face_flow": "30", "bad": float("nan")},
            "image_inputs": {"driver": {"occupied": False}},
            "use_ir_head_fusion": True,
            "apply_geometry": True,
            "initial_params_path": "/tmp/example.json",
        }
    )

    assert signal.amb_t_c == 24.5
    assert signal.raw_amb_t_c is None
    assert signal.rh_percent == 45.0
    assert signal.vehicle_speed_kph == 88.5
    assert signal.measured_driver_head_air_temp_c == 27.5
    assert signal.measured_passenger_head_air_temp_c is None
    assert signal.tma_c == {"FrntFdvTma": 18.0}
    assert signal.flows == {"driver_face_flow": 30.0}
    assert signal.image_inputs.driver.occupied is False
    assert signal.use_ir_head_fusion is True
    assert signal.apply_geometry is True
    assert signal.params_bundle_path == "/tmp/example.json"
    assert signal.use_initial_calibration_params is True


def test_parameter_mode_aliases_and_invalid_mode_fallback():
    assert resolve_chtd_param_mode({}) == CHTD_PARAM_MODE_DEFAULT
    assert (
        resolve_chtd_param_mode({"use_safe_preview_chtd_params": True})
        == CHTD_PARAM_MODE_SAFE_PREVIEW
    )
    assert (
        resolve_chtd_param_mode({"chtd_param_mode": " PHASE3_ENGINEERING_PREVIEW "})
        == CHTD_PARAM_MODE_PHASE3
    )
    assert resolve_chtd_param_mode({"chtd_param_mode": "unknown"}) == CHTD_PARAM_MODE_DEFAULT


def test_preview_parameter_modes_are_self_contained_runtime_imports():
    for mode in (CHTD_PARAM_MODE_SAFE_PREVIEW, CHTD_PARAM_MODE_PHASE3):
        adapted = build_runtime_pipeline_inputs(
            {"amb_t_c": 35.0, "chtd_param_mode": mode}
        )

        assert adapted.provenance["chtd_param_mode"] == mode
        assert adapted.provenance["params_bundle"]["status"] == mode
        assert adapted.pipeline_inputs.chtd_params is not None


def test_socket_parser_drops_invalid_and_non_finite_values():
    assert parse_data_message("not json") is None
    assert parse_data_message('{"type":"heartbeat"}') is None
    parsed = parse_data_message(
        json.dumps(
            {
                "type": "data",
                "data": {
                    "number": 1,
                    "numeric_string": "2.5",
                    "nan": float("nan"),
                    "inf_string": "inf",
                    "bad": "x",
                },
            }
        )
    )
    assert parsed == {"number": 1.0, "numeric_string": 2.5}


def test_previous_state_correction_preserves_deployed_weights():
    state = [20.0] * 28
    corrected = correct_previous_state(state, 30.0)
    assert corrected is state
    assert corrected[1] == 20.5
    assert corrected[15] == 20.5
    assert corrected[16] == 20.5
    assert corrected[3] == 21.5
    assert corrected[4] == 21.5
    assert corrected[0] == 20.0


def test_service_reconnects_after_clean_socket_disconnect(monkeypatch):
    class FakeSocket:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.closed = False

        def recv(self, _size):
            response = next(self.responses)
            if isinstance(response, BaseException):
                raise response
            return response

        def close(self):
            self.closed = True

    first = FakeSocket([b""])
    envelope = json.dumps(
        {
            "type": "data",
            "data": {"VIU_AmbT": 30.0, "AC_FrntInCarT": 25.0},
        }
    ).encode("utf-8") + b"\n"
    second = FakeSocket([envelope, ValueError("stop test loop")])
    sockets = iter((first, second))
    monkeypatch.setattr(runner, "connect_socket", lambda *_args, **_kwargs: next(sockets))
    monkeypatch.setattr(runner, "fetch_image_stats", lambda: None)

    calls = []

    def fake_run_pmv(pmv_input, **kwargs):
        calls.append((pmv_input, kwargs))
        return {
            "status": "ok",
            "pmv": {
                "pmv_driver": 0.1,
                "ppd_driver": 5.0,
                "pmv_passenger": 0.2,
                "ppd_passenger": 5.1,
            },
            "diagnostics": {
                "driver_air_temp_c": 25.0,
                "passenger_air_temp_c": 25.0,
                "driver_mrt_c": 25.0,
                "passenger_mrt_c": 25.0,
                "rh_percent": 50.0,
                "driver_air_speed_m_s": 0.1,
                "passenger_air_speed_m_s": 0.1,
            },
            "model_state": {
                "driver_feet_temp_c": 25.0,
                "passenger_feet_temp_c": 25.0,
            },
            "x_next": [25.0] * 28,
        }

    state = LatestApiData()
    runner.run_service(
        PmvServiceConfig(
            socket_path="/tmp/test.sock",
            client_id="test",
            can_filters=[1],
            interval_s=0.0,
        ),
        state,
        fake_run_pmv,
    )

    assert first.closed is True
    assert second.closed is True
    assert len(calls) == 1
    assert state.snapshot()["run_index"] == 1
