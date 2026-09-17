"""
Vehicle PMV runtime entry point.
Input: SIG/FTE signals only (thermocouple temps forbidden).
Usage: python run_vehicle_pmv.py --input sample.json [--output out.json]
"""
import sys
import json
import argparse
import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from hvac_sim.runtime_adapter import build_runtime_pipeline_inputs
from hvac_sim.pipeline import run_comfort_pipeline_dict
from hvac_sim.chtd.bus_index import X_INDEX

FORBIDDEN_KEYS = {
    "TA_FdHeadTempLe", "TA_FpHeadTempLe",
    "TA_FdFloorTemp1", "TA_FdFloorTemp2",
    "TA_FpFloorTemp1", "TA_FpFloorTemp2",
    "TA_CarbinFrntTempLe", "TA_CarbinFrntTempRi",
    "measured_head_temperature", "measured_feet_temperature",
    "measured_cabin_temperature",
}

SCHEMA_VERSION = "vehicle_runtime_package_v8_l1_candidate_v1"
CALIBRATION_STATUS = "spring_summer_v8_l1_candidate_not_vehicle_calibration"
AIRFLOW_MODE = "v7_ac_mode_adjust_airflow_with_v8_l1_params"


def check_forbidden(raw: dict) -> list:
    return [k for k in raw if k in FORBIDDEN_KEYS]


def _safe_float(v):
    if v is None:
        return None
    try:
        import math
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def run(raw_input: dict, previous_state=None, init_temp_c=None) -> dict:
    forbidden = check_forbidden(raw_input)

    raw_input.setdefault("params_bundle_path", str(ROOT / "config/initial_calibration_params_v8_l1_candidate.json"))
    raw_input.setdefault("chtd_param_mode", "stage2_v4_calibrated")
    raw_input.setdefault("use_next_state", True)

    if previous_state is None and init_temp_c is not None:
        from hvac_sim.runtime_adapter import make_cabin_state_vector
        previous_state = make_cabin_state_vector(
            base_c=float(init_temp_c),
            head_fd_c=float(init_temp_c),
            head_fp_c=float(init_temp_c),
            cabin_fd_c=float(init_temp_c),
            cabin_fp_c=float(init_temp_c),
        )

    try:
        adapter_result = build_runtime_pipeline_inputs(raw_input, previous_state=previous_state)
        pipeline_inputs = adapter_result.pipeline_inputs
        adapter_prov = adapter_result.provenance
        adapter_warnings = list(adapter_result.warnings)

        result = run_comfort_pipeline_dict(pipeline_inputs)
        status = "ok"
        error_msg = None
    except Exception as e:
        result = {}
        adapter_prov = {}
        adapter_warnings = []
        status = "error"
        error_msg = str(e)

    driver = result.get("driver", {}) or {}
    passenger = result.get("passenger", {}) or {}
    x_next = result.get("x_next")  # list of floats

    # Extract zone temps from x_next state vector
    def _x(name):
        if x_next is None:
            return None
        idx = X_INDEX.get(name)
        if idx is None:
            return None
        try:
            return _safe_float(x_next[idx])
        except (IndexError, TypeError):
            return None

    inputs_used = result.get("inputs_used", {}) or {}
    airflow_src = inputs_used.get("airflow_distribution_source", "unknown")

    output = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "model_state": {
            "driver_head_temp_c":               _x("HeadTempFd"),
            "passenger_head_temp_c":            _x("HeadTempFp"),
            "driver_feet_temp_c":               _x("FeetTempFd"),
            "passenger_feet_temp_c":            _x("FeetTempFp"),
            "second_row_driver_head_temp_c":    _x("HeadTempSd"),
            "second_row_passenger_head_temp_c": _x("HeadTempSp"),
            "second_row_driver_feet_temp_c":    _x("FeetTempSd"),
            "second_row_passenger_feet_temp_c": _x("FeetTempSp"),
        "front_cabin_temp_c":               _x("CabinTempFd") or _x("CabinFrntTemp"),
        },
        "pmv": {
            "pmv_driver":               _safe_float(driver.get("pmv")),
            "pmv_passenger":            _safe_float(passenger.get("pmv")),
            "ppd_driver":               _safe_float(driver.get("ppd")),
            "ppd_passenger":            _safe_float(passenger.get("ppd")),
            "pmv_second_row_driver":    None,
            "pmv_second_row_passenger": None,
        },
        "comfort_flags": {
            "driver_comfortable":    bool(driver.get("valid", False)),
            "passenger_comfortable": bool(passenger.get("valid", False)),
        },
        "diagnostics": {
            "bypass_models":               bool(inputs_used.get("bypass_models", False)),
            "airflow_distribution_source": airflow_src,
            "airflow_total_source":        inputs_used.get("airflow_total_source", "unknown"),
            "mode_key":                    inputs_used.get("prov_actuator_mode_key", "unknown"),
            "face_fraction":               _safe_float(inputs_used.get("face_fraction")),
            "foot_fraction":               _safe_float(inputs_used.get("foot_fraction")),
            "defrost_fraction":            _safe_float(inputs_used.get("defrost_fraction")),
            "lr_split_source":             inputs_used.get("prov_lr_split_source", "unknown"),
            "rh_source":                   inputs_used.get("rh_source", "unknown"),
            "forbidden_input_detected":    len(forbidden) > 0,
            "fallback_used":               bool(inputs_used.get("fallback_used", False)),
            "missing_signal_list":         inputs_used.get("missing_signal_list", []),
            "forbidden_keys_found":        forbidden,
            "adapter_warnings":            adapter_warnings,
            "driver_air_temp_source":      inputs_used.get("driver_air_temp_source"),
            "passenger_air_temp_source":   inputs_used.get("passenger_air_temp_source"),
            "driver_air_temp_c":           _safe_float(inputs_used.get("driver_air_temp_c")),
            "passenger_air_temp_c":        _safe_float(inputs_used.get("passenger_air_temp_c")),
            "driver_mrt_source":           inputs_used.get("driver_mrt_source"),
            "passenger_mrt_source":        inputs_used.get("passenger_mrt_source"),
            "driver_mrt_c":                _safe_float(inputs_used.get("driver_mrt_c")),
            "passenger_mrt_c":             _safe_float(inputs_used.get("passenger_mrt_c")),
            "driver_air_speed_m_s":        _safe_float(inputs_used.get("driver_air_speed_m_s")),
            "passenger_air_speed_m_s":     _safe_float(inputs_used.get("passenger_air_speed_m_s")),
            "rh_percent":                  _safe_float(inputs_used.get("rh")),
            "met_driver":                  _safe_float(inputs_used.get("met_driver")),
            "met_passenger":               _safe_float(inputs_used.get("met_passenger")),
            "clo_driver":                  _safe_float(inputs_used.get("clo_driver")),
            "clo_passenger":               _safe_float(inputs_used.get("clo_passenger")),
            "driver_tsv_mrt_formula":      inputs_used.get("driver_tsv_mrt_formula"),
            "passenger_tsv_mrt_formula":   inputs_used.get("passenger_tsv_mrt_formula"),
            "trace_status":                result.get("trace_status"),
            "error": error_msg,
        },
        "provenance": {
            "calibration_status": CALIBRATION_STATUS,
            "airflow_mode":       AIRFLOW_MODE,
            "chtd_param_mode":    raw_input.get("chtd_param_mode", "stage2_v4_calibrated"),
            "package_version":    SCHEMA_VERSION,
            "adapter_provenance": adapter_prov,
        },
    }

    if x_next is not None:
        output["x_next"] = x_next

    return output


def main():
    parser = argparse.ArgumentParser(description="Vehicle PMV runtime")
    parser.add_argument("--input",  required=True, help="Input JSON file (SIG/FTE only)")
    parser.add_argument("--output", default=None,  help="Output JSON file (stdout if omitted)")
    args = parser.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = run(raw)

    out_str = json.dumps(out, indent=2, ensure_ascii=False, default=str)
    if args.output:
        Path(args.output).write_text(out_str, encoding="utf-8")
        print(f"Output written to {args.output}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
