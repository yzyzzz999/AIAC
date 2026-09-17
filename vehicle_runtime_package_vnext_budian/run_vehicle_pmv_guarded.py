"""
Vehicle PMV guarded runtime entry point.
Uses run_runtime_pmv_guarded() which applies additional safety checks.
Usage: python run_vehicle_pmv_guarded.py --input sample.json [--output out.json]
"""
import sys
import json
import argparse
import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from hvac_sim.runtime_guard import run_runtime_pmv_guarded
from run_vehicle_pmv import (
    check_forbidden, load_state,
    SCHEMA_VERSION, CALIBRATION_STATUS, AIRFLOW_MODE,
)


def run_guarded(raw_input: dict, x_init=None) -> dict:
    forbidden = check_forbidden(raw_input)
    if forbidden:
        return {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "status": "blocked",
            "schema_version": SCHEMA_VERSION,
            "diagnostics": {
                "forbidden_input_detected": True,
                "forbidden_keys_found": forbidden,
                "error": f"Forbidden runtime inputs detected: {forbidden}",
            },
            "provenance": {
                "calibration_status": CALIBRATION_STATUS,
                "package_version": SCHEMA_VERSION,
            },
        }

    raw_input.setdefault("chtd_param_mode", "stage2_v4_calibrated")
    raw_input.setdefault("use_next_state", True)
    if x_init is not None:
        raw_input["x_init"] = x_init.tolist() if hasattr(x_init, "tolist") else x_init

    try:
        result = run_runtime_pmv_guarded(raw_input)
        status = "ok"
        error_msg = None
    except Exception as e:
        result = {}
        status = "error"
        error_msg = str(e)

    output = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "model_state": {
            "driver_head_temp_c":               result.get("driver_head_temp"),
            "passenger_head_temp_c":            result.get("passenger_head_temp"),
            "driver_feet_temp_c":               result.get("driver_feet_temp"),
            "passenger_feet_temp_c":            result.get("passenger_feet_temp"),
            "second_row_driver_head_temp_c":    result.get("second_row_driver_head_temp"),
            "second_row_passenger_head_temp_c": result.get("second_row_passenger_head_temp"),
            "second_row_driver_feet_temp_c":    result.get("second_row_driver_feet_temp"),
            "second_row_passenger_feet_temp_c": result.get("second_row_passenger_feet_temp"),
        },
        "pmv": {
            "pmv_driver":               result.get("pmv_driver"),
            "pmv_passenger":            result.get("pmv_passenger"),
            "pmv_second_row_driver":    result.get("pmv_second_row_driver"),
            "pmv_second_row_passenger": result.get("pmv_second_row_passenger"),
        },
        "comfort_flags": {
            "driver_comfortable":    result.get("driver_comfortable"),
            "passenger_comfortable": result.get("passenger_comfortable"),
        },
        "diagnostics": {
            "airflow_distribution_source": result.get("airflow_distribution_source", "unknown"),
            "mode_key":                    result.get("prov_actuator_mode_key", "unknown"),
            "lr_split_source":             result.get("prov_lr_split_source", "unknown"),
            "forbidden_input_detected":    False,
            "fallback_used":               result.get("fallback_used", False),
            "guard_triggered":             result.get("guard_triggered", False),
            "guard_reason":                result.get("guard_reason"),
            "missing_signal_list":         result.get("missing_signal_list", []),
            "error": error_msg,
        },
        "provenance": {
            "calibration_status": CALIBRATION_STATUS,
            "airflow_mode":       AIRFLOW_MODE,
            "chtd_param_mode":    raw_input.get("chtd_param_mode", "stage2_v4_calibrated"),
            "package_version":    SCHEMA_VERSION,
            "entry_point":        "run_runtime_pmv_guarded",
        },
    }
    if result.get("x_next") is not None:
        import numpy as np
        output["x_next"] = np.array(result["x_next"]).tolist()

    return output


def main():
    parser = argparse.ArgumentParser(description="Vehicle PMV guarded runtime")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--state",  default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    x_init = load_state(args.state)
    out = run_guarded(raw, x_init=x_init)

    out_str = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(out_str, encoding="utf-8")
        print(f"Output written to {args.output}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
