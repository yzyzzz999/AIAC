"""AFE v2 offline batch replay: as-found ``afe_calc`` vs M8 dual-layer v2.

Decode/compare only — **not** wired to ``runtime_pipeline`` or default ``afe_calc``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from hvac_sim.afe.v2_diagnostic import AFEV2DiagnosticInputs, run_afe_v2_diagnostic

REPLAY_INPUT_SCHEMA = "afe_v2_replay_input_v1"
REPLAY_OUTPUT_SCHEMA = "afe_v2_replay_output_v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_INPUT_PATH = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "AFE_V2_REPLAY_SAMPLE_INPUT.json"
)
DEFAULT_SAMPLE_OUTPUT_PATH = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "AFE_V2_REPLAY_SAMPLE_OUTPUT.json"
)

_BUILTIN_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "case_id": "v_face_cooling",
        "mode_code": "V",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 50.0,
        "circle_prior_posn": 2.33,
        "driver_temp_door": 0.35,
        "passenger_temp_door": 0.35,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": False,
    },
    {
        "case_id": "f_foot_heating",
        "mode_code": "F",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "driver_temp_door": 0.75,
        "passenger_temp_door": 0.75,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": True,
    },
    {
        "case_id": "f_d_foot_defrost_heating",
        "mode_code": "F_D",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "driver_temp_door": 0.75,
        "passenger_temp_door": 0.75,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": True,
    },
    {
        "case_id": "d_defrost_heating",
        "mode_code": "D",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 50.0,
        "circle_prior_posn": 2.33,
        "driver_temp_door": 0.75,
        "passenger_temp_door": 0.75,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": True,
    },
    {
        "case_id": "v_f_face_foot",
        "mode_code": "V_F",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 50.0,
        "circle_prior_posn": 2.33,
        "driver_temp_door": 0.35,
        "passenger_temp_door": 0.35,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": False,
    },
    {
        "case_id": "v_d_f_face_foot_defrost",
        "mode_code": "V_D_F",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 50.0,
        "circle_prior_posn": 2.33,
        "driver_temp_door": 0.75,
        "passenger_temp_door": 0.75,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": True,
    },
)


class ReplayValidationError(ValueError):
    """Invalid AFE v2 replay input document."""


def default_replay_input_document() -> Dict[str, Any]:
    """Built-in six-case replay input (no external file required)."""
    return {
        "schema": REPLAY_INPUT_SCHEMA,
        "description": "AFE v2 offline replay sample cases (not runtime)",
        "cases": [dict(c) for c in _BUILTIN_CASES],
    }


def _heating_proxy(driver: float, passenger: float) -> bool:
    return 0.5 * (float(driver) + float(passenger)) >= 0.55


def _align_temp_doors_for_heating_mode(case: Mapping[str, Any]) -> Dict[str, float]:
    """Ensure blend-door positions match declared ``heating_mode`` when possible."""
    driver = float(case.get("driver_temp_door", 0.5))
    passenger = float(case.get("passenger_temp_door", 0.5))
    if "heating_mode" not in case:
        return {"driver_temp_door": driver, "passenger_temp_door": passenger}

    heating = bool(case["heating_mode"])
    proxy = _heating_proxy(driver, passenger)
    if heating and not proxy:
        return {"driver_temp_door": 0.75, "passenger_temp_door": 0.75}
    if not heating and proxy:
        return {"driver_temp_door": 0.35, "passenger_temp_door": 0.35}
    return {"driver_temp_door": driver, "passenger_temp_door": passenger}


def case_to_diagnostic_inputs(case: Mapping[str, Any]) -> AFEV2DiagnosticInputs:
    """Map one replay case dict to ``AFEV2DiagnosticInputs``."""
    temps = _align_temp_doors_for_heating_mode(case)
    return AFEV2DiagnosticInputs(
        mode_code=case.get("mode_code", "V"),
        front_blower_voltage=float(case.get("front_blower_voltage", 12.0)),
        rear_blower_voltage=float(case.get("rear_blower_voltage", 12.0)),
        circle_mode_posn=float(case.get("circle_mode_posn", 50.0)),
        circle_prior_posn=float(case.get("circle_prior_posn", 2.33)),
        driver_temp_door_posn=temps["driver_temp_door"],
        passenger_temp_door_posn=temps["passenger_temp_door"],
        rear_mode_posn=float(case.get("rear_mode_posn", 0.5)),
        rear_temp_posn=float(case.get("rear_temp_posn", 0.5)),
        left_motor_voltage_v=case.get("left_motor_voltage_v"),
        temp_C=float(case.get("temp_C", 25.0)),
        rh_percent=float(case.get("rh_percent", 50.0)),
        pressure_Pa=float(case.get("pressure_Pa", 101_325.0)),
    )


def validate_replay_input_document(doc: Any) -> List[Dict[str, Any]]:
    """Validate replay input JSON and return the case list."""
    if not isinstance(doc, dict):
        raise ReplayValidationError("root must be a JSON object")
    schema = doc.get("schema")
    if schema != REPLAY_INPUT_SCHEMA:
        raise ReplayValidationError(
            f"schema must be {REPLAY_INPUT_SCHEMA!r}, got {schema!r}"
        )
    cases = doc.get("cases")
    if not isinstance(cases, list):
        raise ReplayValidationError("'cases' must be a list")
    if len(cases) == 0:
        raise ReplayValidationError("'cases' must not be empty")
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ReplayValidationError(f"cases[{idx}] must be an object")
        case_id = case.get("case_id")
        if not case_id or not str(case_id).strip():
            raise ReplayValidationError(f"cases[{idx}] missing non-empty case_id")
    return cases


def _delta_m3h(delta_block: Mapping[str, Any]) -> float:
    return float(delta_block["delta_m3h"])


def summarize_case_diagnostic(
    diagnostic: Mapping[str, Any],
) -> Dict[str, Any]:
    """Extract per-case rollup fields from a diagnostic report."""
    delta = diagnostic["delta_summary"]
    afe = diagnostic["as_found_afe_result"]
    m8 = diagnostic["m8_dual_layer_v2_result"]
    return {
        "total_flow_delta": _delta_m3h(delta["total_flow"]),
        "face_flow_delta": _delta_m3h(delta["face"]),
        "foot_flow_delta": _delta_m3h(delta["foot"]),
        "defrost_flow_delta": _delta_m3h(delta["defrost"]),
        "rear_flow_delta": _delta_m3h(delta["rear"]),
        "convergence_status": {
            "as_found_afe": "converged" if afe.get("converged") else "not_converged",
            "m8_dual_layer_v2": str(m8.get("convergence_status", "unknown")),
        },
        "pending_items": list(diagnostic.get("pending_items", [])),
        "warnings": list(diagnostic.get("warnings", [])),
    }


def run_afe_v2_replay(
    doc: Optional[Mapping[str, Any]] = None,
    *,
    cases: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run batch AFE v2 diagnostic replay and return aggregated JSON."""
    if doc is not None and cases is not None:
        raise ValueError("provide doc or cases, not both")
    if doc is None and cases is None:
        doc = default_replay_input_document()
    if doc is not None:
        case_list = validate_replay_input_document(doc)
        input_schema = str(doc.get("schema", REPLAY_INPUT_SCHEMA))
        input_description = doc.get("description")
    else:
        case_list = list(cases)  # type: ignore[arg-type]
        input_schema = REPLAY_INPUT_SCHEMA
        input_description = None

    results: List[Dict[str, Any]] = []
    for case in case_list:
        diag = run_afe_v2_diagnostic(case_to_diagnostic_inputs(case))
        results.append(
            {
                "case_id": str(case["case_id"]),
                "inputs": dict(case),
                "diagnostic": diag,
                "summary": summarize_case_diagnostic(diag),
            }
        )

    return {
        "schema": REPLAY_OUTPUT_SCHEMA,
        "not_runtime_active": True,
        "experimental_not_vehicle_calibration": True,
        "afe_calc_default_unchanged": True,
        "input_schema": input_schema,
        "input_description": input_description,
        "case_count": len(results),
        "cases": results,
    }


def load_replay_input(path: Union[str, Path]) -> Dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_replay_output(
    report: Mapping[str, Any],
    path: Union[str, Path],
    *,
    pretty: bool = True,
) -> Path:
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return out
