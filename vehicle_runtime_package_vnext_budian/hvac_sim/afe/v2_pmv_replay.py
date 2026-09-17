"""AFE v2 → CHTD → PMV batch offline replay (experimental, not runtime).

Runs ``run_single_afe_v2_pmv_preview`` over a JSON case list, aggregates
summaries, and optionally validates qualitative trends. Does **not** wire to
``runtime_pipeline`` or replace default ``afe_calc``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from hvac_sim.afe.v2_chtd_preview import build_tma_source
from hvac_sim.afe.v2_pmv_preview import (
    AfeV2PmvPreviewResult,
    run_single_afe_v2_pmv_preview,
)

REPLAY_INPUT_SCHEMA = "afe_v2_pmv_replay_input_v1"
REPLAY_OUTPUT_SCHEMA = "afe_v2_pmv_replay_output_v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_INPUT_PATH = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "AFE_V2_PMV_REPLAY_SAMPLE_INPUT.json"
)
DEFAULT_SAMPLE_OUTPUT_PATH = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "AFE_V2_PMV_REPLAY_SAMPLE_OUTPUT.json"
)

_COLD_HEATING_CASE_IDS = frozenset(
    {
        "foot_heating_cold",
        "foot_defrost_heating_cold",
        "defrost_heating_cold",
    }
)
_FACE_COOLING_CASE_IDS = frozenset({"face_cooling_hot", "face_cooling_mild"})
_HEATING_FALLBACK_CASE_IDS = frozenset(
    {
        "foot_heating_cold",
        "foot_defrost_heating_cold",
        "defrost_heating_cold",
    }
)

_BUILTIN_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "case_id": "face_cooling_hot",
        "mode_code": "V",
        "amb_t": 32.0,
        "rh": 55.0,
        "driver_face_tma": 16.0,
        "passenger_face_tma": 16.0,
        "driver_foot_tma": 20.0,
        "passenger_foot_tma": 20.0,
        "rear_face_tma": 18.0,
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "heating_mode": False,
    },
    {
        "case_id": "face_cooling_mild",
        "mode_code": "V",
        "amb_t": 28.0,
        "rh": 50.0,
        "driver_face_tma": 18.0,
        "passenger_face_tma": 18.0,
        "driver_foot_tma": 22.0,
        "passenger_foot_tma": 22.0,
        "rear_face_tma": 20.0,
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "heating_mode": False,
    },
    {
        "case_id": "foot_heating_cold",
        "mode_code": "F",
        "amb_t": -5.0,
        "rh": 60.0,
        "driver_face_tma": 35.0,
        "passenger_face_tma": 35.0,
        "driver_foot_tma": 42.0,
        "passenger_foot_tma": 42.0,
        "rear_face_tma": 34.0,
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "heating_mode": True,
    },
    {
        "case_id": "foot_defrost_heating_cold",
        "mode_code": "F_D",
        "amb_t": -5.0,
        "rh": 60.0,
        "driver_face_tma": 35.0,
        "passenger_face_tma": 35.0,
        "driver_foot_tma": 42.0,
        "passenger_foot_tma": 42.0,
        "rear_face_tma": 34.0,
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "heating_mode": True,
    },
    {
        "case_id": "defrost_heating_cold",
        "mode_code": "D",
        "amb_t": -5.0,
        "rh": 60.0,
        "driver_face_tma": 35.0,
        "passenger_face_tma": 35.0,
        "driver_foot_tma": 42.0,
        "passenger_foot_tma": 42.0,
        "rear_face_tma": 34.0,
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 50.0,
        "circle_prior_posn": 2.33,
        "heating_mode": True,
    },
    {
        "case_id": "face_foot_transition",
        "mode_code": "V_F",
        "amb_t": 24.0,
        "rh": 50.0,
        "driver_face_tma": 18.0,
        "passenger_face_tma": 18.0,
        "driver_foot_tma": 16.0,
        "passenger_foot_tma": 16.0,
        "rear_face_tma": 20.0,
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 50.0,
        "circle_prior_posn": 2.33,
        "heating_mode": False,
    },
)


class PmvReplayValidationError(ValueError):
    """Invalid AFE v2 PMV replay input document."""


def default_replay_input_document() -> Dict[str, Any]:
    return {
        "schema": REPLAY_INPUT_SCHEMA,
        "description": "AFE v2 PMV offline replay sample cases (not runtime; not sign-off)",
        "cases": [dict(c) for c in _BUILTIN_CASES],
    }


def _heating_proxy(driver: float, passenger: float) -> bool:
    return 0.5 * (float(driver) + float(passenger)) >= 0.55


def case_to_afe_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    """Map PMV replay case fields to AFE v2 diagnostic / preview case dict."""
    heating = bool(case.get("heating_mode", False))
    driver_door = case.get("driver_temp_door")
    passenger_door = case.get("passenger_temp_door")
    if driver_door is None or passenger_door is None:
        default_door = 0.75 if heating else 0.35
        driver_door = default_door if driver_door is None else float(driver_door)
        passenger_door = default_door if passenger_door is None else float(passenger_door)
    else:
        driver_door = float(driver_door)
        passenger_door = float(passenger_door)
        if heating and not _heating_proxy(driver_door, passenger_door):
            driver_door = passenger_door = 0.75
        elif not heating and _heating_proxy(driver_door, passenger_door):
            driver_door = passenger_door = 0.35

    rh = float(case.get("rh", case.get("rh_percent", 50.0)))
    return {
        "case_id": str(case["case_id"]),
        "mode_code": str(case.get("mode_code", "V")),
        "front_blower_voltage": float(case.get("front_blower_voltage", 12.0)),
        "rear_blower_voltage": float(case.get("rear_blower_voltage", 12.0)),
        "circle_mode_posn": float(case.get("circle_mode_posn", 50.0)),
        "circle_prior_posn": float(case.get("circle_prior_posn", 2.33)),
        "driver_temp_door": driver_door,
        "passenger_temp_door": passenger_door,
        "rear_mode_posn": float(case.get("rear_mode_posn", 0.5)),
        "rear_temp_posn": float(case.get("rear_temp_posn", 0.5)),
        "heating_mode": heating,
        "temp_C": float(case.get("amb_t", 24.0)),
        "rh_percent": rh,
    }


def case_to_tma_source(case: Mapping[str, Any]) -> Dict[str, Any]:
    """Build Tma source from PMV replay case (explicit Tma fields optional)."""
    kwargs: Dict[str, Any] = {
        "amb_t": float(case.get("amb_t", 24.0)),
        "heating_mode": bool(case.get("heating_mode", False)),
    }
    for key in (
        "driver_face_tma",
        "passenger_face_tma",
        "driver_foot_tma",
        "passenger_foot_tma",
        "rear_face_tma",
        "rear_foot_tma",
        "defrost_tma",
        "eva_t",
        "hct",
        "posn_fdh",
    ):
        if key in case and case[key] is not None:
            kwargs[key] = float(case[key])
    return build_tma_source(**kwargs)


def validate_replay_input_document(doc: Any) -> List[Dict[str, Any]]:
    if not isinstance(doc, dict):
        raise PmvReplayValidationError("root must be a JSON object")
    schema = doc.get("schema")
    if schema != REPLAY_INPUT_SCHEMA:
        raise PmvReplayValidationError(
            f"schema must be {REPLAY_INPUT_SCHEMA!r}, got {schema!r}"
        )
    cases = doc.get("cases")
    if not isinstance(cases, list):
        raise PmvReplayValidationError("'cases' must be a list")
    if len(cases) == 0:
        raise PmvReplayValidationError("'cases' must not be empty")
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise PmvReplayValidationError(f"cases[{idx}] must be an object")
        case_id = case.get("case_id")
        if not case_id or not str(case_id).strip():
            raise PmvReplayValidationError(f"cases[{idx}] missing non-empty case_id")
    return cases


def summarize_case_pmv_preview(
    preview: AfeV2PmvPreviewResult,
    *,
    inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "case_id": preview.case_id,
        "driver_pmv": float(preview.driver.pmv),
        "passenger_pmv": float(preview.passenger.pmv),
        "driver_pmv_raw": float(preview.driver.pmv_raw),
        "passenger_pmv_raw": float(preview.passenger.pmv_raw),
        "driver_pmv_display": float(preview.driver.pmv_display),
        "passenger_pmv_display": float(preview.passenger.pmv_display),
        "driver_valid_for_comfort_interpretation": bool(
            preview.driver.valid_for_comfort_interpretation
        ),
        "passenger_valid_for_comfort_interpretation": bool(
            preview.passenger.valid_for_comfort_interpretation
        ),
        "driver_pmv_applicability": str(preview.driver.pmv_applicability),
        "passenger_pmv_applicability": str(preview.passenger.pmv_applicability),
        "driver_pmv_applicability_reasons": list(preview.driver.pmv_applicability_reasons),
        "passenger_pmv_applicability_reasons": list(
            preview.passenger.pmv_applicability_reasons
        ),
        "driver_air_speed_m_s": float(preview.driver.air_speed_m_s),
        "passenger_air_speed_m_s": float(preview.passenger.air_speed_m_s),
        "driver_air_temp_c": float(preview.driver.air_temp_c),
        "passenger_air_temp_c": float(preview.passenger.air_temp_c),
        "driver_mrt_c": float(preview.driver.mean_radiant_temp_c),
        "passenger_mrt_c": float(preview.passenger.mean_radiant_temp_c),
        "quality_level": preview.quality_level,
        "fallback_used": dict(preview.fallback_used),
        "warnings": list(preview.warnings),
        "rh_percent": float(preview.rh_percent),
        "inputs": dict(inputs),
    }


def _aggregate_summary(case_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not case_summaries:
        return {"case_count": 0}

    def _stats(key: str) -> Dict[str, float]:
        vals = [float(c[key]) for c in case_summaries]
        return {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(sum(vals) / len(vals)),
        }

    quality_counts: Dict[str, int] = {}
    fallback_case_count = 0
    for c in case_summaries:
        q = str(c.get("quality_level", "unknown"))
        quality_counts[q] = quality_counts.get(q, 0) + 1
        if c.get("fallback_used"):
            fallback_case_count += 1

    return {
        "case_count": len(case_summaries),
        "driver_pmv": _stats("driver_pmv"),
        "passenger_pmv": _stats("passenger_pmv"),
        "driver_air_speed_m_s": _stats("driver_air_speed_m_s"),
        "passenger_air_speed_m_s": _stats("passenger_air_speed_m_s"),
        "quality_level_counts": quality_counts,
        "cases_with_fallback": fallback_case_count,
    }


def _case_summary_by_id(
    case_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    return {str(c["case_id"]): c for c in case_summaries}


def _trend_check(
    check_id: str,
    passed: bool,
    value: Any,
    reason: str,
) -> Dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "value": value,
        "reason": reason,
    }


def validate_afe_v2_pmv_replay_trends(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Qualitative trend checks on a PMV replay result (not accuracy sign-off)."""
    checks: List[Dict[str, Any]] = []
    summaries = list(result.get("case_summary", []))
    by_id = _case_summary_by_id(summaries)

    hot = by_id.get("face_cooling_hot")
    mild = by_id.get("face_cooling_mild")
    if hot and mild:
        hot_drv = float(hot["driver_pmv"])
        mild_drv = float(mild["driver_pmv"])
        passed = mild_drv >= hot_drv or abs(mild_drv) < abs(hot_drv)
        checks.append(
            _trend_check(
                "hot_face_cooling_pmv_vs_mild",
                passed,
                {"hot_driver_pmv": hot_drv, "mild_driver_pmv": mild_drv},
                "mild face cooling PMV should be >= hot or closer to comfort (|pmv| smaller)",
            )
        )

    cold_cases = [by_id[cid] for cid in _COLD_HEATING_CASE_IDS if cid in by_id]
    if cold_cases:
        nan_cases = [
            c["case_id"]
            for c in cold_cases
            if not (
                math.isfinite(float(c["driver_pmv"]))
                and math.isfinite(float(c["passenger_pmv"]))
            )
        ]
        checks.append(
            _trend_check(
                "cold_heating_no_nan",
                len(nan_cases) == 0,
                {"nan_case_ids": nan_cases},
                "cold heating cases must produce finite driver/passenger PMV",
            )
        )

    base_case = by_id.get("face_cooling_mild") or by_id.get("face_cooling_hot")
    if base_case:
        inputs = dict(base_case.get("inputs", {}))
        low_face = dict(inputs)
        high_face = dict(inputs)
        low_face["driver_face_tma"] = float(inputs.get("driver_face_tma", 18.0))
        high_face["driver_face_tma"] = float(inputs.get("driver_face_tma", 18.0)) + 4.0
        low_preview = run_single_afe_v2_pmv_preview(
            case_to_afe_case(low_face),
            tma_source=case_to_tma_source(low_face),
            rh_percent=float(inputs.get("rh", 50.0)),
        )
        high_preview = run_single_afe_v2_pmv_preview(
            case_to_afe_case(high_face),
            tma_source=case_to_tma_source(high_face),
            rh_percent=float(inputs.get("rh", 50.0)),
        )
        low_pmv = float(low_preview.driver.pmv)
        high_pmv = float(high_preview.driver.pmv)
        checks.append(
            _trend_check(
                "face_tma_increases_face_cooling_pmv",
                high_pmv > low_pmv,
                {"low_face_tma_pmv": low_pmv, "high_face_tma_pmv": high_pmv},
                "raising driver_face_tma should increase face-cooling driver PMV",
            )
        )

        low_blower = dict(inputs)
        high_blower = dict(inputs)
        low_blower["front_blower_voltage"] = 8.0
        high_blower["front_blower_voltage"] = 14.0
        low_spd = run_single_afe_v2_pmv_preview(
            case_to_afe_case(low_blower),
            tma_source=case_to_tma_source(low_blower),
            rh_percent=float(inputs.get("rh", 50.0)),
        )
        high_spd = run_single_afe_v2_pmv_preview(
            case_to_afe_case(high_blower),
            tma_source=case_to_tma_source(high_blower),
            rh_percent=float(inputs.get("rh", 50.0)),
        )
        low_as = float(low_spd.driver.air_speed_m_s)
        high_as = float(high_spd.driver.air_speed_m_s)
        checks.append(
            _trend_check(
                "blower_voltage_increases_air_speed",
                high_as > low_as,
                {"low_voltage_air_speed_m_s": low_as, "high_voltage_air_speed_m_s": high_as},
                "higher front_blower_voltage should increase driver air_speed",
            )
        )

    heating_fallback = [
        by_id[cid] for cid in _HEATING_FALLBACK_CASE_IDS if cid in by_id
    ]
    if heating_fallback:
        cases_with_fallback = []
        explainable = []
        for c in heating_fallback:
            fb = c.get("fallback_used") or {}
            if not fb:
                continue
            cases_with_fallback.append(c["case_id"])
            if all(isinstance(v, str) and v.strip() for v in fb.values()):
                explainable.append(c["case_id"])
        passed = all(
            all(isinstance(v, str) and v.strip() for v in (c.get("fallback_used") or {}).values())
            for c in heating_fallback
            if c.get("fallback_used")
        )
        checks.append(
            _trend_check(
                "defrost_foot_heating_fallback_explainable",
                passed,
                {
                    "cases_with_fallback": cases_with_fallback,
                    "explainable_case_ids": explainable,
                    "total_heating_cases": len(heating_fallback),
                },
                "when defrost/foot heating uses fallback, reasons must be non-empty strings",
            )
        )

    out_of_range_typical: List[str] = []
    non_finite: List[str] = []
    for c in summaries:
        cid = str(c["case_id"])
        for seat in ("driver_pmv", "passenger_pmv"):
            val = float(c[seat])
            if not math.isfinite(val):
                non_finite.append(f"{cid}:{seat}")
    checks.append(
        _trend_check(
            "pmv_all_finite",
            len(non_finite) == 0,
            {"non_finite": non_finite},
            "all PMV values must be finite",
        )
    )

    cold_heating = [by_id[cid] for cid in _COLD_HEATING_CASE_IDS if cid in by_id]
    if cold_heating:
        not_marked: List[str] = []
        for c in cold_heating:
            cid = str(c["case_id"])
            if c.get("driver_pmv_applicability") != "out_of_reference_range":
                not_marked.append(f"{cid}:driver")
            if c.get("passenger_pmv_applicability") != "out_of_reference_range":
                not_marked.append(f"{cid}:passenger")
        checks.append(
            _trend_check(
                "cold_heating_applicability_marked",
                len(not_marked) == 0,
                {
                    "not_marked_out_of_reference_range": not_marked,
                    "expected": "out_of_reference_range for cold heating cases",
                },
                "cold heating PMV may exceed [-3,3] but must be marked out_of_reference_range",
            )
        )

    face_cooling_ids = ("face_cooling_hot", "face_cooling_mild")
    face_cases = [by_id[cid] for cid in face_cooling_ids if cid in by_id]
    if face_cases:
        unresolved: List[str] = []
        for c in face_cases:
            cid = str(c["case_id"])
            for seat_prefix in ("driver", "passenger"):
                app = c.get(f"{seat_prefix}_pmv_applicability")
                reasons = c.get(f"{seat_prefix}_pmv_applicability_reasons") or []
                if app == "out_of_reference_range" and not reasons:
                    unresolved.append(f"{cid}:{seat_prefix}")
                if app not in {
                    "within_reference_range",
                    "out_of_reference_range",
                }:
                    unresolved.append(f"{cid}:{seat_prefix}:missing_applicability")
        checks.append(
            _trend_check(
                "face_cooling_applicability_documented",
                len(unresolved) == 0,
                {"unresolved": unresolved, "case_ids": [c["case_id"] for c in face_cases]},
                "face cooling hot/mild must be within range or list applicability reasons",
            )
        )

    return checks


def run_afe_v2_pmv_replay(
    doc: Optional[Mapping[str, Any]] = None,
    *,
    cases: Optional[Sequence[Mapping[str, Any]]] = None,
    include_trend_checks: bool = True,
) -> Dict[str, Any]:
    """Run batch AFE v2 PMV preview replay and return aggregated JSON."""
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

    case_summaries: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    for raw_case in case_list:
        afe_case = case_to_afe_case(raw_case)
        tma = case_to_tma_source(raw_case)
        rh = float(raw_case.get("rh", raw_case.get("rh_percent", 50.0)))
        preview = run_single_afe_v2_pmv_preview(
            afe_case,
            tma_source=tma,
            rh_percent=rh,
        )
        previews.append(preview.to_dict())
        case_summaries.append(
            summarize_case_pmv_preview(preview, inputs=dict(raw_case))
        )

    result: Dict[str, Any] = {
        "schema": REPLAY_OUTPUT_SCHEMA,
        "classification": "experimental_offline_pmv_replay_not_runtime",
        "not_runtime_active": True,
        "experimental_not_vehicle_calibration": True,
        "not_sign_off_result": True,
        "runtime_pipeline_unchanged": True,
        "afe_calc_default_unchanged": True,
        "thermal_formula_unchanged": True,
        "input_schema": input_schema,
        "input_description": input_description,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case_count": len(case_summaries),
        "case_summary": case_summaries,
        "aggregate_summary": _aggregate_summary(case_summaries),
        "previews": previews,
    }
    if include_trend_checks:
        trend_checks = validate_afe_v2_pmv_replay_trends(result)
        result["trend_checks"] = trend_checks
        result["trend_checks_passed"] = all(c["passed"] for c in trend_checks)
    return result


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


def write_sample_files(*, pretty: bool = True) -> Tuple[Path, Path]:
    """Write default sample input and output under python_targets."""
    doc_in = default_replay_input_document()
    write_replay_output(doc_in, DEFAULT_SAMPLE_INPUT_PATH, pretty=pretty)
    doc_out = run_afe_v2_pmv_replay(doc_in)
    write_replay_output(doc_out, DEFAULT_SAMPLE_OUTPUT_PATH, pretty=pretty)
    return DEFAULT_SAMPLE_INPUT_PATH, DEFAULT_SAMPLE_OUTPUT_PATH
