"""Build machine-readable project integration status dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

STATUS_SCHEMA = "pmv_project_integration_status_v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TARGETS = (
    _REPO_ROOT / "simulink_conversion_package" / "python_targets"
)

MODULE_REQUIRED_FIELDS = (
    "status",
    "code_entry",
    "cli_entry",
    "main_output",
    "tests",
    "current_blocker",
    "next_action",
    "can_be_used_for_demo",
    "can_be_used_for_runtime",
    "can_be_used_for_calibration",
)

VALID_STATUSES = frozenset(
    {"ready", "experimental", "no_go", "waiting_data", "demo_only"}
)

KEY_DOCUMENTS = (
    "AFE_CUSTOMER_DATA_REVIEW_CHECKLIST.md",
    "AFE_CUSTOMER_DATA_INGEST_GUIDE.md",
    "CHTD_FIRST_PHASE_DATASET_SCHEMA.md",
    "CHTD_HVAC_SCALE_CALIBRATION_SMOKE.md",
    "PMV_APPLICABILITY_AND_DISPLAY_POLICY.md",
)

CHTD_TRACE_STATUS = "CHTD 28 states: 28 TRACE, 0 FAST_APPROX, 0 GAP"
CHTD_TRACE_SHORT = "28 TRACE / 0 FAST_APPROX / 0 GAP"


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _module(
    *,
    status: str,
    code_entry: Sequence[str],
    cli_entry: Sequence[str],
    main_output: Sequence[str],
    tests: Sequence[str],
    current_blocker: str,
    next_action: str,
    can_be_used_for_demo: bool,
    can_be_used_for_runtime: bool,
    can_be_used_for_calibration: bool,
) -> Dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid module status: {status!r}")
    return {
        "status": status,
        "code_entry": list(code_entry),
        "cli_entry": list(cli_entry),
        "main_output": list(main_output),
        "tests": list(tests),
        "current_blocker": current_blocker,
        "next_action": next_action,
        "can_be_used_for_demo": can_be_used_for_demo,
        "can_be_used_for_runtime": can_be_used_for_runtime,
        "can_be_used_for_calibration": can_be_used_for_calibration,
    }


def _base_modules() -> Dict[str, Dict[str, Any]]:
    return {
        "afe_as_found": _module(
            status="ready",
            code_entry=[
                "python_impl/hvac_sim/afe/flow.py",
                "python_impl/hvac_sim/afe/params.py",
                "python_impl/hvac_sim/airflow_pipeline.py",
            ],
            cli_entry=[
                "python_impl/examples/run_runtime_pmv.py",
                "python_impl/examples/run_pmv_pipeline_demo.py",
                "python_impl/examples/generate_afe_validation_report.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/AFE_VALIDATION_REPORT.json",
                "simulink_conversion_package/python_targets/AFE_VALIDATION_REPORT.md",
            ],
            tests=[
                "python_impl/tests/test_afe.py",
                "python_impl/tests/test_pipeline_smoke.py",
            ],
            current_blocker="Resistance/fan params still placeholder; not vehicle-calibrated",
            next_action="Ingest Monday customer fan curves and airflow distribution via ingest_afe_customer_data.py",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
        "afe_v2_dual_layer": _module(
            status="experimental",
            code_entry=[
                "python_impl/hvac_sim/afe/m8_dual_layer_model.py",
                "python_impl/hvac_sim/afe/m8_dual_layer_solver.py",
                "python_impl/hvac_sim/afe/v2_replay.py",
            ],
            cli_entry=[
                "python_impl/examples/run_afe_v2_replay.py",
                "python_impl/examples/run_afe_v2_diagnostic.py",
                "python_impl/examples/generate_afe_m8_dual_layer_solver_status.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/AFE_M8_DUAL_LAYER_SOLVER_EXPERIMENTAL_STATUS.json",
                "simulink_conversion_package/python_targets/AFE_V2_REPLAY_SAMPLE_OUTPUT.json",
            ],
            tests=[
                "python_impl/tests/test_afe_m8_dual_layer_solver.py",
                "python_impl/tests/test_afe_v2_replay.py",
            ],
            current_blocker="Not wired to runtime; fan P-Q curves and intake thresholds pending customer data",
            next_action="Engineering review of customer ingest output; G1/G2 anchor fit after data",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=False,
            can_be_used_for_calibration=False,
        ),
        "chtd_as_found_28_trace": _module(
            status="ready",
            code_entry=[
                "python_impl/hvac_sim/chtd/thermal.py",
                "python_impl/hvac_sim/chtd/params.py",
                "simulink_conversion_package/golden_cases/chtd_cases.json",
            ],
            cli_entry=[
                "python_impl/examples/run_pmv_pipeline_demo.py",
                "python_impl/examples/run_chtd_stability_audit.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/CHTD_28_TRACE_GOLDEN_AUDIT.md",
                "simulink_conversion_package/python_targets/CHTD_MULTI_STEP_STABILITY_AUDIT.json",
            ],
            tests=[
                "python_impl/tests/test_chtd_28_trace_integrity.py",
                "python_impl/tests/test_chtd_golden_partial.py",
            ],
            current_blocker="Multi-step warm-up blow-up from placeholder thermal mass / LUT=1.0 coupling",
            next_action="First-phase scale calibration after AFE Q and measured transients available",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
        "chtd_physical_capacity_preview": _module(
            status="demo_only",
            code_entry=["python_impl/hvac_sim/chtd/physical_params.py"],
            cli_entry=["python_impl/examples/run_chtd_physical_capacity_preview.py"],
            main_output=[
                "simulink_conversion_package/python_targets/CHTD_PHYSICAL_CAPACITY_PREVIEW.json",
                "simulink_conversion_package/python_targets/CHTD_PHYSICAL_CAPACITY_PREVIEW.md",
            ],
            tests=["python_impl/tests/test_chtd_physical_params.py"],
            current_blocker="estimated_not_calibrated_sanity_preview — not sign-off",
            next_action="Feeds safe_preview bundle; do not use alone as runtime params",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=False,
            can_be_used_for_calibration=False,
        ),
        "chtd_safe_preview_params": _module(
            status="demo_only",
            code_entry=[
                "python_impl/hvac_sim/chtd/safe_preview_params.py",
                "python_impl/hvac_sim/chtd/physical_params.py",
            ],
            cli_entry=[
                "python_impl/examples/run_chtd_safe_preview_check.py",
                "python_impl/examples/run_runtime_pmv.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/CHTD_SAFE_PREVIEW_CHECK.json",
                "simulink_conversion_package/python_targets/CHTD_SAFE_PREVIEW_CHECK.md",
            ],
            tests=[
                "python_impl/tests/test_chtd_safe_preview_params.py",
                "python_impl/tests/test_runtime_safe_preview_params.py",
            ],
            current_blocker="safe_preview_not_vehicle_calibration — opt-in demo/guard only",
            next_action="Use chtd_param_mode=safe_preview for demo; never for production sign-off",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=False,
            can_be_used_for_calibration=False,
        ),
        "chtd_first_phase_calibration": _module(
            status="waiting_data",
            code_entry=[
                "python_impl/hvac_sim/calibration/chtd_first_phase_dataset.py",
                "python_impl/hvac_sim/calibration/chtd_hvac_scale_adapter.py",
            ],
            cli_entry=[
                "python_impl/examples/prepare_chtd_first_phase_dataset.py",
                "python_impl/examples/run_chtd_hvac_scale_calibration_smoke.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/CHTD_FIRST_PHASE_SAMPLE_DATASET.json",
                "simulink_conversion_package/python_targets/CHTD_HVAC_SCALE_CALIBRATION_SMOKE.json",
            ],
            tests=[
                "python_impl/tests/test_chtd_first_phase_dataset.py",
                "python_impl/tests/test_chtd_hvac_scale_calibration.py",
            ],
            current_blocker="Real vehicle time-series not collected; AFE Q must be fixed before meaningful h_scale fit",
            next_action="Populate CHTD_FIRST_PHASE_DATA_COLLECTION_TABLE and run smoke PSO on sample only",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=False,
            can_be_used_for_calibration=False,
        ),
        "pmv_baseline": _module(
            status="ready",
            code_entry=[
                "python_impl/hvac_sim/pipeline.py",
                "python_impl/hvac_sim/pmv/fanger.py",
            ],
            cli_entry=[
                "python_impl/examples/run_pmv_pipeline_demo.py",
                "python_impl/examples/run_runtime_pmv.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/PMV_RUNTIME_INTERFACE_CONTRACT.md",
                "simulink_conversion_package/python_targets/integration_package/",
            ],
            tests=[
                "python_impl/tests/test_pmv_fanger.py",
                "python_impl/tests/test_pipeline_smoke.py",
            ],
            current_blocker="Comfort accuracy limited by placeholder AFE/CHTD params unless safe_preview opt-in",
            next_action="Default runtime uses CHTDParams(); optional chtd_param_mode=safe_preview for demo/guard",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=True,
        ),
        "pmv_applicability": _module(
            status="ready",
            code_entry=["python_impl/hvac_sim/pmv/applicability.py"],
            cli_entry=[
                "python_impl/examples/run_afe_v2_pmv_replay.py",
                "python_impl/examples/run_afe_v2_pmv_preview.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/PMV_APPLICABILITY_AND_DISPLAY_POLICY.md",
                "simulink_conversion_package/python_targets/AFE_V2_PMV_REPLAY_STATUS.md",
            ],
            tests=[
                "python_impl/tests/test_pmv_applicability.py",
                "python_impl/tests/test_afe_v2_pmv_applicability.py",
            ],
            current_blocker="Cold-start pmv_raw can exceed display band; policy handles display not physics",
            next_action="Keep pmv_raw / pmv_display / valid_for_comfort separated in all reports",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=True,
        ),
        "ir_kalman": _module(
            status="experimental",
            code_entry=[
                "python_impl/hvac_sim/ekf/head_temp_filter.py",
                "python_impl/hvac_sim/ir_compensation.py",
            ],
            cli_entry=[
                "python_impl/examples/run_pmv_pipeline_demo.py",
                "python_impl/examples/run_calibration_pso.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/calibration_demo_outputs/ekf_pso_demo_result.json",
            ],
            tests=[
                "python_impl/tests/test_head_temp_filter.py",
                "python_impl/tests/test_calibration_ekf_pso.py",
            ],
            current_blocker="Default off (enable_ir_fusion=false); Q/R/offset placeholder",
            next_action="Collect IR + measured head-air data before enabling in runtime",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=False,
            can_be_used_for_calibration=False,
        ),
        "defrost_tmadef": _module(
            status="experimental",
            code_entry=["python_impl/hvac_sim/defrost.py"],
            cli_entry=["python_impl/examples/run_runtime_pmv.py"],
            main_output=[
                "simulink_conversion_package/python_targets/DEFROST_TMADEF_MODEL_SPEC.md",
            ],
            tests=[
                "python_impl/tests/test_thermal_observer.py",
                "python_impl/tests/test_runtime_adapter.py",
            ],
            current_blocker="Hct SIG and M4 blend curve pending; fallback TmaDef≈EvaT",
            next_action="Close AFE customer checklist M4/M5 items",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
        "post_glass_defrost": _module(
            status="demo_only",
            code_entry=["python_impl/hvac_sim/defrost_post_glass.py"],
            cli_entry=["python_impl/examples/run_runtime_pmv.py"],
            main_output=[
                "simulink_conversion_package/python_targets/DEFROST_POST_GLASS_MODEL_REQUIREMENTS.md",
                "simulink_conversion_package/python_targets/RUNTIME_CONDITIONING_CODE_STATUS.md",
            ],
            tests=[
                "python_impl/tests/test_defrost_post_glass.py",
                "python_impl/tests/test_runtime_degradation.py",
            ],
            current_blocker="Default disabled; k0/k1 not calibrated; not fed into CHTD head-air proxy",
            next_action="Calibrate after glass observer and TmaDef evidence",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=False,
            can_be_used_for_calibration=False,
        ),
        "glass_temp_observer": _module(
            status="experimental",
            code_entry=[
                "python_impl/hvac_sim/glass_temp.py",
                "python_impl/hvac_sim/thermal_observer.py",
            ],
            cli_entry=["python_impl/examples/run_runtime_pmv.py"],
            main_output=[
                "simulink_conversion_package/python_targets/WINDSHIELD_GLASS_TEMP_MODEL_SPEC.md",
            ],
            tests=[
                "python_impl/tests/test_thermal_observer.py",
                "python_impl/tests/test_runtime_adapter.py",
            ],
            current_blocker="Showcase-grade h/k defaults; WinShdTEst falls back to AmbT when missing",
            next_action="Vehicle glass thermal calibration when measurement available",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
        "runtime_adapter": _module(
            status="ready",
            code_entry=[
                "python_impl/hvac_sim/runtime_adapter.py",
                "python_impl/hvac_sim/runtime_diagnostics.py",
            ],
            cli_entry=[
                "python_impl/examples/run_runtime_pmv.py",
                "python_impl/examples/generate_integration_package.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/integration_package/manifest.json",
                "simulink_conversion_package/python_targets/PMV_INTEGRATION_HANDOFF_GUIDE.md",
            ],
            tests=[
                "python_impl/tests/test_runtime_adapter.py",
                "python_impl/tests/test_integration_package.py",
            ],
            current_blocker="Many upstream SIGs use documented fallbacks; enable_corrected_chtd ignored",
            next_action=(
                "Hand off integration_package; supports chtd_param_mode safe_preview "
                "(opt-in, not_vehicle_calibration)"
            ),
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
        "degradation_bypass": _module(
            status="ready",
            code_entry=[
                "python_impl/hvac_sim/runtime_diagnostics.py",
                "python_impl/hvac_sim/chtd/helpers.py",
            ],
            cli_entry=["python_impl/examples/run_runtime_pmv.py"],
            main_output=[
                "simulink_conversion_package/python_targets/RUNTIME_DEGRADATION_AND_BYPASS_STRATEGY.md",
                "simulink_conversion_package/python_targets/RUNTIME_DEGRADATION_CODE_STATUS.md",
            ],
            tests=[
                "python_impl/tests/test_runtime_degradation.py",
                "python_impl/tests/test_runtime_policy.py",
            ],
            current_blocker="enable_runtime_conditioning not wired; CORRECTED CHTD not applied at runtime",
            next_action="Use degradation_report in demos; safe_preview optional via chtd_param_mode",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
        "runtime_guard": _module(
            status="ready",
            code_entry=[
                "python_impl/hvac_sim/runtime_guard.py",
            ],
            cli_entry=[
                "python_impl/examples/run_runtime_guard_smoke.py",
                "python_impl/examples/run_runtime_pmv.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/RUNTIME_GUARD_SMOKE_REPORT.json",
                "simulink_conversion_package/python_targets/RUNTIME_GUARD_SMOKE_REPORT.md",
            ],
            tests=[
                "python_impl/tests/test_runtime_guard.py",
                "python_impl/tests/test_runtime_safe_preview_params.py",
            ],
            current_blocker="Default CHTD params may overflow PMV on clamped extreme solar (overrange_solar)",
            next_action=(
                "Use overrange_solar_with_safe_preview for finite PMV; "
                "set chtd_param_mode=safe_preview in guarded demos"
            ),
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
        "release_integration_package": _module(
            status="ready",
            code_entry=[
                "python_impl/examples/generate_integration_package.py",
                "python_impl/examples/generate_release_package.py",
            ],
            cli_entry=[
                "python_impl/examples/generate_integration_package.py",
                "python_impl/examples/generate_release_package.py",
            ],
            main_output=[
                "simulink_conversion_package/python_targets/integration_package/",
                "simulink_conversion_package/python_targets/release_package/manifest.json",
            ],
            tests=[
                "python_impl/tests/test_integration_package.py",
                "python_impl/tests/test_release_package.py",
            ],
            current_blocker="Package reflects current placeholder calibration state",
            next_action="Regenerate after customer data ingest and first-phase dataset update",
            can_be_used_for_demo=True,
            can_be_used_for_runtime=True,
            can_be_used_for_calibration=False,
        ),
    }


def _go_no_go_table() -> List[Dict[str, str]]:
    return [
        {
            "item": "AFE v2 runtime",
            "decision": "No-Go",
            "rationale": "Dual-layer solver not wired; customer fan curves / intake thresholds pending review",
        },
        {
            "item": "CHTD full-state warmup",
            "decision": "No-Go",
            "rationale": "Until physical params reviewed and first-phase scale calibration on real transients",
        },
        {
            "item": "PMV baseline runtime",
            "decision": "Go with fallback + safe_preview optional",
            "rationale": (
                "End-to-end pipeline runs with default CHTDParams(); "
                "chtd_param_mode=safe_preview opt-in for demo/guard (not calibrated)"
            ),
        },
        {
            "item": "CHTD safe_preview runtime",
            "decision": "Go for demo",
            "rationale": (
                "Wired via runtime_adapter chtd_param_mode=safe_preview; "
                "CHTD_SAFE_PREVIEW_CHECK passes; not_vehicle_calibration"
            ),
        },
        {
            "item": "CHTD safe_preview production",
            "decision": "No-Go",
            "rationale": (
                "safe_preview_not_vehicle_calibration — physical+HVAC+solar scale preview only; "
                "no vehicle sign-off"
            ),
        },
        {
            "item": "IR fusion",
            "decision": "optional / default off unless data valid",
            "rationale": "enable_ir_fusion=false by default; needs measured IR + head-air series",
        },
        {
            "item": "PSO real calibration",
            "decision": "waiting data",
            "rationale": "Scaffold and demo PSO only; vehicle time-series and AFE evidence not signed off",
        },
    ]


def check_key_documents(
    targets_dir: Path,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Return per-document checks and warning strings for missing files."""
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for name in KEY_DOCUMENTS:
        path = targets_dir / name
        exists = path.is_file()
        checks.append({"document": name, "path": str(path), "exists": exists})
        if not exists:
            warnings.append(f"missing_key_document:{name}")
    return checks, warnings


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_key_metrics(targets_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "chtd_trace_status": CHTD_TRACE_STATUS,
        "chtd_trace_short": CHTD_TRACE_SHORT,
        "first_phase_chtd_scale_param_count": None,
        "first_phase_chtd_scale_param_names": [],
        "afe_v2_replay_status": "experimental_offline_not_runtime",
        "pmv_raw_display_policy": "pmv_raw unbounded; pmv_display clamped [-3.5, +3.5]; valid_for_comfort_interpretation separate",
        "feet_hvac_required_lut_scale_for_2c": None,
        "feet_hvac_required_lut_scale_for_5c": None,
        "physical_bundle_600_step_summary": {},
        "safe_preview_runtime_mode": "optional_chtd_param_mode",
        "safe_preview_classification": "safe_preview_not_vehicle_calibration",
        "safe_preview_check_all_passed": None,
        "runtime_guard_overrange_solar_with_safe_preview": None,
    }

    smoke = _read_json(targets_dir / "CHTD_HVAC_SCALE_CALIBRATION_SMOKE.json")
    if smoke and smoke.get("parameter_names"):
        names = list(smoke["parameter_names"])
        metrics["first_phase_chtd_scale_param_count"] = len(names)
        metrics["first_phase_chtd_scale_param_names"] = names
    else:
        try:
            from hvac_sim.calibration.chtd_hvac_scale_adapter import (
                build_chtd_hvac_scale_parameter_specs,
            )

            specs = build_chtd_hvac_scale_parameter_specs(include_optional=True)
            names = [s.name for s in specs]
            metrics["first_phase_chtd_scale_param_count"] = len(names)
            metrics["first_phase_chtd_scale_param_names"] = names
        except Exception:
            metrics["first_phase_chtd_scale_param_count"] = 10

    conv_audit = _read_json(targets_dir / "CHTD_HVAC_CONVECTION_SCALE_AUDIT.json")
    if conv_audit:
        for section in (
            conv_audit.get("summary"),
            conv_audit.get("feet_fd_calibration_targets"),
            conv_audit,
        ):
            if not isinstance(section, dict):
                continue
            for key in ("required_lut_scale_for_2c", "required_lut_scale_for_5c"):
                if key in section and metrics.get(f"feet_hvac_{key}") is None:
                    metrics[f"feet_hvac_{key}"] = section[key]

    preview = _read_json(targets_dir / "CHTD_PHYSICAL_CAPACITY_PREVIEW.json")
    if preview:
        bundle_summary = preview.get("bundle_summary") or {}
        scenarios = preview.get("scenarios") or []
        first = scenarios[0] if scenarios else {}
        pc = first.get("physical_capacity") or {}
        metrics["physical_bundle_600_step_summary"] = {
            "classification": preview.get("classification"),
            "steps": preview.get("steps"),
            "conclusion": bundle_summary.get("conclusion"),
            "first_threshold_step": pc.get("first_threshold_step"),
            "first_out_of_range_step": pc.get("first_out_of_range_step"),
            "all_finite_600s": pc.get("all_finite_600s"),
            "x_in_valid_range_-40_90": pc.get("x_in_valid_range_-40_90"),
            "scenario_id": pc.get("scenario_id"),
        }

    replay_status_md = targets_dir / "AFE_V2_PMV_REPLAY_STATUS.md"
    if replay_status_md.is_file():
        text = replay_status_md.read_text(encoding="utf-8")
        if "runtime_pipeline" in text and "No" in text:
            metrics["afe_v2_replay_status"] = "offline_experimental_runtime_not_wired"
        if "trend_checks_passed: true" in text.replace(" ", ""):
            metrics["afe_v2_replay_trend_checks"] = "passed_on_sample"

    policy_md = targets_dir / "PMV_APPLICABILITY_AND_DISPLAY_POLICY.md"
    if policy_md.is_file():
        metrics["pmv_policy_doc"] = _rel(policy_md, targets_dir.parent.parent)

    safe_preview = _read_json(targets_dir / "CHTD_SAFE_PREVIEW_CHECK.json")
    if safe_preview:
        metrics["safe_preview_check_all_passed"] = bool(safe_preview.get("all_passed"))
        metrics["safe_preview_classification"] = safe_preview.get("classification")

    guard_smoke = _read_json(targets_dir / "RUNTIME_GUARD_SMOKE_REPORT.json")
    if guard_smoke:
        for case in guard_smoke.get("cases", []):
            if case.get("case_id") == "overrange_solar_with_safe_preview":
                metrics["runtime_guard_overrange_solar_with_safe_preview"] = {
                    "pmv_finite": case.get("pmv_finite"),
                    "status": case.get("status"),
                    "pipeline_error": case.get("pipeline_error"),
                }
                break
        default_case = next(
            (c for c in guard_smoke.get("cases", []) if c.get("case_id") == "overrange_solar"),
            None,
        )
        if default_case is not None:
            metrics["runtime_guard_overrange_solar_default"] = {
                "pmv_finite": default_case.get("pmv_finite"),
                "pipeline_error": default_case.get("pipeline_error"),
            }

    return metrics


def build_project_integration_status(
    *,
    repo_root: Optional[Path] = None,
    targets_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Assemble integration status JSON payload."""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    targets = Path(targets_dir) if targets_dir is not None else (_DEFAULT_TARGETS)

    modules = _base_modules()
    doc_checks, warnings = check_key_documents(targets)
    key_metrics = _load_key_metrics(targets)

    for mod in modules.values():
        for field in MODULE_REQUIRED_FIELDS:
            if field not in mod:
                warnings.append(f"internal_module_field_missing:{field}")

    status: Dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "classification": "project_integration_dashboard",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(root.resolve()),
        "targets_dir": str(targets.resolve()),
        "modules": modules,
        "go_no_go": _go_no_go_table(),
        "key_metrics": key_metrics,
        "key_document_checks": doc_checks,
        "warnings": warnings,
    }
    return status


def render_project_integration_status_markdown(status: Mapping[str, Any]) -> str:
    """Render human-readable integration dashboard."""
    lines: List[str] = [
        "# PMV Project Integration Status",
        "",
        f"**Schema:** `{status.get('schema')}`  ",
        f"**Generated (UTC):** {status.get('generated_at_utc')}  ",
        f"**Targets dir:** `{status.get('targets_dir')}`",
        "",
    ]

    if status.get("warnings"):
        lines.extend(["## Warnings", ""])
        for w in status["warnings"]:
            lines.append(f"- `{w}`")
        lines.append("")

    lines.extend(["## 1. Module status", ""])
    lines.append(
        "| Module | Status | Demo | Runtime | Calibration | Blocker |"
    )
    lines.append("|--------|--------|------|---------|-------------|---------|")
    for mod_id, mod in status.get("modules", {}).items():
        blocker = mod["current_blocker"]
        blocker_cell = f"{blocker[:60]}…" if len(blocker) > 60 else blocker
        lines.append(
            f"| {mod_id} | {mod['status']} | "
            f"{'Y' if mod['can_be_used_for_demo'] else 'N'} | "
            f"{'Y' if mod['can_be_used_for_runtime'] else 'N'} | "
            f"{'Y' if mod['can_be_used_for_calibration'] else 'N'} | "
            f"{blocker_cell} |"
        )
    lines.append("")

    lines.extend(["## 2. Go / No-Go summary", ""])
    lines.append("| Item | Decision | Rationale |")
    lines.append("|------|----------|-----------|")
    for row in status.get("go_no_go", []):
        lines.append(f"| {row['item']} | **{row['decision']}** | {row['rationale']} |")
    lines.append("")

    km = status.get("key_metrics") or {}
    lines.extend(["## 3. Key metrics", ""])
    lines.append(f"- **CHTD trace:** `{km.get('chtd_trace_short', CHTD_TRACE_SHORT)}`")
    lines.append(
        f"- **First-phase CHTD scale params:** {km.get('first_phase_chtd_scale_param_count')}"
    )
    lines.append(f"- **AFE v2 replay:** `{km.get('afe_v2_replay_status')}`")
    lines.append(f"- **PMV policy:** {km.get('pmv_raw_display_policy')}")
    lut2 = km.get("feet_hvac_required_lut_scale_for_2c")
    lut5 = km.get("feet_hvac_required_lut_scale_for_5c")
    if lut2 is not None or lut5 is not None:
        lines.append(
            f"- **Feet HVAC LUT scale range:** ~{lut2} (2°C/step target), ~{lut5} (5°C/step target)"
        )
    pb = km.get("physical_bundle_600_step_summary") or {}
    if pb:
        lines.append(
            f"- **Physical bundle 600-step:** steps={pb.get('steps')} "
            f"conclusion={pb.get('conclusion')} "
            f"first_threshold_step={pb.get('first_threshold_step')} "
            f"all_finite_600s={pb.get('all_finite_600s')}"
        )
    sp_pass = km.get("safe_preview_check_all_passed")
    if sp_pass is not None:
        lines.append(
            f"- **CHTD safe_preview check:** all_passed={sp_pass} "
            f"classification={km.get('safe_preview_classification')}"
        )
    sp_guard = km.get("runtime_guard_overrange_solar_with_safe_preview")
    if sp_guard:
        lines.append(
            f"- **Runtime guard overrange_solar_with_safe_preview:** "
            f"pmv_finite={sp_guard.get('pmv_finite')}"
        )
    lines.append("")

    lines.extend(["## 4. Key document checks", ""])
    lines.append("| Document | Exists |")
    lines.append("|----------|--------|")
    for chk in status.get("key_document_checks", []):
        mark = "yes" if chk.get("exists") else "**missing**"
        lines.append(f"| {chk['document']} | {mark} |")
    lines.append("")

    lines.extend(["## Module details", ""])
    for mod_id, mod in status.get("modules", {}).items():
        lines.extend(
            [
                f"### {mod_id}",
                "",
                f"- **Status:** `{mod['status']}`",
                f"- **Code:** {', '.join(f'`{p}`' for p in mod['code_entry'][:3])}",
                f"- **CLI:** {', '.join(f'`{p}`' for p in mod['cli_entry'][:2])}",
                f"- **Output:** {', '.join(f'`{p}`' for p in mod['main_output'][:2])}",
                f"- **Tests:** {', '.join(f'`{p}`' for p in mod['tests'][:2])}",
                f"- **Next action:** {mod['next_action']}",
                "",
            ]
        )

    return "\n".join(lines)


def write_project_integration_status(
    *,
    repo_root: Optional[Path] = None,
    targets_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build and write JSON + Markdown dashboard files."""
    targets = Path(targets_dir) if targets_dir is not None else _DEFAULT_TARGETS
    out_json = Path(json_path) if json_path is not None else targets / "PMV_PROJECT_INTEGRATION_STATUS.json"
    out_md = Path(md_path) if md_path is not None else targets / "PMV_PROJECT_INTEGRATION_STATUS.md"

    status = build_project_integration_status(repo_root=repo_root, targets_dir=targets)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_project_integration_status_markdown(status), encoding="utf-8")
    status["output_json"] = str(out_json.resolve())
    status["output_md"] = str(out_md.resolve())
    return status
