"""AFE customer-data ready pipeline — ingest through offline validation chain.

Chains customer ingest, validation report, G1/G2 smoke, v2 replay, CHTD preview,
and optional project/demo refresh. Does not modify packaged ``calibration_data/``,
``afe_calc`` defaults, ``thermal.py``, or runtime wiring.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Union

from hvac_sim.afe.customer_data_ingest import (
    DEFAULT_CALIB_DIR,
    DEFAULT_SAMPLE_INPUT,
    merge_into_evidence,
    normalize_customer_data,
    validate_customer_data,
)
from hvac_sim.afe.model_version import DEFAULT_AFE_MODEL_VERSION, get_model_version_status
from hvac_sim.afe.validation.airflow_distribution_comparison import (
    compare_airflow_distribution_anchors,
)
from hvac_sim.afe.validation.evidence_validator import validate_calibration_evidence
from hvac_sim.afe.validation.fan_curve_comparison import compare_fan_curves_to_initial_json
from hvac_sim.afe.validation.m8_g1_g2_calibration_smoke import (
    build_g1_g2_smoke_report,
    render_g1_g2_smoke_markdown,
    run_m8_g1_g2_calibration_smoke,
)
from hvac_sim.afe.validation.parameter_groups import build_parameter_groups_for_calibration
from hvac_sim.afe.validation.report import _render_markdown as render_validation_report_md
from hvac_sim.afe.v2_chtd_preview import (
    build_tma_source,
    default_builtin_scenarios,
    run_afe_v2_chtd_preview,
    run_afe_v2_chtd_preview_from_input_doc,
)
from hvac_sim.afe.v2_replay import run_afe_v2_replay, write_replay_output

MANIFEST_SCHEMA = "afe_customer_pipeline_manifest_v1"
CLASSIFICATION = "afe_customer_data_pipeline_not_vehicle_calibration"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "afe_customer_pipeline"
)


def _write_json(path: Path, payload: Mapping[str, Any], *, pretty: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _load_customer_input(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def use_merged_calibration_data(evidence_dir: Path) -> Iterator[None]:
    """Temporarily point calibration JSON loaders at merged evidence."""
    import hvac_sim.afe.calibration_data.loader as loader_mod

    original_dir = loader_mod._DATA_DIR
    loader_mod._DATA_DIR = Path(evidence_dir)
    cached = (
        loader_mod.load_actuator_voltage_targets,
        loader_mod.load_blend_door_voltage_endpoints,
        loader_mod.load_fan_curve_initial,
        loader_mod.load_airflow_distribution_anchors,
        loader_mod.load_physical_outlet_map,
        loader_mod.load_m8_dual_layer_structure_confirmed,
    )
    for fn in cached:
        fn.cache_clear()
    try:
        yield
    finally:
        loader_mod._DATA_DIR = original_dir
        for fn in cached:
            fn.cache_clear()


def _step_result(
    step_id: str,
    *,
    status: str,
    output_paths: Optional[Sequence[Union[str, Path]]] = None,
    warnings: Optional[Sequence[str]] = None,
    blockers: Optional[Sequence[str]] = None,
    skipped_reason: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "step_id": step_id,
        "status": status,
        "output_paths": [str(p) for p in (output_paths or [])],
        "warnings": list(warnings or []),
        "blockers": list(blockers or []),
    }
    if skipped_reason:
        row["skipped_reason"] = skipped_reason
    if extra:
        row.update(dict(extra))
    return row


def _warning_codes(normalized: Mapping[str, Any]) -> List[str]:
    return [str(w.get("code", "")) for w in normalized.get("review_warnings", [])]


def _has_unknown_fan_unit(normalized: Mapping[str, Any]) -> bool:
    if "flow_unit_unknown" in _warning_codes(normalized):
        return True
    return any(
        str(curve.get("flow_unit", "")) == "unknown"
        for curve in normalized.get("fan_curve", [])
    )


def _has_bad_airflow_percent_sum(normalized: Mapping[str, Any]) -> bool:
    return "percentages_not_sum_100" in _warning_codes(normalized)


def _has_missing_mode_code(normalized: Mapping[str, Any]) -> bool:
    return "missing_mode_code" in _warning_codes(normalized)


def _load_anchors_from_merged(merged_dir: Path) -> List[Dict[str, Any]]:
    path = merged_dir / "airflow_distribution_anchors.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return list(doc.get("anchors", []))


def _has_dual_layer_anchors(merged_dir: Path) -> bool:
    anchors = _load_anchors_from_merged(merged_dir)
    if not anchors:
        return False
    return any(a.get("provenance") == "customer_provided" or a.get("anchor_id") for a in anchors)


def build_validation_report_for_evidence_dir(calibration_dir: Path) -> Dict[str, Any]:
    """AFE validation report using merged evidence directory."""
    evidence = validate_calibration_evidence(calibration_dir=calibration_dir)
    with use_merged_calibration_data(calibration_dir):
        fan_rows = compare_fan_curves_to_initial_json()
        airflow_rows = compare_airflow_distribution_anchors()
    groups = build_parameter_groups_for_calibration()
    dual = get_model_version_status("target_vehicle_dual_layer")
    as_found = get_model_version_status(DEFAULT_AFE_MODEL_VERSION.value)
    return {
        "schema": "afe_validation_report_v1",
        "task": "AFE-customer-pipeline",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": (
            "Customer-data pipeline validation — pending engineering review. "
            "Not formal M8 vehicle calibration sign-off."
        ),
        "evidence_dir": str(calibration_dir.resolve()),
        "model_versions": {
            "default": DEFAULT_AFE_MODEL_VERSION.value,
            "afe_as_found": {"implemented": as_found.implemented, "notes": as_found.notes},
            "target_vehicle_dual_layer": {
                "implemented": dual.implemented,
                "notes": dual.notes,
            },
        },
        "evidence_validation": evidence.to_dict(),
        "fan_curve_comparison": [r.to_dict() for r in fan_rows],
        "airflow_distribution_comparison": [r.to_dict() for r in airflow_rows],
        "parameter_groups": [g.to_dict() for g in groups],
        "policy": {
            "runtime_decoder_enabled": False,
            "physical_outlet_layer_enabled": False,
            "afe_calc_default_unchanged": True,
            "default_model_version": DEFAULT_AFE_MODEL_VERSION.value,
            "formal_pso_allowed": False,
            "customer_data_pipeline": True,
        },
    }


def _assess_go_no_go(
    normalized: Mapping[str, Any],
    merged_dir: Path,
    validation_report: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    unknown_fan = _has_unknown_fan_unit(normalized)
    bad_pct = _has_bad_airflow_percent_sum(normalized)
    dual_anchors = _has_dual_layer_anchors(merged_dir)
    validation_errors = int(
        validation_report.get("evidence_validation", {}).get("error_count", 0)
    )
    step_failed = any(s.get("status") == "failed" for s in steps)

    can_run_g1 = not unknown_fan
    can_run_g2 = not bad_pct and dual_anchors
    can_run_chtd = not step_failed and any(
        s.get("step_id") == "afe_v2_replay" and s.get("status") in ("passed", "warning")
        for s in steps
    )
    can_runtime_flag = (
        can_run_g1
        and can_run_g2
        and can_run_chtd
        and dual_anchors
        and validation_errors == 0
        and not unknown_fan
        and not bad_pct
        and not _has_missing_mode_code(normalized)
    )
    return {
        "can_run_g1": can_run_g1,
        "can_run_g2": can_run_g2,
        "can_run_afe_to_chtd_preview": can_run_chtd,
        "can_consider_runtime_feature_flag": can_runtime_flag,
    }


def _next_required_customer_data(
    normalized: Mapping[str, Any],
    g2_report: Optional[Mapping[str, Any]],
) -> List[str]:
    items: List[str] = []
    if _has_unknown_fan_unit(normalized):
        items.append("Confirm rear/front fan curve flow units (m3/h or l/s)")
    if _has_bad_airflow_percent_sum(normalized):
        items.append("Close airflow_distribution outlet_percent to ~100% per anchor")
    if _has_missing_mode_code(normalized):
        items.append("Provide mode_code for all airflow_distribution rows")
    for w in normalized.get("review_warnings", []):
        if w.get("code") == "missing_voltage_threshold":
            items.append("Complete intake left/right motor voltage thresholds")
    if g2_report:
        for item in g2_report.get("next_data_required", []):
            if item not in items:
                items.append(str(item))
    if not items:
        items.append("Engineering review of merged evidence before runtime consideration")
    return items


def render_pipeline_report_md(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# AFE Customer Data Pipeline Report",
        "",
        f"**Schema:** `{manifest.get('schema')}`  ",
        f"**Input:** `{manifest.get('input_path')}`  ",
        f"**Generated:** {manifest.get('generated_at')}",
        "",
        manifest.get("classification_note", ""),
        "",
        "## Go / No-Go",
        "",
    ]
    gng = manifest.get("go_no_go", {})
    for key, val in gng.items():
        lines.append(f"- **{key}:** {'Yes' if val else 'No'}")
    lines.extend(["", "## Steps", ""])
    for step in manifest.get("steps", []):
        lines.append(f"### {step['step_id']} — `{step['status']}`")
        if step.get("skipped_reason"):
            lines.append(f"- Skipped: {step['skipped_reason']}")
        for w in step.get("warnings", []):
            lines.append(f"- Warning: {w}")
        for b in step.get("blockers", []):
            lines.append(f"- Blocker: {b}")
        for p in step.get("output_paths", []):
            lines.append(f"- Output: `{p}`")
        lines.append("")
    summary = manifest.get("summary", {})
    lines.extend(
        [
            "## Summary",
            "",
            f"- Total warnings: {summary.get('total_warnings')}",
            f"- Total blockers: {summary.get('total_blockers')}",
            "",
            "## Next required customer data",
            "",
        ]
    )
    for item in summary.get("next_required_customer_data", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def run_afe_customer_data_pipeline(
    *,
    input_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    pretty: bool = False,
    refresh_demo_bundle: bool = False,
    source_calib_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute full customer-data pipeline; writes under ``output_dir`` only."""
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(input_path) if input_path is not None else DEFAULT_SAMPLE_INPUT
    if not in_path.is_file():
        raise FileNotFoundError(f"customer input not found: {in_path}")

    raw = _load_customer_input(in_path)
    validate_customer_data(raw)
    normalized = normalize_customer_data(raw)

    steps: List[Dict[str, Any]] = []
    all_warnings: List[str] = []
    all_blockers: List[str] = []

    # 1) Ingest
    merged_dir = out_dir / "merged_evidence"
    calib_src = Path(source_calib_dir) if source_calib_dir is not None else DEFAULT_CALIB_DIR
    ingest_manifest = merge_into_evidence(calib_src, normalized, merged_dir)
    ingest_paths = [
        merged_dir / "customer_ingest_manifest.json",
        merged_dir / "fan_curve_initial.json",
        merged_dir / "airflow_distribution_anchors.json",
    ]
    ingest_warnings = [w.get("code", str(w)) for w in normalized.get("review_warnings", [])]
    steps.append(
        _step_result(
            "ingest",
            status="warning" if ingest_warnings else "passed",
            output_paths=[p for p in ingest_paths if Path(p).exists()],
            warnings=ingest_warnings,
            extra={"ingest_manifest": ingest_manifest},
        )
    )
    all_warnings.extend(ingest_warnings)

    # 2) Validation
    validation_report = build_validation_report_for_evidence_dir(merged_dir)
    val_json = out_dir / "AFE_VALIDATION_REPORT.json"
    val_md = out_dir / "AFE_VALIDATION_REPORT.md"
    _write_json(val_json, validation_report, pretty=pretty)
    _write_text(val_md, render_validation_report_md(validation_report))
    val_warnings = [
        f"evidence_validation_warnings={validation_report['evidence_validation']['warning_count']}"
    ]
    if validation_report["evidence_validation"]["error_count"]:
        all_blockers.append("evidence_validation_errors")
    steps.append(
        _step_result(
            "validation",
            status="warning" if validation_report["evidence_validation"]["warning_count"] else "passed",
            output_paths=[val_json, val_md],
            warnings=val_warnings,
        )
    )

    # 3) G1/G2 smoke
    g1_json = out_dir / "AFE_V2_G1_G2_CALIBRATION_SMOKE.json"
    g1_md = out_dir / "AFE_V2_G1_G2_CALIBRATION_SMOKE.md"
    unknown_fan = _has_unknown_fan_unit(normalized)
    bad_pct = _has_bad_airflow_percent_sum(normalized)
    g1_skipped: Optional[str] = None
    g2_skipped: Optional[str] = None
    g2_report: Optional[Dict[str, Any]] = None
    smoke_warnings: List[str] = []
    smoke_blockers: List[str] = []
    smoke_status = "passed"

    if unknown_fan:
        smoke_status = "skipped"
        g1_skipped = "G1 skipped: customer fan curve flow_unit unknown — absolute fan compare blocked"
        g2_skipped = "G2 skipped while G1 blocked (unknown fan unit)"
        smoke_warnings.append("flow_unit_unknown")
        g2_report = {
            "schema": "afe_v2_g1_g2_calibration_smoke_v1",
            "classification": "skipped_g1_unknown_fan_unit",
            "skipped_reason": g1_skipped,
            "g1_status": "skipped",
            "g2_status": "skipped",
        }
    elif bad_pct:
        smoke_status = "skipped"
        g2_skipped = "G2 blocked: airflow_distribution outlet_percent does not sum to ~100%"
        smoke_blockers.append("percentages_not_sum_100")
        g2_report = {
            "schema": "afe_v2_g1_g2_calibration_smoke_v1",
            "classification": "skipped_g2_bad_percent_sum",
            "skipped_reason": g2_skipped,
            "g1_status": "skipped",
            "g2_status": "blocked",
        }
    else:
        anchors = _load_anchors_from_merged(merged_dir)
        usable = [a for a in anchors if str(a.get("mode_code", "")).strip()]
        if _has_missing_mode_code(normalized):
            smoke_warnings.append("missing_mode_code")
        try:
            with use_merged_calibration_data(merged_dir):
                fit = run_m8_g1_g2_calibration_smoke(anchors=usable or None)
            g2_report = build_g1_g2_smoke_report(fit)
            if g2_report.get("warnings"):
                smoke_status = "warning"
        except Exception as exc:  # pragma: no cover
            smoke_status = "failed"
            g2_report = {"schema": "afe_v2_g1_g2_calibration_smoke_v1", "error": str(exc)}
            smoke_blockers.append("g1_g2_smoke_failed")

    if g2_report is not None:
        _write_json(g1_json, g2_report, pretty=pretty)
        if "best_loss" in g2_report:
            _write_text(g1_md, render_g1_g2_smoke_markdown(g2_report))
        else:
            reason = g2_report.get("skipped_reason") or g2_skipped or "skipped"
            _write_text(g1_md, f"# AFE v2 G1/G2 Calibration Smoke\n\nSkipped: {reason}\n")

    steps.append(
        _step_result(
            "g1_g2_smoke",
            status=smoke_status,
            output_paths=[g1_json, g1_md] if g1_json.exists() else [],
            warnings=smoke_warnings,
            blockers=smoke_blockers,
            skipped_reason=g1_skipped or g2_skipped,
        )
    )
    all_warnings.extend(smoke_warnings)
    all_blockers.extend(smoke_blockers)

    # 4) AFE v2 replay
    replay_path = out_dir / "AFE_V2_REPLAY_OUTPUT.json"
    replay_warnings: List[str] = []
    try:
        with use_merged_calibration_data(merged_dir):
            replay_report = run_afe_v2_replay()
        write_replay_output(replay_report, replay_path, pretty=pretty)
        replay_status = "passed"
    except Exception as exc:  # pragma: no cover
        replay_report = {"error": str(exc)}
        _write_json(replay_path, replay_report, pretty=pretty)
        replay_status = "failed"
        replay_warnings.append(str(exc))
        all_blockers.append("afe_v2_replay_failed")

    steps.append(
        _step_result(
            "afe_v2_replay",
            status=replay_status,
            output_paths=[replay_path],
            warnings=replay_warnings,
        )
    )

    # 5) AFE → CHTD preview
    chtd_json = out_dir / "AFE_V2_CHTD_PREVIEW_SUMMARY.json"
    chtd_md = out_dir / "AFE_V2_CHTD_PREVIEW_SUMMARY.md"
    chtd_warnings: List[str] = []
    if replay_status == "failed":
        chtd_status = "skipped"
        chtd_skipped = "CHTD preview skipped: replay failed"
        chtd_doc = {"skipped_reason": chtd_skipped}
    elif not _has_dual_layer_anchors(merged_dir):
        chtd_status = "skipped"
        chtd_skipped = "CHTD preview skipped: insufficient dual-layer anchor evidence"
        chtd_doc = {"skipped_reason": chtd_skipped}
        chtd_warnings.append("no_dual_layer_anchors")
    else:
        try:
            replay_loaded = json.loads(replay_path.read_text(encoding="utf-8"))
            if replay_loaded.get("schema") == "afe_v2_replay_output_v1":
                chtd_doc = run_afe_v2_chtd_preview_from_input_doc(replay_loaded)
            else:
                chtd_doc = run_afe_v2_chtd_preview(
                    default_builtin_scenarios(),
                    tma_source=build_tma_source(amb_t=24.0, heating_mode=False),
                )
            chtd_status = "passed"
            chtd_skipped = None
        except Exception as exc:  # pragma: no cover
            chtd_status = "warning"
            chtd_skipped = str(exc)
            chtd_doc = {"error": str(exc), "warnings": [str(exc)]}
            chtd_warnings.append(str(exc))

    _write_json(chtd_json, chtd_doc, pretty=pretty)
    md_lines = [
        "# AFE v2 → CHTD Preview Summary",
        "",
        f"Status: {chtd_status}",
    ]
    if chtd_skipped:
        md_lines.append(f"Skipped: {chtd_skipped}")
    if isinstance(chtd_doc, dict) and chtd_doc.get("case_count") is not None:
        md_lines.append(f"Cases: {chtd_doc['case_count']}")
    _write_text(chtd_md, "\n".join(md_lines) + "\n")

    steps.append(
        _step_result(
            "afe_to_chtd_preview",
            status=chtd_status,
            output_paths=[chtd_json, chtd_md],
            warnings=chtd_warnings,
            skipped_reason=chtd_skipped if chtd_status == "skipped" else None,
        )
    )
    all_warnings.extend(chtd_warnings)

    # 6) Optional refresh
    refresh_paths: List[str] = []
    if refresh_demo_bundle:
        from hvac_sim.project_integration_status import write_project_integration_status
        from hvac_sim.validation.pmv_demo_bundle import generate_pmv_demo_bundle

        status_dir = out_dir / "integration_status_refresh"
        demo_dir = out_dir / "demo_bundle_refresh"
        write_project_integration_status(
            json_path=status_dir / "PMV_PROJECT_INTEGRATION_STATUS.json",
            md_path=status_dir / "PMV_PROJECT_INTEGRATION_STATUS.md",
        )
        generate_pmv_demo_bundle(output_dir=demo_dir, pretty=pretty)
        refresh_paths = [
            str(status_dir / "PMV_PROJECT_INTEGRATION_STATUS.json"),
            str(demo_dir / "demo_manifest.json"),
        ]
        steps.append(
            _step_result(
                "project_status_demo_refresh",
                status="passed",
                output_paths=refresh_paths,
            )
        )

    go_no_go = _assess_go_no_go(normalized, merged_dir, validation_report, steps)
    if not go_no_go["can_consider_runtime_feature_flag"]:
        all_blockers.append("afe_v2_runtime_feature_flag_no_go")

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "classification": CLASSIFICATION,
        "classification_note": (
            "Customer-data pipeline output — pending engineering review. "
            "Not vehicle calibration sign-off. Default afe_calc and runtime unchanged."
        ),
        "not_vehicle_calibration": True,
        "not_chtd_full_state_prediction": True,
        "not_afe_v2_runtime_wired": True,
        "input_path": str(in_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "merged_evidence_dir": str(merged_dir.resolve()),
        "steps": steps,
        "go_no_go": go_no_go,
        "summary": {
            "total_warnings": len(all_warnings),
            "total_blockers": len(all_blockers),
            "next_required_customer_data": _next_required_customer_data(normalized, g2_report),
            "ingest_review_warnings": list(normalized.get("review_warnings", [])),
        },
        "warnings": all_warnings,
        "blockers": all_blockers,
    }

    manifest_path = out_dir / "afe_customer_pipeline_manifest.json"
    report_md_path = out_dir / "afe_customer_pipeline_report.md"
    _write_json(manifest_path, manifest, pretty=pretty)
    _write_text(report_md_path, render_pipeline_report_md(manifest))

    return {
        "manifest_path": str(manifest_path.resolve()),
        "report_md_path": str(report_md_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "manifest": manifest,
    }
