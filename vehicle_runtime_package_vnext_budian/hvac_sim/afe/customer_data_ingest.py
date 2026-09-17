"""AFE v2 customer data ingestion scaffold (pending engineering review).

Imports customer fan curves, airflow distribution, and voltage thresholds into
copied calibration evidence JSON under an explicit output directory.
Does not modify packaged ``calibration_data/`` unless that path is passed as output.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

CUSTOMER_SCHEMA = "afe_customer_data_ingest_v1"
INGEST_RESULT_SCHEMA = "afe_customer_data_ingest_result_v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIB_DIR = Path(__file__).resolve().parent / "calibration_data"
DEFAULT_SAMPLE_INPUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "AFE_CUSTOMER_DATA_INGEST_SAMPLE_INPUT.json"
)
DEFAULT_SAMPLE_OUTPUT_DIR = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "AFE_CUSTOMER_DATA_INGEST_SAMPLE_OUTPUT"
)
DEFAULT_GUIDE_MD = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "AFE_CUSTOMER_DATA_INGEST_GUIDE.md"
)

EVIDENCE_JSON_FILES = (
    "actuator_voltage_targets.json",
    "blend_door_voltage_endpoints.json",
    "fan_curve_initial.json",
    "airflow_distribution_anchors.json",
    "physical_outlet_map.json",
    "m8_dual_layer_structure_confirmed.json",
)

BLOWER_TO_FAN_ID = {"front": "front_hvac_box", "rear": "rear_booster"}
FLOW_UNIT_ALIASES = {
    "m3h": "m3/h",
    "m3/h": "m3/h",
    "m³/h": "m3/h",
    "lps": "l/s",
    "l/s": "l/s",
    "unknown": "unknown",
}

PENDING_REVIEW_STATUS = "pending_engineering_review"
PENDING_CALIB_STATUS = "pending_review"
CUSTOMER_PROVENANCE = "customer_provided"


class CustomerDataIngestError(ValueError):
    """Invalid customer ingest payload or merge failure."""


def _path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CustomerDataIngestError(f"{path}: must be an object")
    return value


def _require_list(value: Any, *, path: str, min_len: int = 0) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CustomerDataIngestError(f"{path}: must be a list")
    if len(value) < min_len:
        raise CustomerDataIngestError(
            f"{path}: must contain at least {min_len} element(s), got {len(value)}"
        )
    return value


def _slug_anchor_id(mode_code: str, blower_setting: str, recirc: str, temp_door: str) -> str:
    parts = [
        re.sub(r"[^a-zA-Z0-9]+", "_", str(mode_code or "unknown")).strip("_").lower(),
        re.sub(r"[^a-zA-Z0-9]+", "_", str(blower_setting or "blower")).strip("_").lower(),
        re.sub(r"[^a-zA-Z0-9]+", "_", str(recirc or "recirc")).strip("_").lower(),
        re.sub(r"[^a-zA-Z0-9]+", "_", str(temp_door or "temp")).strip("_").lower(),
    ]
    return "__".join(p for p in parts if p)


def _customer_provenance(
    record: Mapping[str, Any],
    *,
    vehicle: str,
    received_date: str,
    default_source: str = "",
) -> Dict[str, Any]:
    return {
        "provenance": CUSTOMER_PROVENANCE,
        "reviewer_status": PENDING_REVIEW_STATUS,
        "calibration_status": PENDING_CALIB_STATUS,
        "vehicle": vehicle,
        "received_date": received_date,
        "source_file": str(record.get("source_file") or default_source or ""),
        "confidence": "medium",
        "vehicle_applicability": f"customer_{vehicle}_pending_review",
        "transcription_method": "customer_data_ingest",
    }


def _section_list(root: Mapping[str, Any], primary: str, *aliases: str) -> List[Any]:
    for key in (primary, *aliases):
        if key in root and root[key] is not None:
            return _require_list(root[key], path=f"root.{key}")
    return _require_list([], path=f"root.{primary}")


def validate_customer_data(data: Mapping[str, Any]) -> None:
    """Validate customer ingest document structure."""
    root = _require_mapping(data, path="root")
    if str(root.get("schema", "")) != CUSTOMER_SCHEMA:
        raise CustomerDataIngestError(
            f"root.schema must be {CUSTOMER_SCHEMA!r}, got {root.get('schema')!r}"
        )
    if not str(root.get("vehicle", "")).strip():
        raise CustomerDataIngestError("root.vehicle: required non-empty string")
    if not str(root.get("received_date", "")).strip():
        raise CustomerDataIngestError("root.received_date: required (YYYY-MM-DD)")

    for idx, fc in enumerate(_section_list(root, "fan_curve", "fan_curves")):
        fp = _path("root.fan_curve", f"[{idx}]")
        rec = _require_mapping(fc, path=fp)
        if rec.get("blower") not in BLOWER_TO_FAN_ID:
            raise CustomerDataIngestError(f"{fp}.blower: must be front or rear")
        if len(rec.get("pressure_pa") or rec.get("flow_value") or []) == 0:
            if not (rec.get("pressure_pa") and rec.get("flow_value")):
                raise CustomerDataIngestError(f"{fp}: pressure_pa and flow_value required")
        if len(rec["pressure_pa"]) != len(rec["flow_value"]):
            raise CustomerDataIngestError(f"{fp}: pressure_pa and flow_value length mismatch")

    for idx, row in enumerate(
        _require_list(root.get("airflow_distribution"), path="root.airflow_distribution")
    ):
        ap = _path("root.airflow_distribution", f"[{idx}]")
        rec = _require_mapping(row, path=ap)
        if not str(rec.get("mode_code", "")).strip():
            raise CustomerDataIngestError(f"{ap}.mode_code: required")
        outlet = _require_mapping(rec.get("outlet_percent"), path=_path(ap, "outlet_percent"))
        if not outlet:
            raise CustomerDataIngestError(f"{ap}.outlet_percent: required non-empty object")

    for idx, row in enumerate(
        _require_list(root.get("intake_voltage_thresholds"), path="root.intake_voltage_thresholds")
    ):
        ip = _path("root.intake_voltage_thresholds", f"[{idx}]")
        rec = _require_mapping(row, path=ip)
        if rec.get("left_motor_voltage_v") is None and rec.get("right_motor_voltage_v") is None:
            raise CustomerDataIngestError(
                f"{ip}: at least one of left_motor_voltage_v / right_motor_voltage_v required"
            )

    for idx, row in enumerate(
        _require_list(root.get("mode_actuator_voltage"), path="root.mode_actuator_voltage")
    ):
        mp = _path("root.mode_actuator_voltage", f"[{idx}]")
        rec = _require_mapping(row, path=mp)
        if not str(rec.get("mode_code", "")).strip():
            raise CustomerDataIngestError(f"{mp}.mode_code: required")


def _normalize_flow_unit(raw: str) -> str:
    key = str(raw or "unknown").strip().lower()
    return FLOW_UNIT_ALIASES.get(key, key)


def _flow_values_to_m3h(values: Sequence[float], unit: str) -> Tuple[List[float], str, Optional[str]]:
    if unit == "m3/h":
        return [float(v) for v in values], "m3/h", None
    if unit in ("l/s", "lps"):
        return [float(v) * 3.6 for v in values], "m3/h", "converted_lps_to_m3h"
    return [float(v) for v in values], "unknown", None


def _collect_review_warnings(normalized: Mapping[str, Any]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    seen_anchors: set[str] = set()

    for idx, fc in enumerate(normalized.get("fan_curve", [])):
        if fc.get("flow_unit") == "unknown":
            warnings.append(
                {
                    "code": "flow_unit_unknown",
                    "path": f"fan_curve[{idx}]",
                    "message": "flow_unit unknown — absolute_compare_allowed=false",
                }
            )

    for idx, row in enumerate(normalized.get("airflow_distribution", [])):
        pct = row.get("outlet_percent") or {}
        total = float(sum(float(v) for v in pct.values()))
        row["measured_sum_percent"] = total
        if abs(total - 100.0) > 1.0:
            warnings.append(
                {
                    "code": "percentages_not_sum_100",
                    "path": f"airflow_distribution[{idx}]",
                    "message": f"outlet_percent sum={total:.2f} (expected ~100)",
                }
            )
        aid = row.get("anchor_id") or _slug_anchor_id(
            row.get("mode_code", ""),
            row.get("blower_setting", ""),
            row.get("recirc_state", ""),
            row.get("temp_door_state", ""),
        )
        if aid in seen_anchors:
            warnings.append(
                {
                    "code": "duplicate_anchor",
                    "path": f"airflow_distribution[{idx}]",
                    "message": f"duplicate anchor_id {aid!r}",
                }
            )
        seen_anchors.add(aid)
        if not str(row.get("mode_code", "")).strip():
            warnings.append(
                {
                    "code": "missing_mode_code",
                    "path": f"airflow_distribution[{idx}]",
                    "message": "mode_code missing after normalize",
                }
            )

    for idx, row in enumerate(normalized.get("intake_voltage_thresholds", [])):
        if row.get("left_motor_voltage_v") is None or row.get("right_motor_voltage_v") is None:
            warnings.append(
                {
                    "code": "missing_voltage_threshold",
                    "path": f"intake_voltage_thresholds[{idx}]",
                    "message": "left or right intake motor voltage missing",
                }
            )
        if not str(row.get("mode_code") or "").strip():
            warnings.append(
                {
                    "code": "missing_mode_code",
                    "path": f"intake_voltage_thresholds[{idx}]",
                    "message": "mode_code optional but absent",
                }
            )

    curve_keys: set[str] = set()
    for idx, fc in enumerate(normalized.get("fan_curve", [])):
        key = f"{fc.get('fan_id')}@{fc.get('operating_voltage_v')}"
        if key in curve_keys:
            warnings.append(
                {
                    "code": "duplicate_anchor",
                    "path": f"fan_curve[{idx}]",
                    "message": f"duplicate fan curve key {key!r}",
                }
            )
        curve_keys.add(key)

    return warnings


def normalize_customer_data(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize units/fields; append review_warnings and normalization_log."""
    validate_customer_data(data)
    out = copy.deepcopy(dict(data))
    vehicle = str(out["vehicle"])
    received = str(out["received_date"])
    fallbacks: List[str] = []

    fan_rows = _section_list(out, "fan_curve", "fan_curves")
    fan_out: List[Dict[str, Any]] = []
    for fc in fan_rows:
        unit_raw = _normalize_flow_unit(str(fc.get("flow_unit", "unknown")))
        flows_m3h, flow_unit, conv = _flow_values_to_m3h(fc["flow_value"], unit_raw)
        if conv:
            fallbacks.append(f"fan_curve: {conv}")
        blower = str(fc["blower"])
        entry = {
            **fc,
            "fan_id": BLOWER_TO_FAN_ID[blower],
            "operating_voltage_v": fc.get("voltage_or_pwm"),
            "curve_points": [
                {"back_pressure_pa": float(p), "flow": float(f)}
                for p, f in zip(fc["pressure_pa"], flows_m3h)
            ],
            "flow_unit": flow_unit,
            "pressure_unit": "Pa",
            "absolute_compare_allowed": flow_unit == "m3/h",
            **_customer_provenance(fc, vehicle=vehicle, received_date=received),
        }
        fan_out.append(entry)
    out["fan_curve"] = fan_out
    out.pop("fan_curves", None)

    dist_out: List[Dict[str, Any]] = []
    for row in out.get("airflow_distribution", []):
        anchor_id = _slug_anchor_id(
            row.get("mode_code", ""),
            row.get("blower_setting", ""),
            row.get("recirc_state", ""),
            row.get("temp_door_state", ""),
        )
        dist_out.append(
            {
                **row,
                "anchor_id": anchor_id,
                "unit": "percent_of_total_flow",
                "outlet_percent": {str(k): float(v) for k, v in row["outlet_percent"].items()},
                **_customer_provenance(
                    row,
                    vehicle=vehicle,
                    received_date=received,
                    default_source=str(row.get("source_file") or row.get("source_page") or ""),
                ),
            }
        )
    out["airflow_distribution"] = dist_out

    intake_out: List[Dict[str, Any]] = []
    for row in out.get("intake_voltage_thresholds", []):
        intake_out.append(
            {
                **row,
                **_customer_provenance(row, vehicle=vehicle, received_date=received),
            }
        )
    out["intake_voltage_thresholds"] = intake_out

    mode_out: List[Dict[str, Any]] = []
    for row in out.get("mode_actuator_voltage", []):
        mode_out.append(
            {
                **row,
                **_customer_provenance(row, vehicle=vehicle, received_date=received),
            }
        )
    out["mode_actuator_voltage"] = mode_out

    out["normalization_log"] = fallbacks
    out["review_warnings"] = _collect_review_warnings(out)
    out["normalized_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def _copy_evidence_tree(source_dir: Path, output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in EVIDENCE_JSON_FILES:
        src = source_dir / name
        if src.is_file():
            dst = output_dir / name
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def _merge_fan_curves(
    evidence: MutableMapping[str, Any],
    curves: Sequence[Mapping[str, Any]],
) -> int:
    added = 0
    entries = list(evidence.get("entries", []))
    for curve in curves:
        fan_id = curve["fan_id"]
        voltage = curve.get("operating_voltage_v")
        new_entry = {
            "fan_id": fan_id,
            "curve_points": curve["curve_points"],
            "flow_unit": curve["flow_unit"],
            "flow_unit_note": curve.get("note", ""),
            "pressure_unit": "Pa",
            "absolute_compare_allowed": curve.get("absolute_compare_allowed", False),
            "operating_voltage_v": voltage,
            "operating_point_note": curve.get("note") or f"customer curve @ {voltage}V",
            "source": curve.get("source_file", ""),
            "page": curve.get("note", ""),
            "condition": f"customer ingest blower={curve.get('blower')}",
            **{k: curve[k] for k in (
                "provenance", "reviewer_status", "calibration_status", "vehicle",
                "received_date", "source_file", "confidence", "vehicle_applicability",
                "transcription_method",
            ) if k in curve},
        }
        replaced = False
        for i, existing in enumerate(entries):
            if existing.get("fan_id") == fan_id and existing.get("operating_voltage_v") == voltage:
                entries[i] = {**existing, **new_entry, "customer_merge": "replaced"}
                replaced = True
                break
        if not replaced:
            entries.append(new_entry)
            added += 1
    evidence["entries"] = entries
    evidence["customer_ingest_note"] = (
        "Contains customer_provided curves — pending_engineering_review, not signed_off"
    )
    evidence["reviewer_status"] = PENDING_REVIEW_STATUS
    evidence["calibration_status"] = PENDING_CALIB_STATUS
    return added


def _merge_airflow_anchors(
    evidence: MutableMapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> int:
    added = 0
    anchors = list(evidence.get("anchors", []))
    existing_ids = {a.get("anchor_id") for a in anchors}
    for row in rows:
        anchor = {
            "anchor_id": row["anchor_id"],
            "mode_code": row.get("mode_code"),
            "blower_setting": row.get("blower_setting"),
            "recirc_state": row.get("recirc_state"),
            "temp_door_state": row.get("temp_door_state"),
            "unit": row.get("unit", "percent_of_total_flow"),
            "outlet_percent": dict(row["outlet_percent"]),
            "measured_sum_percent": row.get("measured_sum_percent"),
            "source": row.get("source_file", ""),
            "page": row.get("source_page", ""),
            "source_page": row.get("source_page", ""),
            "condition": (
                f"mode={row.get('mode_code')} recirc={row.get('recirc_state')} "
                f"temp_door={row.get('temp_door_state')} blower={row.get('blower_setting')}"
            ),
            **{k: row[k] for k in (
                "provenance", "reviewer_status", "calibration_status", "vehicle",
                "received_date", "source_file", "confidence", "vehicle_applicability",
                "transcription_method",
            ) if k in row},
        }
        if anchor["anchor_id"] in existing_ids:
            for i, a in enumerate(anchors):
                if a.get("anchor_id") == anchor["anchor_id"]:
                    anchors[i] = {**a, **anchor, "customer_merge": "replaced"}
                    break
        else:
            anchors.append(anchor)
            existing_ids.add(anchor["anchor_id"])
            added += 1
    evidence["anchors"] = anchors
    evidence["reviewer_status"] = PENDING_REVIEW_STATUS
    evidence["calibration_status"] = PENDING_CALIB_STATUS
    return added


def _merge_mode_actuator_voltage(
    evidence: MutableMapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> int:
    updated = 0
    modes = list(evidence.get("modes", []))
    by_code = {m.get("mode_code"): m for m in modes}
    for row in rows:
        code = str(row["mode_code"])
        actuators = dict((by_code.get(code) or {}).get("actuators") or {})
        mapping = {
            "defrost": row.get("defrost_motor_v"),
            "vent": row.get("face_motor_v"),
            "foot": row.get("foot_motor_v"),
        }
        if row.get("rear_mode_motor_v") is not None:
            actuators["rear_vent"] = {
                **actuators.get("rear_vent", {}),
                "target_v": float(row["rear_mode_motor_v"]),
                "confidence": "medium",
            }
        for act_key, voltage in mapping.items():
            if voltage is None:
                continue
            actuators[act_key] = {
                **actuators.get(act_key, {}),
                "target_v": float(voltage),
                "confidence": "medium",
            }
        mode_entry = {
            **(by_code.get(code) or {"mode_code": code}),
            "actuators": actuators,
            "customer_ingest": True,
            "reviewer_status": PENDING_REVIEW_STATUS,
            "calibration_status": PENDING_CALIB_STATUS,
            "provenance": CUSTOMER_PROVENANCE,
            "received_date": row.get("received_date"),
            "source_file": row.get("source_file"),
        }
        if code in by_code:
            for i, m in enumerate(modes):
                if m.get("mode_code") == code:
                    modes[i] = {**m, **mode_entry}
                    break
        else:
            modes.append(mode_entry)
        by_code[code] = mode_entry
        updated += 1
    evidence["modes"] = modes
    evidence["reviewer_status"] = PENDING_REVIEW_STATUS
    evidence["calibration_status"] = PENDING_CALIB_STATUS
    return updated


def _write_intake_thresholds(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    vehicle: str,
    received_date: str,
) -> Path:
    doc = {
        "schema": "afe_intake_voltage_thresholds_customer_v1",
        "disclaimer": "Customer-provided intake/recirc thresholds — pending_engineering_review",
        "calibration_status": PENDING_CALIB_STATUS,
        "reviewer_status": PENDING_REVIEW_STATUS,
        "vehicle": vehicle,
        "received_date": received_date,
        "provenance": CUSTOMER_PROVENANCE,
        "entries": list(rows),
    }
    path = output_dir / "intake_voltage_thresholds_customer.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def merge_into_evidence(
    existing_json_dir: Union[str, Path],
    data: Mapping[str, Any],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Copy evidence tree to output_dir and merge normalized customer data."""
    src_dir = Path(existing_json_dir)
    out_dir = Path(output_dir)
    if not src_dir.is_dir():
        raise CustomerDataIngestError(f"existing_json_dir not found: {src_dir}")

    normalized = normalize_customer_data(data) if data.get("review_warnings") is None else copy.deepcopy(dict(data))
    copied = _copy_evidence_tree(src_dir, out_dir)

    merge_stats: Dict[str, Any] = {}
    written: List[str] = list(copied)

    fan_path = out_dir / "fan_curve_initial.json"
    if fan_path.is_file() and normalized.get("fan_curve"):
        fan_doc = json.loads(fan_path.read_text(encoding="utf-8"))
        merge_stats["fan_curves_added"] = _merge_fan_curves(fan_doc, normalized["fan_curve"])
        fan_path.write_text(json.dumps(fan_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    anchor_path = out_dir / "airflow_distribution_anchors.json"
    if anchor_path.is_file() and normalized.get("airflow_distribution"):
        anchor_doc = json.loads(anchor_path.read_text(encoding="utf-8"))
        merge_stats["airflow_anchors_added"] = _merge_airflow_anchors(
            anchor_doc, normalized["airflow_distribution"]
        )
        anchor_path.write_text(json.dumps(anchor_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    act_path = out_dir / "actuator_voltage_targets.json"
    if act_path.is_file() and normalized.get("mode_actuator_voltage"):
        act_doc = json.loads(act_path.read_text(encoding="utf-8"))
        merge_stats["mode_actuator_updated"] = _merge_mode_actuator_voltage(
            act_doc, normalized["mode_actuator_voltage"]
        )
        act_path.write_text(json.dumps(act_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    intake_path: Optional[Path] = None
    if normalized.get("intake_voltage_thresholds"):
        intake_path = _write_intake_thresholds(
            out_dir,
            normalized["intake_voltage_thresholds"],
            vehicle=str(normalized["vehicle"]),
            received_date=str(normalized["received_date"]),
        )
        written.append(intake_path.name)

    manifest = {
        "schema": INGEST_RESULT_SCHEMA,
        "classification": "customer_data_ingest_pending_review",
        "not_vehicle_signed_off": True,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dir": str(src_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "vehicle": normalized.get("vehicle"),
        "received_date": normalized.get("received_date"),
        "provenance": CUSTOMER_PROVENANCE,
        "reviewer_status": PENDING_REVIEW_STATUS,
        "calibration_status": PENDING_CALIB_STATUS,
        "files_written": written,
        "merge_stats": merge_stats,
        "review_warnings": normalized.get("review_warnings", []),
        "normalization_log": normalized.get("normalization_log", []),
    }
    manifest_path = out_dir / "customer_ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_sample_customer_input(*, received_date: Optional[str] = None) -> Dict[str, Any]:
    """Synthetic customer bundle for ingest smoke (not signed off)."""
    rd = received_date or date.today().isoformat()
    return {
        "schema": CUSTOMER_SCHEMA,
        "vehicle": "m8",
        "received_date": rd,
        "source_bundle": "AFE_CUSTOMER_DATA_INGEST_SAMPLE_INPUT.json",
        "fan_curve": [
            {
                "blower": "front",
                "voltage_or_pwm": 12.0,
                "pressure_pa": [0, 50, 100, 150, 200],
                "flow_value": [150.0, 145.0, 138.0, 130.0, 120.0],
                "flow_unit": "lps",
                "source_file": "customer/front_fan_12v.csv",
                "note": "Monday delivery sample — linear table",
            },
            {
                "blower": "rear",
                "voltage_or_pwm": 12.0,
                "pressure_pa": [0, 100, 200],
                "flow_value": [55.0, 48.0, 40.0],
                "flow_unit": "unknown",
                "source_file": "customer/rear_fan_12v.csv",
                "note": "unit pending confirmation",
            },
        ],
        "airflow_distribution": [
            {
                "mode_code": "F",
                "blower_setting": "12v_both",
                "recirc_state": "recirc",
                "temp_door_state": "full_cold",
                "outlet_percent": {
                    "frnt_foot_fl": 18.0,
                    "frnt_foot_fr": 17.0,
                    "row2_foot_console": 35.0,
                    "row2_foot_side": 30.0,
                },
                "source_page": "page_7_table_3",
                "source_file": "customer/airflow_linear分配.xlsx",
            },
            {
                "mode_code": "V",
                "blower_setting": "12v_both",
                "recirc_state": "fresh",
                "temp_door_state": "full_cold",
                "outlet_percent": {
                    "frnt_face_fl": 20.0,
                    "frnt_face_fr": 20.0,
                    "row2_face_console": 30.0,
                },
                "source_page": "page_8_table_1",
                "source_file": "customer/airflow_linear分配.xlsx",
            },
        ],
        "intake_voltage_thresholds": [
            {
                "left_motor_voltage_v": 0.42,
                "right_motor_voltage_v": 4.56,
                "recirc_fraction_cmd": 0.0,
                "door_combination": "recirc_stall",
                "mode_code": "V",
                "source_file": "customer/intake_thresholds.pdf",
            }
        ],
        "mode_actuator_voltage": [
            {
                "mode_code": "F",
                "defrost_motor_v": 3.5,
                "face_motor_v": 1.66,
                "foot_motor_v": 1.5,
                "rear_mode_motor_v": 2.5,
                "source_file": "customer/mode_voltages.png",
            }
        ],
    }


def build_ingest_guide_markdown() -> str:
    return "\n".join(
        [
            "# AFE Customer Data Ingest Guide",
            "",
            f"Customer input schema: `{CUSTOMER_SCHEMA}`",
            "",
            "## Supported sections",
            "",
            "1. **fan_curve** — `blower` front/rear, `voltage_or_pwm`, `pressure_pa[]`, `flow_value[]`, `flow_unit` (m3h/lps/unknown)",
            "2. **airflow_distribution** — mode_code, blower/recirc/temp_door, `outlet_percent` dict",
            "3. **intake_voltage_thresholds** — left/right motor V, recirc_fraction_cmd, door_combination",
            "4. **mode_actuator_voltage** — per mode_code defrost/face/foot/rear motor V",
            "",
            "## CLI",
            "",
            "```bash",
            "python python_impl/examples/ingest_afe_customer_data.py",
            "python python_impl/examples/ingest_afe_customer_data.py --input customer.json --output-dir ./out",
            "```",
            "",
            "## Provenance (always applied)",
            "",
            f"- `provenance`: `{CUSTOMER_PROVENANCE}`",
            f"- `reviewer_status`: `{PENDING_REVIEW_STATUS}`",
            f"- `calibration_status`: `{PENDING_CALIB_STATUS}`",
            "- `vehicle`, `received_date`, `source_file` per record",
            "",
            "## Review warnings",
            "",
            "- `flow_unit_unknown`",
            "- `percentages_not_sum_100`",
            "- `missing_mode_code`",
            "- `missing_voltage_threshold`",
            "- `duplicate_anchor`",
            "",
            "## Safety",
            "",
            "- Packaged `calibration_data/` is never modified unless explicitly passed as `--output-dir`",
            "- Customer data is **not** marked signed_off",
            "",
        ]
    )


def write_sample_ingest_artifacts(
    *,
    sample_input: Optional[Mapping[str, Any]] = None,
    input_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    existing_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Write sample input JSON, merged output dir, and guide markdown."""
    doc = sample_input if sample_input is not None else build_sample_customer_input()
    in_path = Path(input_path) if input_path is not None else DEFAULT_SAMPLE_INPUT
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_SAMPLE_OUTPUT_DIR
    src_dir = Path(existing_dir) if existing_dir is not None else DEFAULT_CALIB_DIR

    in_path.parent.mkdir(parents=True, exist_ok=True)
    in_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DEFAULT_GUIDE_MD.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_GUIDE_MD.write_text(build_ingest_guide_markdown(), encoding="utf-8")

    manifest = merge_into_evidence(src_dir, doc, out_dir)
    return {
        "sample_input": str(in_path.resolve()),
        "guide_md": str(DEFAULT_GUIDE_MD.resolve()),
        "output_dir": str(out_dir.resolve()),
        "manifest": manifest,
    }
