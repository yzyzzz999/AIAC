"""Initial calibration param name resolution and mapping audit."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from hvac_sim.afe.params import AFEParams
from hvac_sim.chtd.params import CHTDParams

_SKIP_PREVIEW_LIMIT = 20

# Legacy ParamList / thermal.py names without CHTD_ prefix → canonical field.
_CHTD_LEGACY_TO_CANONICAL: Dict[str, str] = {
    "Dt": "CHTD_Dt_P",
    "QgainCo": "CHTD_QgainCo_P",
    "QlossCo": "CHTD_QlossCo_P",
    "SdlTi": "CHTD_CHTDSdlTi_P",
    "CHTDSdlTi": "CHTD_CHTDSdlTi_P",
    "AirCpAtb": "CHTD_AirCpAtb_P",
    "HeadAirVAtb": "CHTD_HeadAirVAtb_P",
    "FeetAirVAtb": "CHTD_FeetAirVAtb_P",
    "HoodMassAtb": "CHTD_HoodMassAtb_P",
    "CabinFrntMassAtb": "CHTD_CabinFrntMassAtb_P",
    "ConsoleMassAtb": "CHTD_ConsoleMassAtb_P",
}

# Simulink-style suffix variants (missing CHTD_ prefix but with _P).
_CHTD_SUFFIX_ALIASES: Dict[str, str] = {
    "Dt_P": "CHTD_Dt_P",
    "QgainCo_P": "CHTD_QgainCo_P",
    "QlossCo_P": "CHTD_QlossCo_P",
    "CHTDSdlTi_P": "CHTD_CHTDSdlTi_P",
    "AirCpAtb_P": "CHTD_AirCpAtb_P",
    "HeadAirVAtb_P": "CHTD_HeadAirVAtb_P",
    "FeetAirVAtb_P": "CHTD_FeetAirVAtb_P",
}

_AFE_KNOWN: Set[str] = {f.name for f in fields(AFEParams)}
_CHTD_KNOWN: Set[str] = {f.name for f in fields(CHTDParams)}

# Simulink workspace → Python AFEParams (param.xlsx / ParamList.xlsx name column).
_SIMULINK_AFE_PREFIX = "AFE_FH"
_SIMULINK_AFE_SUFFIX = "_P"

# Explicit review buckets for skipped rows (see INITIAL_PARAM_MAPPING_REVIEW.md).
_REVIEW_CATEGORY_BY_STATUS: Dict[str, str] = {
    "skipped_fan_curve_not_afe_field": "not_runtime_param",
    "skipped_target_geometry": "not_runtime_param",
    "skipped_target_pmv": "future_param",
    "skipped_target_ekf": "future_param",
    "skipped_lut_or_axis": "not_runtime_param",
    "skipped_lut_array_field": "not_runtime_param",
    "skipped_non_scalar_list": "not_runtime_param",
    "skipped_non_scalar_value": "unknown_needs_manual_review",
    "skipped_unknown_field": "unknown_needs_manual_review",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def optional_param_table_paths() -> List[Path]:
    """Candidate xlsx paths (optional; may not exist in workspace)."""
    root = _repo_root()
    candidates = [
        root / "param.xlsx",
        root / "ParamList.xlsx",
        root / "simulink_conversion_package" / "params" / "param.xlsx",
        root / "simulink_conversion_package" / "params" / "ParamList.xlsx",
    ]
    return [p for p in candidates if p.is_file()]


def _simulink_afe_python_name(name: str) -> Optional[str]:
    """Map ``AFE_FHEvaRessCo_P``-style Simulink vars to ``AFEParams`` field names."""
    key = name.strip()
    if not key.startswith(_SIMULINK_AFE_PREFIX) or not key.endswith(_SIMULINK_AFE_SUFFIX):
        return None
    stem = key[len(_SIMULINK_AFE_PREFIX) : -len(_SIMULINK_AFE_SUFFIX)]
    if not stem:
        return None
    if stem in _AFE_KNOWN:
        return stem
    return _case_insensitive_lookup(stem, _AFE_KNOWN)


def review_category_for_skipped(
    *,
    name: str,
    status: str,
    target_type: str,
) -> str:
    """Classify a skipped parameter row for mapping review documentation."""
    if status in _REVIEW_CATEGORY_BY_STATUS:
        return _REVIEW_CATEGORY_BY_STATUS[status]
    if target_type == "pmv":
        return "future_param"
    if target_type == "ekf":
        return "future_param"
    if target_type == "geometry":
        return "not_runtime_param"
    return "unknown_needs_manual_review"


def suggested_canonical_for_skipped(
    *,
    name: str,
    status: str,
    target_type: str,
) -> Optional[str]:
    """If a skipped row could map after alias expansion, return canonical Python name."""
    if status != "skipped_unknown_field":
        return None
    if target_type == "airflow":
        resolved, _ = resolve_python_param_name(name, target_type="airflow")
        return resolved
    if target_type == "chtd":
        resolved, _ = resolve_python_param_name(name, target_type="chtd")
        return resolved
    return None


def _case_insensitive_lookup(name: str, known: Set[str]) -> Optional[str]:
    key = name.strip().lower()
    for field_name in known:
        if field_name.lower() == key:
            return field_name
    return None


def resolve_python_param_name(
    raw_name: str,
    *,
    target_type: str,
    model_fields: Optional[Set[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve table name to a dataclass field.

    Returns ``(canonical_name, alias_label)`` where ``alias_label`` describes
  the rewrite (e.g. ``legacy:Dt``), or ``(None, None)`` if unmapped.
    """
    name = str(raw_name).strip()
    if not name:
        return None, None

    if target_type == "airflow":
        known = model_fields or _AFE_KNOWN
    elif target_type == "chtd":
        known = model_fields or _CHTD_KNOWN
    else:
        return None, None

    if target_type == "airflow":
        simulink = _simulink_afe_python_name(name)
        if simulink is not None and simulink in known:
            return simulink, f"simulink_afe:{name}"

    if target_type == "chtd" and name in _CHTD_LEGACY_TO_CANONICAL:
        canonical = _CHTD_LEGACY_TO_CANONICAL[name]
        if canonical in known:
            return canonical, f"legacy:{name}"

    if target_type == "chtd":
        if name in _CHTD_SUFFIX_ALIASES:
            canonical = _CHTD_SUFFIX_ALIASES[name]
            if canonical in known:
                return canonical, f"suffix_alias:{name}"

    if name in known:
        if target_type == "chtd" and name in _CHTD_LEGACY_TO_CANONICAL:
            canonical = _CHTD_LEGACY_TO_CANONICAL[name]
            if canonical in known:
                return canonical, f"legacy:{name}"
        return name, None

    ci = _case_insensitive_lookup(name, known)
    if ci is not None:
        if target_type == "chtd" and ci in _CHTD_LEGACY_TO_CANONICAL:
            canonical = _CHTD_LEGACY_TO_CANONICAL[ci]
            if canonical in known:
                return canonical, f"legacy:{ci}"
        return ci, f"case_insensitive:{name}"

    if target_type == "chtd":
        if not name.startswith("CHTD_") and not name.endswith("_P"):
            prefixed = f"CHTD_{name}_P"
            if prefixed in known:
                return prefixed, f"add_chtd_prefix:{name}"
            ci_pref = _case_insensitive_lookup(prefixed, known)
            if ci_pref is not None:
                return ci_pref, f"add_chtd_prefix_ci:{name}"

    return None, None


def _scalar_param_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _classify_record(
    rec: Any,
) -> Tuple[str, Optional[str], Optional[str], Optional[float]]:
    """Return (status, resolved_name, alias_label, scalar_value)."""
    val = _scalar_param_value(rec.value)
    target = rec.target_type

    if target not in ("airflow", "chtd"):
        return f"skipped_target_{target}", None, None, val

    if val is None:
        if isinstance(rec.value, list):
            return "skipped_non_scalar_list", None, None, None
        return "skipped_non_scalar_value", None, None, None

    known = _AFE_KNOWN if target == "airflow" else _CHTD_KNOWN
    resolved, alias = resolve_python_param_name(rec.name, target_type=target, model_fields=known)
    if resolved is None:
        if target == "chtd" and (rec.name.endswith("_M") or rec.name.endswith("_X")):
            return "skipped_lut_or_axis", None, None, val
        if target == "airflow" and rec.name.startswith(("F1_", "F2_", "S_", "FAN_")):
            return "skipped_fan_curve_not_afe_field", None, None, val
        return "skipped_unknown_field", None, None, val

    if resolved.endswith("_M") or resolved.endswith("_X"):
        return "skipped_lut_array_field", resolved, alias, val

    return f"mapped_{target}", resolved, alias, val


def audit_initial_params_mapping(
    path: Optional[str | Path] = None,
    *,
    skip_preview_limit: int = _SKIP_PREVIEW_LIMIT,
) -> Dict[str, Any]:
    """Audit how an initial-params table maps onto AFE/CHTD dataclass fields."""
    from .param_loader import (
        InitialParamsDocument,
        ParamTableError,
        load_initial_params,
        resolve_initial_params,
    )

    doc: InitialParamsDocument
    primary_source: str
    if path is not None:
        src = Path(path)
        doc = load_initial_params(src)
        primary_source = str(src)
    else:
        doc, primary_source = resolve_initial_params(None)

    xlsx_attempts: List[Dict[str, str]] = []
    for xlsx in optional_param_table_paths():
        try:
            xlsx_doc = load_initial_params(xlsx)
            xlsx_attempts.append(
                {
                    "path": str(xlsx),
                    "status": "loaded",
                    "parameter_count": str(len(xlsx_doc.parameters)),
                }
            )
        except ParamTableError as exc:
            xlsx_attempts.append(
                {"path": str(xlsx), "status": "error", "error": str(exc)}
            )

    mapped_afe: List[Dict[str, Any]] = []
    mapped_chtd: List[Dict[str, Any]] = []
    skipped_params: List[Dict[str, Any]] = []
    skipped_by_reason: Dict[str, int] = {}

    for rec in doc.parameters:
        status, resolved, alias, val = _classify_record(rec)
        skipped_by_reason[status] = skipped_by_reason.get(status, 0) + 1
        entry = {
            "name": rec.name,
            "target_type": rec.target_type,
            "value": rec.value,
            "source": rec.source,
            "status": status,
        }
        if status == "mapped_airflow":
            entry["resolved_name"] = resolved
            if alias:
                entry["alias"] = alias
            mapped_afe.append(entry)
        elif status == "mapped_chtd":
            entry["resolved_name"] = resolved
            if alias:
                entry["alias"] = alias
            mapped_chtd.append(entry)
        else:
            if resolved:
                entry["resolved_hint"] = resolved
            entry["review_category"] = review_category_for_skipped(
                name=rec.name,
                status=status,
                target_type=rec.target_type,
            )
            suggested = suggested_canonical_for_skipped(
                name=rec.name,
                status=status,
                target_type=rec.target_type,
            )
            if suggested:
                entry["suggested_canonical"] = suggested
                entry["review_category"] = "should_map_now"
            skipped_params.append(entry)

    skipped_names = [item["name"] for item in skipped_params]
    review_counts: Dict[str, int] = {}
    for item in skipped_params:
        cat = str(item.get("review_category", "unknown_needs_manual_review"))
        review_counts[cat] = review_counts.get(cat, 0) + 1

    return {
        "primary_source": primary_source,
        "classification": doc.classification,
        "source_files": list(doc.source_files),
        "optional_xlsx": xlsx_attempts,
        "total_params": len(doc.parameters),
        "mapped_afe_params": mapped_afe,
        "mapped_chtd_params": mapped_chtd,
        "skipped_params": skipped_params,
        "skipped_by_reason": skipped_by_reason,
        "applied_afe_count": len(mapped_afe),
        "applied_chtd_count": len(mapped_chtd),
        "skipped_count": len(skipped_params),
        "skipped_preview": skipped_names[:skip_preview_limit],
        "skipped_review_category_counts": review_counts,
    }


def apply_record_with_aliases(
    rec: Any,
    *,
    target_type: str,
    known: Set[str],
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Resolve and return (field_name, scalar_value, alias_label) for one record."""
    if rec.target_type != target_type:
        return None, None, None
    val = _scalar_param_value(rec.value)
    if val is None:
        return None, None, None
    resolved, alias = resolve_python_param_name(
        rec.name, target_type=target_type, model_fields=known
    )
    if resolved is None or resolved not in known:
        return None, None, alias
    if resolved.endswith("_M") or resolved.endswith("_X"):
        return None, None, alias
    return resolved, val, alias
