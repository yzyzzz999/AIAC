"""Read-only loader for calibration_parameter_groups_v1.json (PSO runner prep)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_DEFAULT_REL = (
    Path(__file__).resolve().parents[3]
    / "simulink_conversion_package"
    / "python_targets"
    / "calibration_parameter_groups_v1.json"
)


@dataclass(frozen=True)
class CalibrationParameterRecord:
    group_id: str
    group_name_cn: str
    parameter_name: str
    module: str
    current_source: str
    current_value_summary: str
    is_placeholder_like: str
    can_fix_by_engineering: str
    pso_candidate: str
    pso_phase: str
    first_phase: str
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    transform: str
    share_rule: str
    data_required: str
    dependency: str
    risk_if_wrong: str
    notes_cn: str
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def is_first_phase(self) -> bool:
        return str(self.first_phase).strip().lower() == "yes"

    @property
    def is_pso_candidate(self) -> bool:
        return str(self.pso_candidate).strip().lower() == "yes"


@dataclass(frozen=True)
class CalibrationParameterGroupsDocument:
    schema: str
    date: str
    description: str
    summary: Dict[str, Any]
    group_metadata: Dict[str, Any]
    parameters: Tuple[CalibrationParameterRecord, ...]
    source_path: Path

    def by_group(self, group_id: str) -> Tuple[CalibrationParameterRecord, ...]:
        gid = str(group_id).strip()
        return tuple(p for p in self.parameters if p.group_id == gid)

    def by_phase(self, phase: str) -> Tuple[CalibrationParameterRecord, ...]:
        ph = str(phase).strip().lower()
        return tuple(p for p in self.parameters if p.pso_phase.lower() == ph)


def _parse_bound(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _record_from_dict(row: Mapping[str, Any]) -> CalibrationParameterRecord:
    return CalibrationParameterRecord(
        group_id=str(row.get("group_id", "")),
        group_name_cn=str(row.get("group_name_cn", "")),
        parameter_name=str(row.get("parameter_name", "")),
        module=str(row.get("module", "")),
        current_source=str(row.get("current_source", "")),
        current_value_summary=str(row.get("current_value_summary", "")),
        is_placeholder_like=str(row.get("is_placeholder_like", "")),
        can_fix_by_engineering=str(row.get("can_fix_by_engineering", "")),
        pso_candidate=str(row.get("pso_candidate", "")),
        pso_phase=str(row.get("pso_phase", "")),
        first_phase=str(row.get("first_phase", "")),
        lower_bound=_parse_bound(row.get("lower_bound")),
        upper_bound=_parse_bound(row.get("upper_bound")),
        transform=str(row.get("transform", "")),
        share_rule=str(row.get("share_rule", "")),
        data_required=str(row.get("data_required", "")),
        dependency=str(row.get("dependency", "")),
        risk_if_wrong=str(row.get("risk_if_wrong", "")),
        notes_cn=str(row.get("notes_cn", "")),
        raw=dict(row),
    )


def load_calibration_parameter_groups(
    path: Optional[Path] = None,
) -> CalibrationParameterGroupsDocument:
    """Load structured parameter groups JSON."""
    src = path if path is not None else _DEFAULT_REL
    src = Path(src)
    doc = json.loads(src.read_text(encoding="utf-8"))
    params = tuple(_record_from_dict(r) for r in doc.get("parameters", []))
    return CalibrationParameterGroupsDocument(
        schema=str(doc.get("schema", "")),
        date=str(doc.get("date", "")),
        description=str(doc.get("description", "")),
        summary=dict(doc.get("summary", {})),
        group_metadata=dict(doc.get("group_metadata", {})),
        parameters=params,
        source_path=src,
    )


def select_parameters_by_group(
    group_id: str,
    *,
    path: Optional[Path] = None,
) -> List[CalibrationParameterRecord]:
    return list(load_calibration_parameter_groups(path).by_group(group_id))


def select_parameters_by_phase(
    phase: str,
    *,
    path: Optional[Path] = None,
) -> List[CalibrationParameterRecord]:
    return list(load_calibration_parameter_groups(path).by_phase(phase))


def select_first_phase_parameters(
    *,
    path: Optional[Path] = None,
) -> List[CalibrationParameterRecord]:
    doc = load_calibration_parameter_groups(path)
    return [p for p in doc.parameters if p.is_first_phase]


def validate_parameter_group_bounds(
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate numeric bounds and PSO metadata consistency."""
    doc = load_calibration_parameter_groups(path)
    issues: List[str] = []
    ok_count = 0
    for p in doc.parameters:
        if not p.is_pso_candidate:
            continue
        if p.lower_bound is None or p.upper_bound is None:
            issues.append(f"{p.parameter_name}: missing_bounds")
            continue
        if p.lower_bound >= p.upper_bound:
            issues.append(
                f"{p.parameter_name}: lower_bound>=upper_bound "
                f"({p.lower_bound}>={p.upper_bound})"
            )
            continue
        ok_count += 1
    return {
        "source_path": str(doc.source_path),
        "parameter_count": len(doc.parameters),
        "pso_candidate_count": sum(1 for p in doc.parameters if p.is_pso_candidate),
        "bounds_ok_count": ok_count,
        "issues": issues,
        "valid": len(issues) == 0,
    }
