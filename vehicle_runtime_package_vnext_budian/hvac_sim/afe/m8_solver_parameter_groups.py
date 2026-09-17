"""Map calibration_parameter_groups_v1 (AFE G1/G2/G3) to M8 solver params.

Experimental only — not wired to ``afe_calc`` or ``runtime_pipeline``.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from hvac_sim.afe.m8_dual_layer_solver import (
    M8DualLayerSolverParams,
    default_solver_params_for_mode,
)
from hvac_sim.calibration.parameter_group_loader import (
    CalibrationParameterRecord,
    load_calibration_parameter_groups,
)

_AFE_GROUP_IDS = frozenset({"G1", "G2", "G3"})
_PROVENANCE_SOURCE = "calibration_parameter_groups_v1"
_LEADING_NUMERIC = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)
_OSA_FLAP_RESS_REF = 300.0
_REC_FLAP_RESS_REF = 300.0

# Engineering placeholder defaults when summary is missing / non-numeric.
_PLACEHOLDER_DEFAULTS: Dict[str, float] = {
    "front_fan_V_ref": 12.0,
    "rear_fan_V_ref": 12.0,
    "front_fan_affinity_exponent": 2.0,
    "EvaRessCo": 300.0,
    "R_Def_rel": 1.0,
    "R_Sv_rel": 1.0,
    "R_Fdf_rel": 1.0,
    "R_Row2v_rel": 1.0,
    "R_Row2f_rel": 1.0,
    "OsaFlapRessCo": _OSA_FLAP_RESS_REF,
    "RecFlapRessCo": _REC_FLAP_RESS_REF,
    "FdHexUpRessCo_scale": 1.0,
    "SdHexRessCo_scale": 1.0,
    "R_console_inzone_rel": 1.0,
    "rec_fraction_thresh_osa": 4.24,
    "rec_fraction_thresh_50": 2.33,
    "rec_fraction_thresh_25": 2.90,
    "rec_fraction_thresh_rec": 0.42,
    "alpha_face": 0.5,
    "alpha_rear_face": 0.5,
    "alpha_rear_foot": 0.5,
}

_NON_NUMERIC_PARAMETERS = frozenset({"mode_decode_table"})


def parse_numeric_from_value_summary(text: str) -> Optional[float]:
    """Extract a leading numeric token from ``current_value_summary``."""
    stripped = str(text).strip()
    if not stripped or stripped.lower().startswith("none"):
        return None
    match = _LEADING_NUMERIC.match(stripped)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _resolve_parameter_value(
    name: str,
    record: Optional[CalibrationParameterRecord],
    overrides: Mapping[str, float],
    provenance: Dict[str, Any],
) -> Optional[float]:
    if name in overrides:
        value = float(overrides[name])
        provenance["applied"][name] = {"value": value, "from": "override"}
        return value

    if record is not None:
        parsed = parse_numeric_from_value_summary(record.current_value_summary)
        if parsed is not None:
            provenance["applied"][name] = {"value": parsed, "from": "summary"}
            return parsed

    if name in _PLACEHOLDER_DEFAULTS:
        value = float(_PLACEHOLDER_DEFAULTS[name])
        provenance["applied"][name] = {"value": value, "from": "default_placeholder"}
        provenance["missing_defaults"].append(name)
        return value

    provenance["unresolved"].append(name)
    return None


def _collect_afe_group_records(
    path: Optional[Path],
) -> Tuple[Dict[str, CalibrationParameterRecord], Path]:
    doc = load_calibration_parameter_groups(path)
    records: Dict[str, CalibrationParameterRecord] = {}
    for rec in doc.parameters:
        if rec.group_id not in _AFE_GROUP_IDS:
            continue
        if rec.module != "AFE":
            continue
        records[rec.parameter_name] = rec
    return records, doc.source_path


def _apply_face_alpha(
    params: M8DualLayerSolverParams,
    alpha: float,
) -> None:
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha >= 1.0 - 1e-9:
        params.face_layer_policy = "upper"
        params.alpha_face_upper = 1.0
    elif alpha <= 1e-9:
        params.face_layer_policy = "lower"
        params.alpha_face_upper = 0.0
    else:
        params.face_layer_policy = "mixed"
        params.alpha_face_upper = alpha


def _apply_resolved_values(
    params: M8DualLayerSolverParams,
    base: M8DualLayerSolverParams,
    values: Mapping[str, Optional[float]],
    provenance: Dict[str, Any],
) -> None:
    placeholders = dict(base.pending_placeholders)

    if values.get("front_fan_V_ref") is not None:
        params.reference_voltage_v = float(values["front_fan_V_ref"])

    for key in (
        "rear_fan_V_ref",
        "front_fan_k2_nom",
        "front_fan_k0_nom",
        "rear_fan_k2_nom",
        "rear_fan_k0_nom",
        "front_fan_affinity_exponent",
        "EvaRessCo",
        "FdHexUpRessCo_scale",
        "SdHexRessCo_scale",
        "R_console_inzone_rel",
        "rec_fraction_thresh_osa",
        "rec_fraction_thresh_50",
        "rec_fraction_thresh_25",
        "rec_fraction_thresh_rec",
        "alpha_rear_foot",
    ):
        if values.get(key) is not None:
            placeholders[key] = float(values[key])

    if values.get("R_Def_rel") is not None:
        params.defrost_resistance = base.defrost_resistance * float(values["R_Def_rel"])
    if values.get("R_Sv_rel") is not None:
        params.face_resistance = base.face_resistance * float(values["R_Sv_rel"])
    if values.get("R_Fdf_rel") is not None:
        params.foot_resistance = base.foot_resistance * float(values["R_Fdf_rel"])
    if values.get("R_Row2v_rel") is not None:
        params.rear_face_resistance = base.rear_face_resistance * float(
            values["R_Row2v_rel"]
        )
    if values.get("R_Row2f_rel") is not None:
        params.rear_foot_resistance = base.rear_foot_resistance * float(
            values["R_Row2f_rel"]
        )

    if values.get("OsaFlapRessCo") is not None:
        params.upper_path_resistance = base.upper_path_resistance * (
            float(values["OsaFlapRessCo"]) / _OSA_FLAP_RESS_REF
        )
    if values.get("RecFlapRessCo") is not None:
        params.lower_path_resistance = base.lower_path_resistance * (
            float(values["RecFlapRessCo"]) / _REC_FLAP_RESS_REF
        )

    if values.get("alpha_face") is not None:
        _apply_face_alpha(params, float(values["alpha_face"]))
    if values.get("alpha_rear_face") is not None:
        params.alpha_rear_upper = max(0.0, min(1.0, float(values["alpha_rear_face"])))

    params.pending_placeholders = placeholders
    provenance["solver_fields_touched"] = sorted(
        k for k, v in values.items() if v is not None and k not in _NON_NUMERIC_PARAMETERS
    )


def build_m8_solver_params_from_parameter_groups(
    *,
    base: Optional[M8DualLayerSolverParams] = None,
    mode_code: Union[str, int] = "V",
    heating_mode: bool = False,
    overrides: Optional[Mapping[str, float]] = None,
    path: Optional[Path] = None,
) -> Tuple[M8DualLayerSolverParams, Dict[str, Any]]:
    """Build experimental M8 solver params from AFE G1/G2/G3 calibration groups.

    Returns ``(params, provenance)``. Does not mutate ``base`` or default solver
    templates when this function is not called.
    """
    base_params = (
        deepcopy(base)
        if base is not None
        else default_solver_params_for_mode(mode_code, heating_mode=heating_mode)
    )
    override_map: Dict[str, float] = dict(overrides or {})
    records, source_path = _collect_afe_group_records(path)

    provenance: Dict[str, Any] = {
        "source": _PROVENANCE_SOURCE,
        "source_path": str(source_path),
        "groups_read": sorted(_AFE_GROUP_IDS),
        "mode_code": str(mode_code),
        "heating_mode": bool(heating_mode),
        "applied": {},
        "missing_defaults": [],
        "unresolved": [],
        "skipped_non_numeric": [],
    }

    values: Dict[str, Optional[float]] = {}
    for name, record in records.items():
        if name in _NON_NUMERIC_PARAMETERS:
            provenance["skipped_non_numeric"].append(name)
            continue
        values[name] = _resolve_parameter_value(
            name, record, override_map, provenance
        )

    params = replace(base_params)
    params.pending_placeholders = dict(base_params.pending_placeholders)
    _apply_resolved_values(params, base_params, values, provenance)

    return params, provenance
