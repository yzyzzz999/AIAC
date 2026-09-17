"""Physical baseline v2 — capacity estimates + param.xlsx coupling LUTs (no MAE fitting)."""

from __future__ import annotations

import ast
import copy
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from hvac_sim.chtd.params import CHTDParams, as_lut
from hvac_sim.chtd.physical_params import (
    _DEFAULT_PARAM_TABLE,
    _sync_deprecated_aliases,
    build_physical_capacity_params,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TABLE = _DEFAULT_PARAM_TABLE

# Cabin / inter-zone coupling LUT prefixes from param.xlsx (engineering spec, not PSO).
_CABIN_LUT_KEYWORDS: Tuple[str, ...] = (
    "CabinFd",
    "CabinFp",
    "CabinSd",
    "CabinSp",
    "CabinTd",
    "CabinTp",
    "CabinFrnt",
    "FdCabin",
    "FpCabin",
    "TdCabin",
    "TpCabin",
    "FeetFdCabin",
    "FeetFpCabin",
    "FeetTdCabin",
    "FeetTpCabin",
    "FeetSdCabin",
    "FeetSpCabin",
    "SdCabin",
    "SpCabin",
)


def _lut_matches_cabin_coupling(name: str) -> bool:
    if not name.endswith("_M"):
        return False
    return any(k in name for k in _CABIN_LUT_KEYWORDS)


def load_param_excel_lut_values(
    *,
    table_path: Optional[Path] = None,
    name_filter: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load LUT arrays from parameter_master_table.csv (param.xlsx source)."""
    path = table_path if table_path is not None else _DEFAULT_TABLE
    filt = set(name_filter) if name_filter is not None else None
    out: Dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_status") != "excel_param":
                continue
            if row.get("category") != "lut_map":
                continue
            name = row["parameter_name"]
            if filt is not None and name not in filt:
                continue
            raw = row["current_value"]
            try:
                vals = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                continue
            arr = np.asarray(vals, dtype=float)
            if arr.shape != (7,):
                continue
            out[name] = arr
    return out


_THIRD_ROW_LUT_KEYWORDS: Tuple[str, ...] = (
    "CabinTd", "CabinTp", "TdCabin", "TpCabin", "FeetTdCabin", "FeetTpCabin",
)


def _lut_scope_match(name: str, scope: str) -> bool:
    if scope == "none":
        return False
    if scope == "third_row":
        return any(k in name for k in _THIRD_ROW_LUT_KEYWORDS)
    return _lut_matches_cabin_coupling(name)


def apply_excel_cabin_coupling_luts(
    params: CHTDParams,
    *,
    table_path: Optional[Path] = None,
    lut_scale: float = 1.0,
    lut_names: Optional[Iterable[str]] = None,
    lut_scope: str = "third_row",
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Apply param.xlsx cabin/inter-zone LUT coefficients onto physical_capacity params."""
    p = copy.deepcopy(params)
    all_luts = load_param_excel_lut_values(table_path=table_path)
    applied: Dict[str, list] = {}
    for name, arr in all_luts.items():
        if lut_names is not None:
            if name not in lut_names:
                continue
        elif not _lut_scope_match(name, lut_scope):
            continue
        if not hasattr(p, name):
            continue
        scaled = arr * float(lut_scale) if lut_scale != 1.0 else arr
        setattr(p, name, np.asarray(scaled, dtype=float))
        applied[name] = [float(v) for v in scaled]
    _sync_deprecated_aliases(p)
    meta = {
        "lut_scale": float(lut_scale),
        "lut_scope": lut_scope,
        "applied_lut_count": len(applied),
        "applied_luts": applied,
        "basis": "parameter_master_table.csv ← param.xlsx engineering coupling coefficients",
    }
    return p, meta


def scale_cabin_masses(params: CHTDParams, scale: float) -> CHTDParams:
    p = copy.deepcopy(params)
    for name in (
        "CHTD_CabinFdMassAtb_P", "CHTD_CabinFpMassAtb_P",
        "CHTD_CabinSdMassAtb_P", "CHTD_CabinSpMassAtb_P",
        "CHTD_CabinTdMassAtb_P", "CHTD_CabinTpMassAtb_P",
    ):
        setattr(p, name, float(getattr(p, name)) * scale)
    _sync_deprecated_aliases(p)
    return p


def scale_ambient_leakage(params: CHTDParams, scale: float) -> CHTDParams:
    p = copy.deepcopy(params)
    p.CHTD_QlossCo_P = float(p.CHTD_QlossCo_P) * scale
    return p


def scale_third_row_cabin_masses(params: CHTDParams, scale: float) -> CHTDParams:
    p = copy.deepcopy(params)
    for name in ("CHTD_CabinTdMassAtb_P", "CHTD_CabinTpMassAtb_P"):
        setattr(p, name, float(getattr(p, name)) * scale)
    _sync_deprecated_aliases(p)
    return p


def build_physical_baseline_v2_params(
    *,
    cabin_mass_scale: float = 1.0,
    third_row_mass_scale: float = 1.0,
    cabin_qbus_lut_scale: float = 1.0,
    ambient_leakage_scale: float = 1.0,
    lut_scope: str = "third_row",
    table_path: Optional[Path] = None,
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Physical baseline v2: M8 capacity + scoped param.xlsx cabin coupling LUTs."""
    base, prov = build_physical_capacity_params()
    p = base
    lut_meta: Dict[str, Any] = {"applied_lut_count": 0, "lut_scope": lut_scope}
    if lut_scope != "none":
        p, lut_meta = apply_excel_cabin_coupling_luts(
            base, table_path=table_path, lut_scale=cabin_qbus_lut_scale, lut_scope=lut_scope,
        )
    if cabin_mass_scale != 1.0:
        p = scale_cabin_masses(p, cabin_mass_scale)
    if third_row_mass_scale != 1.0:
        p = scale_third_row_cabin_masses(p, third_row_mass_scale)
    if ambient_leakage_scale != 1.0:
        p = scale_ambient_leakage(p, ambient_leakage_scale)
    meta = {
        "builder": "build_physical_baseline_v2_params",
        "cabin_mass_scale": float(cabin_mass_scale),
        "third_row_mass_scale": float(third_row_mass_scale),
        "cabin_qbus_lut_scale": float(cabin_qbus_lut_scale),
        "ambient_leakage_scale": float(ambient_leakage_scale),
        "lut_scope": lut_scope,
        "capacity_provenance": prov,
        "coupling_lut_meta": lut_meta,
    }
    return p, meta
