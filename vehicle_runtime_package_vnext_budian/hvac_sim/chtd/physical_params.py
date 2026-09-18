"""CHTD physical-capacity parameter bundle (sanity / preview only).

Builds a copied ``CHTDParams`` with engineering-scale Mass/Cp/Area/AirV values.
Does **not** modify ``CHTDParams`` defaults, ``thermal.py``, or golden fixtures.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Tuple

from hvac_sim.chtd.params import CHTDParams

PROVENANCE_ESTIMATED_NOT_CALIBRATED = "estimated_not_calibrated"
PROVENANCE_LITERATURE = "literature_estimate"
PROVENANCE_GEOMETRY = "geometry_estimate"
PROVENANCE_EXCEL = "copied_from_param_excel"
PROVENANCE_PLACEHOLDER = "placeholder_retained"
PROVENANCE_NEEDS_CONFIRM = "needs_engineering_confirmation"

_CAPACITY_KEYWORDS = ("MassAtb", "CpAtb", "AreaAtb", "AirV")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_default_param_table() -> Path:
    candidates = [
        _PACKAGE_ROOT / "config" / "parameter_master_table.csv",
        _REPO_ROOT / "simulink_conversion_package" / "python_targets" / "parameter_master_table.csv",
        Path("/home/data/simulink_conversion_package/python_targets/parameter_master_table.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


_DEFAULT_PARAM_TABLE = _resolve_default_param_table()

# M8-class SUV engineering estimates (CHTD_DYNAMIC_STABILITY_ENGINEERING_ANALYSIS.md §S1).
# Not vehicle sign-off calibration — sanity magnitudes only.
_M8_ESTIMATED_OVERRIDES: Dict[str, Tuple[float, str, str]] = {
    # cabin air lumps (0.35–0.5 m³ effective × ρ≈1.2 → kg, cp air)
    "CHTD_CabinFdMassAtb_P": (0.50, PROVENANCE_GEOMETRY, "≈0.42 m³ cabin air lump, driver front"),
    "CHTD_CabinFdCpAtb_P": (1006.0, PROVENANCE_LITERATURE, "dry air cp @ near-room"),
    "CHTD_CabinFpMassAtb_P": (0.50, PROVENANCE_GEOMETRY, "passenger front cabin air lump"),
    "CHTD_CabinFpCpAtb_P": (1006.0, PROVENANCE_LITERATURE, "dry air cp"),
    "CHTD_CabinSdMassAtb_P": (0.45, PROVENANCE_GEOMETRY, "2nd-row cabin air lump"),
    "CHTD_CabinSdCpAtb_P": (1006.0, PROVENANCE_LITERATURE, "dry air cp"),
    "CHTD_CabinSpMassAtb_P": (0.45, PROVENANCE_GEOMETRY, "2nd-row center cabin air lump"),
    "CHTD_CabinSpCpAtb_P": (1006.0, PROVENANCE_LITERATURE, "dry air cp"),
    "CHTD_CabinTdMassAtb_P": (0.35, PROVENANCE_GEOMETRY, "3rd-row driver-side air lump"),
    "CHTD_CabinTdCpAtb_P": (1006.0, PROVENANCE_LITERATURE, "dry air cp"),
    "CHTD_CabinTpMassAtb_P": (0.35, PROVENANCE_GEOMETRY, "3rd-row passenger-side air lump"),
    "CHTD_CabinTpCpAtb_P": (1006.0, PROVENANCE_LITERATURE, "dry air cp"),
    # structural / trim lumps
    "CHTD_RoofMassAtb_P": (5.0, PROVENANCE_GEOMETRY, "lightweight roof panel + headliner lump"),
    "CHTD_RoofCpAtb_P": (500.0, PROVENANCE_LITERATURE, "aluminum/foam composite order-of-magnitude"),
    "CHTD_ConsoleMassAtb_P": (2.0, PROVENANCE_GEOMETRY, "IP center stack + console trim"),
    "CHTD_ConsoleCpAtb_P": (800.0, PROVENANCE_LITERATURE, "plastic/electronics blend"),
    "CHTD_CabinFrntMassAtb_P": (3.0, PROVENANCE_GEOMETRY, "dash + firewall air film lump"),
    "CHTD_CabinFrntCpAtb_P": (800.0, PROVENANCE_LITERATURE, "mixed trim/sheet metal"),
    "CHTD_HoodMassAtb_P": (8.0, PROVENANCE_GEOMETRY, "hood outer panel lump"),
    "CHTD_HoodCpAtb_P": (500.0, PROVENANCE_LITERATURE, "sheet steel order-of-magnitude"),
    # glazing (CHTD_DYNAMIC_STABILITY_ENGINEERING_ANALYSIS.md)
    "CHTD_WinFdMassAtb_P": (6.0, PROVENANCE_GEOMETRY, "windshield glass lump"),
    "CHTD_WinFdCpAtb_P": (750.0, PROVENANCE_LITERATURE, "glass cp"),
    "CHTD_WinFpMassAtb_P": (6.0, PROVENANCE_GEOMETRY, "windshield passenger-side symmetry"),
    "CHTD_WinFpCpAtb_P": (750.0, PROVENANCE_LITERATURE, "glass cp"),
    "CHTD_WinSdMassAtb_P": (3.0, PROVENANCE_GEOMETRY, "2nd-row side glass"),
    "CHTD_WinSdCpAtb_P": (750.0, PROVENANCE_LITERATURE, "glass cp"),
    "CHTD_WinSpMassAtb_P": (3.0, PROVENANCE_GEOMETRY, "2nd-row side glass symmetry"),
    "CHTD_WinSpCpAtb_P": (750.0, PROVENANCE_LITERATURE, "glass cp"),
    "CHTD_WinTdMassAtb_P": (2.5, PROVENANCE_GEOMETRY, "3rd-row quarter glass"),
    "CHTD_WinTdCpAtb_P": (750.0, PROVENANCE_LITERATURE, "glass cp"),
    "CHTD_WinTpMassAtb_P": (2.5, PROVENANCE_GEOMETRY, "3rd-row quarter glass symmetry"),
    "CHTD_WinTpCpAtb_P": (750.0, PROVENANCE_LITERATURE, "glass cp"),
}

# Deprecated alias ← canonical sync (params.__post_init__ only handles Dt/QgainCo/...).
_DEP_CANONICAL_MASS_CP_AREA: Tuple[Tuple[str, str], ...] = (
    ("HoodMassAtb", "CHTD_HoodMassAtb_P"),
    ("HoodCpAtb", "CHTD_HoodCpAtb_P"),
    ("HoodAreaAtb", "CHTD_HoodAreaAtb_P"),
    ("CabinFrntMassAtb", "CHTD_CabinFrntMassAtb_P"),
    ("CabinFrntCpAtb", "CHTD_CabinFrntCpAtb_P"),
    ("CabinFrntAreaAtb", "CHTD_CabinFrntAreaAtb_P"),
    ("ConsoleMassAtb", "CHTD_ConsoleMassAtb_P"),
    ("ConsoleCpAtb", "CHTD_ConsoleCpAtb_P"),
    ("ConsoleAreaAtb", "CHTD_ConsoleAreaAtb_P"),
    ("CabinFdMassAtb", "CHTD_CabinFdMassAtb_P"),
    ("CabinFdCpAtb", "CHTD_CabinFdCpAtb_P"),
    ("CabinFdAreaAtb", "CHTD_CabinFdAreaAtb_P"),
    ("CabinFpMassAtb", "CHTD_CabinFpMassAtb_P"),
    ("CabinFpCpAtb", "CHTD_CabinFpCpAtb_P"),
    ("CabinFpAreaAtb", "CHTD_CabinFpAreaAtb_P"),
    ("CabinSdMassAtb", "CHTD_CabinSdMassAtb_P"),
    ("CabinSdCpAtb", "CHTD_CabinSdCpAtb_P"),
    ("CabinSdAreaAtb", "CHTD_CabinSdAreaAtb_P"),
    ("CabinSpMassAtb", "CHTD_CabinSpMassAtb_P"),
    ("CabinSpCpAtb", "CHTD_CabinSpCpAtb_P"),
    ("CabinSpAreaAtb", "CHTD_CabinSpAreaAtb_P"),
    ("CabinTdMassAtb", "CHTD_CabinTdMassAtb_P"),
    ("CabinTdCpAtb", "CHTD_CabinTdCpAtb_P"),
    ("CabinTdAreaAtb", "CHTD_CabinTdAreaAtb_P"),
    ("CabinTpMassAtb", "CHTD_CabinTpMassAtb_P"),
    ("CabinTpCpAtb", "CHTD_CabinTpCpAtb_P"),
    ("CabinTpAreaAtb", "CHTD_CabinTpAreaAtb_P"),
    ("WinFdMassAtb", "CHTD_WinFdMassAtb_P"),
    ("WinFdCpAtb", "CHTD_WinFdCpAtb_P"),
    ("WinFdAreaAtb", "CHTD_WinFdAreaAtb_P"),
    ("WinFpMassAtb", "CHTD_WinFpMassAtb_P"),
    ("WinFpCpAtb", "CHTD_WinFpCpAtb_P"),
    ("WinFpAreaAtb", "CHTD_WinFpAreaAtb_P"),
    ("WinSdMassAtb", "CHTD_WinSdMassAtb_P"),
    ("WinSdCpAtb", "CHTD_WinSdCpAtb_P"),
    ("WinSdAreaAtb", "CHTD_WinSdAreaAtb_P"),
    ("WinSpMassAtb", "CHTD_WinSpMassAtb_P"),
    ("WinSpCpAtb", "CHTD_WinSpCpAtb_P"),
    ("WinSpAreaAtb", "CHTD_WinSpAreaAtb_P"),
    ("WinTdMassAtb", "CHTD_WinTdMassAtb_P"),
    ("WinTdCpAtb", "CHTD_WinTdCpAtb_P"),
    ("WinTdAreaAtb", "CHTD_WinTdAreaAtb_P"),
    ("WinTpMassAtb", "CHTD_WinTpMassAtb_P"),
    ("WinTpCpAtb", "CHTD_WinTpCpAtb_P"),
    ("WinTpAreaAtb", "CHTD_WinTpAreaAtb_P"),
    ("RoofMassAtb", "CHTD_RoofMassAtb_P"),
    ("RoofCpAtb", "CHTD_RoofCpAtb_P"),
    ("RoofAreaAtb", "CHTD_RoofAreaAtb_P"),
    ("HeadAirVAtb", "CHTD_HeadAirVAtb_P"),
    ("FeetAirVAtb", "CHTD_FeetAirVAtb_P"),
    ("HeadAreaAtb", "CHTD_HeadAreaAtb_P"),
    ("FeetAreaAtb", "CHTD_FeetAreaAtb_P"),
)


def load_param_excel_capacity_values(
    *,
    table_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Load capacity-related scalar params from parameter_master_table.csv."""
    path = table_path if table_path is not None else _DEFAULT_PARAM_TABLE
    out: Dict[str, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["parameter_name"]
            if row.get("source_status") != "excel_param":
                continue
            if not name.startswith("CHTD_") or not name.endswith("_P"):
                continue
            if not any(key in name for key in _CAPACITY_KEYWORDS):
                continue
            out[name] = float(row["current_value"])
    return out


def _sync_deprecated_aliases(params: CHTDParams) -> None:
    for dep, canonical in _DEP_CANONICAL_MASS_CP_AREA:
        if hasattr(params, canonical):
            setattr(params, dep, float(getattr(params, canonical)))


def _record_provenance(
    registry: MutableMapping[str, Dict[str, Any]],
    name: str,
    value: float,
    *,
    category: str,
    note: str,
    vehicle_class: str,
) -> None:
    registry[name] = {
        "value": float(value),
        "category": category,
        "note": note,
        "vehicle_class": vehicle_class,
        "calibration_status": PROVENANCE_ESTIMATED_NOT_CALIBRATED,
    }


def build_physical_capacity_params(
    base: Optional[CHTDParams] = None,
    *,
    vehicle_class: str = "m8_estimated",
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Return copied params + provenance registry for physical-capacity sanity runs."""
    if vehicle_class != "m8_estimated":
        raise ValueError(f"unsupported vehicle_class: {vehicle_class!r}")

    params = copy.deepcopy(base if base is not None else CHTDParams())
    provenance: Dict[str, Dict[str, Any]] = {}
    defaults = CHTDParams()

    excel_values = load_param_excel_capacity_values()
    for name, value in excel_values.items():
        if not hasattr(params, name):
            continue
        setattr(params, name, float(value))
        _record_provenance(
            provenance,
            name,
            float(value),
            category=PROVENANCE_EXCEL,
            note="parameter_master_table.csv ← param.xlsx",
            vehicle_class=vehicle_class,
        )

    for name, (value, category, note) in _M8_ESTIMATED_OVERRIDES.items():
        if not hasattr(params, name):
            continue
        setattr(params, name, float(value))
        _record_provenance(
            provenance,
            name,
            float(value),
            category=category,
            note=note,
            vehicle_class=vehicle_class,
        )

    # Head/feet zone deprecated mass fields — not used by cp_v_ex path; mark retained.
    for dep_name in (
        "HeadFdMassAtb",
        "HeadFpMassAtb",
        "HeadSdMassAtb",
        "HeadTdMassAtb",
        "HeadTpMassAtb",
        "FeetFdMassAtb",
        "FeetFpMassAtb",
        "FeetSdMassAtb",
        "FeetSpMassAtb",
        "FeetTdMassAtb",
        "FeetTpMassAtb",
    ):
        val = float(getattr(params, dep_name))
        if dep_name not in provenance:
            _record_provenance(
                provenance,
                dep_name,
                val,
                category=PROVENANCE_PLACEHOLDER,
                note="deprecated alias; Head/Feet TRACE uses HeadAirV/FeetAirV not mass",
                vehicle_class=vehicle_class,
            )

    # Document untouched placeholder LUTs (coupling strength still default=1.0).
    lut_placeholder_count = sum(
        1
        for f in fields(params)
        if f.name.endswith("_M") and float(getattr(params, f.name)[0]) == 1.0
    )
    provenance["_meta"] = {
        "vehicle_class": vehicle_class,
        "calibration_status": PROVENANCE_ESTIMATED_NOT_CALIBRATED,
        "lut_maps_still_placeholder_like": lut_placeholder_count,
        "lut_note": (
            "RadCo/ConvCo LUT maps remain placeholder=1.0 in this bundle; "
            "capacitance-only uplift improves step-1/2 but may not stabilize 600-step "
            "runs until coupling coefficients are calibrated."
        ),
        "needs_engineering_confirmation": [
            "All Mass/Cp/Area/AirV magnitudes",
            "Whether param.xlsx Cp units are literal J/(kg·K) or model-normalized",
            "Coupling LUT calibration (not included in this bundle)",
        ],
    }

    _sync_deprecated_aliases(params)

    # Baseline default snapshot for diff reporting.
    provenance["_default_snapshot"] = {
        name: float(getattr(defaults, name))
        for name in provenance
        if name.startswith("CHTD_") and hasattr(defaults, name)
    }

    return params, provenance
