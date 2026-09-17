"""CHTD physical-capacity parameter bundle (sanity / preview only).

Builds a copied ``CHTDParams`` with engineering-scale Mass/Cp/Area/AirV values.
Does **not** modify ``CHTDParams`` defaults, ``thermal.py``, or golden fixtures.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from hvac_sim.chtd.bus_index import CHTD_X_NAMES, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.stability_audit import (
    build_state_and_input_from_afe_case,
    build_uniform_equilibrium,
)
from hvac_sim.chtd.thermal import compute_chtd_delta

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


def effective_capacitance_summary(params: CHTDParams) -> Dict[str, float]:
    """Rough C=M×Cp or C=ρV×Cp for key zones (diagnostic)."""
    from hvac_sim.chtd.helpers import cp_m_ex, cp_v_ex

    amb_like_t = 20.0
    return {
        "ConsoleTemp_J_per_K": cp_m_ex(params.CHTD_ConsoleMassAtb_P, params.CHTD_ConsoleCpAtb_P),
        "CabinTempFd_J_per_K": cp_m_ex(params.CHTD_CabinFdMassAtb_P, params.CHTD_CabinFdCpAtb_P),
        "RoofTemp_J_per_K": cp_m_ex(params.CHTD_RoofMassAtb_P, params.CHTD_RoofCpAtb_P),
        "HeadTempFd_J_per_K": cp_v_ex(
            amb_like_t, params.CHTD_HeadAirVAtb_P, params.CHTD_AirCpAtb_P
        ),
        "FeetTempFd_J_per_K": cp_v_ex(
            amb_like_t, params.CHTD_FeetAirVAtb_P, params.CHTD_AirCpAtb_P
        ),
    }


# ── Multi-step preview runner ───────────────────────────────────────────────

PREVIEW_SCHEMA = "chtd_physical_capacity_preview_v1"
PREVIEW_CLASSIFICATION = "estimated_not_calibrated_sanity_preview"
PREVIEW_DISCLAIMER = (
    "Physical-capacity sanity bundle only. Not vehicle sign-off calibration. "
    "Default CHTDParams, thermal.py, and AS_FOUND golden unchanged."
)

PREVIEW_JSON_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_PHYSICAL_CAPACITY_PREVIEW.json"
)
PREVIEW_MD_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_PHYSICAL_CAPACITY_PREVIEW.md"
)

_THRESHOLD_C = 20.0
_X_VALID_MIN = -40.0
_X_VALID_MAX = 90.0
_HISTORY_ZONES = (
    "HeadTempFd",
    "FeetTempFd",
    "ConsoleTemp",
    "CabinTempFd",
    "RoofTemp",
    "WinTempFd",
)

_STEP_NUMBERING_NOTE = (
    "Step indexing matches stability_audit / stability_experiments: "
    "step 0 = pre-integration baseline (delta=0); step 1 = first compute_chtd_delta "
    "at cold-soak x0; step 2 = second delta after x+=delta once. "
    "CHTD_DYNAMIC_STABILITY_ENGINEERING_ANALYSIS.md table labels 'Step0' for the first "
    "integration as Cursor step 1 (~HeadTempFd 2–4°C/step). Its 'Step1' (RoofTemp ~481°C) "
    "used a different audit snapshot / dominant zone — not the same as Cursor step 2 "
    "(ConsoleTemp ~181°C with placeholder C=1). CC equilibrium 62.8°C was a separate "
    "u-bus wiring issue (RawAmbT=0), not step numbering."
)

_PREVIEW_SCENARIOS: Tuple[Dict[str, Any], ...] = (
    {
        "scenario_id": "cold_foot_heating_full_state",
        "description": "AFE v2 foot heating cold soak; full 28-state integration",
        "case": {
            "case_id": "cold_foot_heating_full_state",
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
    },
    {
        "scenario_id": "cold_foot_defrost_full_state",
        "description": "AFE v2 foot+defrost heating cold soak; full 28-state integration",
        "case": {
            "case_id": "cold_foot_defrost_full_state",
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
    },
    {
        "scenario_id": "uniform_equilibrium_control",
        "description": "Uniform x/u at 25°C, zero flow — expect near-zero delta",
        "equilibrium_temp_c": 25.0,
    },
)


def _run_multistep_scenario(
    *,
    scenario_id: str,
    description: str,
    params: CHTDParams,
    x0: np.ndarray,
    u: np.ndarray,
    steps: int,
    dt_seconds: float,
    threshold_c: float,
) -> Dict[str, Any]:
    x = np.asarray(x0, dtype=float).copy()
    max_abs_delta_by_step: List[float] = []
    max_zone_by_step: List[str] = []
    zone_history: Dict[str, List[float]] = {
        z: [float(x[X_INDEX[z]])] for z in _HISTORY_ZONES
    }
    first_threshold_step: Optional[int] = None
    first_nonfinite_step: Optional[int] = None
    first_out_of_range_step: Optional[int] = None
    steps_completed = 0
    top_zones_at_step2: List[Dict[str, Any]] = []

    for step in range(1, steps + 1):
        delta = compute_chtd_delta(x, u, params, mode=AS_FOUND)
        if not np.all(np.isfinite(delta)):
            first_nonfinite_step = step
            break
        abs_delta = np.abs(delta)
        max_abs = float(np.max(abs_delta))
        max_idx = int(np.argmax(abs_delta))
        max_zone = CHTD_X_NAMES[max_idx]
        max_abs_delta_by_step.append(max_abs)
        max_zone_by_step.append(max_zone)
        steps_completed = step

        if max_abs > threshold_c and first_threshold_step is None:
            first_threshold_step = step

        if step == 2:
            ranked = sorted(
                ((CHTD_X_NAMES[i], float(abs_delta[i])) for i in range(len(abs_delta))),
                key=lambda t: t[1],
                reverse=True,
            )
            top_zones_at_step2 = [
                {"zone": name, "abs_delta_c": val} for name, val in ranked[:10]
            ]

        x = x + delta
        if not np.all(np.isfinite(x)):
            first_nonfinite_step = step
            break
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if (x_min < _X_VALID_MIN or x_max > _X_VALID_MAX) and first_out_of_range_step is None:
            first_out_of_range_step = step
        for z in _HISTORY_ZONES:
            zone_history[z].append(float(x[X_INDEX[z]]))

    all_finite = first_nonfinite_step is None and steps_completed == steps
    final_x_min = float(np.min(x)) if np.all(np.isfinite(x)) else None
    final_x_max = float(np.max(x)) if np.all(np.isfinite(x)) else None
    in_valid_range = (
        all_finite
        and final_x_min is not None
        and final_x_max is not None
        and final_x_min >= _X_VALID_MIN
        and final_x_max <= _X_VALID_MAX
    )

    if not all_finite:
        conclusion = "unstable_nonfinite"
    elif first_threshold_step is not None and first_threshold_step <= 2:
        conclusion = "unstable_early_threshold"
    elif not in_valid_range:
        conclusion = "unstable_out_of_range"
    elif first_threshold_step is None:
        conclusion = "stable_600s"
    else:
        conclusion = "marginal_late_threshold"

    return {
        "scenario_id": scenario_id,
        "description": description,
        "steps_requested": int(steps),
        "steps_completed": int(steps_completed),
        "dt_seconds": float(dt_seconds),
        "threshold_delta_c_per_step": float(threshold_c),
        "first_threshold_step": first_threshold_step,
        "first_nonfinite_step": first_nonfinite_step,
        "first_out_of_range_step": first_out_of_range_step,
        "all_finite_600s": bool(all_finite),
        "x_in_valid_range_-40_90": bool(in_valid_range),
        "max_abs_delta_by_step": max_abs_delta_by_step,
        "max_zone_by_step": max_zone_by_step,
        "step1_max_abs_delta_c": max_abs_delta_by_step[0] if max_abs_delta_by_step else None,
        "step2_max_abs_delta_c": max_abs_delta_by_step[1] if len(max_abs_delta_by_step) > 1 else None,
        "step2_max_zone": max_zone_by_step[1] if len(max_zone_by_step) > 1 else None,
        "top_zones_at_step2": top_zones_at_step2,
        "final_x_min_c": final_x_min,
        "final_x_max_c": final_x_max,
        "zone_history": zone_history,
        "conclusion": conclusion,
    }


def run_physical_capacity_preview(
    *,
    steps: int = 600,
    dt_seconds: float = 1.0,
    threshold_c: float = _THRESHOLD_C,
    vehicle_class: str = "m8_estimated",
    include_default_baseline: bool = True,
) -> Dict[str, Any]:
    """Run 600-step preview with physical-capacity bundle (+ optional default baseline)."""
    physical_params, provenance = build_physical_capacity_params(vehicle_class=vehicle_class)
    physical_params.CHTD_Dt_P = float(dt_seconds)

    default_params = CHTDParams()
    default_params.CHTD_Dt_P = float(dt_seconds)

    scenario_runs: List[Dict[str, Any]] = []
    for spec in _PREVIEW_SCENARIOS:
        sid = str(spec["scenario_id"])
        if "case" in spec:
            x0, u, _ = build_state_and_input_from_afe_case(spec["case"])
        else:
            x0, u = build_uniform_equilibrium(temp_c=float(spec.get("equilibrium_temp_c", 25.0)))

        physical_run = _run_multistep_scenario(
            scenario_id=sid,
            description=str(spec.get("description", "")),
            params=physical_params,
            x0=x0,
            u=u,
            steps=steps,
            dt_seconds=dt_seconds,
            threshold_c=threshold_c,
        )
        entry: Dict[str, Any] = {
            "scenario_id": sid,
            "physical_capacity": physical_run,
        }
        if include_default_baseline and "case" in spec:
            entry["default_params_baseline"] = _run_multistep_scenario(
                scenario_id=sid,
                description="default CHTDParams baseline for comparison",
                params=default_params,
                x0=x0,
                u=u,
                steps=min(steps, 10),
                dt_seconds=dt_seconds,
                threshold_c=threshold_c,
            )
        scenario_runs.append(entry)

    eq_check = _run_multistep_scenario(
        scenario_id="equilibrium_true_check",
        description="True equilibrium @25C with physical params",
        params=physical_params,
        x0=build_uniform_equilibrium(temp_c=25.0)[0],
        u=build_uniform_equilibrium(temp_c=25.0)[1],
        steps=5,
        dt_seconds=dt_seconds,
        threshold_c=threshold_c,
    )

    suspect_params: List[str] = []
    if provenance.get("_meta", {}).get("lut_maps_still_placeholder_like", 0) > 0:
        suspect_params.append("CHTD_*RadCo_M / CHTD_*ConvCo_M LUT maps (placeholder=1.0)")
    if any(r["physical_capacity"]["conclusion"].startswith("unstable") for r in scenario_runs):
        suspect_params.extend(
            [
                "Coupling coefficient LUTs not scaled with capacitance uplift",
                "ConsoleFeet convection path (see stability_experiments qbus_gain)",
            ]
        )

    return {
        "schema": PREVIEW_SCHEMA,
        "classification": PREVIEW_CLASSIFICATION,
        "disclaimer": PREVIEW_DISCLAIMER,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vehicle_class": vehicle_class,
        "steps": int(steps),
        "dt_seconds": float(dt_seconds),
        "step_numbering_note": _STEP_NUMBERING_NOTE,
        "effective_capacitance_J_per_K": effective_capacitance_summary(physical_params),
        "provenance": provenance,
        "equilibrium_check": eq_check,
        "scenarios": scenario_runs,
        "suspect_params_if_unstable": suspect_params,
    }


def build_physical_capacity_markdown(doc: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# CHTD Physical Capacity Preview",
        "",
        f"**{doc.get('disclaimer', PREVIEW_DISCLAIMER)}**",
        "",
        f"Generated: `{doc.get('generated_at_utc', '')}` | vehicle_class: `{doc.get('vehicle_class')}`",
        "",
        "## Step numbering (Cursor vs CC)",
        "",
        doc.get("step_numbering_note", ""),
        "",
        "## Effective capacitance (key zones)",
        "",
    ]
    for k, v in doc.get("effective_capacitance_J_per_K", {}).items():
        lines.append(f"- {k}: **{v:.1f}** J/K")
    lines.extend(["", "## Scenario results", ""])

    for sc in doc.get("scenarios", []):
        sid = sc["scenario_id"]
        pr = sc["physical_capacity"]
        lines.extend(
            [
                f"### {sid}",
                "",
                f"- Conclusion: **{pr['conclusion']}**",
                f"- Steps completed: {pr['steps_completed']}/{pr['steps_requested']}",
                f"- Step-1 max |delta|: {pr.get('step1_max_abs_delta_c')} °C",
                f"- Step-2 max |delta|: {pr.get('step2_max_abs_delta_c')} °C ({pr.get('step2_max_zone')})",
                f"- First threshold step: {pr.get('first_threshold_step')}",
                f"- All finite: {pr.get('all_finite_600s')} | x in [-40,90]: {pr.get('x_in_valid_range_-40_90')}",
            ]
        )
        if pr.get("top_zones_at_step2"):
            lines.append("- Step-2 top zones:")
            for z in pr["top_zones_at_step2"][:5]:
                lines.append(f"  - {z['zone']}: {z['abs_delta_c']:.3g} °C/step")
        bl = sc.get("default_params_baseline")
        if bl:
            lines.append(
                f"- Default baseline step-2 (10-step cap): **{bl.get('step2_max_abs_delta_c')}** °C "
                f"({bl.get('step2_max_zone')})"
            )
        lines.append("")

    if doc.get("suspect_params_if_unstable"):
        lines.extend(["## Suspect parameters (if still unstable)", ""])
        for s in doc["suspect_params_if_unstable"]:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)


def write_physical_capacity_preview_outputs(
    *,
    bundle: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Path]:
    doc = bundle if bundle is not None else run_physical_capacity_preview()
    json_out = json_path if json_path is not None else PREVIEW_JSON_OUT
    md_out = md_path if md_path is not None else PREVIEW_MD_OUT
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(build_physical_capacity_markdown(doc), encoding="utf-8")
    return {"json": json_out, "md": md_out}
