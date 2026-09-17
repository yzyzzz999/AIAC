"""AFE v2 + CHTD offline one-step preview (experimental, not runtime).

Pipeline: ``run_afe_v2_diagnostic`` → ``v2_to_chtd_adapter`` →
``compute_chtd_delta`` → ``x_next = x0 + delta``.

Does **not** modify ``runtime_adapter``, ``pipeline``, or ``thermal.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.afe.v2_diagnostic import AFEV2DiagnosticInputs, run_afe_v2_diagnostic
from hvac_sim.afe.v2_replay import case_to_diagnostic_inputs
from hvac_sim.afe.v2_to_chtd_adapter import (
    FLOW_U_SLOTS,
    TMA_U_SLOTS,
    build_chtd_u_preview_from_afe_v2,
    default_tma_source_from_amb,
)
from hvac_sim.chtd.bus_index import N_U, N_X_STATES, U_INDEX, X_INDEX
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.thermal import compute_chtd_delta
from hvac_sim.pipeline import (
    _driver_air_temp_c,
    _driver_mrt_c,
    _passenger_air_temp_c,
    _passenger_mrt_c,
)

PREVIEW_SCHEMA = "afe_v2_chtd_preview_v1"
CLASSIFICATION = "experimental_offline_preview_not_runtime"

_KEY_ZONE_NAMES = (
    "HeadTempFd",
    "HeadTempFp",
    "HeadTempSd",
    "HeadTempSp",
    "CabinTempFd",
    "CabinTempFp",
    "WinTempFd",
    "WinTempFp",
    "RoofTemp",
)

_BUILTIN_SCENARIOS: Tuple[Dict[str, Any], ...] = (
    {
        "case_id": "face_cooling",
        "mode_code": "V",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 50.0,
        "circle_prior_posn": 2.33,
        "driver_temp_door": 0.35,
        "passenger_temp_door": 0.35,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": False,
    },
    {
        "case_id": "foot_heating",
        "mode_code": "F",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "driver_temp_door": 0.75,
        "passenger_temp_door": 0.75,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": True,
    },
    {
        "case_id": "foot_defrost_heating",
        "mode_code": "F_D",
        "front_blower_voltage": 12.0,
        "rear_blower_voltage": 12.0,
        "circle_mode_posn": 100.0,
        "circle_prior_posn": 4.56,
        "driver_temp_door": 0.75,
        "passenger_temp_door": 0.75,
        "rear_mode_posn": 0.5,
        "rear_temp_posn": 0.5,
        "heating_mode": True,
    },
)


@dataclass
class AfeV2ChtdPreviewResult:
    case_id: str
    x0: np.ndarray
    u: np.ndarray
    x_delta: np.ndarray
    x_next: np.ndarray
    u_preview_summary: Dict[str, Any]
    x_delta_summary: Dict[str, float]
    key_zone_deltas: Dict[str, float]
    pmv_temperature_inputs_preview: Dict[str, float]
    fallback_used: Dict[str, str]
    warnings: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)
    rh_percent: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PREVIEW_SCHEMA,
            "classification": CLASSIFICATION,
            "not_runtime_active": True,
            "afe_calc_default_unchanged": True,
            "thermal_formula_unchanged": True,
            "case_id": self.case_id,
            "rh_percent": self.rh_percent,
            "x0": self.x0.tolist(),
            "u": self.u.tolist(),
            "x_delta": self.x_delta.tolist(),
            "x_next": self.x_next.tolist(),
            "u_preview_summary": dict(self.u_preview_summary),
            "x_delta_summary": dict(self.x_delta_summary),
            "key_zone_deltas": dict(self.key_zone_deltas),
            "pmv_temperature_inputs_preview": dict(self.pmv_temperature_inputs_preview),
            "fallback_used": dict(self.fallback_used),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


def _repo_python_targets() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "simulink_conversion_package"
        / "python_targets"
    )


def default_builtin_scenarios() -> List[Dict[str, Any]]:
    return [dict(c) for c in _BUILTIN_SCENARIOS]


def build_tma_source(
    *,
    amb_t: float,
    driver_face_tma: Optional[float] = None,
    passenger_face_tma: Optional[float] = None,
    driver_foot_tma: Optional[float] = None,
    passenger_foot_tma: Optional[float] = None,
    rear_face_tma: Optional[float] = None,
    rear_foot_tma: Optional[float] = None,
    defrost_tma: Optional[float] = None,
    eva_t: Optional[float] = None,
    hct: Optional[float] = None,
    posn_fdh: Optional[float] = None,
    heating_mode: bool = False,
) -> Dict[str, Any]:
    """Build Tma source dict with sensible defaults from ``amb_t``."""
    base_eva = float(eva_t) if eva_t is not None else (amb_t + 8.0 if heating_mode else amb_t - 8.0)
    supply_hot = float(hct) if hct is not None else amb_t + 35.0
    supply_cool = base_eva
    face = float(driver_face_tma) if driver_face_tma is not None else (
        supply_hot - 5.0 if heating_mode else supply_cool + 2.0
    )
    foot = float(driver_foot_tma) if driver_foot_tma is not None else (
        supply_hot if heating_mode else supply_cool
    )
    src: Dict[str, Any] = {
        "driver_face_tma": face,
        "passenger_face_tma": float(passenger_face_tma) if passenger_face_tma is not None else face,
        "driver_foot_tma": foot,
        "passenger_foot_tma": (
            float(passenger_foot_tma) if passenger_foot_tma is not None else foot
        ),
        "rear_face_tma": float(rear_face_tma) if rear_face_tma is not None else face,
        "amb_t": float(amb_t),
        "eva_t": base_eva,
    }
    if rear_foot_tma is not None:
        src["rear_foot_tma"] = float(rear_foot_tma)
    if defrost_tma is not None:
        src["defrost_tma"] = float(defrost_tma)
    if hct is not None:
        src["hct"] = float(hct)
    if posn_fdh is not None:
        src["posn_fdh"] = float(posn_fdh)
    return src


def _initial_state_x0(
    x0: Optional[np.ndarray],
    *,
    amb_t: float,
) -> np.ndarray:
    if x0 is not None:
        arr = np.asarray(x0, dtype=float).reshape(N_X_STATES)
        if arr.shape != (N_X_STATES,):
            raise ValueError(f"x0 must have shape ({N_X_STATES},)")
        return arr.copy()
    return np.full(N_X_STATES, float(amb_t), dtype=float)


def _diagnostic_from_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    if "diagnostic" in case and isinstance(case["diagnostic"], Mapping):
        return dict(case["diagnostic"])
    if "m8_dual_layer_v2_result" in case:
        return dict(case)
    inputs = case_to_diagnostic_inputs(case)
    return run_afe_v2_diagnostic(inputs)


def _u_preview_summary(populated: Mapping[str, float]) -> Dict[str, Any]:
    flows = {k: float(populated[k]) for k in FLOW_U_SLOTS if k in populated}
    tmas = {k: float(populated[k]) for k in TMA_U_SLOTS if k in populated}
    return {
        "flow_slots_m3h": flows,
        "tma_slots_c": tmas,
        "flow_total_m3h": float(sum(flows.values())),
    }


def _x_delta_summary(delta: np.ndarray) -> Dict[str, float]:
    return {
        "shape": float(N_X_STATES),
        "max_abs_c": float(np.max(np.abs(delta))),
        "l2_norm_c": float(np.linalg.norm(delta)),
        "sum_c": float(np.sum(delta)),
    }


def _key_zone_deltas(delta: np.ndarray) -> Dict[str, float]:
    return {name: float(delta[X_INDEX[name]]) for name in _KEY_ZONE_NAMES}


def _pmv_temperature_preview(x: np.ndarray) -> Dict[str, float]:
    return {
        "driver_air_temp_candidate": float(_driver_air_temp_c(x)),
        "passenger_air_temp_candidate": float(_passenger_air_temp_c(x)),
        "driver_mrt_candidate": float(_driver_mrt_c(x)),
        "passenger_mrt_candidate": float(_passenger_mrt_c(x)),
    }


def run_single_afe_v2_chtd_preview(
    case: Mapping[str, Any],
    *,
    tma_source: Mapping[str, Any],
    x0: Optional[np.ndarray] = None,
    rh_percent: float = 50.0,
    chtd_params: Optional[CHTDParams] = None,
) -> AfeV2ChtdPreviewResult:
    """Run one offline AFE v2 → CHTD preview step for a case or diagnostic."""
    case_id = str(case.get("case_id", case.get("mode_code", "preview")))
    amb_t = float(tma_source["amb_t"])
    warnings: List[str] = [
        CLASSIFICATION,
        "offline_preview_not_wired_to_runtime_pipeline",
        "one_step_forward_euler_preview_only",
    ]

    diagnostic = _diagnostic_from_case(case)
    if diagnostic.get("warnings"):
        warnings.extend(str(w) for w in diagnostic["warnings"])

    adapter_result = build_chtd_u_preview_from_afe_v2(diagnostic, tma_source)
    u = adapter_result.u.copy()
    x_init = _initial_state_x0(x0, amb_t=amb_t)

    params = chtd_params if chtd_params is not None else CHTDParams()
    x_delta = compute_chtd_delta(x_init, u, params)
    if not np.all(np.isfinite(x_delta)):
        warnings.append("chtd_delta_contains_non_finite_values")
    x_next = x_init + x_delta

    fallback_used = dict(adapter_result.fallback_used)
    provenance: Dict[str, Any] = {
        "pipeline": [
            "run_afe_v2_diagnostic",
            "build_chtd_u_preview_from_afe_v2",
            "compute_chtd_delta",
            "x_next = x0 + delta",
        ],
        "diagnostic_schema": diagnostic.get("schema"),
        "adapter": dict(adapter_result.provenance),
        "x0_source": "user_supplied" if x0 is not None else "amb_t_uniform",
        "chtd_params": "default_CHTDParams",
        "thermal_api": "compute_chtd_delta",
    }

    return AfeV2ChtdPreviewResult(
        case_id=case_id,
        x0=x_init,
        u=u,
        x_delta=x_delta,
        x_next=x_next,
        u_preview_summary=_u_preview_summary(adapter_result.populated_slots),
        x_delta_summary=_x_delta_summary(x_delta),
        key_zone_deltas=_key_zone_deltas(x_delta),
        pmv_temperature_inputs_preview=_pmv_temperature_preview(x_next),
        fallback_used=fallback_used,
        warnings=warnings,
        provenance=provenance,
        rh_percent=float(rh_percent),
    )


def run_afe_v2_chtd_preview(
    cases: Sequence[Mapping[str, Any]],
    *,
    tma_source: Mapping[str, Any],
    x0: Optional[np.ndarray] = None,
    rh_percent: float = 50.0,
) -> Dict[str, Any]:
    """Run preview for multiple cases; return aggregate JSON document."""
    previews = [
        run_single_afe_v2_chtd_preview(
            case,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
        ).to_dict()
        for case in cases
    ]
    return {
        "schema": PREVIEW_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "afe_calc_default_unchanged": True,
        "thermal_formula_unchanged": True,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case_count": len(previews),
        "rh_percent": float(rh_percent),
        "previews": previews,
    }


def run_afe_v2_chtd_preview_from_input_doc(
    doc: Mapping[str, Any],
    *,
    tma_source: Optional[Mapping[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    rh_percent: float = 50.0,
) -> Dict[str, Any]:
    """Accept replay output, replay case, diagnostic, or builtin-style case."""
    schema = str(doc.get("schema", ""))
    if schema == "afe_v2_replay_output_v1":
        cases = list(doc.get("cases", []))
        if not cases:
            raise ValueError("replay document has no cases")
        amb = 24.0
        if tma_source is None:
            first = cases[0].get("diagnostic", {}).get("inputs", {})
            if isinstance(first, dict) and "temp_C" in first:
                amb = float(first["temp_C"])
            tma_source = default_tma_source_from_amb(amb_t=amb)
        return run_afe_v2_chtd_preview(
            cases,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
        )
    if schema == "afe_v2_diagnostic_v1" or "m8_dual_layer_v2_result" in doc:
        if tma_source is None:
            inputs = doc.get("inputs", {})
            amb = float(inputs.get("temp_C", 24.0)) if isinstance(inputs, dict) else 24.0
            tma_source = default_tma_source_from_amb(amb_t=amb)
        single = run_single_afe_v2_chtd_preview(
            doc,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
        )
        return single.to_dict()
    if "mode_code" in doc:
        if tma_source is None:
            heating = bool(doc.get("heating_mode", False))
            amb = float(doc.get("amb_t", 24.0))
            tma_source = build_tma_source(amb_t=amb, heating_mode=heating)
        single = run_single_afe_v2_chtd_preview(
            doc,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
        )
        return single.to_dict()
    raise ValueError(f"unsupported input document schema: {schema!r}")


def write_default_preview_bundle(
    output_dir: Optional[Path] = None,
    *,
    amb_t: float = 24.0,
    rh_percent: float = 50.0,
) -> Path:
    """Write builtin three-scenario preview JSON under python_targets."""
    out_dir = output_dir or _repo_python_targets()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = default_builtin_scenarios()
    previews: List[Dict[str, Any]] = []
    for case in cases:
        tma = build_tma_source(
            amb_t=amb_t,
            heating_mode=bool(case.get("heating_mode", False)),
        )
        previews.append(
            run_single_afe_v2_chtd_preview(
                case,
                tma_source=tma,
                rh_percent=rh_percent,
            ).to_dict()
        )
    doc = {
        "schema": PREVIEW_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "afe_calc_default_unchanged": True,
        "thermal_formula_unchanged": True,
        "case_count": len(previews),
        "previews": previews,
    }
    path = out_dir / "AFE_V2_CHTD_PREVIEW_SAMPLE.json"
    import json

    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
