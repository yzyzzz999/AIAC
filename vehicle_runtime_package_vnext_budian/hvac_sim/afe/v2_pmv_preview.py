"""AFE v2 + CHTD + PMV offline preview (experimental, not runtime).

Full offline chain: ``run_single_afe_v2_chtd_preview`` → air speed → baseline
Fanger PMV/PPD. Does **not** call ``runtime_pipeline`` or modify core modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from hvac_sim.afe.v2_chtd_preview import (
    build_tma_source,
    default_builtin_scenarios,
    run_single_afe_v2_chtd_preview,
)
from hvac_sim.air_speed import AirSpeedInputs, estimate_driver_passenger_air_speed
from hvac_sim.chtd.bus_index import U_INDEX
from hvac_sim.ekf.head_temp_filter import fuse_front_row_heads
from hvac_sim.occupant import ImageModuleInputs, resolve_image_module_inputs
from hvac_sim.pipeline import (
    _driver_air_temp_c,
    _driver_mrt_c,
    _passenger_air_temp_c,
    _passenger_mrt_c,
    air_speed_inputs_from_chtd_u,
)
from hvac_sim.pmv.applicability import (
    applicability_seat_fields,
    evaluate_pmv_applicability,
)
from hvac_sim.pmv.interface import VehicleComfortInputs, compute_vehicle_pmv

PREVIEW_SCHEMA = "afe_v2_pmv_preview_v1"
CLASSIFICATION = "experimental_offline_pmv_preview_not_runtime"

_PMV_ABS_LIMIT = 5.0


@dataclass
class SeatPmvPreview:
    air_temp_c: float
    mean_radiant_temp_c: float
    air_speed_m_s: float
    pmv: float
    ppd: float
    valid: bool = True
    pmv_raw: float = 0.0
    pmv_display: float = 0.0
    valid_for_comfort_interpretation: bool = True
    pmv_applicability: str = "within_reference_range"
    pmv_applicability_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "air_temp_c": float(self.air_temp_c),
            "mean_radiant_temp_c": float(self.mean_radiant_temp_c),
            "air_speed_m_s": float(self.air_speed_m_s),
            "pmv": float(self.pmv),
            "ppd": float(self.ppd),
            "valid": bool(self.valid),
            "pmv_raw": float(self.pmv_raw),
            "pmv_display": float(self.pmv_display),
            "valid_for_comfort_interpretation": bool(self.valid_for_comfort_interpretation),
            "pmv_applicability": str(self.pmv_applicability),
            "pmv_applicability_reasons": list(self.pmv_applicability_reasons),
        }


@dataclass
class AfeV2PmvPreviewResult:
    case_id: str
    driver: SeatPmvPreview
    passenger: SeatPmvPreview
    chtd_preview_summary: Dict[str, Any]
    x_delta_summary: Dict[str, float]
    u_preview_summary: Dict[str, Any]
    fallback_used: Dict[str, str]
    warnings: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)
    quality_level: str = "medium"
    rh_percent: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PREVIEW_SCHEMA,
            "classification": CLASSIFICATION,
            "not_runtime_active": True,
            "runtime_pipeline_unchanged": True,
            "afe_calc_default_unchanged": True,
            "thermal_formula_unchanged": True,
            "case_id": self.case_id,
            "rh_percent": self.rh_percent,
            "driver": self.driver.to_dict(),
            "passenger": self.passenger.to_dict(),
            "ch_preview_summary": dict(self.chtd_preview_summary),
            "chtd_preview_summary": dict(self.chtd_preview_summary),
            "x_delta_summary": dict(self.x_delta_summary),
            "u_preview_summary": dict(self.u_preview_summary),
            "fallback_used": dict(self.fallback_used),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "quality_level": self.quality_level,
        }


def _repo_python_targets() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "simulink_conversion_package"
        / "python_targets"
    )


def default_builtin_scenarios_pmv() -> List[Dict[str, Any]]:
    return default_builtin_scenarios()


def _air_speed_inputs_from_u_preview(
    u: np.ndarray,
    *,
    geometry: Optional[AirSpeedInputs] = None,
) -> AirSpeedInputs:
    """Build ``AirSpeedInputs`` from preview ``u`` including rear-foot slots."""
    base = air_speed_inputs_from_chtd_u(u, fallback_inputs=geometry)
    return replace(
        base,
        rear_driver_foot_flow=float(u[U_INDEX["RearSdfFlow"]]),
        rear_passenger_foot_flow=float(u[U_INDEX["RearSpfFlow"]]),
    )


def _assess_quality_level(
    *,
    fallback_used: Mapping[str, str],
    warnings: Sequence[str],
    driver_pmv: float,
    passenger_pmv: float,
) -> str:
    if not (
        math.isfinite(driver_pmv)
        and math.isfinite(passenger_pmv)
        and -_PMV_ABS_LIMIT <= driver_pmv <= _PMV_ABS_LIMIT
        and -_PMV_ABS_LIMIT <= passenger_pmv <= _PMV_ABS_LIMIT
    ):
        return "low"
    if len(fallback_used) >= 3 or any("did_not_converge" in w for w in warnings):
        return "low"
    if fallback_used:
        return "medium"
    return "high"


def run_single_afe_v2_pmv_preview(
    case: Mapping[str, Any],
    *,
    tma_source: Mapping[str, Any],
    x0: Optional[np.ndarray] = None,
    rh_percent: float = 50.0,
    image_inputs: Optional[ImageModuleInputs] = None,
    air_speed_geometry: Optional[AirSpeedInputs] = None,
    use_ir_head_fusion: bool = False,
    met_driver: float = 1.0,
    clo_driver: float = 0.5,
    met_passenger: float = 1.0,
    clo_passenger: float = 0.5,
) -> AfeV2PmvPreviewResult:
    """Run offline AFE v2 → CHTD → PMV baseline for one case."""
    chtd = run_single_afe_v2_chtd_preview(
        case,
        tma_source=tma_source,
        x0=x0,
        rh_percent=rh_percent,
    )
    x_comfort = chtd.x_next
    u = chtd.u

    warnings = list(chtd.warnings)
    warnings.append(CLASSIFICATION)
    warnings.append("offline_pmv_preview_not_runtime_pipeline")

    air_in = _air_speed_inputs_from_u_preview(u, geometry=air_speed_geometry)
    air_speeds = estimate_driver_passenger_air_speed(air_in)

    if image_inputs is not None:
        driver_occ, passenger_occ = resolve_image_module_inputs(
            image_inputs,
            driver_default_met=met_driver,
            driver_default_clo=clo_driver,
            passenger_default_met=met_passenger,
            passenger_default_clo=clo_passenger,
        )
        met_d, clo_d = driver_occ.met, driver_occ.clo
        met_p, clo_p = passenger_occ.met, passenger_occ.clo
    else:
        driver_occ = None
        passenger_occ = None
        met_d, clo_d = met_driver, clo_driver
        met_p, clo_p = met_passenger, clo_passenger

    driver_model_air = _driver_air_temp_c(x_comfort)
    passenger_model_air = _passenger_air_temp_c(x_comfort)
    driver_mrt = _driver_mrt_c(x_comfort)
    passenger_mrt = _passenger_mrt_c(x_comfort)

    if use_ir_head_fusion and image_inputs is not None:
        driver_fusion, passenger_fusion = fuse_front_row_heads(
            driver_model_air,
            passenger_model_air,
            image_inputs,
        )
        driver_air = driver_fusion.fused_temp_c
        passenger_air = passenger_fusion.fused_temp_c
        warnings.append("ir_head_fusion_enabled_offline_preview")
    else:
        driver_air = driver_model_air
        passenger_air = passenger_model_air

    driver_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=driver_air,
            mean_radiant_temp_c=driver_mrt,
            air_velocity_m_s=air_speeds.driver_air_speed_m_s,
            relative_humidity_pct=rh_percent,
            metabolic_rate_met=met_d,
            clothing_insulation_clo=clo_d,
        )
    )
    passenger_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=passenger_air,
            mean_radiant_temp_c=passenger_mrt,
            air_velocity_m_s=air_speeds.passenger_air_speed_m_s,
            relative_humidity_pct=rh_percent,
            metabolic_rate_met=met_p,
            clothing_insulation_clo=clo_p,
        )
    )

    driver_valid = driver_occ.occupied if driver_occ is not None else True
    passenger_valid = passenger_occ.occupied if passenger_occ is not None else True

    driver_app = evaluate_pmv_applicability(
        ta=driver_air,
        tr=driver_mrt,
        vel=air_speeds.driver_air_speed_m_s,
        rh=rh_percent,
        met=met_d,
        clo=clo_d,
        pmv=driver_comfort.pmv,
    )
    passenger_app = evaluate_pmv_applicability(
        ta=passenger_air,
        tr=passenger_mrt,
        vel=air_speeds.passenger_air_speed_m_s,
        rh=rh_percent,
        met=met_p,
        clo=clo_p,
        pmv=passenger_comfort.pmv,
    )
    driver_app_fields = applicability_seat_fields(driver_app)
    passenger_app_fields = applicability_seat_fields(passenger_app)

    driver = SeatPmvPreview(
        air_temp_c=driver_air,
        mean_radiant_temp_c=driver_mrt,
        air_speed_m_s=air_speeds.driver_air_speed_m_s,
        pmv=driver_comfort.pmv,
        ppd=driver_comfort.ppd,
        valid=driver_valid,
        **driver_app_fields,
    )
    passenger = SeatPmvPreview(
        air_temp_c=passenger_air,
        mean_radiant_temp_c=passenger_mrt,
        air_speed_m_s=air_speeds.passenger_air_speed_m_s,
        pmv=passenger_comfort.pmv,
        ppd=passenger_comfort.ppd,
        valid=passenger_valid,
        **passenger_app_fields,
    )

    fallback_used = dict(chtd.fallback_used)
    quality = _assess_quality_level(
        fallback_used=fallback_used,
        warnings=warnings,
        driver_pmv=driver.pmv,
        passenger_pmv=passenger.pmv,
    )

    provenance: Dict[str, Any] = {
        "pipeline": [
            "run_single_afe_v2_chtd_preview",
            "air_speed_inputs_from_chtd_u",
            "estimate_driver_passenger_air_speed",
            "compute_vehicle_pmv",
        ],
        "chtd_provenance": dict(chtd.provenance),
        "air_speed_source": "chtd_u_preview_with_rear_foot",
        "pmv_method": "compute_vehicle_pmv_baseline",
        "use_ir_head_fusion": use_ir_head_fusion,
        "comfort_state": "x_next",
    }

    chtd_summary = {
        "key_zone_deltas": dict(chtd.key_zone_deltas),
        "x0_mean_c": float(np.mean(chtd.x0)),
        "x_next_mean_c": float(np.mean(chtd.x_next)),
        "pmv_temperature_inputs_preview": dict(chtd.pmv_temperature_inputs_preview),
    }

    return AfeV2PmvPreviewResult(
        case_id=chtd.case_id,
        driver=driver,
        passenger=passenger,
        chtd_preview_summary=chtd_summary,
        x_delta_summary=dict(chtd.x_delta_summary),
        u_preview_summary=dict(chtd.u_preview_summary),
        fallback_used=fallback_used,
        warnings=warnings,
        provenance=provenance,
        quality_level=quality,
        rh_percent=float(rh_percent),
    )


def run_afe_v2_pmv_preview(
    cases: Sequence[Mapping[str, Any]],
    *,
    tma_source: Mapping[str, Any],
    x0: Optional[np.ndarray] = None,
    rh_percent: float = 50.0,
    image_inputs: Optional[ImageModuleInputs] = None,
    air_speed_geometry: Optional[AirSpeedInputs] = None,
    use_ir_head_fusion: bool = False,
) -> Dict[str, Any]:
    previews = [
        run_single_afe_v2_pmv_preview(
            case,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
            image_inputs=image_inputs,
            air_speed_geometry=air_speed_geometry,
            use_ir_head_fusion=use_ir_head_fusion,
        ).to_dict()
        for case in cases
    ]
    return {
        "schema": PREVIEW_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "runtime_pipeline_unchanged": True,
        "afe_calc_default_unchanged": True,
        "thermal_formula_unchanged": True,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case_count": len(previews),
        "rh_percent": float(rh_percent),
        "previews": previews,
    }


def run_afe_v2_pmv_preview_from_input_doc(
    doc: Mapping[str, Any],
    *,
    tma_source: Optional[Mapping[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    rh_percent: float = 50.0,
    image_inputs: Optional[ImageModuleInputs] = None,
    air_speed_geometry: Optional[AirSpeedInputs] = None,
) -> Dict[str, Any]:
    """Accept replay output, diagnostic, or bare AFE v2 case dict."""
    schema = str(doc.get("schema", ""))
    if schema == "afe_v2_replay_output_v1":
        cases = list(doc.get("cases", []))
        if not cases:
            raise ValueError("replay document has no cases")
        if tma_source is None:
            from hvac_sim.afe.v2_to_chtd_adapter import default_tma_source_from_amb

            amb = 24.0
            first = cases[0].get("diagnostic", {}).get("inputs", {})
            if isinstance(first, dict) and "temp_C" in first:
                amb = float(first["temp_C"])
            tma_source = default_tma_source_from_amb(amb_t=amb)
        return run_afe_v2_pmv_preview(
            cases,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
            image_inputs=image_inputs,
            air_speed_geometry=air_speed_geometry,
        )
    if schema in {"afe_v2_diagnostic_v1", "afe_v2_chtd_preview_v1", "afe_v2_pmv_preview_v1"}:
        if schema == "afe_v2_chtd_preview_v1" or schema == "afe_v2_pmv_preview_v1":
            inner = doc.get("previews", [doc])[0]
            if tma_source is None:
                amb = float(inner.get("chtd_preview_summary", {}).get("x0_mean_c", 24.0))
                tma_source = build_tma_source(amb_t=amb)
            return run_single_afe_v2_pmv_preview(
                inner,
                tma_source=tma_source,
                x0=x0,
                rh_percent=rh_percent,
                image_inputs=image_inputs,
                air_speed_geometry=air_speed_geometry,
            ).to_dict()
        if tma_source is None:
            inputs = doc.get("inputs", {})
            amb = float(inputs.get("temp_C", 24.0)) if isinstance(inputs, dict) else 24.0
            tma_source = build_tma_source(amb_t=amb)
        return run_single_afe_v2_pmv_preview(
            doc,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
            image_inputs=image_inputs,
            air_speed_geometry=air_speed_geometry,
        ).to_dict()
    if "mode_code" in doc:
        if tma_source is None:
            tma_source = build_tma_source(
                amb_t=float(doc.get("amb_t", 24.0)),
                heating_mode=bool(doc.get("heating_mode", False)),
            )
        return run_single_afe_v2_pmv_preview(
            doc,
            tma_source=tma_source,
            x0=x0,
            rh_percent=rh_percent,
            image_inputs=image_inputs,
            air_speed_geometry=air_speed_geometry,
        ).to_dict()
    raise ValueError(f"unsupported input document schema: {schema!r}")


def write_default_pmv_preview_bundle(
    output_dir: Optional[Path] = None,
    *,
    amb_t: float = 24.0,
    rh_percent: float = 50.0,
) -> Path:
    out_dir = output_dir or _repo_python_targets()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = default_builtin_scenarios_pmv()
    previews = []
    for case in cases:
        tma = build_tma_source(
            amb_t=amb_t,
            heating_mode=bool(case.get("heating_mode", False)),
        )
        previews.append(
            run_single_afe_v2_pmv_preview(
                case,
                tma_source=tma,
                rh_percent=rh_percent,
            ).to_dict()
        )
    doc = {
        "schema": PREVIEW_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "runtime_pipeline_unchanged": True,
        "afe_calc_default_unchanged": True,
        "thermal_formula_unchanged": True,
        "case_count": len(previews),
        "rh_percent": float(rh_percent),
        "previews": previews,
    }
    path = out_dir / "AFE_V2_PMV_PREVIEW_SAMPLE.json"
    import json

    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
