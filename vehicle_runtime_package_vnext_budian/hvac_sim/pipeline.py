"""Runnable PMV pipeline: CHTD + air_speed + Fanger (T7-3).

Single entry point for joint debug: advance CHTD one step, map seat
temperatures, estimate local air speed, compute driver/passenger **baseline**
Fanger PMV/PPD (ISO 7730). Per-seat outputs are reference values for a future
recommendation module; individual/scenario/subjective offsets are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np

from hvac_sim.air_speed import AirSpeedInputs, estimate_driver_passenger_air_speed
from hvac_sim.chtd.bus_index import N_U, N_X_STATES, U_INDEX, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.runtime_policy import apply_two_row_vehicle_policy
from hvac_sim.chtd.thermal import compute_chtd_delta, one_step_chtd
from hvac_sim.chtd.zone_config import CHTD_ZONE_CONFIG, ZoneImplementationMode
from hvac_sim.ekf.head_temp_filter import HeadTempFilterResult, fuse_front_row_heads
from hvac_sim.occupant import ImageModuleInputs, SeatOccupantResolved, resolve_image_module_inputs
from hvac_sim.pmv.interface import VehicleComfortInputs, compute_vehicle_pmv


@dataclass(frozen=True)
class PipelineInputs:
    """Inputs for one comfort-pipeline evaluation step."""

    x: np.ndarray
    u: np.ndarray
    chtd_params: Optional[CHTDParams] = None
    rh: float = 50.0
    met_driver: float = 1.0
    clo_driver: float = 0.5
    met_passenger: float = 1.0
    clo_passenger: float = 0.5
    driver_face_flow: Optional[float] = None
    passenger_face_flow: Optional[float] = None
    driver_floor_flow: Optional[float] = None
    passenger_floor_flow: Optional[float] = None
    driver_defrost_flow: Optional[float] = None
    passenger_defrost_flow: Optional[float] = None
    rear_driver_foot_flow: Optional[float] = None
    rear_passenger_foot_flow: Optional[float] = None
    use_next_state: bool = True
    air_speed: Optional[AirSpeedInputs] = None
    image_inputs: Optional[ImageModuleInputs] = None
    measured_driver_head_air_temp_c: Optional[float] = None
    measured_passenger_head_air_temp_c: Optional[float] = None
    use_ir_head_fusion: bool = False
    ir_process_var: float = 1.0
    ir_measurement_var: float = 1.0
    ir_surface_to_air_offset_c: float = 2.0
    ir_min_confidence: float = 0.5
    ir_initial_variance: float = 1.0
    apply_runtime_policy: bool = True
    chtd_param_mode: str = "default"
    safe_preview_active: bool = False
    phase3_engineering_preview_active: bool = False
    phase3_capacity_active: bool = False
    solar_shell_routing_active: bool = False
    actuator_distribution_active: bool = False
    foot_leakage_active: bool = False
    phase3_classification: Optional[str] = None
    params_provenance: Optional[Dict[str, Any]] = None
    bypass_models: bool = False
    driver_air_temp_override_c: Optional[float] = None
    passenger_air_temp_override_c: Optional[float] = None
    driver_mrt_override_c: Optional[float] = None
    passenger_mrt_override_c: Optional[float] = None
    driver_air_speed_override_m_s: Optional[float] = None
    passenger_air_speed_override_m_s: Optional[float] = None


@dataclass(frozen=True)
class SeatComfortResult:
    """Per-seat comfort outputs for PMV assessment."""

    air_temp_c: float
    mean_radiant_temp_c: float
    air_speed_m_s: float
    pmv: float
    ppd: float
    valid: bool = True


@dataclass(frozen=True)
class PipelineResult:
    """Full pipeline output for one step."""

    x_next: np.ndarray
    x_delta: np.ndarray
    driver: SeatComfortResult
    passenger: SeatComfortResult
    trace_status: str
    implementation_modes: Dict[str, list[str]]
    inputs_used: Dict[str, Any] = field(default_factory=dict)


def _validate_chtd_vectors(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    if x_arr.shape != (N_X_STATES,):
        raise ValueError(
            f"pipeline: x must have shape ({N_X_STATES},), got {x_arr.shape}"
        )
    if u_arr.shape != (N_U,):
        raise ValueError(f"pipeline: u must have shape ({N_U},), got {u_arr.shape}")
    return x_arr, u_arr


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _pick_air_temp(primary: float, fallback: float) -> float:
    if _is_finite(primary):
        return float(primary)
    if _is_finite(fallback):
        return float(fallback)
    return 20.0


def _mean_radiant(*temps: float) -> float:
    finite = [float(t) for t in temps if _is_finite(t)]
    if not finite:
        return 20.0
    return sum(finite) / len(finite)


def _driver_air_temp_c(x: np.ndarray) -> float:
    return _pick_air_temp(x[X_INDEX["HeadTempFd"]], x[X_INDEX["CabinTempFd"]])


def _passenger_air_temp_c(x: np.ndarray) -> float:
    return _pick_air_temp(x[X_INDEX["HeadTempFp"]], x[X_INDEX["CabinTempFd"]])


MRT_HOT_SURFACE_THRESHOLD_C = 35.0


def _mrt_surface_or_cabin(surface_temp: float, cabin_temp: float) -> float:
    """Return cabin_temp when surface is hotter than threshold (summer glass/roof).

    Hot glass and roof surfaces radiate disproportionately into the cabin, but
    the Fanger MRT formula treats them as uniform-environment components.  Above
    the threshold the occupant's limited view factor means the raw surface
    temperature overstates the effective radiant load, so cabin air temperature
    is a better proxy.
    """
    if surface_temp > MRT_HOT_SURFACE_THRESHOLD_C:
        return cabin_temp
    return surface_temp


def _driver_mrt_c(x: np.ndarray) -> float:
    cabin = x[X_INDEX["CabinTempFd"]]
    return _mean_radiant(
        x[X_INDEX["HeadTempFd"]],
        cabin,
        _mrt_surface_or_cabin(x[X_INDEX["WinTempFd"]], cabin),
        _mrt_surface_or_cabin(x[X_INDEX["RoofTemp"]], cabin),
    )


def _passenger_mrt_c(x: np.ndarray) -> float:
    cabin = x[X_INDEX["CabinTempFp"]]
    return _mean_radiant(
        x[X_INDEX["HeadTempFp"]],
        cabin,
        _mrt_surface_or_cabin(x[X_INDEX["WinTempFp"]], cabin),
        _mrt_surface_or_cabin(x[X_INDEX["RoofTemp"]], cabin),
    )


def _resolve_pmv_air_temp(
    measured_temp_c: Optional[float],
    fused_temp_c: float,
) -> tuple[float, str]:
    if measured_temp_c is not None and math.isfinite(float(measured_temp_c)):
        return float(measured_temp_c), "measured_head_points"
    return float(fused_temp_c), "model_state"


def _build_implementation_modes() -> Dict[str, list[str]]:
    modes: Dict[str, list[str]] = {
        ZoneImplementationMode.TRACE.value: [],
        ZoneImplementationMode.FAST_APPROX.value: [],
        ZoneImplementationMode.GAP.value: [],
    }
    for entry in CHTD_ZONE_CONFIG:
        modes[entry.mode.value].append(entry.name)
    return modes


def _trace_status_summary(modes: Dict[str, list[str]]) -> str:
    n_trace = len(modes[ZoneImplementationMode.TRACE.value])
    n_gap = len(modes[ZoneImplementationMode.GAP.value])
    n_fast = len(modes[ZoneImplementationMode.FAST_APPROX.value])
    return (
        f"CHTD {N_X_STATES} states: "
        f"{n_trace} TRACE, {n_fast} FAST_APPROX, {n_gap} GAP"
    )


_PIPELINE_FLOW_FIELDS = (
    "driver_face_flow",
    "passenger_face_flow",
    "driver_floor_flow",
    "passenger_floor_flow",
    "driver_defrost_flow",
    "passenger_defrost_flow",
    "rear_driver_foot_flow",
    "rear_passenger_foot_flow",
)


def air_speed_inputs_from_chtd_u(
    u: np.ndarray,
    fallback_inputs: Optional[AirSpeedInputs] = None,
) -> AirSpeedInputs:
    """Build ``AirSpeedInputs`` front-row vent flows from ``Bus_CHTD_u``.

    Maps FrntFdvFlow / FrntFpvFlow / FrntFdfFlow / FrntFpfFlow /
    FrntFdDefFlow / FrntFpDefFlow.  Rear-foot flows stay ``None`` until a
    canonical bus mapping is confirmed.

    When *fallback_inputs* is given, non-``None`` optional flow fields and
    face-flow floats from *fallback_inputs* override the values read from *u*.
    """
    u_arr = np.asarray(u, dtype=float)
    if u_arr.shape != (N_U,):
        raise ValueError(
            f"pipeline: u must have shape ({N_U},), got {u_arr.shape}"
        )

    base = AirSpeedInputs(
        driver_face_flow=float(u_arr[U_INDEX["FrntFdvFlow"]]),
        passenger_face_flow=float(u_arr[U_INDEX["FrntFpvFlow"]]),
        driver_floor_flow=float(u_arr[U_INDEX["FrntFdfFlow"]]),
        passenger_floor_flow=float(u_arr[U_INDEX["FrntFpfFlow"]]),
        driver_defrost_flow=float(u_arr[U_INDEX["FrntFdDefFlow"]]),
        passenger_defrost_flow=float(u_arr[U_INDEX["FrntFpDefFlow"]]),
        rear_driver_foot_flow=None,
        rear_passenger_foot_flow=None,
    )

    if fallback_inputs is None:
        return base

    overrides: Dict[str, Any] = {
        "driver_face_flow": fallback_inputs.driver_face_flow,
        "passenger_face_flow": fallback_inputs.passenger_face_flow,
    }
    for name in (
        "driver_floor_flow",
        "passenger_floor_flow",
        "driver_defrost_flow",
        "passenger_defrost_flow",
        "rear_driver_foot_flow",
        "rear_passenger_foot_flow",
    ):
        fb_val = getattr(fallback_inputs, name)
        if fb_val is not None:
            overrides[name] = fb_val
    return replace(base, **overrides)


def _pipeline_has_explicit_flows(inputs: PipelineInputs) -> bool:
    return any(getattr(inputs, name) is not None for name in _PIPELINE_FLOW_FIELDS)


def _flows_used_from_air_speed(air_in: AirSpeedInputs) -> Dict[str, Any]:
    return {name: getattr(air_in, name) for name in _PIPELINE_FLOW_FIELDS}


def _resolve_air_speed_inputs(
    inputs: PipelineInputs, u: np.ndarray
) -> tuple[AirSpeedInputs, str]:
    """Resolve air-speed inputs with explicit-air_speed > pipeline fields > u."""
    if inputs.air_speed is not None:
        return inputs.air_speed, "explicit_air_speed"

    u_base = air_speed_inputs_from_chtd_u(u)
    if _pipeline_has_explicit_flows(inputs):
        overrides = {
            name: getattr(inputs, name)
            for name in _PIPELINE_FLOW_FIELDS
            if getattr(inputs, name) is not None
        }
        return replace(u_base, **overrides), "explicit_pipeline_fields"

    return u_base, "chtd_u"


def _occupant_resolved_to_dict(resolved: SeatOccupantResolved) -> Dict[str, Any]:
    return {
        "occupied": resolved.occupied,
        "met": float(resolved.met),
        "clo": float(resolved.clo),
        "ir_head_surface_temp_c": resolved.ir_head_surface_temp_c,
        "confidence": resolved.confidence,
        "source": dict(resolved.source),
    }


def _head_fusion_to_dict(
    result: HeadTempFilterResult,
    *,
    ir_surface_temp_c: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "source": result.source,
        "kalman_gain": float(result.kalman_gain),
        "model_air_temp_c": float(result.model_temp_c),
        "ir_surface_temp_c": ir_surface_temp_c,
        "fused_air_temp_c": float(result.fused_temp_c),
        "fused_temp_c": float(result.fused_temp_c),
        "model_temp_c": float(result.model_temp_c),
        "observation_temp_c": result.observation_temp_c,
        "innovation_c": result.innovation_c,
        "variance": float(result.variance),
        "valid": bool(result.valid),
    }


def _resolve_pipeline_occupants(
    inputs: PipelineInputs,
) -> tuple[float, float, float, float, SeatOccupantResolved, SeatOccupantResolved]:
    """Return met/clo for each seat and resolved occupant diagnostics."""
    if inputs.image_inputs is not None:
        driver_res, passenger_res = resolve_image_module_inputs(
            inputs.image_inputs,
            driver_default_met=inputs.met_driver,
            driver_default_clo=inputs.clo_driver,
            passenger_default_met=inputs.met_passenger,
            passenger_default_clo=inputs.clo_passenger,
        )
        return (
            driver_res.met,
            driver_res.clo,
            passenger_res.met,
            passenger_res.clo,
            driver_res,
            passenger_res,
        )

    default_driver = SeatOccupantResolved(
        occupied=True,
        met=float(inputs.met_driver),
        clo=float(inputs.clo_driver),
        ir_head_surface_temp_c=None,
        confidence=None,
        source={"occupied": "pipeline_default", "met": "pipeline", "clo": "pipeline"},
    )
    default_passenger = SeatOccupantResolved(
        occupied=True,
        met=float(inputs.met_passenger),
        clo=float(inputs.clo_passenger),
        ir_head_surface_temp_c=None,
        confidence=None,
        source={"occupied": "pipeline_default", "met": "pipeline", "clo": "pipeline"},
    )
    return (
        inputs.met_driver,
        inputs.clo_driver,
        inputs.met_passenger,
        inputs.clo_passenger,
        default_driver,
        default_passenger,
    )


_BYPASS_DEFAULT_AIR_TEMP_C = 25.0
_BYPASS_DEFAULT_MRT_C = 25.0
_BYPASS_DEFAULT_AIR_SPEED_M_S = 0.1


def _or_default(value: Optional[float], default: float) -> float:
    if value is not None and math.isfinite(float(value)):
        return float(value)
    return default


def _run_comfort_pipeline_bypass(
    inputs: PipelineInputs,
    x: np.ndarray,
    u: np.ndarray,
    met_driver: float,
    clo_driver: float,
    met_passenger: float,
    clo_passenger: float,
    driver_occ: SeatOccupantResolved,
    passenger_occ: SeatOccupantResolved,
) -> PipelineResult:
    """Bypass CHTD + air-speed models; compute PMV directly from override inputs."""
    driver_pmv_air_temp = _or_default(
        inputs.driver_air_temp_override_c, _BYPASS_DEFAULT_AIR_TEMP_C
    )
    passenger_pmv_air_temp = _or_default(
        inputs.passenger_air_temp_override_c, _BYPASS_DEFAULT_AIR_TEMP_C
    )
    driver_mrt = _or_default(inputs.driver_mrt_override_c, _BYPASS_DEFAULT_MRT_C)
    passenger_mrt = _or_default(inputs.passenger_mrt_override_c, _BYPASS_DEFAULT_MRT_C)
    driver_air_speed = _or_default(
        inputs.driver_air_speed_override_m_s, _BYPASS_DEFAULT_AIR_SPEED_M_S
    )
    passenger_air_speed = _or_default(
        inputs.passenger_air_speed_override_m_s, _BYPASS_DEFAULT_AIR_SPEED_M_S
    )

    driver_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=driver_pmv_air_temp,
            mean_radiant_temp_c=driver_mrt,
            air_velocity_m_s=driver_air_speed,
            relative_humidity_pct=inputs.rh,
            metabolic_rate_met=met_driver,
            clothing_insulation_clo=clo_driver,
        )
    )
    passenger_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=passenger_pmv_air_temp,
            mean_radiant_temp_c=passenger_mrt,
            air_velocity_m_s=passenger_air_speed,
            relative_humidity_pct=inputs.rh,
            metabolic_rate_met=met_passenger,
            clothing_insulation_clo=clo_passenger,
        )
    )

    x_delta = np.zeros(N_X_STATES, dtype=float)
    x_next = np.asarray(x, dtype=float).copy()

    impl_modes = _build_implementation_modes()
    inputs_used: Dict[str, Any] = {
        "bypass_models": True,
        "use_next_state": False,
        "rh": inputs.rh,
        "met_driver": met_driver,
        "clo_driver": clo_driver,
        "met_passenger": met_passenger,
        "clo_passenger": clo_passenger,
        "air_speed_source": "bypass_override",
        "flows_used": {},
        "occupant_driver": _occupant_resolved_to_dict(driver_occ),
        "occupant_passenger": _occupant_resolved_to_dict(passenger_occ),
        "ir_head_fusion_enabled": False,
        "ir_head_surface_temp_driver_c": driver_occ.ir_head_surface_temp_c,
        "ir_head_surface_temp_passenger_c": passenger_occ.ir_head_surface_temp_c,
        "driver_air_temp_source": "bypass_override",
        "passenger_air_temp_source": "bypass_override",
        "driver_air_temp_c": driver_pmv_air_temp,
        "passenger_air_temp_c": passenger_pmv_air_temp,
        "driver_mrt_c": driver_mrt,
        "passenger_mrt_c": passenger_mrt,
        "driver_mrt_source": "bypass_override",
        "passenger_mrt_source": "bypass_override",
        "driver_air_speed_m_s": driver_air_speed,
        "passenger_air_speed_m_s": passenger_air_speed,
        "runtime_policy": {"applied": False, "bypass": True},
        "chtd_param_mode": "bypass",
        "classification": "bypass_models",
    }

    return PipelineResult(
        x_next=x_next,
        x_delta=x_delta,
        driver=SeatComfortResult(
            air_temp_c=driver_pmv_air_temp,
            mean_radiant_temp_c=driver_mrt,
            air_speed_m_s=driver_air_speed,
            pmv=driver_comfort.pmv,
            ppd=driver_comfort.ppd,
            valid=driver_occ.occupied,
        ),
        passenger=SeatComfortResult(
            air_temp_c=passenger_pmv_air_temp,
            mean_radiant_temp_c=passenger_mrt,
            air_speed_m_s=passenger_air_speed,
            pmv=passenger_comfort.pmv,
            ppd=passenger_comfort.ppd,
            valid=passenger_occ.occupied,
        ),
        trace_status="CHTD bypassed — direct PMV inputs used",
        implementation_modes=impl_modes,
        inputs_used=inputs_used,
    )


def run_comfort_pipeline(inputs: PipelineInputs) -> PipelineResult:
    """Run CHTD -> air_speed -> PMV/PPD for driver and passenger.

    When ``inputs.bypass_models`` is True, CHTD and air-speed estimation are
    skipped entirely.  PMV uses the explicit override fields directly, falling
    back to 25 °C / 0.1 m/s defaults when an override is not provided.
    """
    x, u = _validate_chtd_vectors(inputs.x, inputs.u)

    met_driver, clo_driver, met_passenger, clo_passenger, driver_occ, passenger_occ = (
        _resolve_pipeline_occupants(inputs)
    )

    if inputs.bypass_models:
        return _run_comfort_pipeline_bypass(
            inputs, x, u,
            met_driver, clo_driver, met_passenger, clo_passenger,
            driver_occ, passenger_occ,
        )

    params = inputs.chtd_params if inputs.chtd_params is not None else CHTDParams()

    runtime_policy: Dict[str, Any] = {"applied": False}
    if inputs.apply_runtime_policy:
        x, u, runtime_policy = apply_two_row_vehicle_policy(x, u)
        runtime_policy["applied"] = True

    x_delta = compute_chtd_delta(x, u, params, mode=AS_FOUND)
    x_next = one_step_chtd(x, u, params, mode=AS_FOUND)
    x_comfort = x_next if inputs.use_next_state else x

    air_speed_in, air_speed_source = _resolve_air_speed_inputs(inputs, u)
    air_speeds = estimate_driver_passenger_air_speed(air_speed_in)
    flows_used = _flows_used_from_air_speed(air_speed_in)

    driver_air_temp = _driver_air_temp_c(x_comfort)
    passenger_air_temp = _passenger_air_temp_c(x_comfort)
    driver_mrt = _driver_mrt_c(x_comfort)
    passenger_mrt = _passenger_mrt_c(x_comfort)
    driver_head_fusion, passenger_head_fusion = fuse_front_row_heads(
        driver_air_temp,
        passenger_air_temp,
        inputs.image_inputs if inputs.use_ir_head_fusion else None,
        process_var=inputs.ir_process_var,
        measurement_var=inputs.ir_measurement_var,
        surface_to_air_offset_c=inputs.ir_surface_to_air_offset_c,
        min_confidence=inputs.ir_min_confidence,
        initial_variance=inputs.ir_initial_variance,
    )
    driver_pmv_air_temp, driver_air_temp_source = _resolve_pmv_air_temp(
        inputs.measured_driver_head_air_temp_c,
        driver_head_fusion.fused_temp_c,
    )
    passenger_pmv_air_temp, passenger_air_temp_source = _resolve_pmv_air_temp(
        inputs.measured_passenger_head_air_temp_c,
        passenger_head_fusion.fused_temp_c,
    )

    driver_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=driver_pmv_air_temp,
            mean_radiant_temp_c=driver_mrt,
            air_velocity_m_s=air_speeds.driver_air_speed_m_s,
            relative_humidity_pct=inputs.rh,
            metabolic_rate_met=met_driver,
            clothing_insulation_clo=clo_driver,
        )
    )
    passenger_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=passenger_pmv_air_temp,
            mean_radiant_temp_c=passenger_mrt,
            air_velocity_m_s=air_speeds.passenger_air_speed_m_s,
            relative_humidity_pct=inputs.rh,
            metabolic_rate_met=met_passenger,
            clothing_insulation_clo=clo_passenger,
        )
    )

    # ---- PMV debug: print detailed inputs every 20 calls ----
    _pmv_debug_counter = getattr(run_comfort_pipeline, "_pmv_debug_counter", 0) + 1
    run_comfort_pipeline._pmv_debug_counter = _pmv_debug_counter  # type: ignore[attr-defined]
    if _pmv_debug_counter == 1 or _pmv_debug_counter % 20 == 0:
        import sys as _sys
        pas_head_raw = x_comfort[X_INDEX["HeadTempFp"]]
        pas_cabin_raw = x_comfort[X_INDEX["CabinTempFp"]]
        pas_win_raw = x_comfort[X_INDEX["WinTempFp"]]
        pas_roof_raw = x_comfort[X_INDEX["RoofTemp"]]
        pas_win_mrt = _mrt_surface_or_cabin(pas_win_raw, pas_cabin_raw)
        pas_roof_mrt = _mrt_surface_or_cabin(pas_roof_raw, pas_cabin_raw)
        pas_win_clamped = pas_win_raw > MRT_HOT_SURFACE_THRESHOLD_C
        pas_roof_clamped = pas_roof_raw > MRT_HOT_SURFACE_THRESHOLD_C
        print(
            f"\n[PMV-DEBUG #{_pmv_debug_counter}] "
            f"========== 副驾 Passenger PMV 输入明细 ==========",
            file=_sys.stderr,
        )
        print(
            f"  PMV结果: PMV={passenger_comfort.pmv:+.3f}  PPD={passenger_comfort.ppd:.1f}%",
            file=_sys.stderr,
        )
        print(
            f"  ① air_temp_c       = {passenger_pmv_air_temp:.2f} °C  "
            f"(source={passenger_air_temp_source})",
            file=_sys.stderr,
        )
        print(
            f"  ② mean_radiant_temp = {passenger_mrt:.2f} °C",
            file=_sys.stderr,
        )
        print(
            f"     ├ HeadTempFp  = {pas_head_raw:.2f} °C",
            file=_sys.stderr,
        )
        print(
            f"     ├ CabinTempFp = {pas_cabin_raw:.2f} °C",
            file=_sys.stderr,
        )
        print(
            f"     ├ WinTempFp   = {pas_win_raw:.2f} °C"
            + (" ← 已截断(>35°C)" if pas_win_clamped else ""),
            file=_sys.stderr,
        )
        print(
            f"     └ RoofTemp    = {pas_roof_raw:.2f} °C"
            + (" ← 已截断(>35°C)" if pas_roof_clamped else ""),
            file=_sys.stderr,
        )
        print(
            f"  MRT计算值(截断后): mean({pas_head_raw:.1f}, "
            f"{pas_cabin_raw:.1f}, "
            f"{pas_win_mrt:.1f}, "
            f"{pas_roof_mrt:.1f})",
            file=_sys.stderr,
        )
        print(
            f"  ③ air_velocity     = {air_speeds.passenger_air_speed_m_s:.3f} m/s",
            file=_sys.stderr,
        )
        print(
            f"  ④ rh               = {inputs.rh:.1f} %",
            file=_sys.stderr,
        )
        print(
            f"  ⑤ met              = {met_passenger:.2f} met",
            file=_sys.stderr,
        )
        print(
            f"  ⑥ clo              = {clo_passenger:.2f} clo",
            file=_sys.stderr,
        )
        print(
            f"  操作温度 ≈ (ta+tr)/2 = {(passenger_pmv_air_temp + passenger_mrt) / 2:.2f} °C",
            file=_sys.stderr,
        )
        drv_head_raw = x_comfort[X_INDEX["HeadTempFd"]]
        drv_feet_raw = x_comfort[X_INDEX["FeetTempFd"]]
        pas_feet_raw = x_comfort[X_INDEX["FeetTempFp"]]
        print(
            f"  主驾头温={drv_head_raw:.1f}°C  主驾脚温={drv_feet_raw:.1f}°C  "
            f"副驾脚温={pas_feet_raw:.1f}°C  "
            f"主驾PMV={driver_comfort.pmv:+.3f}",
            file=_sys.stderr,
        )
        print(
            f"  MRT阈值={MRT_HOT_SURFACE_THRESHOLD_C}°C  "
            f"头脚温差={pas_head_raw - pas_feet_raw:.1f}°C",
            file=_sys.stderr,
        )
        print(
            f"==============================================================\n",
            file=_sys.stderr,
        )

    impl_modes = _build_implementation_modes()
    inputs_used: Dict[str, Any] = {
        "use_next_state": inputs.use_next_state,
        "rh": inputs.rh,
        "met_driver": met_driver,
        "clo_driver": clo_driver,
        "met_passenger": met_passenger,
        "clo_passenger": clo_passenger,
        "air_speed_source": air_speed_source,
        "flows_used": flows_used,
        "occupant_driver": _occupant_resolved_to_dict(driver_occ),
        "occupant_passenger": _occupant_resolved_to_dict(passenger_occ),
        "ir_head_surface_temp_driver_c": driver_occ.ir_head_surface_temp_c,
        "ir_head_surface_temp_passenger_c": passenger_occ.ir_head_surface_temp_c,
        "ir_fusion_enabled": inputs.use_ir_head_fusion,
        "ir_head_fusion_enabled": inputs.use_ir_head_fusion,
        "ir_process_var": inputs.ir_process_var,
        "ir_measurement_var": inputs.ir_measurement_var,
        "ir_surface_to_air_offset_c": inputs.ir_surface_to_air_offset_c,
        "ir_min_confidence": inputs.ir_min_confidence,
        "driver_ir_fusion": _head_fusion_to_dict(
            driver_head_fusion,
            ir_surface_temp_c=driver_occ.ir_head_surface_temp_c,
        ),
        "passenger_ir_fusion": _head_fusion_to_dict(
            passenger_head_fusion,
            ir_surface_temp_c=passenger_occ.ir_head_surface_temp_c,
        ),
        "driver_head_temp_fusion": _head_fusion_to_dict(
            driver_head_fusion,
            ir_surface_temp_c=driver_occ.ir_head_surface_temp_c,
        ),
        "passenger_head_temp_fusion": _head_fusion_to_dict(
            passenger_head_fusion,
            ir_surface_temp_c=passenger_occ.ir_head_surface_temp_c,
        ),
        "driver_air_temp_model_c": driver_air_temp,
        "passenger_air_temp_model_c": passenger_air_temp,
        "driver_measured_head_air_temp_c": inputs.measured_driver_head_air_temp_c,
        "passenger_measured_head_air_temp_c": inputs.measured_passenger_head_air_temp_c,
        "driver_air_temp_source": driver_air_temp_source,
        "passenger_air_temp_source": passenger_air_temp_source,
        "driver_air_temp_c": driver_pmv_air_temp,
        "passenger_air_temp_c": passenger_pmv_air_temp,
        "driver_mrt_c": driver_mrt,
        "passenger_mrt_c": passenger_mrt,
        "driver_mrt_source": "model_state",
        "passenger_mrt_source": "model_state",
        "driver_air_speed_m_s": air_speeds.driver_air_speed_m_s,
        "passenger_air_speed_m_s": air_speeds.passenger_air_speed_m_s,
        "runtime_policy": runtime_policy,
        "chtd_param_mode": inputs.chtd_param_mode,
        "safe_preview_active": inputs.safe_preview_active,
        "phase3_engineering_preview_active": inputs.phase3_engineering_preview_active,
        "phase3_capacity_active": inputs.phase3_capacity_active,
        "solar_shell_routing_active": inputs.solar_shell_routing_active,
        "actuator_distribution_active": inputs.actuator_distribution_active,
        "foot_leakage_active": inputs.foot_leakage_active,
        "classification": inputs.phase3_classification or (
            (inputs.params_provenance or {}).get("classification")
        ),
        "params_provenance": inputs.params_provenance,
    }

    return PipelineResult(
        x_next=x_next,
        x_delta=x_delta,
        driver=SeatComfortResult(
            air_temp_c=driver_pmv_air_temp,
            mean_radiant_temp_c=driver_mrt,
            air_speed_m_s=air_speeds.driver_air_speed_m_s,
            pmv=driver_comfort.pmv,
            ppd=driver_comfort.ppd,
            valid=driver_occ.occupied,
        ),
        passenger=SeatComfortResult(
            air_temp_c=passenger_pmv_air_temp,
            mean_radiant_temp_c=passenger_mrt,
            air_speed_m_s=air_speeds.passenger_air_speed_m_s,
            pmv=passenger_comfort.pmv,
            ppd=passenger_comfort.ppd,
            valid=passenger_occ.occupied,
        ),
        trace_status=_trace_status_summary(impl_modes),
        implementation_modes=impl_modes,
        inputs_used=inputs_used,
    )


def _seat_comfort_to_dict(seat: SeatComfortResult) -> Dict[str, Any]:
    return {
        "air_temp_c": float(seat.air_temp_c),
        "mean_radiant_temp_c": float(seat.mean_radiant_temp_c),
        "air_speed_m_s": float(seat.air_speed_m_s),
        "pmv": float(seat.pmv),
        "ppd": float(seat.ppd),
        "valid": bool(seat.valid),
    }


def _to_json_safe(value: Any) -> Any:
    """Convert pipeline values to JSON-serializable Python builtins."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return [float(v) for v in np.asarray(value, dtype=float).tolist()]
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return int(item)
        return float(item)
    if isinstance(value, SeatComfortResult):
        return _seat_comfort_to_dict(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    raise TypeError(
        f"pipeline: value not JSON-serializable: {type(value)!r}"
    )


def pipeline_result_to_dict(result: PipelineResult) -> Dict[str, Any]:
    """Convert a ``PipelineResult`` to a JSON-safe plain dict."""
    x_delta = np.asarray(result.x_delta, dtype=float)
    x_next = np.asarray(result.x_next, dtype=float)
    return {
        "trace_status": result.trace_status,
        "driver": _seat_comfort_to_dict(result.driver),
        "passenger": _seat_comfort_to_dict(result.passenger),
        "x_delta": [float(v) for v in x_delta.tolist()],
        "x_next": [float(v) for v in x_next.tolist()],
        "x_delta_summary": {
            "min": float(np.min(x_delta)),
            "max": float(np.max(x_delta)),
            "mean": float(np.mean(x_delta)),
        },
        "implementation_modes": {
            mode: list(names)
            for mode, names in result.implementation_modes.items()
        },
        "inputs_used": _to_json_safe(result.inputs_used),
    }


def run_comfort_pipeline_dict(inputs: PipelineInputs) -> Dict[str, Any]:
    """Run the comfort pipeline and return a JSON-safe result dict."""
    return pipeline_result_to_dict(run_comfort_pipeline(inputs))
