"""Runtime upstream → ``PipelineInputs`` adapter (T10).

Maps **SIG / AFE / FTE / image-module** engineering quantities onto CHTD
``x[28]`` / ``u[54]``. PMV does **not** parse CAN or DBC.

Expected producers:
- **SIG**: ``amb_t_c``, ``raw_amb_t_c``, ``vehicle_speed_kph`` (km/h),
  ``rh_percent`` (%), ``solar_driver_w_m2`` / ``solar_passenger_w_m2`` or
  single ``solar_w_m2`` duplicated to both sides
- **AFE / FTE**: vent ``*_flow`` (m³/h), ``eva_t_c``, optional ``tma`` map
- **Image / FTE**: ``image_inputs`` v1 — ``ir_head_surface_temp_c`` per seat;
  ``confidence`` optional (gates IR fusion when present); optional
  ``occupied``, ``clothing_class``, ``activity_met``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from hvac_sim.chtd.bus_index import CHTD_U_NAMES, N_U, N_X_STATES, U_INDEX, X_INDEX
from hvac_sim.config.geometry import merge_geometry_with_air_speed_inputs
from hvac_sim.defrost import estimate_tma_def
from hvac_sim.defrost_post_glass import (
    DefrostPostGlassInputs,
    estimate_defrost_post_glass_temperature,
)
from hvac_sim.glass_temp import estimate_windshield_glass_temp_step
from hvac_sim.pipeline import (
    PipelineInputs,
    air_speed_inputs_from_chtd_u,
)
from hvac_sim.runtime_parameter_mode import (
    CHTD_PARAM_MODE_DEFAULT,
    CHTD_PARAM_MODE_PHASE3,
    CHTD_PARAM_MODE_SAFE_PREVIEW,
    SAFE_PREVIEW_NOT_VEHICLE_CALIBRATION,
    VALID_CHTD_PARAM_MODES,
    apply_chtd_parameter_mode,
    resolve_chtd_param_mode,
)
from hvac_sim.runtime_input import (
    DEFAULT_RH_PERCENT as _DEFAULT_RH_PERCENT,
    RuntimeFeatureFlags,
    RuntimeSignalInput,
    as_optional_float as _as_optional_float,
    feature_flags_from_signal,
    first_present as _first_present,
    resolve_runtime_feature_flags,
    runtime_mapping_to_signal_input,
)

RUNTIME_INPUT_SCHEMA_VERSION = "runtime_signal_input_v1"
RUNTIME_PMV_OUTPUT_SCHEMA_VERSION = "runtime_pmv_output_v1"

PHASE3_NOT_VEHICLE_CALIBRATION = "phase3_engineering_preview_not_vehicle_calibration"

_DEFAULT_VEH_SPD_KPH = 0.0

_FLOW_RUNTIME_TO_U = {
    "driver_face_flow": "FrntFdvFlow",
    "passenger_face_flow": "FrntFpvFlow",
    "driver_floor_flow": "FrntFdfFlow",
    "passenger_floor_flow": "FrntFpfFlow",
    "driver_defrost_flow": "FrntFdDefFlow",
    "passenger_defrost_flow": "FrntFpDefFlow",
    "frnt_fd_def_flow": "FrntFdDefFlow",
    "frnt_fdv_flow": "FrntFdvFlow",
    "frnt_fdf_flow": "FrntFdfFlow",
    "frnt_fp_def_flow": "FrntFpDefFlow",
    "frnt_fpv_flow": "FrntFpvFlow",
    "frnt_fpf_flow": "FrntFpfFlow",
    "frnt_sdv_flow": "FrntSdvFlow",
    "frnt_sdf_flow": "FrntSdfFlow",
    "frnt_spv_flow": "FrntSpvFlow",
    "frnt_spf_flow": "FrntSpfFlow",
}


@dataclass(frozen=True)
class RuntimeAdapterResult:
    """Adapter output: pipeline inputs plus integration diagnostics."""

    pipeline_inputs: PipelineInputs
    provenance: Dict[str, Any]
    warnings: tuple[str, ...]


def make_cabin_state_vector(
    *,
    base_c: float = 24.0,
    head_fd_c: Optional[float] = None,
    head_fp_c: Optional[float] = None,
    cabin_fd_c: Optional[float] = None,
    cabin_fp_c: Optional[float] = None,
    roof_c: Optional[float] = None,
) -> List[float]:
    """Build a 28-state CHTD vector with uniform base and optional zone overrides."""
    x = [float(base_c)] * N_X_STATES
    if head_fd_c is not None:
        x[3] = float(head_fd_c)
    if head_fp_c is not None:
        x[4] = float(head_fp_c)
    if cabin_fd_c is not None:
        x[15] = float(cabin_fd_c)
    if cabin_fp_c is not None:
        x[16] = float(cabin_fp_c)
    if roof_c is not None:
        x[27] = float(roof_c)
    return x


def _resolve_solar_w_m2(
    raw: Mapping[str, Any],
    solar_driver: Optional[float],
    solar_passenger: Optional[float],
) -> tuple[float, float, Dict[str, Any]]:
    """Resolve left/right solar; duplicate single ``solar_w_m2`` to both sides."""
    single = _as_optional_float(
        _first_present(
            raw,
            "solar_w_m2",
            "solar_total_w_m2",
            "sig_solar_w_m2",
            "solar_irradiance_w_m2",
        )
    )
    meta: Dict[str, Any] = {}
    if solar_driver is not None and solar_passenger is not None:
        return float(solar_driver), float(solar_passenger), meta
    if single is not None:
        meta["source"] = "single_sensor_duplicated"
        meta["solar_w_m2"] = single
        return float(single), float(single), meta
    if solar_driver is not None:
        meta["source"] = "driver_only_passenger_zero"
        return float(solar_driver), 0.0, meta
    if solar_passenger is not None:
        meta["source"] = "passenger_only_driver_zero"
        return 0.0, float(solar_passenger), meta
    return 0.0, 0.0, meta


def _build_x_vector(
    amb_t: float,
    previous_state: Optional[Sequence[float]],
    provenance: Dict[str, Any],
) -> np.ndarray:
    if previous_state is not None:
        x = np.asarray(previous_state, dtype=float)
        if x.shape != (N_X_STATES,):
            raise ValueError(
                f"previous_state must have length {N_X_STATES}, got {x.shape}"
            )
        provenance["x_source"] = "previous_state"
        return x

    provenance["x_source"] = "ambient_uniform"
    return np.full(N_X_STATES, float(amb_t))


def _resolve_eva_t(signal: RuntimeSignalInput, amb_t: float) -> float:
    if signal.eva_t_c is not None:
        return float(signal.eva_t_c)
    return float(amb_t)


def _compute_frnt_def_tma_est(
    signal: RuntimeSignalInput,
    eva_t: float,
    amb_t: float,
    provenance: Dict[str, Any],
    warnings: List[str],
) -> float:
    if signal.frnt_def_tma_est_c is not None:
        provenance["TmaDef"] = {
            "source": "runtime.frnt_def_tma_est_c",
            "TmaDef_approx": False,
        }
        return float(signal.frnt_def_tma_est_c)

    if signal.hct_c is None:
        provenance["Hct"] = {
            "status": "precision_fallback",
            "fallback": "eva_t",
            "source": "AFE/SIG",
            "note": "Optional at runtime; improves TmaDef when supplied.",
        }
        warnings.append(
            "Hct missing (AFE/SIG): heater-core outlet not supplied; "
            "FrntDefTmaEst uses EvaT/AmbT precision fallback."
        )
    if signal.posn_fdh is None and signal.blend_request is None:
        provenance["PosnFdh"] = {
            "status": "precision_fallback",
            "fallback": "eva_t_only",
            "note": "Optional at runtime; improves TmaDef when supplied.",
        }
        warnings.append(
            "PosnFdh missing (AFE): defrost mix position not supplied; "
            "FrntDefTmaEst uses EvaT/AmbT precision fallback."
        )

    tma = estimate_tma_def(
        eva_temp_c=float(eva_t),
        hct_temp_c=signal.hct_c,
        posn_fdh=signal.posn_fdh,
        blend_request=signal.blend_request,
        fallback="eva",
    )
    provenance["TmaDef"] = {
        "source": tma.method,
        "TmaDef_approx": tma.tmadef_approx,
        "posn_fdh": tma.posn_fdh,
        "hct_c": tma.hct_temp_c,
        "eva_t_c": tma.eva_temp_c,
        **tma.diagnostics,
    }
    if tma.tmadef_approx:
        provenance["TmaDef"]["status"] = "precision_fallback"
        provenance["TmaDef"]["fallback"] = (
            "eva_t" if signal.eva_t_c is not None else "amb_t"
        )
        warnings.append(
            "TmaDef precision fallback: FrntDefTmaEst set to "
            + ("EvaT" if signal.eva_t_c is not None else "AmbT")
            + " (PMV still runs; heating+defrost accuracy reduced)."
        )
    return float(tma.tma_def_c)


def _resolve_defrost_flow_m3h(signal: RuntimeSignalInput) -> float:
    if signal.driver_defrost_flow is not None:
        return float(signal.driver_defrost_flow)
    if "FrntFdDefFlow" in signal.flows:
        return float(signal.flows["FrntFdDefFlow"])
    if "frnt_fd_def_flow" in signal.flows:
        return float(signal.flows["frnt_fd_def_flow"])
    return 0.0


def _compute_win_shd_t_est(
    signal: RuntimeSignalInput,
    x: np.ndarray,
    frnt_def_tma: float,
    amb_t: float,
    veh_spd_kph: float,
    solar_fd: float,
    solar_fp: float,
    provenance: Dict[str, Any],
    warnings: List[str],
) -> float:
    if signal.win_shd_t_est_c is not None:
        provenance["WinShdTEst"] = {
            "source": "runtime.win_shd_t_est_c",
            "estimated": False,
        }
        return float(signal.win_shd_t_est_c)

    glass_init = signal.windshield_glass_temp_c
    if glass_init is None:
        glass_init = float(x[X_INDEX["WinTempFd"]])
        if abs(glass_init) < 1e-9:
            glass_init = amb_t

    defrost_flow = _resolve_defrost_flow_m3h(signal)
    cabin_t = float(x[X_INDEX["CabinFrntTemp"]]) if x.shape == (N_X_STATES,) else amb_t
    dash_t = float(x[X_INDEX["ConsoleTemp"]]) if x.shape == (N_X_STATES,) else amb_t
    solar = max(float(solar_fd), float(solar_fp))

    try:
        step = estimate_windshield_glass_temp_step(
            glass_temp_c=float(glass_init),
            tma_def_c=float(frnt_def_tma),
            defrost_flow_m3h=defrost_flow,
            cabin_temp_c=cabin_t,
            dashboard_temp_c=dash_t,
            ambient_temp_c=float(amb_t),
            vehicle_speed_kph=float(veh_spd_kph),
            solar_w_m2=solar,
        )
    except (TypeError, ValueError) as exc:
        provenance["WinShdTEst"] = {"status": "error", "error": str(exc)}
        warnings.append(f"WinShdTEst glass observer failed: {exc}; using 0.")
        return 0.0

    provenance["WinShdTEst"] = {
        "source": "estimate_windshield_glass_temp_step",
        "estimated": True,
        "glass_temp_init_c": float(glass_init),
        "delta_c": step.delta_c,
        "heat_terms": dict(step.heat_terms),
        **step.diagnostics,
    }
    if signal.windshield_glass_temp_c is None and signal.win_shd_t_est_c is None:
        warnings.append(
            "WinShdTEst estimated via first-order glass observer "
            "(windshield_glass_temp_c not supplied; init from state or AmbT)."
        )
    return float(step.glass_temp_next_c)


def _compute_defrost_post_glass_diagnostic(
    signal: RuntimeSignalInput,
    x: np.ndarray,
    frnt_def_tma: float,
    amb_t: float,
    veh_spd_kph: float,
    provenance: Dict[str, Any],
) -> None:
    """Record post-glass defrost effective temperature in provenance only."""
    glass = signal.windshield_glass_temp_c
    if glass is None:
        glass = float(x[X_INDEX["WinTempFd"]]) if x.shape == (N_X_STATES,) else amb_t
        if abs(glass) < 1e-9:
            glass = amb_t

    defrost_flow = _resolve_defrost_flow_m3h(signal)
    cabin_t = float(x[X_INDEX["CabinFrntTemp"]]) if x.shape == (N_X_STATES,) else amb_t

    try:
        result = estimate_defrost_post_glass_temperature(
            DefrostPostGlassInputs(
                tma_def_outlet_c=float(frnt_def_tma),
                windshield_glass_temp_c=float(glass),
                defrost_flow_m3h=float(defrost_flow),
                cabin_air_temp_c=cabin_t,
                vehicle_speed_kph=float(veh_spd_kph),
            )
        )
        provenance["DefrostPostGlass"] = {
            "t_after_glass_c": result.t_after_glass_c,
            "eta": result.eta,
            "diagnostic_only": True,
            "note": "Not written to CHTD u/x; for head effective-temperature review.",
            **result.provenance,
        }
    except (TypeError, ValueError) as exc:
        provenance["DefrostPostGlass"] = {
            "status": "error",
            "error": str(exc),
            "diagnostic_only": True,
        }


def _fill_tma_bus(
    u: np.ndarray,
    signal: RuntimeSignalInput,
    eva_t: float,
    amb_t: float,
    frnt_def_tma: float,
    provenance: Dict[str, Any],
    warnings: List[str],
) -> None:
    tma_values: Dict[str, float] = {}
    for name in CHTD_U_NAMES:
        if not name.endswith("Tma") and name != "FrntDefTmaEst":
            continue
        if name == "FrntDefTmaEst":
            tma_values[name] = frnt_def_tma
            continue
        if name in signal.tma_c:
            tma_values[name] = float(signal.tma_c[name])
            continue
        snake = "".join(
            ["_" + c.lower() if c.isupper() else c for c in name]
        ).lstrip("_")
        alt = f"{snake}_c"
        if alt in signal.tma_c:
            tma_values[name] = float(signal.tma_c[alt])
            continue
        tma_values[name] = eva_t

    for name, value in tma_values.items():
        u[U_INDEX[name]] = value

    provenance["tma_fallback"] = {
        "default": "eva_t_or_amb_t",
        "eva_t_c": eva_t,
        "amb_t_c": amb_t,
        "explicit_count": len(signal.tma_c),
    }
    if not signal.tma_c:
        warnings.append(
            "Tma not supplied (AFE/FTE): duct supply temperatures filled from "
            "EvaT or AmbT fallback."
        )


def _fill_flow_bus(
    u: np.ndarray,
    signal: RuntimeSignalInput,
    provenance: Dict[str, Any],
    warnings: List[str],
) -> None:
    mapped: Dict[str, float] = {}
    for runtime_key, u_name in _FLOW_RUNTIME_TO_U.items():
        value = getattr(signal, runtime_key, None)
        if value is None and runtime_key in signal.flows:
            value = signal.flows[runtime_key]
        if value is not None:
            mapped[u_name] = float(value)

    for u_name, value in mapped.items():
        u[U_INDEX[u_name]] = value

    provenance["flows_mapped"] = dict(mapped)
    if not mapped:
        provenance["vent_flows"] = {
            "status": "missing",
            "source": "AFE/FTE",
            "note": "PMV does not decode CAN; supply m³/h flows explicitly.",
        }
        warnings.append(
            "Vent flows missing (AFE/FTE): no face/floor/defrost flows supplied; "
            "unspecified u-bus flow slots remain 0."
        )
    else:
        provenance["vent_flows"] = {"status": "explicit", "count": len(mapped)}


def _apply_chtd_parameter_mode(
    runtime: Union[Mapping[str, Any], RuntimeSignalInput],
    signal: RuntimeSignalInput,
    pipeline_kw: Dict[str, Any],
    provenance: Dict[str, Any],
    warnings: List[str],
) -> None:
    raw = runtime if isinstance(runtime, Mapping) else {"chtd_param_mode": signal.chtd_param_mode}
    apply_chtd_parameter_mode(raw, signal, pipeline_kw, provenance, warnings)

def _apply_direct_pmv_overrides(
    signal: RuntimeSignalInput,
    pipeline_kw: Dict[str, Any],
    provenance: Dict[str, Any],
) -> None:
    """Copy the explicit bypass contract without interpreting its values."""
    pipeline_kw["bypass_models"] = signal.bypass_models
    for field_name in (
        "driver_air_temp_override_c",
        "passenger_air_temp_override_c",
        "driver_mrt_override_c",
        "passenger_mrt_override_c",
        "driver_air_speed_override_m_s",
        "passenger_air_speed_override_m_s",
    ):
        value = getattr(signal, field_name)
        if value is not None:
            pipeline_kw[field_name] = value
    provenance["bypass_models"] = signal.bypass_models


def build_runtime_pipeline_inputs(
    runtime: Union[Mapping[str, Any], RuntimeSignalInput],
    previous_state: Optional[Sequence[float]] = None,
) -> RuntimeAdapterResult:
    """Build ``PipelineInputs`` and integration diagnostics from runtime dict."""
    signal = (
        runtime
        if isinstance(runtime, RuntimeSignalInput)
        else runtime_mapping_to_signal_input(runtime)
    )

    provenance: Dict[str, Any] = {"schema": "runtime_adapter_v1"}
    warnings: List[str] = []
    flags = feature_flags_from_signal(signal)
    provenance["feature_flags"] = flags.to_dict()
    provenance["disabled_features"] = flags.disabled_feature_names
    provenance["active_features"] = flags.active_feature_names
    if not flags.enable_corrected_chtd:
        provenance["chtd_mode"] = "AS_FOUND"
    else:
        provenance["chtd_mode"] = "CORRECTED_requested_not_applied_in_runtime"
        warnings.append(
            "enable_corrected_chtd=true ignored at runtime adapter; pipeline still AS_FOUND."
        )
    if not flags.enable_image_occupancy:
        provenance["image_module"] = "disabled"
    elif signal.image_inputs is None:
        provenance["image_module"] = "absent"
    else:
        provenance["image_module"] = "present"

    amb_t = float(signal.amb_t_c)
    raw_amb = signal.raw_amb_t_c if signal.raw_amb_t_c is not None else amb_t
    if signal.raw_amb_t_c is None:
        provenance["RawAmbT"] = {"status": "missing", "fallback": "amb_t_c"}
        warnings.append(
            "RawAmbT missing (SIG): using amb_t_c for RawAmbT u-bus slot."
        )

    rh_provided = (
        not isinstance(runtime, RuntimeSignalInput)
        and _first_present(runtime, "rh_percent", "rh", "relative_humidity_percent")
        is not None
    )
    rh = float(signal.rh_percent)
    if not rh_provided:
        provenance["RH"] = {
            "status": "default",
            "value_percent": _DEFAULT_RH_PERCENT,
        }
        warnings.append(
            f"RH default (SIG): relative humidity set to {_DEFAULT_RH_PERCENT}%."
        )

    veh_spd = (
        float(signal.vehicle_speed_kph)
        if signal.vehicle_speed_kph is not None
        else _DEFAULT_VEH_SPD_KPH
    )
    if signal.vehicle_speed_kph is None:
        provenance["VehSpd"] = {
            "status": "missing",
            "fallback_kph": _DEFAULT_VEH_SPD_KPH,
            "source": "SIG",
        }
        warnings.append(
            "VehSpd missing (SIG): vehicle speed set to 0 km/h (parked default)."
        )
    else:
        provenance["VehSpd"] = {"status": "sig", "value_kph": veh_spd}

    raw_map: Mapping[str, Any] = runtime if isinstance(runtime, Mapping) else {}
    solar_fd, solar_fp, solar_meta = _resolve_solar_w_m2(
        raw_map,
        signal.solar_driver_w_m2,
        signal.solar_passenger_w_m2,
    )
    provenance["Solar"] = {
        "solar_fd_w_m2": solar_fd,
        "solar_fp_w_m2": solar_fp,
        **solar_meta,
    }
    if (
        signal.solar_driver_w_m2 is None
        and signal.solar_passenger_w_m2 is None
        and solar_meta.get("source") != "single_sensor_duplicated"
    ):
        provenance["Solar"]["status"] = "missing"
        provenance["Solar"]["fallback_w_m2"] = 0.0
        warnings.append("Solar missing (SIG): SolarFd/SolarFp set to 0 W/m².")

    x = _build_x_vector(amb_t, previous_state, provenance)
    u = np.zeros(N_U, dtype=float)

    u[U_INDEX["ICT"]] = float(signal.ict_c if signal.ict_c is not None else amb_t)
    u[U_INDEX["RawAmbT"]] = float(raw_amb)
    u[U_INDEX["AmbT"]] = amb_t
    u[U_INDEX["SolarFd"]] = solar_fd
    u[U_INDEX["SolarFp"]] = solar_fp
    u[U_INDEX["VehSpd"]] = veh_spd

    eva_t = _resolve_eva_t(signal, amb_t)
    frnt_def_tma = _compute_frnt_def_tma_est(signal, eva_t, amb_t, provenance, warnings)
    u[U_INDEX["WinShdTEst"]] = _compute_win_shd_t_est(
        signal,
        x,
        frnt_def_tma,
        amb_t,
        veh_spd,
        solar_fd,
        solar_fp,
        provenance,
        warnings,
    )
    if flags.enable_post_glass_defrost:
        _compute_defrost_post_glass_diagnostic(
            signal, x, frnt_def_tma, amb_t, veh_spd, provenance
        )
    else:
        provenance["DefrostPostGlass"] = {
            "status": "disabled",
            "diagnostic_only": True,
            "note": "Set enable_post_glass_defrost=true to compute post-glass T.",
        }
    _fill_tma_bus(u, signal, eva_t, amb_t, frnt_def_tma, provenance, warnings)
    _fill_flow_bus(u, signal, provenance, warnings)

    image_for_pipeline = (
        signal.image_inputs if flags.enable_image_occupancy else None
    )
    use_ir_fusion = bool(
        flags.enable_ir_fusion and flags.enable_image_occupancy
    )

    pipeline_kw: Dict[str, Any] = {
        "x": x,
        "u": u,
        "rh": rh,
        "image_inputs": image_for_pipeline,
        "use_ir_head_fusion": use_ir_fusion,
        "apply_runtime_policy": signal.apply_runtime_policy,
        "use_next_state": signal.use_next_state,
    }
    for attr in (
        "driver_face_flow",
        "passenger_face_flow",
        "driver_floor_flow",
        "passenger_floor_flow",
        "driver_defrost_flow",
        "passenger_defrost_flow",
    ):
        value = getattr(signal, attr)
        if value is not None:
            pipeline_kw[attr] = value
    if signal.measured_driver_head_air_temp_c is not None:
        pipeline_kw["measured_driver_head_air_temp_c"] = (
            signal.measured_driver_head_air_temp_c
        )
    if signal.measured_passenger_head_air_temp_c is not None:
        pipeline_kw["measured_passenger_head_air_temp_c"] = (
            signal.measured_passenger_head_air_temp_c
        )

    if flags.enable_geometry_merge and signal.apply_geometry:
        base_air = air_speed_inputs_from_chtd_u(u)
        geom = merge_geometry_with_air_speed_inputs(
            base_air,
            geometry_path=signal.geometry_path,
        )
        pipeline_kw["air_speed"] = geom.air_speed
        provenance["geometry"] = geom.provenance
        if geom.provenance.get("placeholder_field_names"):
            warnings.append(
                "Geometry placeholder (estimated): outlet areas/distances use "
                "packaged measured=false defaults; replace when bench data exists."
            )

    _apply_chtd_parameter_mode(
        runtime,
        signal,
        pipeline_kw,
        provenance,
        warnings,
    )

    _apply_direct_pmv_overrides(signal, pipeline_kw, provenance)

    pipeline_inputs = PipelineInputs(**pipeline_kw)
    provenance["ir_fusion_enabled"] = use_ir_fusion
    provenance["enable_ir_fusion_alias"] = flags.enable_ir_fusion

    return RuntimeAdapterResult(
        pipeline_inputs=pipeline_inputs,
        provenance=provenance,
        warnings=tuple(warnings),
    )


def build_pipeline_inputs_from_runtime(
    runtime: Mapping[str, Any],
    previous_state: Optional[Sequence[float]] = None,
) -> PipelineInputs:
    """Convert runtime upstream dict to ``PipelineInputs`` (no x/u bus details)."""
    return build_runtime_pipeline_inputs(runtime, previous_state).pipeline_inputs


def runtime_input_to_dict_example() -> Dict[str, Any]:
    """JSON-friendly example runtime payload for integration docs/tests."""
    return {
        "schema": RUNTIME_INPUT_SCHEMA_VERSION,
        "description": (
            "SIG/AFE/FTE engineering quantities; PMV does not parse CAN. "
            "Measured cabin temperature time series is not required for runtime PMV "
            "(required only for CHTD calibration / PSO)."
        ),
        "amb_t_c": 24.0,
        "raw_amb_t_c": 24.0,
        "rh_percent": 50.0,
        "vehicle_speed_kph": 30.0,
        "solar_driver_w_m2": 40.0,
        "solar_passenger_w_m2": 35.0,
        "eva_t_c": 18.0,
        "driver_face_flow": 95.0,
        "passenger_face_flow": 90.0,
        "driver_floor_flow": 55.0,
        "passenger_floor_flow": 50.0,
        "tma": {
            "FrntFdvTma": 20.0,
            "FrntFpvTma": 20.0,
            "FrntFdfTma": 19.0,
            "FrntFpfTma": 19.0,
        },
        "previous_state": make_cabin_state_vector(
            base_c=25.0,
            head_fd_c=25.2,
            head_fp_c=25.1,
            cabin_fd_c=25.0,
            cabin_fp_c=24.8,
            roof_c=25.5,
        ),
        "image_inputs": {
            "driver": {
                "occupied": True,
                "confidence": 0.85,
                "ir_head_surface_temp_c": 33.5,
            },
            "passenger": {
                "occupied": True,
                "confidence": 0.8,
                "ir_head_surface_temp_c": 33.0,
            },
        },
        "use_ir_head_fusion": False,
        "apply_geometry": True,
        "use_next_state": True,
    }


def runtime_signal_input_to_dict(signal: RuntimeSignalInput) -> Dict[str, Any]:
    """Serialize ``RuntimeSignalInput``; image_inputs as nested dicts."""
    data = asdict(signal)
    if signal.image_inputs is not None:
        data["image_inputs"] = {
            "driver": asdict(signal.image_inputs.driver),
            "passenger": asdict(signal.image_inputs.passenger),
        }
    return data
