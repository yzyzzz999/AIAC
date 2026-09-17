"""AFE v2 + CHTD + PMV multi-step warm-up preview (experimental, not runtime).

Integrates ``compute_chtd_delta`` over multiple steps from a cold initial state
to diagnose heating warm-up trajectories. Does **not** wire ``runtime_pipeline``
or modify ``thermal.py`` / PMV core algorithms.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.afe.v2_chtd_preview import (
    _diagnostic_from_case,
    _initial_state_x0,
    _x_delta_summary,
    build_tma_source,
)
from hvac_sim.afe.v2_pmv_preview import _air_speed_inputs_from_u_preview
from hvac_sim.afe.v2_pmv_replay import case_to_afe_case, case_to_tma_source
from hvac_sim.afe.v2_to_chtd_adapter import build_chtd_u_preview_from_afe_v2
from hvac_sim.air_speed import AirSpeedInputs, estimate_driver_passenger_air_speed
from hvac_sim.chtd.bus_index import CHTD_X_NAMES, N_X_STATES, U_INDEX, X_INDEX
from hvac_sim.chtd.params import CHTDParams
from hvac_sim.chtd.thermal import compute_chtd_delta
from hvac_sim.pipeline import (
    _driver_air_temp_c,
    _driver_mrt_c,
    _passenger_air_temp_c,
    _passenger_mrt_c,
)
from hvac_sim.pmv.applicability import evaluate_pmv_applicability
from hvac_sim.pmv.interface import VehicleComfortInputs, compute_vehicle_pmv

WARMUP_SCHEMA = "afe_v2_pmv_warmup_preview_v1"
CLASSIFICATION = "experimental_offline_warmup_preview_not_runtime"

_X_TEMP_MIN = -40.0
_X_TEMP_MAX = 90.0
_DELTA_ABS_LIMIT_C = 20.0

_INTEGRATION_FULL = "full_state"
_INTEGRATION_HEAD_FEET = "head_feet_preview_stable"

_HEAD_FEET_INTEGRATE_ZONES = frozenset(
    name for name in CHTD_X_NAMES if name.startswith("Head") or name.startswith("Feet")
)
_MRT_SYNC_ZONES = ("CabinTempFd", "WinTempFd", "RoofTemp")

_DEFAULT_STEPS = 600
_DEFAULT_DT_SECONDS = 1.0
_DEFAULT_SAMPLE_EVERY = 60

_BUILTIN_SCENARIOS: Tuple[Dict[str, Any], ...] = (
    {
        "case_id": "cold_foot_heating_10min",
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
        "integration_mode": _INTEGRATION_HEAD_FEET,
        "steps": _DEFAULT_STEPS,
        "dt_seconds": _DEFAULT_DT_SECONDS,
        "sample_every": _DEFAULT_SAMPLE_EVERY,
    },
    {
        "case_id": "cold_foot_defrost_heating_10min",
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
        "integration_mode": _INTEGRATION_HEAD_FEET,
        "steps": _DEFAULT_STEPS,
        "dt_seconds": _DEFAULT_DT_SECONDS,
        "sample_every": _DEFAULT_SAMPLE_EVERY,
    },
)


def default_builtin_warmup_scenarios() -> List[Dict[str, Any]]:
    return [dict(c) for c in _BUILTIN_SCENARIOS]


def _repo_python_targets() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "simulink_conversion_package"
        / "python_targets"
    )


def _resolve_case_and_tma(
    case: Mapping[str, Any],
    *,
    tma_source: Optional[Mapping[str, Any]] = None,
    amb_t: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    if "amb_t" in case or "driver_face_tma" in case:
        afe_case = case_to_afe_case(case)
        tma = dict(tma_source) if tma_source is not None else case_to_tma_source(case)
    else:
        afe_case = dict(case)
        resolved_amb = float(amb_t if amb_t is not None else afe_case.get("temp_C", 24.0))
        if tma_source is not None:
            tma = dict(tma_source)
        else:
            tma = build_tma_source(
                amb_t=resolved_amb,
                heating_mode=bool(afe_case.get("heating_mode", False)),
            )
    amb = float(tma.get("amb_t", amb_t if amb_t is not None else -5.0))
    rh = float(case.get("rh", case.get("rh_percent", 50.0)))
    return afe_case, tma, amb, rh


def _x_out_of_bounds_zones(x: np.ndarray) -> List[str]:
    from hvac_sim.chtd.bus_index import CHTD_X_NAMES

    return [
        CHTD_X_NAMES[i]
        for i, val in enumerate(x)
        if float(val) < _X_TEMP_MIN or float(val) > _X_TEMP_MAX
    ]


def _x_for_pmv_evaluation(x: np.ndarray, *, integration_mode: str) -> np.ndarray:
    """Optional PMV-only MRT zone sync (does not alter integrated state)."""
    if integration_mode != _INTEGRATION_HEAD_FEET:
        return x
    x_pmv = x.copy()
    head = float(x[X_INDEX["HeadTempFd"]])
    for name in _MRT_SYNC_ZONES:
        x_pmv[X_INDEX[name]] = head
    return x_pmv


def _pin_ambient_coupling_zones(x: np.ndarray, *, amb: float, integration_mode: str) -> np.ndarray:
    if integration_mode != _INTEGRATION_HEAD_FEET:
        return x
    x_in = x.copy()
    for name in CHTD_X_NAMES:
        if name not in _HEAD_FEET_INTEGRATE_ZONES:
            x_in[X_INDEX[name]] = float(amb)
    return x_in


def _apply_delta(
    x: np.ndarray,
    x_delta: np.ndarray,
    *,
    amb: float,
    integration_mode: str,
) -> np.ndarray:
    if integration_mode == _INTEGRATION_FULL:
        return x + x_delta
    x_next = x.copy()
    for name in _HEAD_FEET_INTEGRATE_ZONES:
        idx = X_INDEX[name]
        x_next[idx] = x[idx] + x_delta[idx]
    for name in CHTD_X_NAMES:
        if name not in _HEAD_FEET_INTEGRATE_ZONES:
            x_next[X_INDEX[name]] = float(amb)
    return x_next


def _delta_abs_limit_for_mode(x_delta: np.ndarray, *, integration_mode: str) -> float:
    if integration_mode == _INTEGRATION_FULL:
        return float(np.max(np.abs(x_delta)))
    vals = [abs(float(x_delta[X_INDEX[name]])) for name in _HEAD_FEET_INTEGRATE_ZONES]
    return float(max(vals)) if vals else 0.0


def _pmv_for_state(
    x: np.ndarray,
    *,
    u: np.ndarray,
    rh_percent: float,
    integration_mode: str,
    met_driver: float = 1.0,
    clo_driver: float = 0.5,
    met_passenger: float = 1.0,
    clo_passenger: float = 0.5,
    air_speed_geometry: Optional[AirSpeedInputs] = None,
) -> Tuple[float, float, Dict[str, Any], Dict[str, Any]]:
    x_eval = _x_for_pmv_evaluation(x, integration_mode=integration_mode)
    air_in = _air_speed_inputs_from_u_preview(u, geometry=air_speed_geometry)
    air_speeds = estimate_driver_passenger_air_speed(air_in)

    driver_air = _driver_air_temp_c(x_eval)
    passenger_air = _passenger_air_temp_c(x_eval)
    driver_mrt = _driver_mrt_c(x_eval)
    passenger_mrt = _passenger_mrt_c(x_eval)

    driver_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=driver_air,
            mean_radiant_temp_c=driver_mrt,
            air_velocity_m_s=air_speeds.driver_air_speed_m_s,
            relative_humidity_pct=rh_percent,
            metabolic_rate_met=met_driver,
            clothing_insulation_clo=clo_driver,
        )
    )
    passenger_comfort = compute_vehicle_pmv(
        VehicleComfortInputs(
            air_temp_c=passenger_air,
            mean_radiant_temp_c=passenger_mrt,
            air_velocity_m_s=air_speeds.passenger_air_speed_m_s,
            relative_humidity_pct=rh_percent,
            metabolic_rate_met=met_passenger,
            clothing_insulation_clo=clo_passenger,
        )
    )

    driver_app = evaluate_pmv_applicability(
        ta=driver_air,
        tr=driver_mrt,
        vel=air_speeds.driver_air_speed_m_s,
        rh=rh_percent,
        met=met_driver,
        clo=clo_driver,
        pmv=driver_comfort.pmv,
    )
    passenger_app = evaluate_pmv_applicability(
        ta=passenger_air,
        tr=passenger_mrt,
        vel=air_speeds.passenger_air_speed_m_s,
        rh=rh_percent,
        met=met_passenger,
        clo=clo_passenger,
        pmv=passenger_comfort.pmv,
    )
    return driver_comfort.pmv, passenger_comfort.pmv, driver_app, passenger_app


def _sample_record(
    *,
    step: int,
    dt_seconds: float,
    x: np.ndarray,
    x_delta: Optional[np.ndarray],
    u: np.ndarray,
    rh_percent: float,
    integration_mode: str,
    air_speed_geometry: Optional[AirSpeedInputs] = None,
) -> Dict[str, Any]:
    drv_pmv, psg_pmv, drv_app, psg_app = _pmv_for_state(
        x,
        u=u,
        rh_percent=rh_percent,
        integration_mode=integration_mode,
        air_speed_geometry=air_speed_geometry,
    )
    record: Dict[str, Any] = {
        "step": int(step),
        "time_s": float(step * dt_seconds),
        "driver_pmv_raw": float(drv_pmv),
        "driver_pmv_display": float(drv_app["pmv_display"]),
        "driver_valid_for_comfort_interpretation": bool(
            drv_app["valid_for_comfort_interpretation"]
        ),
        "driver_pmv_applicability": str(drv_app["applicability"]),
        "driver_pmv_applicability_reasons": list(drv_app["reasons"]),
        "passenger_pmv_raw": float(psg_pmv),
        "passenger_pmv_display": float(psg_app["pmv_display"]),
        "passenger_valid_for_comfort_interpretation": bool(
            psg_app["valid_for_comfort_interpretation"]
        ),
        "passenger_pmv_applicability": str(psg_app["applicability"]),
        "passenger_pmv_applicability_reasons": list(psg_app["reasons"]),
        "HeadTempFd": float(x[X_INDEX["HeadTempFd"]]),
        "FeetTempFd": float(x[X_INDEX["FeetTempFd"]]),
        "CabinTempFd": float(x[X_INDEX["CabinTempFd"]]),
        "x_out_of_bounds_zones": _x_out_of_bounds_zones(x),
    }
    if x_delta is not None:
        record["x_delta_summary"] = _x_delta_summary(x_delta)
    return record


def _should_sample(step: int, *, steps: int, sample_every: int) -> bool:
    if step == 0:
        return True
    if step == steps:
        return True
    return sample_every > 0 and step % sample_every == 0


def run_afe_v2_pmv_warmup_preview(
    case: Mapping[str, Any],
    *,
    steps: Optional[int] = None,
    dt_seconds: Optional[float] = None,
    sample_every: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    tma_source: Optional[Mapping[str, Any]] = None,
    amb_t: Optional[float] = None,
    rh_percent: Optional[float] = None,
    chtd_params: Optional[CHTDParams] = None,
    air_speed_geometry: Optional[AirSpeedInputs] = None,
    met_driver: float = 1.0,
    clo_driver: float = 0.5,
    met_passenger: float = 1.0,
    clo_passenger: float = 0.5,
    integration_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Run multi-step warm-up preview for one case."""
    afe_case, tma, amb, rh = _resolve_case_and_tma(case, tma_source=tma_source, amb_t=amb_t)
    if rh_percent is not None:
        rh = float(rh_percent)

    mode = str(integration_mode or case.get("integration_mode", _INTEGRATION_FULL))
    if mode not in {_INTEGRATION_FULL, _INTEGRATION_HEAD_FEET}:
        raise ValueError(f"unsupported integration_mode: {mode!r}")

    n_steps = int(steps if steps is not None else case.get("steps", _DEFAULT_STEPS))
    dt = float(dt_seconds if dt_seconds is not None else case.get("dt_seconds", _DEFAULT_DT_SECONDS))
    sample_n = int(
        sample_every if sample_every is not None else case.get("sample_every", _DEFAULT_SAMPLE_EVERY)
    )
    if n_steps < 0:
        raise ValueError("steps must be >= 0")
    if sample_n <= 0:
        raise ValueError("sample_every must be > 0")

    warnings: List[str] = [
        CLASSIFICATION,
        "offline_warmup_preview_not_wired_to_runtime_pipeline",
        "multi_step_forward_euler_not_steady_state",
        f"integration_mode_{mode}",
    ]
    if mode == _INTEGRATION_HEAD_FEET:
        warnings.extend(
            [
                "head_feet_zones_only_integration_preview",
                "structural_and_cabin_zones_pinned_to_amb_for_stability",
                "pmv_mrt_zones_synced_to_head_for_preview_only",
            ]
        )

    diagnostic = _diagnostic_from_case(afe_case)
    if diagnostic.get("warnings"):
        warnings.extend(str(w) for w in diagnostic["warnings"])

    adapter = build_chtd_u_preview_from_afe_v2(diagnostic, tma)
    u = adapter.u.copy()
    params = chtd_params if chtd_params is not None else CHTDParams()

    x = _initial_state_x0(x0, amb_t=amb)
    samples: List[Dict[str, Any]] = []
    max_abs_delta = 0.0
    stopped_early = False
    stop_reason: Optional[str] = None
    completed_steps = 0

    if _should_sample(0, steps=n_steps, sample_every=sample_n):
        samples.append(
            _sample_record(
                step=0,
                dt_seconds=dt,
                x=x,
                x_delta=None,
                u=u,
                rh_percent=rh,
                integration_mode=mode,
                air_speed_geometry=air_speed_geometry,
            )
        )

    for step in range(1, n_steps + 1):
        x_in = _pin_ambient_coupling_zones(x, amb=amb, integration_mode=mode)
        x_delta = compute_chtd_delta(x_in, u, params)
        step_max_abs = _delta_abs_limit_for_mode(x_delta, integration_mode=mode)
        max_abs_delta = max(max_abs_delta, step_max_abs)

        if not np.all(np.isfinite(x_delta)):
            warnings.append(f"non_finite_delta_step_{step}")
            stopped_early = True
            stop_reason = "non_finite_delta"
            break

        if step_max_abs > _DELTA_ABS_LIMIT_C:
            warnings.append(f"unstable_large_delta_step_{step}_max_abs_{step_max_abs:.3f}c")
            stopped_early = True
            stop_reason = "unstable_delta_exceeds_20c_per_step"
            break

        x = _apply_delta(x_in, x_delta, amb=amb, integration_mode=mode)
        completed_steps = step

        if not np.all(np.isfinite(x)):
            warnings.append(f"non_finite_x_step_{step}")
            stopped_early = True
            stop_reason = "non_finite_x"
            break

        oob = _x_out_of_bounds_zones(x)
        if oob:
            warnings.append(f"x_out_of_bounds_step_{step}_zones_{','.join(oob[:5])}")

        if _should_sample(step, steps=n_steps, sample_every=sample_n):
            samples.append(
                _sample_record(
                    step=step,
                    dt_seconds=dt,
                    x=x,
                    x_delta=x_delta,
                    u=u,
                    rh_percent=rh,
                    integration_mode=mode,
                    air_speed_geometry=air_speed_geometry,
                )
            )

    if not samples:
        samples.append(
            _sample_record(
                step=0,
                dt_seconds=dt,
                x=x,
                x_delta=None,
                u=u,
                rh_percent=rh,
                integration_mode=mode,
                air_speed_geometry=air_speed_geometry,
            )
        )

    initial = samples[0]
    final = samples[-1]
    first_valid: Optional[int] = None
    for s in samples:
        if s.get("driver_valid_for_comfort_interpretation"):
            first_valid = int(s["step"])
            break

    summary = {
        "initial_driver_pmv_raw": float(initial["driver_pmv_raw"]),
        "final_driver_pmv_raw": float(final["driver_pmv_raw"]),
        "initial_driver_pmv_display": float(initial["driver_pmv_display"]),
        "final_driver_pmv_display": float(final["driver_pmv_display"]),
        "initial_HeadTempFd": float(initial["HeadTempFd"]),
        "final_HeadTempFd": float(final["HeadTempFd"]),
        "initial_FeetTempFd": float(initial["FeetTempFd"]),
        "final_FeetTempFd": float(final["FeetTempFd"]),
        "first_valid_step": first_valid,
        "max_abs_delta": float(max_abs_delta),
        "completed_steps": int(completed_steps),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "sample_count": len(samples),
        "warnings": list(warnings),
    }

    return {
        "schema": WARMUP_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "runtime_pipeline_unchanged": True,
        "afe_calc_default_unchanged": True,
        "thermal_formula_unchanged": True,
        "pmv_core_unchanged": True,
        "case_id": str(case.get("case_id", afe_case.get("case_id", "warmup"))),
        "mode_code": str(afe_case.get("mode_code", "")),
        "integration_mode": mode,
        "amb_t_c": float(amb),
        "rh_percent": float(rh),
        "steps": int(n_steps),
        "dt_seconds": float(dt),
        "sample_every": int(sample_n),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fallback_used": dict(adapter.fallback_used),
        "samples": samples,
        "summary": summary,
        "warnings": list(warnings),
    }


def run_builtin_warmup_previews() -> Dict[str, Any]:
    previews = []
    for scenario in default_builtin_warmup_scenarios():
        cfg = dict(scenario)
        steps = cfg.pop("steps", _DEFAULT_STEPS)
        dt = cfg.pop("dt_seconds", _DEFAULT_DT_SECONDS)
        sample = cfg.pop("sample_every", _DEFAULT_SAMPLE_EVERY)
        integration_mode = cfg.pop("integration_mode", _INTEGRATION_HEAD_FEET)
        previews.append(
            run_afe_v2_pmv_warmup_preview(
                cfg,
                steps=steps,
                dt_seconds=dt,
                sample_every=sample,
                integration_mode=integration_mode,
            )
        )
    return {
        "schema": WARMUP_SCHEMA,
        "classification": CLASSIFICATION,
        "not_runtime_active": True,
        "case_count": len(previews),
        "previews": previews,
    }
