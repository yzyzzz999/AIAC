"""AFE v2 side-by-side diagnostic: as-found ``afe_calc`` vs experimental M8 dual-layer solver.

Decode/compare only — **not** wired to ``runtime_pipeline``, PMV, or CHTD.
Does not alter ``afe_calc`` defaults.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from hvac_sim.afe.calibration_data import load_actuator_voltage_targets
from hvac_sim.afe.fan import FanSet
from hvac_sim.afe.fan_reference import (
    fit_front_fan_reference_curve,
    fit_rear_booster_reference_curve,
    scale_fan_curve_by_voltage,
)
from hvac_sim.afe.flow import FlowInputs, afe_calc
from hvac_sim.afe.m8_dual_layer_model import (
    confirmed_outlet_layer_assignments,
    decode_m8_intake_state,
    decode_m8_mode_layer_distribution,
    intake_motor_control_ownership,
    is_heating_mode,
)
from hvac_sim.afe.m8_dual_layer_solver import (
    M8DualLayerSolverInputs,
    default_solver_params_for_mode,
    solve_m8_dual_layer,
)
from hvac_sim.afe.params import AFEParams
from hvac_sim.afe.resistance import FlapSet
from hvac_sim.afe.runtime_decoder import build_flapset_proposal

_M3S_TO_M3H = 3600.0
_Q_OUT_LABELS = (
    "Q_total",
    "Q_Osa",
    "Q_Rec",
    "Q_F",
    "Q_S",
    "Q_Fd",
    "Q_Fp",
    "Q_Sd",
    "Q_Sp",
    "Q_FdDef",
    "Q_FdSv",
    "Q_Fdv",
    "Q_Fdf",
    "Q_FpDef",
    "Q_FpSv",
    "Q_Fpv",
    "Q_Fpf",
    "Q_Sdv",
    "Q_Sdf",
    "Q_Spv",
    "Q_Spf",
    "constant_1",
)

_MODE_CODE_TO_ID: Dict[str, int] = {
    "V": 1,
    "V_F": 2,
    "V_D_F": 3,
    "F": 4,
    "F_D": 5,
    "D": 6,
    "V_D": 7,
    "OFF": 8,
}


@dataclass
class AFEV2DiagnosticInputs:
    mode_code: Union[str, int]
    front_blower_voltage: float = 12.0
    rear_blower_voltage: float = 12.0
    circle_mode_posn: float = 50.0
    circle_prior_posn: float = 2.33
    driver_temp_door_posn: float = 0.5
    passenger_temp_door_posn: float = 0.5
    rear_mode_posn: float = 0.5
    rear_temp_posn: float = 0.5
    temp_C: float = 25.0
    rh_percent: float = 50.0
    pressure_Pa: float = 101_325.0
    left_motor_voltage_v: Optional[float] = None


def _normalize_mode_code(mode_code: Union[str, int]) -> str:
    key = str(mode_code).strip().upper()
    if key in _MODE_CODE_TO_ID:
        return key
    aliases = {
        "1": "V",
        "2": "V_F",
        "3": "V_D_F",
        "4": "F",
        "5": "F_D",
        "6": "D",
        "7": "V_D",
        "8": "OFF",
    }
    return aliases.get(key, key)


def _circle_mode_pct(circle_mode_posn: float) -> float:
    v = float(circle_mode_posn)
    if 0.0 <= v <= 1.0 and not math.isclose(v, round(v), rel_tol=0, abs_tol=1e-9):
        return v * 100.0
    return v


def _fan_speeds_from_voltage(front_v: float, rear_v: float) -> Dict[str, Any]:
    front_ref = fit_front_fan_reference_curve()
    rear_ref = fit_rear_booster_reference_curve()
    front_scaled = scale_fan_curve_by_voltage(front_ref, front_v)
    rear_scaled = scale_fan_curve_by_voltage(rear_ref, rear_v)
    return {
        "n_F1": front_scaled.equivalent_n_rev_s,
        "n_F2": front_scaled.equivalent_n_rev_s,
        "n_S": rear_scaled.equivalent_n_rev_s,
        "provenance": {
            "method": "fan_reference_voltage_affinity",
            "front_voltage_v": front_v,
            "rear_voltage_v": rear_v,
            "front_meta": dict(front_scaled.provenance),
            "rear_meta": dict(rear_scaled.provenance),
        },
    }


def _mode_actuator_voltages(mode_code: str) -> Dict[str, float]:
    mode_id = _MODE_CODE_TO_ID.get(mode_code)
    if mode_id is None:
        return {}
    doc = load_actuator_voltage_targets()
    mode = next((m for m in doc["modes"] if m["mode_id"] == mode_id), None)
    if mode is None:
        return {}
    act = mode["actuators"]
    out: Dict[str, float] = {}
    for key in ("defrost", "vent", "foot"):
        entry = act.get(key)
        if entry and entry.get("target_v") is not None:
            out[key] = float(entry["target_v"])
    return out


def _build_as_found_flapset(
    diag: AFEV2DiagnosticInputs,
    mode_code: str,
    intake_state,
    warnings: List[str],
) -> Tuple[FlapSet, Dict[str, Any]]:
    mode_id = _MODE_CODE_TO_ID.get(mode_code, 1)
    feedback = _mode_actuator_voltages(mode_code)
    proposal = build_flapset_proposal(hvac_mode_id=mode_id, feedback_voltages=feedback)
    warnings.extend(proposal.get("warnings", []))

    flaps = FlapSet()
    for flap_name, travel in proposal.get("flaps", {}).items():
        if hasattr(flaps, flap_name):
            setattr(flaps, flap_name, float(travel))

    recirc = max(0.0, min(1.0, float(intake_state.recirc_fraction_total)))
    flaps.FHRecFlapPosn = recirc
    flaps.FHOsaFlapPosn = 1.0 - recirc
    flaps.FHFdhFlapPosn = float(diag.driver_temp_door_posn)
    flaps.FHFphFlapPosn = float(diag.passenger_temp_door_posn)
    flaps.FHSdhFlapPosn = float(diag.rear_temp_posn)
    flaps.FHSphFlapPosn = float(diag.rear_temp_posn)
    flaps.FHSpvFlapPosn = float(diag.rear_mode_posn)
    flaps.FHSdvFlapPosn = float(diag.rear_mode_posn)

    prov = {
        "method": "diagnostic_flapset_proposal",
        "mode_id": mode_id,
        "mode_code": mode_code,
        "actuator_feedback_proxy_v": feedback,
        "flapset_proposal": proposal,
        "intake_recirc_fraction": recirc,
        "note": "Placeholder flap map from mode voltage table + intake decode; not runtime signed-off",
    }
    return flaps, prov


def _serialize_afe_result(result) -> Dict[str, Any]:
    q_out = {label: float(result.Q_out[i]) for i, label in enumerate(_Q_OUT_LABELS)}
    q_out_m3h = {k: v * _M3S_TO_M3H for k, v in q_out.items()}
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "Q_out_m3s": q_out,
        "Q_out_m3h": q_out_m3h,
        "aggregates_m3h": {
            "q_total": q_out_m3h["Q_total"],
            "q_face_fd_fp": q_out_m3h["Q_Fdv"] + q_out_m3h["Q_Fpv"],
            "q_foot_fd_fp": q_out_m3h["Q_Fdf"] + q_out_m3h["Q_Fpf"],
            "q_defrost": (
                q_out_m3h["Q_FdDef"]
                + q_out_m3h["Q_FdSv"]
                + q_out_m3h["Q_FpDef"]
                + q_out_m3h["Q_FpSv"]
            ),
            "q_rear": q_out_m3h["Q_Sd"] + q_out_m3h["Q_Sp"],
        },
        "rho_kg_m3": float(result.rho),
    }


def _serialize_m8_result(result) -> Dict[str, Any]:
    return {
        "convergence_status": result.convergence_status,
        "q_upper_m3h": float(result.q_upper_m3h),
        "q_lower_m3h": float(result.q_lower_m3h),
        "q_defrost_m3h": float(result.q_defrost_m3h),
        "q_face_m3h": float(result.q_face_m3h),
        "q_foot_m3h": float(result.q_foot_m3h),
        "q_rear_face_m3h": float(result.q_rear_face_m3h),
        "q_rear_foot_m3h": float(result.q_rear_foot_m3h),
        "aggregates_m3h": {
            "q_total_layers": float(result.q_upper_m3h + result.q_lower_m3h),
            "q_face": float(result.q_face_m3h),
            "q_foot": float(result.q_foot_m3h),
            "q_defrost": float(result.q_defrost_m3h),
            "q_rear": float(result.q_rear_face_m3h + result.q_rear_foot_m3h),
        },
        "provenance": dict(result.provenance),
    }


def _delta_summary(afe: Mapping[str, Any], m8: Mapping[str, Any]) -> Dict[str, Any]:
    a = afe["aggregates_m3h"]
    m = m8["aggregates_m3h"]

    def _delta(key_a: str, key_m: str) -> Dict[str, float]:
        av = float(a[key_a])
        mv = float(m[key_m])
        return {
            "as_found_m3h": av,
            "m8_v2_m3h": mv,
            "delta_m3h": mv - av,
            "delta_pct": (mv - av) / av * 100.0 if abs(av) > 1e-9 else float("nan"),
        }

    return {
        "note": "Coarse aggregate comparison only; models use different topology and units mapping",
        "total_flow": _delta("q_total", "q_total_layers"),
        "face": _delta("q_face_fd_fp", "q_face"),
        "foot": _delta("q_foot_fd_fp", "q_foot"),
        "defrost": _delta("q_defrost", "q_defrost"),
        "rear": _delta("q_rear", "q_rear"),
    }


def _collect_pending_items(
    intake_prov: Mapping[str, Any],
    m8_prov: Mapping[str, Any],
    flap_prov: Mapping[str, Any],
) -> List[str]:
    pending: List[str] = []
    if intake_prov.get("upper_lower_split_pending_customer_confirmation"):
        pending.append("upper_lower_fresh_recirc_split_pending_calibration")
    if intake_prov.get("upper_fresh_fraction_status") == "placeholder_pending":
        pending.append("upper_fresh_fraction_placeholder")
    if intake_prov.get("lower_recirc_fraction_status") == "placeholder_pending":
        pending.append("lower_recirc_fraction_placeholder")
    if m8_prov.get("not_vehicle_signed_off"):
        pending.append("m8_dual_layer_solver_not_vehicle_signed_off")
    if flap_prov.get("note"):
        pending.append("as_found_flapset_diagnostic_proxy")
    pending.append("not_wired_to_runtime_pipeline")
    return pending


def run_afe_v2_diagnostic(inputs: AFEV2DiagnosticInputs) -> Dict[str, Any]:
    """Run as-found AFE vs M8 v2 experimental solver and return comparison JSON."""
    warnings: List[str] = []
    mode_code = _normalize_mode_code(inputs.mode_code)
    mode_pct = _circle_mode_pct(inputs.circle_mode_posn)

    if mode_code not in _MODE_CODE_TO_ID:
        warnings.append(f"unknown_mode_code:{mode_code}")

    intake = decode_m8_intake_state(
        mode_pct,
        float(inputs.circle_prior_posn),
        mode_code=mode_code,
        left_motor_voltage_v=inputs.left_motor_voltage_v,
    )
    heating = is_heating_mode(
        inputs.driver_temp_door_posn,
        inputs.passenger_temp_door_posn,
    )
    mode_layers = decode_m8_mode_layer_distribution(mode_code, heating_mode=heating)

    flaps, flap_prov = _build_as_found_flapset(inputs, mode_code, intake, warnings)
    fan_meta = _fan_speeds_from_voltage(
        inputs.front_blower_voltage,
        inputs.rear_blower_voltage,
    )

    afe_result = afe_calc(
        FlowInputs(
            flaps=flaps,
            n_F1=fan_meta["n_F1"],
            n_F2=fan_meta["n_F2"],
            n_S=fan_meta["n_S"],
            temp_C=inputs.temp_C,
            rh_percent=inputs.rh_percent,
            pressure_Pa=inputs.pressure_Pa,
        ),
        AFEParams(),
        FanSet(),
    )
    if not afe_result.converged:
        warnings.append("as_found_afe_calc_did_not_converge")

    m8_inputs = M8DualLayerSolverInputs(
        front_blower_voltage=float(inputs.front_blower_voltage),
        rear_blower_voltage=float(inputs.rear_blower_voltage),
        mode_code=mode_code,
        circle_mode_posn_pct=mode_pct,
        circle_prior_posn_v=float(inputs.circle_prior_posn),
        driver_temp_door_posn=float(inputs.driver_temp_door_posn),
        passenger_temp_door_posn=float(inputs.passenger_temp_door_posn),
        rear_mode_posn=float(inputs.rear_mode_posn),
        rear_temp_posn=float(inputs.rear_temp_posn),
    )
    m8_params = default_solver_params_for_mode(mode_code, heating_mode=heating)
    m8_result = solve_m8_dual_layer(m8_inputs, m8_params)
    if m8_result.convergence_status != "converged":
        warnings.append(f"m8_solver_status:{m8_result.convergence_status}")

    as_found = _serialize_afe_result(afe_result)
    as_found["flap_provenance"] = flap_prov
    as_found["fan_provenance"] = fan_meta["provenance"]

    m8_serialized = _serialize_m8_result(m8_result)

    layer_assignment = {
        "mode_code": mode_code,
        "heating_mode": heating,
        "confirmed_outlet_layers": confirmed_outlet_layer_assignments(),
        "mode_layers": {
            "face_layer": mode_layers.face_layer,
            "defrost_layer": mode_layers.defrost_layer,
            "foot_layer": mode_layers.foot_layer,
            "rear_layer": mode_layers.rear_layer,
            "rear_face_policy": mode_layers.rear_face_policy,
            "rear_foot_policy": mode_layers.rear_foot_policy,
            "dual_layer_active": mode_layers.dual_layer_active,
        },
        "intake_motor_ownership": intake_motor_control_ownership(),
        "intake_state": {
            "recirc_fraction_total": intake.recirc_fraction_total,
            "upper_recirc_fraction": intake.upper_recirc_fraction,
            "lower_recirc_fraction": intake.lower_recirc_fraction,
            "confidence": intake.confidence,
        },
    }

    pending = _collect_pending_items(
        intake.provenance,
        m8_result.provenance,
        flap_prov,
    )

    return {
        "schema": "afe_v2_diagnostic_v1",
        "not_runtime_active": True,
        "wired_to_runtime_pipeline": False,
        "wired_to_afe_calc_default": False,
        "wired_to_pmv": False,
        "wired_to_chtd": False,
        "inputs": asdict(inputs),
        "as_found_afe_result": as_found,
        "m8_dual_layer_v2_result": m8_serialized,
        "delta_summary": _delta_summary(as_found, m8_serialized),
        "layer_assignment": layer_assignment,
        "pending_items": pending,
        "warnings": warnings,
    }
