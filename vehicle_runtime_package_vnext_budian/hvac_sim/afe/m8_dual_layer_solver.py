"""M8 dual-layer AFE v2 experimental Newton solver skeleton.

Uses ``fan_reference`` scaled curves and placeholder resistances. **Not wired**
to ``afe_calc`` or ``runtime_pipeline``. Pending customer answers Q1–Q8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

import numpy as np

from hvac_sim.afe.fan import fan_k_coeffs
from hvac_sim.afe.fan_reference import (
    fit_front_fan_reference_curve,
    fit_rear_booster_reference_curve,
    scale_fan_curve_by_voltage,
)
from hvac_sim.afe.m8_dual_layer_model import (
    compute_dual_layer_active,
    decode_m8_intake_state,
    decode_m8_mode_layer_distribution,
    effective_intake_layer_fractions,
    is_heating_mode,
)

_M3H_TO_M3S = 1.0 / 3600.0
_M3S_TO_M3H = 3600.0
_CLOSED_R_SCALE = 1.0e4
LayerPolicy = Literal["upper", "lower", "mixed"]


@dataclass(frozen=True)
class M8DualLayerSolverInputs:
    front_blower_voltage: float
    rear_blower_voltage: float
    mode_code: Union[str, int]
    circle_mode_posn_pct: float
    circle_prior_posn_v: float
    driver_temp_door_posn: float
    passenger_temp_door_posn: float
    rear_mode_posn: float
    rear_temp_posn: float


@dataclass
class M8DualLayerSolverParams:
    fan_curve_source: str = "fan_reference_scaled"
    upper_path_resistance: float = 55.0
    lower_path_resistance: float = 55.0
    defrost_resistance: float = 85.0
    face_resistance: float = 42.0
    foot_resistance: float = 38.0
    rear_path_resistance: float = 62.0
    rear_face_resistance: float = 48.0
    rear_foot_resistance: float = 48.0
    face_layer_policy: LayerPolicy = "upper"
    rear_layer_policy: LayerPolicy = "lower"
    alpha_face_upper: float = 1.0
    alpha_rear_upper: float = 0.0
    rho_kg_m3: float = 1.2
    reference_voltage_v: float = 12.0
    newton_max_iter: int = 50
    newton_tol: float = 1e-6
    pending_placeholders: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class M8DualLayerSolverResult:
    q_upper_m3h: float
    q_lower_m3h: float
    q_defrost_m3h: float
    q_face_m3h: float
    q_foot_m3h: float
    q_rear_face_m3h: float
    q_rear_foot_m3h: float
    convergence_status: str
    provenance: Dict[str, Any] = field(default_factory=dict)


def _sq(q: float) -> float:
    return q * abs(q)


def _dsq(q: float) -> float:
    return 2.0 * abs(q)


def _fan_dp(q: float, k: np.ndarray) -> Tuple[float, float]:
    dp = float(k[0] * q * q + k[1] * q + k[2])
    ddp = float(2.0 * k[0] * q + k[1])
    return dp, ddp


def _finite_or_raise(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _normalize_mode(mode_code: Union[str, int]) -> str:
    return decode_m8_mode_layer_distribution(mode_code).mode_code


def _mode_outlet_scales(
    mode_code: str,
    params: Optional[M8DualLayerSolverParams] = None,
) -> Dict[str, float]:
    """Placeholder outlet openness multipliers (1=open, ~0=closed)."""
    m = mode_code.upper()
    scales = {
        "defrost": 0.05,
        "face": 0.05,
        "foot": 0.05,
        "rear_face": 0.3,
        "rear_foot": 0.3,
    }
    if m in ("D", "V_D"):
        scales.update(defrost=1.0, face=0.08, foot=0.05)
    elif m == "F":
        scales.update(defrost=0.12, face=0.01, foot=1.0)
    elif m in ("F_D", "V_D_F"):
        scales.update(defrost=1.0, face=0.12, foot=0.85)
    elif m in ("V", "V_F"):
        scales.update(defrost=0.05, face=1.0, foot=0.35 if m == "V_F" else 0.08)
    elif m == "OFF":
        scales = {k: 0.02 for k in scales}
    if params is not None:
        ph = params.pending_placeholders or {}
        for scale_key, ph_key in (
            ("defrost", "defrost_open_scale"),
            ("foot", "foot_open_scale"),
            ("rear_face", "rear_face_scale"),
            ("rear_foot", "rear_foot_scale"),
        ):
            if ph_key in ph:
                scales[scale_key] *= max(float(ph[ph_key]), 1e-3)
    return scales


def _effective_r(base_r: float, openness: float, temp_door: float = 0.5) -> float:
    """Map placeholder resistance with outlet openness and temp-door bias."""
    o = max(float(openness), 1e-3)
    door = 0.75 + 0.5 * _finite_or_raise(temp_door, "temp_door")
    return float(base_r) * door / o


def _intake_path_resistances(
    params: M8DualLayerSolverParams,
    upper_fresh_fraction: float,
    lower_recirc_fraction: float,
) -> Tuple[float, float]:
    uf = max(float(upper_fresh_fraction), 0.05)
    lr = max(float(lower_recirc_fraction), 0.05)
    r_upper = params.upper_path_resistance / uf
    r_lower = params.lower_path_resistance / lr
    return r_upper, r_lower


def default_solver_params_for_mode(
    mode_code: Union[str, int],
    *,
    heating_mode: bool,
) -> M8DualLayerSolverParams:
    """Structural defaults: face upper_confirmed; rear lower_confirmed (2026-06-07)."""
    layers = decode_m8_mode_layer_distribution(mode_code, heating_mode=heating_mode)
    p = M8DualLayerSolverParams()
    p.rear_layer_policy = "lower"
    p.alpha_rear_upper = 0.0
    if layers.face_layer == "upper_confirmed":
        p.face_layer_policy = "upper"
        p.alpha_face_upper = 1.0
    elif layers.face_layer == "closed_or_minimal":
        p.face_layer_policy = "upper"
        p.alpha_face_upper = 1.0
    else:
        p.face_layer_policy = "lower"
        p.alpha_face_upper = 0.0
    return p


def _scaled_fan_k(
    params: M8DualLayerSolverParams,
    *,
    voltage_v: float,
    fan_id: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    v_ref = params.reference_voltage_v
    if params.fan_curve_source != "fan_reference_scaled":
        raise ValueError(f"unsupported fan_curve_source: {params.fan_curve_source!r}")
    if fan_id == "front_hvac_box":
        ref = fit_front_fan_reference_curve(reference_voltage_v=v_ref)
    elif fan_id == "rear_booster":
        ref = fit_rear_booster_reference_curve(reference_voltage_v=v_ref)
    else:
        raise ValueError(f"unknown fan_id: {fan_id!r}")
    scaled = scale_fan_curve_by_voltage(ref, voltage_v, reference_voltage_v=v_ref)
    k = fan_k_coeffs(scaled.fitted_params, scaled.equivalent_n_rev_s, params.rho_kg_m3)
    meta = {
        "fan_id": fan_id,
        "voltage_v": voltage_v,
        "equivalent_n_rev_s": scaled.equivalent_n_rev_s,
        "speed_ratio": scaled.speed_ratio,
    }
    return k, meta


def _newton_dual_layer_parallel(
    *,
    k_front: np.ndarray,
    r_upper: float,
    r_lower: float,
    x0: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, bool, int, float]:
    """Two-path parallel network: dp_fan(Q_u+Q_l) balances each branch R·sq(Q)."""
    x = np.array(x0, dtype=float).reshape(2)
    x = np.maximum(x, 1e-6)
    final_residual = float("inf")

    for it in range(1, max_iter + 1):
        qu, ql = x
        qt = qu + ql
        dp_f, ddp_f = _fan_dp(qt, k_front)

        f1 = dp_f - r_upper * _sq(qu)
        f2 = dp_f - r_lower * _sq(ql)
        F = np.array([f1, f2])
        final_residual = float(np.max(np.abs(F)))

        if final_residual < tol:
            return x, True, it, final_residual

        j11 = ddp_f - r_upper * _dsq(qu)
        j12 = ddp_f
        j21 = ddp_f
        j22 = ddp_f - r_lower * _dsq(ql)
        J = np.array([[j11, j12], [j21, j22]])

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break

        alpha = 1.0
        f_norm = float(np.dot(F, F))
        for _ in range(6):
            x_new = np.maximum(x + alpha * dx, 1e-6)
            qu_n, ql_n = x_new
            dp_n, _ = _fan_dp(qu_n + ql_n, k_front)
            F_new = np.array([
                dp_n - r_upper * _sq(qu_n),
                dp_n - r_lower * _sq(ql_n),
            ])
            if float(np.dot(F_new, F_new)) < f_norm:
                x = x_new
                break
            alpha *= 0.5
        else:
            x = np.maximum(x + 0.25 * dx, 1e-6)

    return x, False, max_iter, final_residual


def _solve_rear_tap_flow(
    *,
    k_rear: np.ndarray,
    q_lower_m3s: float,
    r_rear: float,
    rear_voltage_v: float,
) -> float:
    """One-dimensional balance for rear booster tap from lower layer (experimental)."""
    if rear_voltage_v <= 0.0:
        return max(0.0, q_lower_m3s * 0.15)

    q_hi = max(q_lower_m3s * 0.95, 1e-4)
    q_lo = 1e-6
    q = q_hi * 0.5
    for _ in range(40):
        dp_f, ddp = _fan_dp(q, k_rear)
        g = dp_f - r_rear * _sq(q)
        if abs(g) < 1e-8:
            break
        dg = ddp - r_rear * _dsq(q)
        if abs(dg) < 1e-12:
            break
        q = min(max(q - g / dg, q_lo), q_hi)
    return float(q)


def _conductance(r: float) -> float:
    if r <= 0.0 or not math.isfinite(r):
        return 0.0
    return 1.0 / math.sqrt(r)


def _split_by_conductance(total_q: float, *resistances: float) -> Tuple[float, ...]:
    gs = [_conductance(r) for r in resistances]
    g_sum = sum(gs)
    if g_sum <= 0.0:
        return tuple(0.0 for _ in resistances)
    return tuple(total_q * g / g_sum for g in gs)


def _face_supply_m3s(
    q_upper: float,
    q_lower: float,
    params: M8DualLayerSolverParams,
) -> float:
    policy = params.face_layer_policy
    if policy == "upper":
        return q_upper
    if policy == "lower":
        return q_lower
    alpha = min(max(params.alpha_face_upper, 0.0), 1.0)
    return alpha * q_upper + (1.0 - alpha) * q_lower


def _rear_supply_m3s(
    q_upper: float,
    q_lower: float,
    q_rear_tap: float,
    params: M8DualLayerSolverParams,
) -> float:
    policy = params.rear_layer_policy
    if policy == "upper":
        return min(q_rear_tap, q_upper)
    if policy == "lower":
        return q_rear_tap
    alpha = min(max(params.alpha_rear_upper, 0.0), 1.0)
    blended = alpha * q_upper + (1.0 - alpha) * q_lower
    return min(q_rear_tap, max(blended, 0.0))


def solve_m8_dual_layer(
    inputs: M8DualLayerSolverInputs,
    params: Optional[M8DualLayerSolverParams] = None,
) -> M8DualLayerSolverResult:
    """Run experimental dual-layer Newton skeleton (not production signed-off)."""
    p = params if params is not None else M8DualLayerSolverParams()
    mode = _normalize_mode(inputs.mode_code)
    heating = is_heating_mode(
        inputs.driver_temp_door_posn,
        inputs.passenger_temp_door_posn,
    )
    dual_layer_active = compute_dual_layer_active(mode, heating)
    intake = decode_m8_intake_state(
        inputs.circle_mode_posn_pct,
        inputs.circle_prior_posn_v,
        mode_code=mode,
    )
    mode_layers = decode_m8_mode_layer_distribution(mode, heating_mode=heating)
    ur, lr, uf, lf = effective_intake_layer_fractions(intake, dual_layer_active)
    outlet_scales = _mode_outlet_scales(mode, p)

    k_front, front_meta = _scaled_fan_k(
        p, voltage_v=inputs.front_blower_voltage, fan_id="front_hvac_box"
    )
    k_rear, rear_meta = _scaled_fan_k(
        p, voltage_v=inputs.rear_blower_voltage, fan_id="rear_booster"
    )

    r_upper, r_lower = _intake_path_resistances(p, uf, lr)
    if not dual_layer_active:
        r_blend = 0.5 * (r_upper + r_lower)
        r_upper = r_lower = r_blend

    k2_shutoff = float(k_front[2])
    if k2_shutoff <= 0.0 or not math.isfinite(k2_shutoff):
        return _zero_result(
            convergence_status="fan_off_or_invalid",
            provenance={"reason": "invalid front fan shutoff", "front_meta": front_meta},
        )

    q_guess = math.sqrt(max(k2_shutoff / max(r_upper, 1e-6), 1e-12)) * 0.35
    x0 = np.array([q_guess, q_guess], dtype=float)

    q_vec, converged, iterations, residual = _newton_dual_layer_parallel(
        k_front=k_front,
        r_upper=r_upper,
        r_lower=r_lower,
        x0=x0,
        max_iter=p.newton_max_iter,
        tol=p.newton_tol,
    )
    q_upper_m3s, q_lower_m3s = float(q_vec[0]), float(q_vec[1])

    temp_avg = 0.5 * (
        _finite_or_raise(inputs.driver_temp_door_posn, "driver_temp_door_posn")
        + _finite_or_raise(inputs.passenger_temp_door_posn, "passenger_temp_door_posn")
    )
    r_def = _effective_r(p.defrost_resistance, outlet_scales["defrost"], temp_avg)
    r_face = _effective_r(p.face_resistance, outlet_scales["face"], temp_avg)
    r_foot = _effective_r(p.foot_resistance, outlet_scales["foot"], temp_avg)
    r_rear = _effective_r(
        p.rear_path_resistance,
        0.5 * (outlet_scales["rear_face"] + outlet_scales["rear_foot"]),
        inputs.rear_temp_posn,
    )

    if p.face_layer_policy == "upper":
        q_def_m3s, q_face_m3s = _split_by_conductance(q_upper_m3s, r_def, r_face)
        q_foot_m3s, q_rear_branch_m3s = _split_by_conductance(q_lower_m3s, r_foot, r_rear)
    elif p.face_layer_policy == "mixed":
        q_def_m3s, q_face_upper_m3s = _split_by_conductance(q_upper_m3s, r_def, r_face)
        q_face_lower_m3s, q_foot_m3s, q_rear_branch_m3s = _split_by_conductance(
            q_lower_m3s, r_face, r_foot, r_rear
        )
        alpha = min(max(p.alpha_face_upper, 0.0), 1.0)
        q_face_m3s = alpha * q_face_upper_m3s + (1.0 - alpha) * q_face_lower_m3s
    else:
        q_def_m3s, q_face_upper_m3s = _split_by_conductance(q_upper_m3s, r_def, r_face)
        q_face_lower_m3s, q_foot_m3s, q_rear_branch_m3s = _split_by_conductance(
            q_lower_m3s, r_face, r_foot, r_rear
        )
        if mode_layers.face_layer == "closed_or_minimal":
            q_face_m3s = q_face_lower_m3s
        else:
            q_face_m3s = q_face_lower_m3s + q_face_upper_m3s

    q_rear_tap_m3s = _solve_rear_tap_flow(
        k_rear=k_rear,
        q_lower_m3s=max(q_rear_branch_m3s, 1e-9),
        r_rear=r_rear,
        rear_voltage_v=inputs.rear_blower_voltage,
    )
    q_rear_supply = _rear_supply_m3s(q_upper_m3s, q_lower_m3s, q_rear_tap_m3s, p)

    rear_mode = _finite_or_raise(inputs.rear_mode_posn, "rear_mode_posn")
    r_rf = _effective_r(p.rear_face_resistance, outlet_scales["rear_face"] * (1.1 - 0.2 * rear_mode))
    r_rfoot = _effective_r(p.rear_foot_resistance, outlet_scales["rear_foot"] * (0.9 + 0.2 * rear_mode))
    q_rear_face_m3s, q_rear_foot_m3s = _split_by_conductance(q_rear_supply, r_rf, r_rfoot)

    status = "converged" if converged else "max_iter"
    prov: Dict[str, Any] = {
        "status": "experimental_skeleton",
        "not_vehicle_signed_off": True,
        "not_default_runtime": True,
        "not_wired_to": ["afe_calc", "runtime_pipeline"],
        "waiting_customer_answers": "Q1-Q8",
        "customer_confirmed": "customer_confirmed_2026_06_07",
        "solver": "m8_dual_layer_newton_v0_parallel_two_path",
        "converged": converged,
        "iterations": iterations,
        "residual_max": residual,
        "mode_code": mode,
        "mode_layer_decode": {
            "defrost_layer": mode_layers.defrost_layer,
            "face_layer": mode_layers.face_layer,
            "foot_layer": mode_layers.foot_layer,
            "rear_layer": mode_layers.rear_layer,
            "rear_face_policy": mode_layers.rear_face_policy,
            "rear_foot_policy": mode_layers.rear_foot_policy,
        },
        "heating_mode": heating,
        "dual_layer_active": dual_layer_active,
        "face_layer_policy": p.face_layer_policy,
        "face_policy_confirmed": "upper_confirmed",
        "rear_policy_confirmed": "lower_confirmed",
        "face_mixed_superseded_by_customer_structure": mode_layers.provenance.get(
            "face_mixed_superseded", {}
        ),
        "rear_layer_policy": p.rear_layer_policy,
        "blower_model": {
            "front_hvac_blower": front_meta,
            "rear_booster_blower": rear_meta,
        },
        "heat_model": mode_layers.provenance.get("heat_model"),
        "intake_layer_fractions_applied": {
            "upper_recirc": ur,
            "lower_recirc": lr,
            "upper_fresh": uf,
            "lower_fresh": lf,
            "dual_layer_active": dual_layer_active,
        },
        "intake": intake.provenance,
        "path_resistances_pa_s2_m6": {
            "upper": r_upper,
            "lower": r_lower,
            "defrost": r_def,
            "face": r_face,
            "foot": r_foot,
            "rear_path": r_rear,
        },
        "pending_placeholders": dict(p.pending_placeholders),
    }

    return M8DualLayerSolverResult(
        q_upper_m3h=q_upper_m3s * _M3S_TO_M3H,
        q_lower_m3h=q_lower_m3s * _M3S_TO_M3H,
        q_defrost_m3h=q_def_m3s * _M3S_TO_M3H,
        q_face_m3h=q_face_m3s * _M3S_TO_M3H,
        q_foot_m3h=q_foot_m3s * _M3S_TO_M3H,
        q_rear_face_m3h=q_rear_face_m3s * _M3S_TO_M3H,
        q_rear_foot_m3h=q_rear_foot_m3s * _M3S_TO_M3H,
        convergence_status=status,
        provenance=prov,
    )


def _zero_result(*, convergence_status: str, provenance: Dict[str, Any]) -> M8DualLayerSolverResult:
    return M8DualLayerSolverResult(
        q_upper_m3h=0.0,
        q_lower_m3h=0.0,
        q_defrost_m3h=0.0,
        q_face_m3h=0.0,
        q_foot_m3h=0.0,
        q_rear_face_m3h=0.0,
        q_rear_foot_m3h=0.0,
        convergence_status=convergence_status,
        provenance=provenance,
    )


def build_m8_dual_layer_solver_status_report() -> Dict[str, Any]:
    """Structured status for markdown/JSON export."""
    low = solve_m8_dual_layer(
        M8DualLayerSolverInputs(
            front_blower_voltage=6.0,
            rear_blower_voltage=6.0,
            mode_code="V",
            circle_mode_posn_pct=50.0,
            circle_prior_posn_v=2.33,
            driver_temp_door_posn=0.5,
            passenger_temp_door_posn=0.5,
            rear_mode_posn=0.5,
            rear_temp_posn=0.5,
        )
    )
    high = solve_m8_dual_layer(
        M8DualLayerSolverInputs(
            front_blower_voltage=12.0,
            rear_blower_voltage=12.0,
            mode_code="V",
            circle_mode_posn_pct=50.0,
            circle_prior_posn_v=2.33,
            driver_temp_door_posn=0.5,
            passenger_temp_door_posn=0.5,
            rear_mode_posn=0.5,
            rear_temp_posn=0.5,
        )
    )
    defrost = solve_m8_dual_layer(
        M8DualLayerSolverInputs(
            front_blower_voltage=12.0,
            rear_blower_voltage=12.0,
            mode_code="D",
            circle_mode_posn_pct=0.0,
            circle_prior_posn_v=4.24,
            driver_temp_door_posn=0.8,
            passenger_temp_door_posn=0.8,
            rear_mode_posn=0.2,
            rear_temp_posn=0.5,
        )
    )
    foot = solve_m8_dual_layer(
        M8DualLayerSolverInputs(
            front_blower_voltage=12.0,
            rear_blower_voltage=12.0,
            mode_code="F",
            circle_mode_posn_pct=100.0,
            circle_prior_posn_v=4.56,
            driver_temp_door_posn=0.3,
            passenger_temp_door_posn=0.3,
            rear_mode_posn=0.7,
            rear_temp_posn=0.5,
        )
    )
    q_low = low.q_upper_m3h + low.q_lower_m3h
    q_high = high.q_upper_m3h + high.q_lower_m3h
    return {
        "schema": "afe_m8_dual_layer_solver_experimental_status_v1",
        "experimental": True,
        "vehicle_signed_off": False,
        "default_runtime": False,
        "wired_to_afe_calc": False,
        "wired_to_runtime_pipeline": False,
        "waiting_customer_answers": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
        "module": "hvac_sim.afe.m8_dual_layer_solver",
        "entrypoint": "solve_m8_dual_layer",
        "sample_12v_total_m3h": q_high,
        "sample_6v_total_m3h": q_low,
        "voltage_monotonic_total_flow": q_high > q_low,
        "sample_defrost_mode": {
            "q_defrost_m3h": defrost.q_defrost_m3h,
            "q_foot_m3h": defrost.q_foot_m3h,
            "defrost_gt_foot": defrost.q_defrost_m3h > defrost.q_foot_m3h,
        },
        "sample_foot_mode": {
            "q_foot_m3h": foot.q_foot_m3h,
            "q_face_m3h": foot.q_face_m3h,
            "q_defrost_m3h": foot.q_defrost_m3h,
            "foot_gt_face": foot.q_foot_m3h > foot.q_face_m3h,
            "defrost_intentional_bleed": foot.q_defrost_m3h > 0.0,
        },
    }


def render_m8_dual_layer_solver_status_markdown(
    report: Optional[Mapping[str, Any]] = None,
) -> str:
    r = dict(report) if report is not None else build_m8_dual_layer_solver_status_report()
    waiting = ", ".join(r["waiting_customer_answers"])
    lines = [
        "# AFE M8 Dual-Layer Solver — Experimental Status",
        "",
        "> **Experimental skeleton only.** Not vehicle signed-off. "
        "Not default runtime. Not wired to `afe_calc` or `runtime_pipeline`.",
        "",
        f"> **Blocked on customer answers:** {waiting}",
        "",
        "## Scope",
        "",
        "- Module: `python_impl/hvac_sim/afe/m8_dual_layer_solver.py`",
        "- Entry: `solve_m8_dual_layer(inputs, params)`",
        "- Fan curves: `fan_reference.scale_fan_curve_by_voltage()` (provisional V∝n)",
        "- Resistances: placeholder constants; mode/outlet gating is heuristic",
        "",
        "- Blowers: **front HVAC blower** + **rear booster blower** (not coaxial model)",
        "- Heat: **heater core / water PTC pending** (not air PTC)",
        "",
        "## Structure (customer confirmed 2026-06-07)",
        "",
        "- **Face / defrost** → **upper_confirmed**",
        "- **Foot / rear face / rear foot** → **lower_confirmed**",
        "- F mode: foot lower main + **defrost upper intentional bleed** (not a contradiction)",
        "- `face=mixed` in anchor/fit reports: **historical smoke-fit only**, superseded by customer feedback",
        "- Dual-layer active: heating mode AND mode in `{F_D, V_D_F, F}`",
        "- Default solver face policy: **upper**; rear policy: **lower**",
        "",
        "## Current intake mechanism hypothesis",
        "",
        "Decode-only draft (`m8_dual_layer_model.py`) — **not wired to runtime**.",
        "",
        "| Motor | Role | Signal proxy |",
        "|-------|------|--------------|",
        "| **Left** | `fresh_air_door`, `left_25_recirc_leaf` | "
        "`left_motor_voltage_v` or inferred from `circle_mode_posn_pct` |",
        "| **Right** | `middle_50_recirc_leaf`, `right_25_recirc_leaf` | "
        "`circle_prior_posn_v` (right motor voltage) |",
        "",
        "Left motor voltage anchors (tolerance ~0.45 V): **0.42 V** → recirc, "
        "**4.24 V** → fresh, **2.5 V** → dual_layer.",
        "",
        "Dual-layer mode: fresh opening fixed at left-motor middle position; "
        "recirc effective opening from right-motor leaf command.",
        "",
        "Upper/lower fresh–recirc split: **pending customer confirmation** "
        "(LUT placeholder only; not sign-off physics).",
        "",
        "API: `decode_recirc_leaf_state`, `decode_left_intake_motor_state`, "
        "`build_m8_intake_mechanism_state`, `decode_m8_intake_state`.",
        "",
        "## Solver topology (v0)",
        "",
        "Two-path parallel Newton balance at front plenum:",
        "",
        "```",
        "F1: dp_front(Q_upper + Q_lower) − R_upper·sq(Q_upper) = 0",
        "F2: dp_front(Q_upper + Q_lower) − R_lower·sq(Q_lower) = 0",
        "```",
        "",
        "Post-solve distribution (draft policy):",
        "",
        "- Defrost ← upper layer (intentional bleed allowed in F mode)",
        "- Foot ← lower layer",
        "- Face ← **upper_confirmed** when active; minimal in F mode",
        "- Rear face/foot ← **lower_confirmed**",
        "",
        "## Smoke checks",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| 12 V total flow [m³/h] | {r['sample_12v_total_m3h']:.2f} |",
        f"| 6 V total flow [m³/h] | {r['sample_6v_total_m3h']:.2f} |",
        f"| Higher voltage → higher total | {r['voltage_monotonic_total_flow']} |",
        f"| Defrost mode: q_defrost > q_foot | {r['sample_defrost_mode']['defrost_gt_foot']} |",
        f"| Foot mode: q_foot > q_face | {r['sample_foot_mode']['foot_gt_face']} |",
        f"| Foot mode: defrost intentional bleed | {r['sample_foot_mode'].get('defrost_intentional_bleed', False)} |",
        "",
        "## Customer questions (Q1–Q8)",
        "",
        "| ID | Topic | Impact on solver |",
        "|----|-------|------------------|",
        "| Q1 | Intake voltage thresholds | `decode_m8_intake_state` accuracy |",
        "| Q2 | Inlet motor physical function | Upper/lower path topology |",
        "| Q3 | Face layer | **upper_confirmed** (2026-06-07); mixed is fit-only |",
        "| Q4 | Rear layer | **lower_confirmed** (2026-06-07) |",
        "| Q5 | Temp door linkage | Hex resistance split upper/lower |",
        "| Q6 | Defrost Tma inputs | Not in flow solve (downstream) |",
        "| Q7 | Rear foot Tma | Not in flow solve (downstream) |",
        "| Q8 | Dual-layer bench data | PSO / resistance calibration |",
        "",
        "## Policy",
        "",
        f"- Experimental: `{r['experimental']}`",
        f"- Vehicle signed-off: `{r['vehicle_signed_off']}`",
        f"- Default runtime: `{r['default_runtime']}`",
        f"- Wired to `afe_calc`: `{r['wired_to_afe_calc']}`",
        f"- Wired to `runtime_pipeline`: `{r['wired_to_runtime_pipeline']}`",
        "",
    ]
    return "\n".join(lines)
