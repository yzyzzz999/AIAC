"""Reference fan P-Q curves from supplier JSON and voltage affinity scaling.

Fits ``fan_curve_initial.json`` to the existing ``FanParams`` / ``fan_k_coeffs``
representation. Does **not** change ``FanSet()`` defaults used by ``afe_calc``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hvac_sim.afe.calibration_data import load_fan_curve_initial
from hvac_sim.afe.fan import FanParams, fan_k_coeffs
from hvac_sim.core.fluid import air_density

_M3H_TO_M3S = 1.0 / 3600.0
_DEFAULT_N_REF = 1000.0
_DEFAULT_V_REF = 12.0
_DEFAULT_RHO_REF = 1.2


@dataclass(frozen=True)
class FanReferenceCurve:
    """Supplier reference P-Q curve at one provisional reference voltage."""

    fan_id: str
    reference_voltage_v: float
    n_ref_rev_s: float
    rho_ref_kg_m3: float
    points: Tuple[Tuple[float, float], ...]  # (back_pressure_pa, flow_m3h)
    fitted_params: FanParams
    fit_k0: float
    fit_k2: float
    fit_max_residual_pa: float
    flow_unit: str
    absolute_compare_allowed: bool
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScaledFanCurve:
    """P-Q curve scaled from reference by voltage (provisional V∝n affinity)."""

    fan_id: str
    voltage_v: float
    reference_voltage_v: float
    speed_ratio: float
    points: Tuple[Tuple[float, float], ...]  # (back_pressure_pa, flow_m3h)
    equivalent_n_rev_s: float
    fitted_params: FanParams
    provenance: Dict[str, Any] = field(default_factory=dict)


def _entry_by_fan_id(fan_id: str) -> Dict[str, Any]:
    doc = load_fan_curve_initial()
    for entry in doc.get("entries", []):
        if str(entry["fan_id"]) == fan_id:
            return entry
    raise KeyError(f"fan_id not found in fan_curve_initial.json: {fan_id!r}")


def _points_from_entry(entry: Mapping[str, Any]) -> Tuple[Tuple[float, float], ...]:
    pts: List[Tuple[float, float]] = []
    for row in entry.get("curve_points", []):
        pts.append((float(row["back_pressure_pa"]), float(row["flow"])))
    return tuple(pts)


def _fit_k0_k2_from_pq_points(
    points: Sequence[Tuple[float, float]],
    *,
    rho_kg_m3: float,
) -> Tuple[float, float, float]:
    """Least-squares fit dp = k0·Q² + k2 with Q in m³/s, dp in Pa."""
    if len(points) < 2:
        raise ValueError("need at least two P-Q points to fit fan curve")
    p_arr = np.array([p for p, _q in points], dtype=float)
    q_m3h = np.array([q for _p, q in points], dtype=float)
    q_m3s = q_m3h * _M3H_TO_M3S
    x = q_m3s ** 2
    # dp = k0*x + k2
    a = np.column_stack([x, np.ones_like(x)])
    k0, k2 = np.linalg.lstsq(a, p_arr, rcond=None)[0]
    pred = k0 * x + k2
    max_res = float(np.max(np.abs(pred - p_arr)))
    return float(k0), float(k2), max_res


def _fan_params_from_k_at_n_ref(
    k0: float,
    k2: float,
    *,
    n_ref: float,
    rho_kg_m3: float,
) -> FanParams:
    """Map fitted (k0,k2) at n=n_ref to FanParams for fan_k_coeffs."""
    if k0 >= 0.0:
        raise ValueError(f"fitted k0 must be negative for a fan curve, got {k0}")
    if k2 <= 0.0:
        raise ValueError(f"fitted k2 (shutoff) must be positive, got {k2}")
    a0 = -k0 * (rho_kg_m3 ** 2)
    return FanParams(n0=float(n_ref), p0=float(k2), a0=float(a0))


def _flow_m3s_at_back_pressure(k: np.ndarray, back_pressure_pa: float) -> float:
    """Solve k0·Q² + k2 = P for Q ≥ 0 (k1=0, k0<0)."""
    k0, _, k2 = float(k[0]), float(k[1]), float(k[2])
    if k0 >= 0.0 or not math.isfinite(k0):
        return 0.0
    numer = back_pressure_pa - k2
    ratio = numer / k0
    if ratio < 0.0:
        return 0.0
    return math.sqrt(ratio)


def fit_front_fan_reference_curve(
    *,
    n_ref: float = _DEFAULT_N_REF,
    rho_kg_m3: float = _DEFAULT_RHO_REF,
    reference_voltage_v: float = _DEFAULT_V_REF,
) -> FanReferenceCurve:
    """Fit front HVAC box reference curve from ``fan_curve_initial.json``."""
    entry = _entry_by_fan_id("front_hvac_box")
    points = _points_from_entry(entry)
    k0, k2, max_res = _fit_k0_k2_from_pq_points(points, rho_kg_m3=rho_kg_m3)
    params = _fan_params_from_k_at_n_ref(k0, k2, n_ref=n_ref, rho_kg_m3=rho_kg_m3)
    return FanReferenceCurve(
        fan_id="front_hvac_box",
        reference_voltage_v=float(reference_voltage_v),
        n_ref_rev_s=float(n_ref),
        rho_ref_kg_m3=float(rho_kg_m3),
        points=points,
        fitted_params=params,
        fit_k0=k0,
        fit_k2=k2,
        fit_max_residual_pa=max_res,
        flow_unit=str(entry.get("flow_unit", "unknown")),
        absolute_compare_allowed=bool(entry.get("absolute_compare_allowed", False)),
        provenance={
            "source": entry.get("source"),
            "reference_voltage_assumption": (
                "Bench curve voltage unspecified; using provisional 12 V reference"
            ),
            "fit_method": "least_squares_dp_equals_k0_q2_plus_k2",
            "flow_unit_note": entry.get("flow_unit_note"),
        },
    )


def fit_rear_booster_reference_curve(
    *,
    n_ref: float = _DEFAULT_N_REF,
    rho_kg_m3: float = _DEFAULT_RHO_REF,
    reference_voltage_v: float = _DEFAULT_V_REF,
) -> FanReferenceCurve:
    """Fit rear booster reference curve from ``fan_curve_initial.json``."""
    entry = _entry_by_fan_id("rear_booster")
    points = _points_from_entry(entry)
    fit_points = tuple(pt for pt in points if pt[0] >= 0.0)
    k0, k2, max_res = _fit_k0_k2_from_pq_points(fit_points, rho_kg_m3=rho_kg_m3)
    params = _fan_params_from_k_at_n_ref(k0, k2, n_ref=n_ref, rho_kg_m3=rho_kg_m3)
    return FanReferenceCurve(
        fan_id="rear_booster",
        reference_voltage_v=float(reference_voltage_v),
        n_ref_rev_s=float(n_ref),
        rho_ref_kg_m3=float(rho_kg_m3),
        points=points,
        fitted_params=params,
        fit_k0=k0,
        fit_k2=k2,
        fit_max_residual_pa=max_res,
        flow_unit=str(entry.get("flow_unit", "m3/h")),
        absolute_compare_allowed=bool(entry.get("absolute_compare_allowed", True)),
        provenance={
            "source": entry.get("source"),
            "reference_voltage_assumption": (
                "Bench curve voltage unspecified; using provisional 12 V reference"
            ),
            "fit_method": "least_squares_dp_equals_k0_q2_plus_k2",
            "fit_points_policy": "non_negative_back_pressure_only",
            "negative_pressure_points_excluded_from_fit": len(points) - len(fit_points),
        },
    )


def scale_fan_curve_by_voltage(
    reference: FanReferenceCurve,
    voltage_v: float,
    *,
    reference_voltage_v: Optional[float] = None,
    voltage_to_speed: str = "linear",
) -> ScaledFanCurve:
    """Scale reference P-Q points by fan affinity with provisional V∝n."""
    v = float(voltage_v)
    v_ref = float(reference_voltage_v if reference_voltage_v is not None else reference.reference_voltage_v)
    if v <= 0.0 or v_ref <= 0.0:
        raise ValueError("voltages must be positive")
    if voltage_to_speed != "linear":
        raise ValueError(f"unsupported voltage_to_speed: {voltage_to_speed!r}")

    speed_ratio = v / v_ref
    scaled_pts = tuple(
        (p_pa * speed_ratio ** 2, q_m3h * speed_ratio)
        for p_pa, q_m3h in reference.points
    )
    n_eq = reference.n_ref_rev_s * speed_ratio
    return ScaledFanCurve(
        fan_id=reference.fan_id,
        voltage_v=v,
        reference_voltage_v=v_ref,
        speed_ratio=speed_ratio,
        points=scaled_pts,
        equivalent_n_rev_s=n_eq,
        fitted_params=reference.fitted_params,
        provenance={
            "method": "fan_affinity_with_linear_voltage_to_speed",
            "affinity_laws": {"Q": "proportional_to_n", "delta_P": "proportional_to_n_squared"},
            "speed_ratio": speed_ratio,
            "not_wired_to_afe_calc_defaults": True,
        },
    )


def model_flow_m3h_at_back_pressure(
    params: FanParams,
    *,
    back_pressure_pa: float,
    n_rev_s: float,
    rho_kg_m3: float = _DEFAULT_RHO_REF,
) -> float:
    """Evaluate ``fan_k_coeffs`` model flow (m³/h) at a back-pressure point."""
    k = fan_k_coeffs(params, n_rev_s, rho_kg_m3)
    q_m3s = _flow_m3s_at_back_pressure(k, back_pressure_pa)
    return q_m3s / _M3H_TO_M3S


def audit_fan_k_coeffs_affinity(
    fan: FanParams,
    *,
    n_ref: float = _DEFAULT_N_REF,
    n_scaled: float = 500.0,
    rho_kg_m3: float = _DEFAULT_RHO_REF,
) -> Dict[str, Any]:
    """Audit how ``fan_k_coeffs`` scales shutoff pressure and free-flow with speed."""
    k_ref = fan_k_coeffs(fan, n_ref, rho_kg_m3)
    k_s = fan_k_coeffs(fan, n_scaled, rho_kg_m3)
    ratio = n_scaled / n_ref

    shutoff_ref = float(k_ref[2])
    shutoff_s = float(k_s[2])
    q_free_ref = _flow_m3s_at_back_pressure(k_ref, 0.0)
    q_free_s = _flow_m3s_at_back_pressure(k_s, 0.0)

    dp_ratio = shutoff_s / shutoff_ref if shutoff_ref else float("nan")
    q_ratio = q_free_s / q_free_ref if q_free_ref else float("nan")

    return {
        "n_ref": n_ref,
        "n_scaled": n_scaled,
        "speed_ratio": ratio,
        "shutoff_pressure_ref_pa": shutoff_ref,
        "shutoff_pressure_scaled_pa": shutoff_s,
        "shutoff_ratio": dp_ratio,
        "shutoff_matches_n_squared": math.isclose(dp_ratio, ratio ** 2, rel_tol=1e-9),
        "free_flow_q_ref_m3s": q_free_ref,
        "free_flow_q_scaled_m3s": q_free_s,
        "free_flow_ratio": q_ratio,
        "free_flow_matches_n": math.isclose(q_ratio, ratio, rel_tol=1e-6),
        "free_flow_matches_n_squared": math.isclose(q_ratio, ratio ** 2, rel_tol=1e-6),
        "k0_scales_as": "n_ref_over_n_squared",
        "note": (
            "fan_k_coeffs uses k2∝(n/n0)² (ΔP shutoff ∝ n²). "
            "k0∝(n0/n)² yields Q_free∝(n/n0)² in this parameterization, "
            "not strict affinity Q∝n unless k0 is held constant."
        ),
    }


def build_fan_curve_model_status_report(
    *,
    n_ref: float = _DEFAULT_N_REF,
    rho_kg_m3: float = _DEFAULT_RHO_REF,
) -> Dict[str, Any]:
    """Assemble structured status for markdown/JSON export."""
    front = fit_front_fan_reference_curve(n_ref=n_ref, rho_kg_m3=rho_kg_m3)
    rear = fit_rear_booster_reference_curve(n_ref=n_ref, rho_kg_m3=rho_kg_m3)
    default_fan = FanParams()
    affinity_default = audit_fan_k_coeffs_affinity(default_fan, n_ref=n_ref, n_scaled=n_ref * 0.5)
    affinity_front = audit_fan_k_coeffs_affinity(front.fitted_params, n_ref=n_ref, n_scaled=n_ref * 0.5)
    scaled_6v = scale_fan_curve_by_voltage(rear, 6.0)
    return {
        "schema": "afe_fan_curve_model_status_v1",
        "default_fan_params_unchanged": True,
        "wired_to_afe_calc_defaults": False,
        "affinity_audit_default_fanparams": affinity_default,
        "affinity_audit_fitted_rear": audit_fan_k_coeffs_affinity(
            rear.fitted_params, n_ref=n_ref, n_scaled=n_ref * 0.5
        ),
        "front_reference": {
            "fan_id": front.fan_id,
            "fit_max_residual_pa": front.fit_max_residual_pa,
            "flow_unit": front.flow_unit,
            "absolute_compare_allowed": front.absolute_compare_allowed,
            "fitted_p0": front.fitted_params.p0,
            "fitted_a0": front.fitted_params.a0,
        },
        "rear_reference": {
            "fan_id": rear.fan_id,
            "fit_max_residual_pa": rear.fit_max_residual_pa,
            "fitted_p0": rear.fitted_params.p0,
            "fitted_a0": rear.fitted_params.a0,
        },
        "rear_scaled_6v_from_12v": {
            "speed_ratio": scaled_6v.speed_ratio,
            "free_flow_q_m3h": scaled_6v.points[2][1] if len(scaled_6v.points) > 2 else None,
            "shutoff_pressure_pa": scaled_6v.points[-1][0] if scaled_6v.points else None,
        },
    }


def render_fan_curve_model_status_markdown(
    report: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render human-readable fan curve model status."""
    r = dict(report) if report is not None else build_fan_curve_model_status_report()
    aff = r["affinity_audit_default_fanparams"]
    lines = [
        "# AFE Fan Curve Model Status",
        "",
        "Structure and affinity audit for `hvac_sim/afe/fan.py` reference-curve fitting.",
        "**Default `FanSet()` / `afe_calc` behavior is unchanged.**",
        "",
        "## 1. Affinity law audit (`fan_k_coeffs`)",
        "",
        f"- Speed ratio audited: {aff['speed_ratio']:.3f} (n={aff['n_scaled']} vs n={aff['n_ref']})",
        f"- Shutoff ΔP ratio: {aff['shutoff_ratio']:.4f} (expected n²={aff['speed_ratio']**2:.4f})",
        f"- **ΔP ∝ n² at Q=0:** {'yes' if aff['shutoff_matches_n_squared'] else 'no'}",
        f"- Free-flow Q ratio: {aff['free_flow_ratio']:.4f}",
        f"- **Q ∝ n (free flow via `fan_k_coeffs`):** {'yes' if aff['free_flow_matches_n'] else 'no'}",
        f"- **Q ∝ n² (observed parameterization):** {'yes' if aff['free_flow_matches_n_squared'] else 'no'}",
        "",
        aff["note"],
        "",
        "## 2. Reference curve fit (`fan_curve_initial.json`)",
        "",
        "| Fan | fit max residual [Pa] | flow unit | abs compare |",
        "|-----|----------------------|-----------|-------------|",
        f"| front | {r['front_reference']['fit_max_residual_pa']:.2f} | "
        f"{r['front_reference']['flow_unit']} | "
        f"{r['front_reference']['absolute_compare_allowed']} |",
        f"| rear | {r['rear_reference']['fit_max_residual_pa']:.2f} | "
        f"m3/h | {r['rear_reference'].get('absolute_compare_allowed', True)} |",
        "",
        "Functions: `fit_front_fan_reference_curve()`, `fit_rear_booster_reference_curve()`, "
        "`scale_fan_curve_by_voltage()`.",
        "",
        "## 3. Voltage scaling (provisional)",
        "",
        "Assumption: `n / n_ref = V / V_ref` (linear, pending multi-voltage bench data).",
        "",
        f"- Rear 6 V from 12 V reference: speed_ratio={r['rear_scaled_6v_from_12v']['speed_ratio']:.2f}",
        "- Affinity scaling on reference **points**: Q×(V/Vref), ΔP×(V/Vref)²",
        "- **Mismatch:** scaled points follow Q∝n; `fan_k_coeffs` free-flow at the same "
        "equivalent speed follows Q∝n². Do not mix the two without reconciling k0 scaling.",
        "",
        "## 4. Policy",
        "",
        f"- Default FanParams unchanged: {r['default_fan_params_unchanged']}",
        f"- Wired to afe_calc defaults: {r['wired_to_afe_calc_defaults']}",
        "- Formal PSO / default parameter replacement: **not enabled**",
        "",
    ]
    return "\n".join(lines)
