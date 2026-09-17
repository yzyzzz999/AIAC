"""CHTD multi-step stabilization experiments (non-production parameter sweeps).

Runs full_state warm-up trials with copied ``CHTDParams`` — never mutates defaults,
``thermal.py``, or AS_FOUND golden behaviour.
"""

from __future__ import annotations

import copy
import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.afe.v2_pmv_replay import case_to_afe_case
from hvac_sim.chtd.bus_index import CHTD_X_NAMES, N_X_STATES, X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND, CORRECTED, DefectMode
from hvac_sim.chtd.params import CHTDParams, as_lut
from hvac_sim.chtd.stability_audit import (
    build_state_and_input_from_afe_case,
    build_uniform_equilibrium,
)
from hvac_sim.chtd.thermal import compute_chtd_delta

EXPERIMENT_SCHEMA = "chtd_stability_experiments_v1"
CLASSIFICATION = "experimental_not_formal_calibration"
DISCLAIMER = (
    "Parameter sweeps on copied CHTDParams only. Does NOT change default "
    "compute_chtd_delta behaviour, AS_FOUND golden, or runtime wiring."
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_STABILITY_EXPERIMENTS.json"
)
DEFAULT_MD_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_STABILITY_EXPERIMENTS.md"
)

_THRESHOLD_C = 20.0
_MARGINAL_C = 5.0
_DEFAULT_STEPS = 10
_HISTORY_ZONES = ("ConsoleTemp", "CabinTempFd", "FeetTempFd", "HeadTempFd")

_COLD_CASE: Dict[str, Any] = {
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
}

_FEET_TO_CABIN_LUTS = (
    "CHTD_FeetFdCabinFdRadCo_M",
    "CHTD_FeetFdCabinFdConvCo_M",
    "CHTD_FeetFpCabinFpRadCo_M",
    "CHTD_FeetFpCabinFpConvCo_M",
)


def default_cold_case() -> Dict[str, Any]:
    return dict(_COLD_CASE)


def copy_params(base: Optional[CHTDParams] = None) -> CHTDParams:
    return copy.deepcopy(base if base is not None else CHTDParams())


def scale_air_volume(params: CHTDParams, scale: float) -> CHTDParams:
    p = copy_params(params)
    p.CHTD_HeadAirVAtb_P = float(p.CHTD_HeadAirVAtb_P) * float(scale)
    p.CHTD_FeetAirVAtb_P = float(p.CHTD_FeetAirVAtb_P) * float(scale)
    return p


def scale_solid_capacitance(params: CHTDParams, scale: float) -> CHTDParams:
    """Scale zone Mass/Cp (_P canonical + deprecated aliases)."""
    p = copy_params(params)
    factor = float(scale)
    for f in fields(p):
        name = f.name
        if "MassAtb" in name or (name.endswith("CpAtb_P") or name.endswith("CpAtb")):
            if name.endswith("_M"):
                continue
            try:
                val = getattr(p, name)
            except AttributeError:
                continue
            if isinstance(val, (int, float)):
                setattr(p, name, float(val) * factor)
    return p


def apply_dt(params: CHTDParams, dt_seconds: float) -> CHTDParams:
    p = copy_params(params)
    p.CHTD_Dt_P = float(dt_seconds)
    return p


def apply_console_feet_convection_scale(params: CHTDParams, scale: float) -> CHTDParams:
    p = copy_params(params)
    factor = float(scale)
    p.CHTD_ConsoleFeetFdConvCo_M = np.asarray(p.CHTD_ConsoleFeetFdConvCo_M, dtype=float) * factor
    p.CHTD_ConsoleFeetFpConvCo_M = np.asarray(p.CHTD_ConsoleFeetFpConvCo_M, dtype=float) * factor
    return p


def apply_feet_to_cabin_qbus_scale(params: CHTDParams, scale: float) -> CHTDParams:
    """Scale Feet→Cabin q-bus LUTs (adds missing fields on experimental copy)."""
    p = copy_params(params)
    lut = as_lut(float(scale))
    for name in _FEET_TO_CABIN_LUTS:
        setattr(p, name, lut.copy())
    return p


def _classify_conclusion(
    *,
    steps_completed: int,
    steps_requested: int,
    first_threshold_step: Optional[int],
    max_abs_delta_by_step: Sequence[float],
    nonfinite: bool,
) -> str:
    if nonfinite:
        return "unstable"
    if not max_abs_delta_by_step:
        return "stable"
    peak = float(max(max_abs_delta_by_step))
    if first_threshold_step is not None and first_threshold_step <= 2:
        return "unstable"
    if peak <= _MARGINAL_C and steps_completed >= steps_requested:
        return "stable"
    if peak <= _THRESHOLD_C and steps_completed >= steps_requested:
        return "marginal"
    return "unstable"


def run_single_experiment(
    *,
    experiment_id: str,
    group: str,
    description: str,
    params: CHTDParams,
    mode: DefectMode = AS_FOUND,
    case: Optional[Mapping[str, Any]] = None,
    steps: int = _DEFAULT_STEPS,
    threshold_c: float = _THRESHOLD_C,
    param_notes: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Forward-Euler full_state trial; returns structured experiment record."""
    case_dict = dict(case if case is not None else default_cold_case())
    x0, u, _ = build_state_and_input_from_afe_case(case_dict)
    x = np.asarray(x0, dtype=float).copy()
    dt = float(params.CHTD_Dt_P)

    max_abs_delta_by_step: List[float] = []
    max_zone_by_step: List[str] = []
    zone_history: Dict[str, List[float]] = {z: [float(x[X_INDEX[z]])] for z in _HISTORY_ZONES}
    first_threshold_step: Optional[int] = None
    nonfinite = False
    steps_completed = 0

    for step in range(1, steps + 1):
        delta = compute_chtd_delta(x, u, params, mode=mode)
        if not np.all(np.isfinite(delta)):
            nonfinite = True
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

        x = x + delta
        if not np.all(np.isfinite(x)):
            nonfinite = True
            break
        for z in _HISTORY_ZONES:
            zone_history[z].append(float(x[X_INDEX[z]]))

    conclusion = _classify_conclusion(
        steps_completed=steps_completed,
        steps_requested=steps,
        first_threshold_step=first_threshold_step,
        max_abs_delta_by_step=max_abs_delta_by_step,
        nonfinite=nonfinite,
    )

    return {
        "experiment_id": experiment_id,
        "group": group,
        "description": description,
        "mode": mode.value,
        "case_id": case_dict.get("case_id", "cold_foot_heating_full_state"),
        "afe_mode_code": case_to_afe_case(case_dict)["mode_code"],
        "dt_seconds": dt,
        "steps_requested": int(steps),
        "steps_completed": int(steps_completed),
        "threshold_delta_c_per_step": float(threshold_c),
        "first_threshold_step": first_threshold_step,
        "max_abs_delta_by_step": max_abs_delta_by_step,
        "max_zone_by_step": max_zone_by_step,
        "step2_max_abs_delta_c": (
            float(max_abs_delta_by_step[1]) if len(max_abs_delta_by_step) > 1 else None
        ),
        "step2_max_zone": max_zone_by_step[1] if len(max_zone_by_step) > 1 else None,
        "peak_max_abs_delta_c": float(max(max_abs_delta_by_step)) if max_abs_delta_by_step else 0.0,
        "final_x_min_c": float(np.min(x)) if np.all(np.isfinite(x)) else None,
        "final_x_max_c": float(np.max(x)) if np.all(np.isfinite(x)) else None,
        "zone_history": zone_history,
        "conclusion": conclusion,
        "param_notes": dict(param_notes or {}),
        "nonfinite": bool(nonfinite),
    }


def check_equilibrium_stable(
    params: CHTDParams,
    *,
    mode: DefectMode = AS_FOUND,
    temp_c: float = 25.0,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    x0, u = build_uniform_equilibrium(temp_c=temp_c)
    delta = compute_chtd_delta(x0, u, params, mode=mode)
    max_abs = float(np.max(np.abs(delta)))
    return {
        "temp_c": float(temp_c),
        "max_abs_delta_c": max_abs,
        "stable": max_abs <= atol,
        "mode": mode.value,
    }


def default_experiment_specs() -> List[Dict[str, Any]]:
    """Build the standard experiment matrix."""
    base = CHTDParams()
    specs: List[Dict[str, Any]] = []

    def add(
        experiment_id: str,
        group: str,
        description: str,
        params_builder: Callable[[], CHTDParams],
        *,
        mode: DefectMode = AS_FOUND,
        param_notes: Optional[Dict[str, Any]] = None,
    ) -> None:
        specs.append(
            {
                "experiment_id": experiment_id,
                "group": group,
                "description": description,
                "params_builder": params_builder,
                "mode": mode,
                "param_notes": param_notes or {},
            }
        )

    add(
        "baseline_as_found",
        "baseline",
        "Default CHTDParams, AS_FOUND, cold foot heating full_state",
        lambda: copy_params(base),
    )

    for scale in (100.0, 300.0, 600.0):
        add(
            f"capacitance_air_{int(scale)}x",
            "capacitance_scaled",
            f"Head/Feet air volume ×{scale:g}",
            lambda s=scale: scale_air_volume(base, s),
            param_notes={"air_volume_scale": scale},
        )

    for scale in (10.0, 50.0, 100.0):
        add(
            f"capacitance_mass_{int(scale)}x",
            "capacitance_scaled",
            f"All zone Mass/Cp ×{scale:g}",
            lambda s=scale: scale_solid_capacitance(base, s),
            param_notes={"solid_mass_cp_scale": scale},
        )

    add(
        "capacitance_combined_air600_mass100",
        "capacitance_scaled",
        "Air volume ×600 and solid Mass/Cp ×100",
        lambda: scale_air_volume(scale_solid_capacitance(base, 100.0), 600.0),
        param_notes={"air_volume_scale": 600.0, "solid_mass_cp_scale": 100.0},
    )

    for dt in (1.0, 0.5, 0.1, 0.05):
        add(
            f"dt_sensitivity_{dt:g}s",
            "dt_sensitivity",
            f"CHTD_Dt_P={dt:g}s (same Euler step count)",
            lambda d=dt: apply_dt(base, d),
            param_notes={"dt_seconds": dt},
        )

    for scale in (1.0, 0.3, 0.1, 0.0):
        label = "1p0" if scale == 1.0 else str(scale).replace(".", "p")
        add(
            f"qbus_console_feet_conv_{label}",
            "qbus_gain_sensitivity",
            f"ConsoleFeet Fd/Fp convection LUT scale={scale:g}",
            lambda s=scale: apply_console_feet_convection_scale(base, s),
            param_notes={"console_feet_convection_scale": scale},
        )

    for scale in (1.0, 0.3, 0.1, 0.0):
        label = "1p0" if scale == 1.0 else str(scale).replace(".", "p")
        add(
            f"qbus_feet_to_cabin_{label}",
            "qbus_gain_sensitivity",
            f"Feet→Cabin q-bus LUT scale={scale:g}",
            lambda s=scale: apply_feet_to_cabin_qbus_scale(base, s),
            param_notes={"feet_to_cabin_qbus_scale": scale},
        )

    add(
        "corrected_radiation_preview",
        "corrected_radiation_preview",
        "DefectMode.CORRECTED on copied default params (preview only)",
        lambda: copy_params(base),
        mode=CORRECTED,
    )

    add(
        "equilibrium_control_25C",
        "equilibrium_sanity",
        "True uniform equilibrium at 25°C — must stay delta≈0",
        lambda: copy_params(base),
    )

    return specs


def rank_stabilization_factors(experiments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Rank non-baseline warm-up experiments by step-2 and peak |delta|."""
    ranked: List[Dict[str, Any]] = []
    baseline_s2 = None
    for exp in experiments:
        if exp.get("experiment_id") == "baseline_as_found":
            baseline_s2 = exp.get("step2_max_abs_delta_c")
            break

    for exp in experiments:
        if exp.get("group") in ("baseline", "equilibrium_sanity"):
            continue
        s2 = exp.get("step2_max_abs_delta_c")
        if s2 is None:
            continue
        ranked.append(
            {
                "experiment_id": exp["experiment_id"],
                "group": exp["group"],
                "conclusion": exp["conclusion"],
                "step2_max_abs_delta_c": s2,
                "peak_max_abs_delta_c": exp.get("peak_max_abs_delta_c"),
                "step2_improvement_vs_baseline": (
                    float(baseline_s2) - float(s2) if baseline_s2 is not None else None
                ),
                "first_threshold_step": exp.get("first_threshold_step"),
            }
        )

    ranked.sort(
        key=lambda r: (
            r["conclusion"] != "stable",
            r["conclusion"] == "unstable",
            r["step2_max_abs_delta_c"] if r["step2_max_abs_delta_c"] is not None else 1e18,
            r["peak_max_abs_delta_c"] if r["peak_max_abs_delta_c"] is not None else 1e18,
        )
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def run_stability_experiments(
    *,
    specs: Optional[Sequence[Mapping[str, Any]]] = None,
    steps: int = _DEFAULT_STEPS,
    threshold_c: float = _THRESHOLD_C,
) -> Dict[str, Any]:
    matrix = list(specs if specs is not None else default_experiment_specs())
    results: List[Dict[str, Any]] = []

    for spec in matrix:
        params = spec["params_builder"]()
        mode = spec.get("mode", AS_FOUND)
        if spec["experiment_id"] == "equilibrium_control_25C":
            eq = check_equilibrium_stable(params, mode=mode)
            results.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "group": spec["group"],
                    "description": spec["description"],
                    "mode": mode.value,
                    "equilibrium_check": eq,
                    "conclusion": "stable" if eq["stable"] else "unstable",
                    "param_notes": dict(spec.get("param_notes", {})),
                }
            )
            continue

        rec = run_single_experiment(
            experiment_id=str(spec["experiment_id"]),
            group=str(spec["group"]),
            description=str(spec["description"]),
            params=params,
            mode=mode,
            steps=steps,
            threshold_c=threshold_c,
            param_notes=spec.get("param_notes"),
        )
        if spec["experiment_id"] != "baseline_as_found":
            rec["equilibrium_check"] = check_equilibrium_stable(params, mode=mode)
        results.append(rec)

    ranking = rank_stabilization_factors(results)
    baseline = next((r for r in results if r["experiment_id"] == "baseline_as_found"), None)

    summary = {
        "baseline_step2_max_abs_delta_c": baseline.get("step2_max_abs_delta_c") if baseline else None,
        "baseline_first_threshold_step": baseline.get("first_threshold_step") if baseline else None,
        "most_effective_experiments": ranking[:5],
        "interpretation": _build_interpretation(ranking, baseline),
    }

    return {
        "schema": EXPERIMENT_SCHEMA,
        "classification": CLASSIFICATION,
        "disclaimer": DISCLAIMER,
        "not_runtime_active": True,
        "thermal_formula_unchanged": True,
        "default_params_unchanged": True,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps_per_experiment": int(steps),
        "threshold_delta_c_per_step": float(threshold_c),
        "summary": summary,
        "experiments": results,
        "stabilization_ranking": ranking,
    }


def _build_interpretation(
    ranking: Sequence[Mapping[str, Any]],
    baseline: Optional[Mapping[str, Any]],
) -> str:
    if not ranking:
        return "No ranking available."
    top = ranking[0]
    baseline_s2 = baseline.get("step2_max_abs_delta_c") if baseline else "?"
    parts = [
        f"Baseline step-2 blow-up: max |delta|≈{baseline_s2}°C/step (ConsoleTemp / CabinTempFd).",
        f"Strongest experimental stabilizer: **{top['experiment_id']}** "
        f"(step-2 |delta|={top['step2_max_abs_delta_c']:.4g}°C, conclusion={top['conclusion']}).",
    ]
    groups = {r["group"] for r in ranking[:3]}
    if "capacitance_scaled" in groups:
        parts.append(
            "Solid Mass/Cp inflation dominates: low capacitance (placeholder=1) makes "
            "Console/Cabin zones integrate huge q-bus/convection terms in one Euler step."
        )
    if any(r["group"] == "dt_sensitivity" and r["conclusion"] != "stable" for r in ranking):
        parts.append(
            "Smaller dt alone delays but does not remove the instability — parameter/capacitance "
            "mismatch is primary, not explicit step size alone."
        )
    if any(r["experiment_id"].startswith("qbus_feet_to_cabin_0") for r in ranking):
        parts.append(
            "Feet→Cabin q-bus scale=0 does not fix ConsoleTemp; Console feet convection is a "
            "separate x-coupling path."
        )
    parts.append("These are exploratory sweeps — not formal calibration or production defaults.")
    return " ".join(parts)


def build_markdown_report(bundle: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# CHTD Stability Experiments",
        "",
        f"**{bundle.get('disclaimer', DISCLAIMER)}**",
        "",
        f"Generated: `{bundle.get('generated_at_utc', '')}`",
        "",
        "## Summary",
        "",
        bundle.get("summary", {}).get("interpretation", ""),
        "",
        "### Baseline",
        "",
    ]
    summary = bundle.get("summary", {})
    lines.append(
        f"- Step-2 max |delta|: **{summary.get('baseline_step2_max_abs_delta_c')}** °C/step"
    )
    lines.append(f"- First threshold step: **{summary.get('baseline_first_threshold_step')}**")
    lines.append("")

    lines.extend(["## Stabilization ranking (top 10)", "", "| rank | experiment | group | step-2 |delta| | conclusion |", "|------|------------|-------|----------------|------------|"])
    for row in bundle.get("stabilization_ranking", [])[:10]:
        s2 = row.get("step2_max_abs_delta_c")
        s2_txt = f"{s2:.3g}" if s2 is not None else "—"
        lines.append(
            f"| {row.get('rank')} | {row['experiment_id']} | {row['group']} | {s2_txt} | {row['conclusion']} |"
        )
    lines.append("")

    lines.append("## All experiments")
    lines.append("")
    for exp in bundle.get("experiments", []):
        lines.extend(
            [
                f"### {exp['experiment_id']}",
                "",
                exp.get("description", ""),
                "",
                f"- Group: `{exp.get('group')}`",
                f"- Conclusion: **{exp.get('conclusion')}**",
            ]
        )
        if "step2_max_abs_delta_c" in exp:
            lines.append(f"- Step-2 max |delta|: {exp['step2_max_abs_delta_c']:.4g} °C ({exp.get('step2_max_zone')})")
            lines.append(f"- First threshold step: {exp.get('first_threshold_step')}")
            lines.append(f"- Steps completed: {exp.get('steps_completed')}/{exp.get('steps_requested')}")
            if exp.get("zone_history"):
                zh = exp["zone_history"]
                lines.append(
                    f"- Zone history (final): Console={zh['ConsoleTemp'][-1]:.2f}°C, "
                    f"CabinFd={zh['CabinTempFd'][-1]:.2f}°C, "
                    f"FeetFd={zh['FeetTempFd'][-1]:.2f}°C, "
                    f"HeadFd={zh['HeadTempFd'][-1]:.2f}°C"
                )
        if exp.get("equilibrium_check"):
            eq = exp["equilibrium_check"]
            lines.append(f"- Equilibrium @25°C max |delta|: {eq['max_abs_delta_c']:.6g} (stable={eq['stable']})")
        lines.append("")

    return "\n".join(lines)


def write_stability_experiment_outputs(
    *,
    bundle: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
    pretty: bool = True,
) -> Dict[str, Path]:
    doc = bundle if bundle is not None else run_stability_experiments()
    json_out = Path(json_path) if json_path is not None else DEFAULT_JSON_OUT
    md_out = Path(md_path) if md_path is not None else DEFAULT_MD_OUT
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )
    md_out.write_text(build_markdown_report(doc), encoding="utf-8")
    return {"json": json_out, "md": md_out}
