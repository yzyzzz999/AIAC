"""First-phase CHTD HVAC convection scale calibration scaffold (demo only).

Applies scalar scale factors to tied LUT / capacity groups on copied ``CHTDParams``.
Does not modify ``CHTDParams`` defaults, ``thermal.py``, or AS_FOUND golden.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.calibration.pso import PSOOptions, pso_optimize
from hvac_sim.calibration.schema import (
    ParameterSpec,
    specs_to_bounds,
    specs_to_initial_vector,
    specs_to_names,
)
from hvac_sim.chtd.bus_index import X_INDEX
from hvac_sim.chtd.helpers import AS_FOUND
from hvac_sim.chtd.params import CHTDParams, as_lut
from hvac_sim.chtd.physical_params import build_physical_capacity_params
from hvac_sim.chtd.stability_audit import (
    build_state_and_input_from_afe_case,
    build_uniform_equilibrium,
)
from hvac_sim.chtd.thermal import compute_chtd_delta

SCHEMA = "chtd_hvac_scale_calibration_scaffold_v1"
CLASSIFICATION = "demo_only"
NOT_VEHICLE_CALIBRATION = True

WHY_NOT_200_PARAMS = (
    "First phase binds ~10 scalar scales across symmetric front/2nd-row HVAC conv LUTs "
    "and lumped capacity groups — not 200+ individual placeholders or 7-point AmbT LUT "
    "surfaces. Third row (Td/Tp) excluded. Real vehicle sign-off needs measured transients."
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_HVAC_SCALE_CALIBRATION_SMOKE.json"
)
DEFAULT_MD_OUT = (
    _REPO_ROOT
    / "simulink_conversion_package"
    / "python_targets"
    / "CHTD_HVAC_SCALE_CALIBRATION_SMOKE.md"
)

_CONV_SCALE_BOUNDS = (0.01, 1.0)
_AIR_VOLUME_SCALE_BOUNDS = (0.3, 30.0)
_SOLID_CAPACITY_SCALE_BOUNDS = (0.3, 100.0)

# ── Tied LUT / capacity groups (third row Td/Tp excluded) ───────────────────

TIED_PARAM_GROUPS: Dict[str, Tuple[str, ...]] = {
    "front_face_head_conv_scale": (
        "CHTD_FdvFdConvCo_M",
        "CHTD_FpvFpConvCo_M",
    ),
    "front_foot_feet_conv_scale": (
        "CHTD_FdfFeetFdConvCo_M",
        "CHTD_FpfFeetFpConvCo_M",
    ),
    "second_row_face_head_conv_scale": (
        "CHTD_SdvSdConvCo_M",
        "CHTD_RearSdvSdConvCo_M",
        "CHTD_SpvSpConvCo_M",
        "CHTD_RearSpvSpConvCo_M",
    ),
    "second_row_foot_feet_conv_scale": (
        "CHTD_SdfFeetSdConvCo_M",
        "CHTD_RearSdfFeetSdConvCo_M",
        "CHTD_SpfFeetSpConvCo_M",
        "CHTD_RearSpfFeetSpConvCo_M",
    ),
    "feet_to_cabin_coupling_scale": (
        "CHTD_FeetFdCabinFdRadCo_M",
        "CHTD_FeetFdCabinFdConvCo_M",
        "CHTD_FeetFpCabinFpRadCo_M",
        "CHTD_FeetFpCabinFpConvCo_M",
    ),
    "console_feet_coupling_scale": (
        "CHTD_ConsoleFeetFdRadCo_M",
        "CHTD_ConsoleFeetFdConvCo_M",
        "CHTD_ConsoleFeetFpRadCo_M",
        "CHTD_ConsoleFeetFpConvCo_M",
    ),
}

HEAD_AIR_VOLUME_ATTR = "CHTD_HeadAirVAtb_P"
FEET_AIR_VOLUME_ATTR = "CHTD_FeetAirVAtb_P"

CABIN_SOLID_CAPACITY_ATTRS: Tuple[str, ...] = (
    "CHTD_CabinFdMassAtb_P",
    "CHTD_CabinFdCpAtb_P",
    "CHTD_CabinFpMassAtb_P",
    "CHTD_CabinFpCpAtb_P",
    "CHTD_CabinSdMassAtb_P",
    "CHTD_CabinSdCpAtb_P",
    "CHTD_CabinSpMassAtb_P",
    "CHTD_CabinSpCpAtb_P",
    "CHTD_CabinFrntMassAtb_P",
    "CHTD_CabinFrntCpAtb_P",
)

ROOF_WIN_CONSOLE_CAPACITY_ATTRS: Tuple[str, ...] = (
    "CHTD_RoofMassAtb_P",
    "CHTD_RoofCpAtb_P",
    "CHTD_ConsoleMassAtb_P",
    "CHTD_ConsoleCpAtb_P",
    "CHTD_HoodMassAtb_P",
    "CHTD_HoodCpAtb_P",
    "CHTD_WinFdMassAtb_P",
    "CHTD_WinFdCpAtb_P",
    "CHTD_WinFpMassAtb_P",
    "CHTD_WinFpCpAtb_P",
    "CHTD_WinSdMassAtb_P",
    "CHTD_WinSdCpAtb_P",
    "CHTD_WinSpMassAtb_P",
    "CHTD_WinSpCpAtb_P",
)

SCALAR_CAPACITY_BINDINGS: Dict[str, Tuple[str, ...]] = {
    "head_air_volume_scale": (HEAD_AIR_VOLUME_ATTR,),
    "feet_air_volume_scale": (FEET_AIR_VOLUME_ATTR,),
    "cabin_solid_capacity_scale": CABIN_SOLID_CAPACITY_ATTRS,
    "roof_win_console_capacity_scale": ROOF_WIN_CONSOLE_CAPACITY_ATTRS,
}

DEMO_COLD_FOOT_HEATING_CASE: Dict[str, Any] = {
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

DEMO_FACE_COOLING_CASE: Dict[str, Any] = {
    "case_id": "synthetic_face_cooling_demo",
    "mode_code": "F",
    "amb_t": 30.0,
    "rh": 50.0,
    "driver_face_tma": 16.0,
    "passenger_face_tma": 16.0,
    "driver_foot_tma": 30.0,
    "passenger_foot_tma": 30.0,
    "rear_face_tma": 30.0,
    "front_blower_voltage": 12.0,
    "rear_blower_voltage": 12.0,
    "circle_mode_posn": 100.0,
    "circle_prior_posn": 4.56,
    "heating_mode": False,
}


@dataclass(frozen=True)
class CHTDHvacScaleParameterSpec:
    """One scalar scale knob with tied CHTD param names."""

    name: str
    initial: float
    lower: float
    upper: float
    tied_params: Tuple[str, ...]
    reason: str
    optional: bool = False

    def to_parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(
            name=self.name,
            initial=self.initial,
            lower=self.lower,
            upper=self.upper,
            description=self.reason,
            target_model="CHTD",
            source="chtd_hvac_scale_scaffold",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "initial": self.initial,
            "lower": self.lower,
            "upper": self.upper,
            "tied_params": list(self.tied_params),
            "reason": self.reason,
            "optional": self.optional,
        }


def build_chtd_hvac_scale_parameter_specs(
    *,
    include_optional: bool = True,
) -> Tuple[CHTDHvacScaleParameterSpec, ...]:
    """Return first-phase scale parameter definitions."""
    specs: List[CHTDHvacScaleParameterSpec] = [
        CHTDHvacScaleParameterSpec(
            "front_face_head_conv_scale",
            0.2,
            *_CONV_SCALE_BOUNDS,
            TIED_PARAM_GROUPS["front_face_head_conv_scale"],
            "HVAC convection audit: face vent (Fdv/Fpv) → head; symmetric Fd/Fp bind",
        ),
        CHTDHvacScaleParameterSpec(
            "front_foot_feet_conv_scale",
            0.1,
            *_CONV_SCALE_BOUNDS,
            TIED_PARAM_GROUPS["front_foot_feet_conv_scale"],
            "Audit required_lut_scale_for_2c≈0.08; foot duct LUT placeholder bind Fdf/Fpf→feet",
        ),
        CHTDHvacScaleParameterSpec(
            "second_row_face_head_conv_scale",
            0.2,
            *_CONV_SCALE_BOUNDS,
            TIED_PARAM_GROUPS["second_row_face_head_conv_scale"],
            "2nd-row face (Sdv/Spv) → head; symmetric left/right + center bind",
        ),
        CHTDHvacScaleParameterSpec(
            "second_row_foot_feet_conv_scale",
            0.1,
            *_CONV_SCALE_BOUNDS,
            TIED_PARAM_GROUPS["second_row_foot_feet_conv_scale"],
            "2nd-row foot (Sdf/Spf) → feet; third row excluded",
        ),
        CHTDHvacScaleParameterSpec(
            "head_air_volume_scale",
            1.0,
            *_AIR_VOLUME_SCALE_BOUNDS,
            (HEAD_AIR_VOLUME_ATTR,),
            "Scales CHTD_HeadAirVAtb_P lumped head air capacitance",
        ),
        CHTDHvacScaleParameterSpec(
            "feet_air_volume_scale",
            1.0,
            *_AIR_VOLUME_SCALE_BOUNDS,
            (FEET_AIR_VOLUME_ATTR,),
            "Audit: excel FeetAirV small → low C; scale without 7-point LUT",
        ),
        CHTDHvacScaleParameterSpec(
            "cabin_solid_capacity_scale",
            1.0,
            *_SOLID_CAPACITY_SCALE_BOUNDS,
            CABIN_SOLID_CAPACITY_ATTRS,
            "Lumped cabin trim/air mass×cp (Fd/Fp/Sd/Sp + CabinFrnt); no 3rd row",
        ),
        CHTDHvacScaleParameterSpec(
            "roof_win_console_capacity_scale",
            1.0,
            *_SOLID_CAPACITY_SCALE_BOUNDS,
            ROOF_WIN_CONSOLE_CAPACITY_ATTRS,
            "Roof/console/hood + front/2nd-row glazing solids; Td/Tp win excluded",
        ),
    ]
    if include_optional:
        specs.extend(
            [
                CHTDHvacScaleParameterSpec(
                    "feet_to_cabin_coupling_scale",
                    1.0,
                    *_CONV_SCALE_BOUNDS,
                    TIED_PARAM_GROUPS["feet_to_cabin_coupling_scale"],
                    "Optional: feet↔cabin inter-zone conv/rad LUT scale (front row)",
                    optional=True,
                ),
                CHTDHvacScaleParameterSpec(
                    "console_feet_coupling_scale",
                    1.0,
                    *_CONV_SCALE_BOUNDS,
                    TIED_PARAM_GROUPS["console_feet_coupling_scale"],
                    "Optional: console↔feet q-bus LUT scale (warm-up coupling)",
                    optional=True,
                ),
            ]
        )
    return tuple(specs)


def _scale_lut_on_params(params: CHTDParams, lut_name: str, scale: float) -> None:
    factor = float(scale)
    if hasattr(params, lut_name):
        arr = np.asarray(getattr(params, lut_name), dtype=float)
        setattr(params, lut_name, arr * factor)
    else:
        setattr(params, lut_name, as_lut(factor))


def _scale_scalar_attrs(params: CHTDParams, attrs: Sequence[str], scale: float) -> None:
    factor = float(scale)
    for attr in attrs:
        if hasattr(params, attr):
            setattr(params, attr, float(getattr(params, attr)) * factor)


def _vector_or_dict_to_mapping(
    values: Union[np.ndarray, Mapping[str, float], Sequence[float]],
    names: Sequence[str],
) -> Dict[str, float]:
    if isinstance(values, Mapping):
        return {str(k): float(v) for k, v in values.items()}
    vec = np.asarray(values, dtype=float).reshape(-1)
    if vec.shape[0] != len(names):
        raise ValueError(f"scale vector length {vec.shape[0]} != {len(names)} parameters")
    return {name: float(vec[i]) for i, name in enumerate(names)}


def _complete_scale_mapping(
    scale_values: Union[np.ndarray, Mapping[str, float], Sequence[float]],
    specs: Sequence[CHTDHvacScaleParameterSpec],
) -> Dict[str, float]:
    names = [s.name for s in specs]
    defaults = {s.name: float(s.initial) for s in specs}
    if isinstance(scale_values, Mapping):
        out = dict(defaults)
        for key, val in scale_values.items():
            if key not in defaults:
                raise KeyError(f"unknown scale parameter {key!r}")
            out[key] = float(val)
        return out
    return _vector_or_dict_to_mapping(scale_values, names)


def apply_chtd_hvac_scale_params(
    base_params: CHTDParams,
    scale_values: Union[np.ndarray, Mapping[str, float], Sequence[float]],
    *,
    include_optional: bool = True,
) -> CHTDParams:
    """Return deep-copied ``CHTDParams`` with scale factors applied; default unchanged."""
    specs = build_chtd_hvac_scale_parameter_specs(include_optional=include_optional)
    mapping = _complete_scale_mapping(scale_values, specs)

    out = copy.deepcopy(base_params)
    for spec in specs:
        scale = mapping[spec.name]
        if spec.name in TIED_PARAM_GROUPS or spec.name in (
            "feet_to_cabin_coupling_scale",
            "console_feet_coupling_scale",
        ):
            for lut_name in spec.tied_params:
                _scale_lut_on_params(out, lut_name, scale)
        elif spec.name in SCALAR_CAPACITY_BINDINGS:
            _scale_scalar_attrs(out, spec.tied_params, scale)
        else:
            _scale_scalar_attrs(out, spec.tied_params, scale)
    return out


def _feet_in_band_penalty(delta_c: float, lo: float = 2.0, hi: float = 5.0) -> float:
    target = 0.5 * (lo + hi)
    band_err = max(0.0, lo - delta_c, delta_c - hi)
    return float((delta_c - target) ** 2 + 4.0 * band_err ** 2)


def demo_objective(
    scale_values: Union[np.ndarray, Mapping[str, float], Sequence[float]],
    base_params: CHTDParams,
    *,
    include_optional: bool = True,
    feet_dt_band: Tuple[float, float] = (2.0, 5.0),
    face_cooling_target_delta_c: float = -0.5,
    equilibrium_temp_c: float = 25.0,
) -> float:
    """Synthetic pseudo-target loss for scaffold smoke (not vehicle calibration)."""
    params = apply_chtd_hvac_scale_params(
        base_params, scale_values, include_optional=include_optional
    )

    x_heat, u_heat, _ = build_state_and_input_from_afe_case(DEMO_COLD_FOOT_HEATING_CASE)
    delta_heat = compute_chtd_delta(x_heat, u_heat, params, AS_FOUND)
    feet_dt = float(delta_heat[X_INDEX["FeetTempFd"]])
    loss_feet = _feet_in_band_penalty(feet_dt, *feet_dt_band)

    x_cool, u_cool, _ = build_state_and_input_from_afe_case(DEMO_FACE_COOLING_CASE)
    delta_cool = compute_chtd_delta(x_cool, u_cool, params, AS_FOUND)
    head_fd_dt = float(delta_cool[X_INDEX["HeadTempFd"]])
    # Want cooling: negative delta when Tma < T_zone
    loss_face = float(max(0.0, head_fd_dt - face_cooling_target_delta_c) ** 2)
    if head_fd_dt >= 0.0:
        loss_face += float(head_fd_dt ** 2)

    x_eq, u_eq = build_uniform_equilibrium(temp_c=equilibrium_temp_c)
    delta_eq = compute_chtd_delta(x_eq, u_eq, params, AS_FOUND)
    loss_eq = float(np.max(np.abs(delta_eq)) ** 2)

    return loss_feet + 0.5 * loss_face + 2.0 * loss_eq


def _evaluate_demo_terms(
    params: CHTDParams,
) -> Dict[str, float]:
    x_heat, u_heat, _ = build_state_and_input_from_afe_case(DEMO_COLD_FOOT_HEATING_CASE)
    feet_dt = float(
        compute_chtd_delta(x_heat, u_heat, params, AS_FOUND)[X_INDEX["FeetTempFd"]]
    )
    x_cool, u_cool, _ = build_state_and_input_from_afe_case(DEMO_FACE_COOLING_CASE)
    head_fd_dt = float(
        compute_chtd_delta(x_cool, u_cool, params, AS_FOUND)[X_INDEX["HeadTempFd"]]
    )
    x_eq, u_eq = build_uniform_equilibrium(temp_c=25.0)
    eq_max = float(np.max(np.abs(compute_chtd_delta(x_eq, u_eq, params, AS_FOUND))))
    return {
        "feet_temp_fd_step1_delta_c": feet_dt,
        "head_temp_fd_face_cooling_step1_delta_c": head_fd_dt,
        "equilibrium_max_abs_delta_c": eq_max,
    }


def run_demo_calibration_smoke(
    *,
    base_params: Optional[CHTDParams] = None,
    vehicle_class: str = "m8_estimated",
    include_optional: bool = True,
    pso_swarm_size: int = 12,
    pso_max_iter: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run small PSO on demo objective; returns smoke bundle for JSON/MD export."""
    if base_params is None:
        base_params, _prov = build_physical_capacity_params(vehicle_class=vehicle_class)
    else:
        base_params = copy.deepcopy(base_params)
        _prov = {}

    specs = build_chtd_hvac_scale_parameter_specs(include_optional=include_optional)
    param_specs = [s.to_parameter_spec() for s in specs]
    names = specs_to_names(param_specs)
    x0 = specs_to_initial_vector(param_specs)
    bounds = specs_to_bounds(param_specs)

    baseline_params = apply_chtd_hvac_scale_params(
        base_params, dict(zip(names, x0)), include_optional=include_optional
    )
    unscaled_terms = _evaluate_demo_terms(base_params)
    baseline_loss = demo_objective(x0, base_params, include_optional=include_optional)
    baseline_terms = _evaluate_demo_terms(baseline_params)

    def objective(vec: np.ndarray) -> float:
        return demo_objective(vec, base_params, include_optional=include_optional)

    pso_result = pso_optimize(
        objective,
        bounds,
        options=PSOOptions(swarm_size=pso_swarm_size, max_iter=pso_max_iter, seed=seed),
    )

    best_dict = {name: float(pso_result.best_x[i]) for i, name in enumerate(names)}
    best_params = apply_chtd_hvac_scale_params(
        base_params, best_dict, include_optional=include_optional
    )
    best_terms = _evaluate_demo_terms(best_params)

    tied_groups = {
        spec.name: list(spec.tied_params) for spec in specs
    }

    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "not_vehicle_calibration": NOT_VEHICLE_CALIBRATION,
        "why_not_200_params": WHY_NOT_200_PARAMS,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vehicle_class": vehicle_class,
        "base_params_bundle": "physical_capacity_params",
        "parameter_names": names,
        "parameter_specs": [s.to_dict() for s in specs],
        "tied_param_groups": tied_groups,
        "demo_targets": {
            "cold_foot_heating_feet_temp_fd_step1_delta_c_band": [2.0, 5.0],
            "face_cooling_head_temp_fd_direction": "negative_delta_c",
            "equilibrium_max_abs_delta_c": 0.0,
        },
        "baseline": {
            "scale_params": {n: float(x0[i]) for i, n in enumerate(names)},
            "loss": baseline_loss,
            "terms": baseline_terms,
        },
        "physical_unscaled": {
            "terms": unscaled_terms,
            "note": "physical_capacity_params before any scale adapter (reference only)",
        },
        "best_params": best_dict,
        "best_loss": float(pso_result.best_loss),
        "best_terms": best_terms,
        "pso": {
            "swarm_size": pso_swarm_size,
            "max_iter": pso_max_iter,
            "n_iter": pso_result.n_iter,
            "converged": pso_result.converged,
            "history_last_5": [float(x) for x in pso_result.history[-5:]],
        },
        "provenance_note": _prov.get("_meta", {}),
    }


def build_smoke_markdown(doc: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# CHTD HVAC Scale Calibration Smoke (demo only)",
        "",
        f"**classification:** `{doc.get('classification')}` | "
        f"**not_vehicle_calibration:** `{doc.get('not_vehicle_calibration')}`",
        "",
        doc.get("why_not_200_params", ""),
        "",
        f"Generated: `{doc.get('generated_at_utc', '')}`",
        "",
        "## Parameter scales",
        "",
        "| name | initial | bounds | tied count |",
        "|------|---------|--------|------------|",
    ]
    for spec in doc.get("parameter_specs", []):
        lines.append(
            f"| {spec['name']} | {spec['initial']} | [{spec['lower']}, {spec['upper']}] | "
            f"{len(spec['tied_params'])} |"
        )
    lines.extend(
        [
            "",
            "## Demo results",
            "",
            f"- Baseline loss: **{doc['baseline']['loss']:.4f}**",
            f"- Best loss: **{doc['best_loss']:.4f}**",
            f"- Baseline FeetTempFd ΔT: {doc['baseline']['terms']['feet_temp_fd_step1_delta_c']:.3f} °C",
            f"- Best FeetTempFd ΔT: {doc['best_terms']['feet_temp_fd_step1_delta_c']:.3f} °C",
            f"- Best HeadTempFd face-cooling ΔT: {doc['best_terms']['head_temp_fd_face_cooling_step1_delta_c']:.3f} °C",
            f"- Best equilibrium max |Δ|: {doc['best_terms']['equilibrium_max_abs_delta_c']:.6f} °C",
            "",
            "### best_params",
            "",
            "```json",
            json.dumps(doc.get("best_params", {}), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_smoke_outputs(
    *,
    bundle: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Union[str, Path]] = None,
    md_path: Optional[Union[str, Path]] = None,
    **run_kwargs: Any,
) -> Dict[str, Path]:
    doc = bundle if bundle is not None else run_demo_calibration_smoke(**run_kwargs)
    json_out = Path(json_path) if json_path is not None else DEFAULT_JSON_OUT
    md_out = Path(md_path) if md_path is not None else DEFAULT_MD_OUT
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(build_smoke_markdown(doc), encoding="utf-8")
    return {"json": json_out, "md": md_out}
