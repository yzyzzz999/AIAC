"""First-phase CHTD HVAC convection scale calibration scaffold (demo only).

Applies scalar scale factors to tied LUT / capacity groups on copied ``CHTDParams``.
Does not modify ``CHTDParams`` defaults, ``thermal.py``, or AS_FOUND golden.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hvac_sim.calibration.schema import ParameterSpec
from hvac_sim.chtd.params import CHTDParams, as_lut
from hvac_sim.chtd.physical_params import build_physical_capacity_params

SCHEMA = "chtd_hvac_scale_calibration_scaffold_v1"
CLASSIFICATION = "demo_only"
NOT_VEHICLE_CALIBRATION = True

WHY_NOT_200_PARAMS = (
    "First phase binds ~10 scalar scales across symmetric front/2nd-row HVAC conv LUTs "
    "and lumped capacity groups — not 200+ individual placeholders or 7-point AmbT LUT "
    "surfaces. Third row (Td/Tp) excluded. Real vehicle sign-off needs measured transients."
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
