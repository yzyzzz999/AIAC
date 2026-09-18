"""Opt-in CHTD safe_preview parameter bundle for demo / guard preview.

Builds on ``physical_capacity_params`` + HVAC conv scales + solar LUT scales.
Does not modify default ``CHTDParams``, AS_FOUND golden, or ``thermal.py``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from hvac_sim.calibration.chtd_hvac_scale_adapter import (
    apply_chtd_hvac_scale_params,
)
from hvac_sim.chtd.params import CHTDParams, as_lut
from hvac_sim.chtd.physical_params import build_physical_capacity_params

SAFE_PREVIEW_SCHEMA = "chtd_safe_preview_params_v1"
CLASSIFICATION = "safe_preview_not_vehicle_calibration"
NOT_VEHICLE_CALIBRATION = True

DEFAULT_HVAC_SCALES: Dict[str, float] = {
    "front_foot_feet_conv_scale": 0.1,
    "front_face_head_conv_scale": 0.2,
    "second_row_face_head_conv_scale": 0.2,
    "second_row_foot_feet_conv_scale": 0.1,
}
DEFAULT_SOLAR_GLOBAL_SCALE = 0.05
DEFAULT_SOLAR_HEAD_PROXY_SCALE = 0.03

HEAD_SOLAR_LUT_NAMES = frozenset(
    {
        "CHTD_FdSolarRadCo_M",
        "CHTD_FpSolarRadCo_M",
        "CHTD_SdSolarRadCo_M",
        "CHTD_SpSolarRadCo_M",
        "CHTD_TdSolarRadCo_M",
        "CHTD_TpSolarRadCo_M",
    }
)

def _all_solar_lut_field_names(params: CHTDParams) -> List[str]:
    return [f.name for f in fields(params) if f.name.endswith("SolarRadCo_M")]


def _scale_lut_field(params: CHTDParams, lut_name: str, factor: float) -> None:
    fac = float(factor)
    if hasattr(params, lut_name):
        arr = np.asarray(getattr(params, lut_name), dtype=float)
        setattr(params, lut_name, arr * fac)
    else:
        setattr(params, lut_name, as_lut(fac))


def apply_solar_lut_scales(
    params: CHTDParams,
    *,
    global_scale: float = DEFAULT_SOLAR_GLOBAL_SCALE,
    head_proxy_scale: Optional[float] = DEFAULT_SOLAR_HEAD_PROXY_SCALE,
) -> Dict[str, float]:
    """Scale all ``*SolarRadCo_M`` LUTs on copied params; returns per-LUT factors applied."""
    applied: Dict[str, float] = {}
    for lut_name in _all_solar_lut_field_names(params):
        if head_proxy_scale is not None and lut_name in HEAD_SOLAR_LUT_NAMES:
            factor = float(head_proxy_scale)
        else:
            factor = float(global_scale)
        _scale_lut_field(params, lut_name, factor)
        applied[lut_name] = factor
    return applied


def build_safe_preview_chtd_params(
    base: Optional[CHTDParams] = None,
    *,
    vehicle_class: str = "m8_estimated",
    hvac_scales: Optional[Mapping[str, float]] = None,
    solar_global_scale: float = DEFAULT_SOLAR_GLOBAL_SCALE,
    solar_head_proxy_scale: Optional[float] = DEFAULT_SOLAR_HEAD_PROXY_SCALE,
) -> Tuple[CHTDParams, Dict[str, Any]]:
    """Build opt-in safe_preview ``CHTDParams`` (physical capacity + conv + solar scales)."""
    physical, physical_prov = build_physical_capacity_params(
        base=base,
        vehicle_class=vehicle_class,
    )
    scales = dict(DEFAULT_HVAC_SCALES)
    if hvac_scales:
        scales.update(dict(hvac_scales))

    params = apply_chtd_hvac_scale_params(physical, scales, include_optional=False)
    solar_applied = apply_solar_lut_scales(
        params,
        global_scale=solar_global_scale,
        head_proxy_scale=solar_head_proxy_scale,
    )

    provenance: Dict[str, Any] = {
        "schema": SAFE_PREVIEW_SCHEMA,
        "classification": CLASSIFICATION,
        "not_vehicle_calibration": NOT_VEHICLE_CALIBRATION,
        "calibration_status": "safe_preview_not_vehicle_calibration",
        "vehicle_class": vehicle_class,
        "base_chain": [
            "CHTDParams() defaults (unchanged on disk)",
            "build_physical_capacity_params (cabin/roof/win/console physical bundle)",
            "apply_chtd_hvac_scale_params (HVAC conv scales)",
            "apply_solar_lut_scales (solar_global_scale + optional head proxy)",
        ],
        "hvac_scales_applied": scales,
        "solar_global_scale": float(solar_global_scale),
        "solar_head_proxy_scale": solar_head_proxy_scale,
        "solar_lut_scales_applied": solar_applied,
        "physical_capacity_provenance_meta": physical_prov.get("_meta", {}),
        "opt_in_only": True,
        "as_found_golden_unchanged": True,
        "thermal_py_unchanged": True,
    }
    return params, provenance
