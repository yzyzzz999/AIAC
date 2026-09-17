"""Air-speed geometry configuration loader and merge helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from hvac_sim.air_speed import AirSpeedInputs

_DEFAULT_GEOMETRY_PATH = Path(__file__).with_name("geometry_defaults.json")

_OUTLET_FIELD_MAP = {
    "driver_face": (
        "driver_face_outlet_area_m2",
        "driver_face_distance_m",
        "driver_face_distribution_ratio",
    ),
    "passenger_face": (
        "passenger_face_outlet_area_m2",
        "passenger_face_distance_m",
        "passenger_face_distribution_ratio",
    ),
    "driver_floor": (
        "driver_floor_outlet_area_m2",
        "driver_floor_distance_m",
        "driver_floor_distribution_ratio",
    ),
    "passenger_floor": (
        "passenger_floor_outlet_area_m2",
        "passenger_floor_distance_m",
        "passenger_floor_distribution_ratio",
    ),
    "driver_defrost": (
        "driver_defrost_outlet_area_m2",
        "driver_defrost_distance_m",
        "driver_defrost_distribution_ratio",
    ),
    "passenger_defrost": (
        "passenger_defrost_outlet_area_m2",
        "passenger_defrost_distance_m",
        "passenger_defrost_distribution_ratio",
    ),
    "rear_foot": (
        "rear_foot_outlet_area_m2",
        "rear_foot_distance_m",
        "rear_foot_distribution_ratio",
    ),
}


class GeometryConfigError(ValueError):
    """Invalid geometry configuration."""


@dataclass(frozen=True)
class GeometryMergeResult:
    air_speed: AirSpeedInputs
    provenance: Dict[str, Any]


def load_geometry_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load geometry JSON; defaults to packaged ``geometry_defaults.json``."""
    src = Path(path) if path is not None else _DEFAULT_GEOMETRY_PATH
    if not src.exists():
        raise GeometryConfigError(f"geometry config not found: {src}")
    doc = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise GeometryConfigError("geometry config root must be an object")
    return doc


def merge_geometry_with_air_speed_inputs(
    base: AirSpeedInputs,
    geometry: Optional[Mapping[str, Any]] = None,
    *,
    geometry_path: Optional[Union[str, Path]] = None,
) -> GeometryMergeResult:
    """Apply geometry placeholders/overrides onto ``AirSpeedInputs``.

    Returns merged inputs plus provenance showing which fields were placeholders
    (``measured=false``). This does **not** imply formal calibration.
    """
    cfg = dict(geometry) if geometry is not None else load_geometry_config(geometry_path)
    overrides: Dict[str, float] = {}
    field_provenance: Dict[str, Dict[str, Any]] = {}
    placeholder_fields: list[str] = []

    for outlet_name, fields in _OUTLET_FIELD_MAP.items():
        outlet = cfg.get("outlets", {}).get(outlet_name)
        if not isinstance(outlet, dict):
            continue
        measured = bool(outlet.get("measured", False))
        area_key, dist_key, ratio_key = fields
        if "effective_area_m2" in outlet:
            overrides[area_key] = float(outlet["effective_area_m2"])
            field_provenance[area_key] = {
                "outlet": outlet_name,
                "measured": measured,
                "source": cfg.get("schema", "air_speed_geometry_v1"),
            }
            if not measured:
                placeholder_fields.append(area_key)
        if "distance_to_head_m" in outlet:
            overrides[dist_key] = float(outlet["distance_to_head_m"])
            field_provenance[dist_key] = {
                "outlet": outlet_name,
                "measured": measured,
                "source": cfg.get("schema", "air_speed_geometry_v1"),
            }
            if not measured:
                placeholder_fields.append(dist_key)
        if "distribution_ratio" in outlet:
            overrides[ratio_key] = float(outlet["distribution_ratio"])
            field_provenance[ratio_key] = {
                "outlet": outlet_name,
                "measured": measured,
                "source": cfg.get("schema", "air_speed_geometry_v1"),
            }
            if not measured:
                placeholder_fields.append(ratio_key)

    att = cfg.get("global", {}).get("attenuation_k", {})
    if isinstance(att, dict) and "value" in att:
        measured = bool(att.get("measured", False))
        overrides["attenuation_k"] = float(att["value"])
        field_provenance["attenuation_k"] = {
            "outlet": "global",
            "measured": measured,
            "source": cfg.get("schema", "air_speed_geometry_v1"),
        }
        if not measured:
            placeholder_fields.append("attenuation_k")

    merged = replace(base, **overrides)
    provenance = {
        "classification": str(
            cfg.get("classification", "placeholder_until_measured")
        ),
        "note": str(
            cfg.get(
                "note",
                "Geometry placeholders are not formal calibration results.",
            )
        ),
        "fields": field_provenance,
        "placeholder_field_names": sorted(set(placeholder_fields)),
        "all_measured": len(placeholder_fields) == 0,
    }
    return GeometryMergeResult(air_speed=merged, provenance=provenance)
