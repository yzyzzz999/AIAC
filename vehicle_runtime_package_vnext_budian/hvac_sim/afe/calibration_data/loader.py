"""Load AFE-T19-B calibration JSON with provenance metadata."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_DATA_DIR = Path(__file__).resolve().parent


def _load_json(name: str) -> Dict[str, Any]:
    path = _DATA_DIR / name
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_actuator_voltage_targets() -> Dict[str, Any]:
    return _load_json("actuator_voltage_targets.json")


@lru_cache(maxsize=1)
def load_blend_door_voltage_endpoints() -> Dict[str, Any]:
    return _load_json("blend_door_voltage_endpoints.json")


@lru_cache(maxsize=1)
def load_fan_curve_initial() -> Dict[str, Any]:
    return _load_json("fan_curve_initial.json")


@lru_cache(maxsize=1)
def load_airflow_distribution_anchors() -> Dict[str, Any]:
    return _load_json("airflow_distribution_anchors.json")


@lru_cache(maxsize=1)
def load_physical_outlet_map() -> Dict[str, Any]:
    return _load_json("physical_outlet_map.json")


@lru_cache(maxsize=1)
def load_m8_dual_layer_structure_confirmed() -> Dict[str, Any]:
    return _load_json("m8_dual_layer_structure_confirmed.json")
