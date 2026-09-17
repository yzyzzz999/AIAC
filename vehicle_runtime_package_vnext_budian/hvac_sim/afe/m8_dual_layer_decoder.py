"""M8 dual-layer AFE structure decoder (decode-only, future AFE v2).

Maps runtime intake/mode signals to layer fractions per engineer-confirmed structure.
Not wired into ``afe_calc`` — use with ``target_vehicle_dual_layer`` when solver exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from hvac_sim.afe.calibration_data import load_m8_dual_layer_structure_confirmed

_RECIRC_STATES: Tuple[int, ...] = (0, 25, 50, 75, 100)

_MODE_ALIASES: Dict[str, str] = {
    "1": "V",
    "2": "V_F",
    "3": "V_D_F",
    "4": "F",
    "5": "F_D",
    "6": "D",
    "7": "V_D",
    "8": "OFF",
    "吹面": "V",
    "吹面吹脚": "V_F",
    "吹面吹脚除霜": "V_D_F",
    "吹脚": "F",
    "吹脚除霜": "F_D",
    "除霜": "D",
    "吹面除霜": "V_D",
}


@dataclass(frozen=True)
class IntakeStateResult:
    upper_recirc_ratio: float
    lower_recirc_ratio: float
    upper_fresh_ratio: float
    lower_fresh_ratio: float
    state_index: int
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModeLayerDistributionResult:
    defrost_layer: str
    face_layer: str
    foot_layer: str
    rear_layer: str
    provenance: Dict[str, Any] = field(default_factory=dict)


def _finite_or_raise(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _normalize_mode(front_mode: Union[str, int]) -> str:
    key = str(front_mode).strip()
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    return key.upper().replace(" ", "_")


def _snap_recirc_state_index(circle_prior_posn: float) -> int:
    """Map prior-door position to 0/25/50/75/100 recirc percent state."""
    p = _finite_or_raise(circle_prior_posn, "circle_prior_posn")
    if 0.0 <= p <= 1.0 and p != int(p):
        p = p * 100.0
    p = max(0.0, min(100.0, p))
    return min(_RECIRC_STATES, key=lambda s: abs(s - p))


def decode_intake_state(
    circle_mode_posn: float,
    circle_prior_posn: float,
    mode: Optional[Union[str, int]] = None,
) -> IntakeStateResult:
    """Decode dual-layer intake state from runtime positions.

    Parameters
    ----------
    circle_mode_posn:
        ``AC_CircleModeVentilaPosn`` (normalized travel or voltage-scaled; recorded only).
    circle_prior_posn:
        ``AC_CirclePriorVentilaPosn`` — wide dual-layer door; 0/25/50/75/100 = recirc %.
    mode:
        Optional HVAC mode id/code for provenance cross-check (not used to override LUT).
    """
    doc = load_m8_dual_layer_structure_confirmed()
    intake = doc["intake_actuators"]
    lut = doc["intake_state_lut"]

    mode_posn = _finite_or_raise(circle_mode_posn, "circle_mode_posn")
    state_index = _snap_recirc_state_index(circle_prior_posn)
    row = lut[str(state_index)]

    prov: Dict[str, Any] = {
        "method": "m8_dual_layer_intake_lut",
        "source": doc.get("source"),
        "circle_mode_signal": intake["circle_mode_signal"],
        "circle_prior_signal": intake["circle_prior_signal"],
        "circle_mode_posn": mode_posn,
        "circle_prior_posn_raw": float(circle_prior_posn),
        "state_index": state_index,
        "layer_split_rule_id": intake["layer_split_rule_id"],
        "linked_motors": intake["linked_motors"],
        "not_wired_to": doc.get("not_wired_to"),
    }
    if mode is not None:
        prov["mode_normalized"] = _normalize_mode(mode)

    return IntakeStateResult(
        upper_recirc_ratio=float(row["upper_recirc_ratio"]),
        lower_recirc_ratio=float(row["lower_recirc_ratio"]),
        upper_fresh_ratio=float(row["upper_fresh_ratio"]),
        lower_fresh_ratio=float(row["lower_fresh_ratio"]),
        state_index=int(row["state_index"]),
        provenance=prov,
    )


def decode_mode_layer_distribution(
    front_mode: Union[str, int],
) -> ModeLayerDistributionResult:
    """Decode outlet-layer assignment for a front HVAC mode."""
    doc = load_m8_dual_layer_structure_confirmed()
    assign = doc["mode_layer_assignment"]
    mode_code = _normalize_mode(front_mode)

    prov: Dict[str, Any] = {
        "method": "m8_mode_layer_assignment",
        "source": doc.get("source"),
        "front_mode": mode_code,
        "isolated_until": doc["layer_separation"]["isolated_until"],
        "rear_supply_layer": assign["rear_supply"]["layer"],
        "not_wired_to": doc.get("not_wired_to"),
    }

    defrost_layer = assign["defrost"]["layer"]
    face_layer = assign["face"]["layer"]
    foot_layer = assign["foot"]["layer"]
    rear_layer = assign["rear_supply"]["layer"]

    if mode_code in ("F_D", "V_D_F"):
        fd = assign["foot_defrost"]
        defrost_layer = fd["defrost_layer"]
        foot_layer = fd["foot_layer"]
        prov["foot_defrost_split"] = True
        prov["foot_defrost_note"] = fd["note"]
    elif mode_code == "D":
        foot_layer = "closed_or_minimal"
        prov["foot_in_defrost_only"] = False
    elif mode_code == "F":
        defrost_layer = "closed_or_minimal"
        face_layer = "closed_or_minimal"
        prov["foot_only_lower"] = True
    elif mode_code in ("V", "V_F", "V_D"):
        prov["face_active"] = True
        if mode_code == "V_F":
            prov["face_and_foot_simultaneous"] = True

    prov["temperature_policy"] = {
        "rear_face_sensor": doc["temperature_sensors"]["rear_face"]["placement"],
        "rear_foot_sensor": doc["temperature_sensors"]["rear_foot"]["available"],
        "defrost_sensor": doc["temperature_sensors"]["defrost"]["available"],
        "rear_face_shared": doc["temperature_sensors"]["rear_face"]["shared_across_outlets"],
    }

    return ModeLayerDistributionResult(
        defrost_layer=defrost_layer,
        face_layer=face_layer,
        foot_layer=foot_layer,
        rear_layer=rear_layer,
        provenance=prov,
    )


def rear_temperature_sharing_policy() -> Dict[str, Any]:
    """Return confirmed second-row temperature sensor sharing rules."""
    doc = load_m8_dual_layer_structure_confirmed()
    rear = doc["temperature_sensors"]["rear_face"]
    foot = doc["temperature_sensors"]["rear_foot"]
    return {
        "sensor_placement": rear["placement"],
        "sensor_placement_cn": rear["placement_cn"],
        "shared_tma_across_outlets": rear["shared_across_outlets"],
        "outlets_shared": list(rear["outlets_shared"]),
        "rear_foot_independent_tma": foot["available"],
        "provenance": {
            "source": doc.get("source"),
            "confirmed_by": doc.get("confirmed_by"),
        },
    }
