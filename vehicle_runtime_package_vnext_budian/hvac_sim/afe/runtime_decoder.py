"""AFE runtime input decoder draft (AFE-T19-B).

Maps actuator feedback voltages to normalized door travel [0, 1].
Does **not** equate door travel with hot-air mix fraction (see ``blend_hot_air_fraction_note``).

Not wired into ``afe_calc`` — optional layer for future runtime integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from hvac_sim.afe.calibration_data import (
    load_actuator_voltage_targets,
    load_blend_door_voltage_endpoints,
)


@dataclass(frozen=True)
class DecoderResult:
    """Single decoded scalar with provenance."""

    value: float
    provenance: Dict[str, Any] = field(default_factory=dict)


def _finite_or_raise(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def voltage_to_normalized_travel(
    voltage_v: float,
    *,
    cold_endpoint_v: float,
    hot_endpoint_v: float,
    increasing_hot: bool,
) -> DecoderResult:
    """Map feedback voltage to normalized mechanical travel (0=cold end, 1=hot end).

    This is **door travel**, not hot-air mass fraction.
    """
    v = _finite_or_raise(voltage_v, "voltage_v")
    cold = float(cold_endpoint_v)
    hot = float(hot_endpoint_v)
    prov: Dict[str, Any] = {
        "method": "endpoint_linear_travel",
        "cold_endpoint_v": cold,
        "hot_endpoint_v": hot,
        "increasing_hot": increasing_hot,
        "note": "travel_only_not_mix_ratio",
    }
    if increasing_hot:
        span = hot - cold
        if abs(span) < 1e-9:
            prov["fallback"] = "zero_span_endpoints"
            prov["status"] = "degraded"
            return DecoderResult(0.0, prov)
        raw = (v - cold) / span
    else:
        span = cold - hot
        if abs(span) < 1e-9:
            prov["fallback"] = "zero_span_endpoints"
            prov["status"] = "degraded"
            return DecoderResult(0.0, prov)
        raw = (cold - v) / span

    travel = max(0.0, min(1.0, raw))
    if travel != raw:
        prov["clamped"] = True
    prov["normalized_travel"] = travel
    return DecoderResult(travel, prov)


def decode_blend_door_travel(side: str, voltage_v: float) -> DecoderResult:
    """Decode left/right blend-door voltage with opposite directions."""
    data = load_blend_door_voltage_endpoints()
    side_key = side.strip().lower()
    if side_key not in ("left", "right"):
        prov = {"status": "unknown_side", "fallback": "mid_travel_0.5", "side": side}
        return DecoderResult(0.5, prov)

    entries = {e["side"]: e for e in data["entries"] if "endpoint" in e}
    by_side = [e for e in data["entries"] if e["side"] == side_key]
    cold = next(e for e in by_side if e["endpoint"] == "full_cold_open")
    hot = next(e for e in by_side if e["endpoint"] == "full_heat_closed")
    increasing_hot = side_key == "right"
    result = voltage_to_normalized_travel(
        voltage_v,
        cold_endpoint_v=cold["voltage_v"],
        hot_endpoint_v=hot["voltage_v"],
        increasing_hot=increasing_hot,
    )
    prov = dict(result.provenance)
    prov.update(
        {
            "actuator": "blend_door",
            "side": side_key,
            "source": cold["source"],
            "confidence": cold["confidence"],
        }
    )
    return DecoderResult(result.value, prov)


def blend_hot_air_fraction_note() -> str:
    """Explicit policy: do not use normalized travel as mix ratio without bench map."""
    return (
        "PosnFdh / blend normalized travel is mechanical door position only. "
        "Hot-air fraction requires Hct/EvaT blend model or measured outlet temperature."
    )


def _actuator_open_closed_targets(
    actuator: str, mode_id: int
) -> Optional[tuple[float, float]]:
    """Return (closed_v, open_v) stall targets for simple flaps when both ends are stalls."""
    doc = load_actuator_voltage_targets()
    mode = next((m for m in doc["modes"] if m["mode_id"] == mode_id), None)
    if mode is None:
        return None
    act = mode["actuators"].get(actuator)
    if act is None or act.get("target_v") is None:
        return None
    state = str(act.get("state_cn", ""))
    tv = float(act["target_v"])
    if "全关" in state:
        return tv, None
    if "全开" in state:
        return None, tv
    return None


def decode_distribution_flap_travel(
    actuator: str,
    voltage_v: float,
    *,
    hvac_mode_id: int,
    open_is_high_voltage: bool = True,
) -> DecoderResult:
    """Decode defrost/vent/foot/rear actuators vs mode target table."""
    _finite_or_raise(voltage_v, "voltage_v")
    doc = load_actuator_voltage_targets()
    mode = next((m for m in doc["modes"] if m["mode_id"] == hvac_mode_id), None)
    prov: Dict[str, Any] = {
        "actuator": actuator,
        "hvac_mode_id": hvac_mode_id,
        "source": doc.get("source"),
    }
    if mode is None:
        prov.update({"status": "unknown_mode", "fallback": "mid_travel_0.5"})
        return DecoderResult(0.5, prov)

    act = mode["actuators"].get(actuator)
    if act is None or act.get("target_v") is None:
        prov.update({"status": "unknown_actuator_for_mode", "fallback": "mid_travel_0.5"})
        return DecoderResult(0.5, prov)

    target_v = float(act["target_v"])
    state = str(act.get("state_cn", ""))
    prov["target_v"] = target_v
    prov["state_cn"] = state
    prov["confidence"] = act.get("confidence", "medium")

    if "全开" in state and "堵转" in state:
        travel = 1.0 if abs(voltage_v - target_v) <= float(act.get("tolerance_v", 0.2)) + 0.05 else 0.5
        prov["method"] = "stall_open_match"
        return DecoderResult(travel, prov)
    if "全关" in state and "堵转" in state:
        travel = 0.0 if abs(voltage_v - target_v) <= float(act.get("tolerance_v", 0.2)) + 0.05 else 0.5
        prov["method"] = "stall_closed_match"
        return DecoderResult(travel, prov)

    prov.update(
        {
            "status": "intermediate_position_unspecified",
            "fallback": "mid_travel_0.5",
            "note": "PDF table lists target only; no segment map between stalls",
        }
    )
    return DecoderResult(0.5, prov)


def decode_hvac_mode(mode_id: int) -> Dict[str, Any]:
    """Resolve HVAC main mode metadata; unknown modes return explicit fallback."""
    doc = load_actuator_voltage_targets()
    mode = next((m for m in doc["modes"] if m["mode_id"] == mode_id), None)
    if mode is None:
        return {
            "mode_id": mode_id,
            "status": "unknown_mode",
            "fallback": "no_flap_targets",
            "provenance": {"source": doc.get("source")},
        }
    return {
        "mode_id": mode_id,
        "mode_code": mode.get("mode_code"),
        "mode_name_cn": mode.get("mode_name_cn"),
        "status": "ok",
        "provenance": {"source": doc.get("source"), "confidence": "high"},
    }


def build_flapset_proposal(
    *,
    hvac_mode_id: int,
    feedback_voltages: Mapping[str, float],
) -> Dict[str, Any]:
    """Build a diagnostic FlapSet proposal dict (not applied to ``afe_calc``)."""
    warnings: List[str] = []
    flaps: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {"hvac_mode": decode_hvac_mode(hvac_mode_id)}

    mapping = {
        "defrost": "FHFdDefFlapPosn",
        "vent": "FHFdvFlapPosn",
        "foot": "FHFdfFlapPosn",
    }
    for act_key, flap_name in mapping.items():
        if act_key not in feedback_voltages:
            continue
        dec = decode_distribution_flap_travel(
            act_key, feedback_voltages[act_key], hvac_mode_id=hvac_mode_id
        )
        flaps[flap_name] = dec.value
        provenance[flap_name] = dec.provenance
        if dec.provenance.get("status"):
            warnings.append(f"{flap_name}:{dec.provenance.get('status')}")

    if "blend_left_v" in feedback_voltages:
        dec = decode_blend_door_travel("left", feedback_voltages["blend_left_v"])
        provenance["blend_left_travel"] = dec.provenance
        provenance["blend_left_travel"]["hot_air_fraction_policy"] = blend_hot_air_fraction_note()
    if "blend_right_v" in feedback_voltages:
        dec = decode_blend_door_travel("right", feedback_voltages["blend_right_v"])
        provenance["blend_right_travel"] = dec.provenance

    return {
        "flaps": flaps,
        "warnings": warnings,
        "provenance": provenance,
        "hot_air_fraction_policy": blend_hot_air_fraction_note(),
    }
