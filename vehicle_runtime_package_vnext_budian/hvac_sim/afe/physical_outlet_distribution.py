"""Physical outlet airflow distribution layer (AFE-T19-B / T21-A).

Splits AFE zone branch totals into supplier physical outlets using PDF anchor
ratios. Does not modify CHTD ``u`` bus wiring or ``afe_calc`` defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from hvac_sim.afe.calibration_data import load_airflow_distribution_anchors, load_physical_outlet_map

# Zone-level inputs (m³/h) — matches AFE_Calc Q_out branch semantics + center row2 face
ZONE_FLOW_KEYS = (
    "frnt_fd_face",
    "frnt_fp_face",
    "frnt_fd_foot",
    "frnt_fp_foot",
    "frnt_fd_def",
    "frnt_fp_def",
    "frnt_fd_sv",
    "frnt_fp_sv",
    "row2_face_center",
    "row2_sd_face",
    "row2_sp_face",
    "row2_sd_foot",
    "row2_sp_foot",
)

# Direct outlet → single zone mapping
_OUTLET_TO_ZONE: Dict[str, str] = {
    "frnt_face_fl": "frnt_fd_face",
    "frnt_face_flc": "frnt_fd_face",
    "frnt_face_frc": "frnt_fp_face",
    "frnt_face_fr": "frnt_fp_face",
    "frnt_foot_fl": "frnt_fd_foot",
    "frnt_foot_fr": "frnt_fp_foot",
    "row2_face_console": "row2_face_center",
    "row2_face_bp_left": "row2_sd_face",
    "row2_face_bp_right": "row2_sp_face",
    "row2_foot_left": "row2_sd_foot",
    "row2_foot_right": "row2_sp_foot",
}

# PDF combined outlets → split across Fd/Fp AFE zones (50/50 per physical_outlet_map rule)
_SPLIT_OUTLET_ZONES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "defrost_main": (
        ("frnt_fd_def", "defrost_main"),
        ("frnt_fp_def", "defrost_main"),
    ),
    "defrost_side": (
        ("frnt_fd_sv", "defrost_side"),
        ("frnt_fp_sv", "defrost_side"),
    ),
}


def _split_fractions() -> Dict[str, Tuple[float, ...]]:
    """Load split fractions from map when present; default equal Fd/Fp."""
    doc = load_physical_outlet_map()
    rules = doc.get("zone_split_rules", {})
    out: Dict[str, Tuple[float, ...]] = {}
    for outlet_id, zones in _SPLIT_OUTLET_ZONES.items():
        rule = rules.get(outlet_id, {})
        fracs = rule.get("fractions")
        if isinstance(fracs, list) and len(fracs) == len(zones):
            out[outlet_id] = tuple(float(x) for x in fracs)
        else:
            n = len(zones)
            out[outlet_id] = tuple(1.0 / n for _ in range(n))
    return out


@dataclass(frozen=True)
class PhysicalOutletDistributionResult:
    """Physical outlet flows with provenance."""

    outlet_flows_m3h: Dict[str, float]
    provenance: Dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _anchor_index() -> Dict[str, Dict[str, Any]]:
    doc = load_airflow_distribution_anchors()
    return {a["anchor_id"]: a for a in doc["anchors"]}


def _zone_weights_from_anchor(anchor: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Build per-zone outlet weight tables from anchor outlet_percent."""
    outlet_pct: Dict[str, float] = dict(anchor["outlet_percent"])
    zone_outlet_pct: Dict[str, Dict[str, float]] = {}
    split_fracs = _split_fractions()

    for outlet_id, pct in outlet_pct.items():
        if outlet_id in _SPLIT_OUTLET_ZONES:
            zones = _SPLIT_OUTLET_ZONES[outlet_id]
            fracs = split_fracs[outlet_id]
            for (zone, oid), frac in zip(zones, fracs):
                zone_outlet_pct.setdefault(zone, {})[oid] = (
                    zone_outlet_pct.get(zone, {}).get(oid, 0.0) + float(pct) * frac
                )
            continue
        zone = _OUTLET_TO_ZONE.get(outlet_id)
        if zone is None:
            continue
        zone_outlet_pct.setdefault(zone, {})[outlet_id] = float(pct)
    return zone_outlet_pct


def distribute_physical_outlets(
    zone_flows_m3h: Mapping[str, float],
    *,
    anchor_id: Optional[str] = None,
) -> PhysicalOutletDistributionResult:
    """Distribute zone branch totals to physical outlets.

    Parameters
    ----------
    zone_flows_m3h:
        AFE zone totals, e.g. ``frnt_fd_face``, ``row2_face_center`` (m³/h).
    anchor_id:
        Key from ``airflow_distribution_anchors.json``. When omitted or unknown,
        uses equal split within each zone branch (explicit fallback).
    """
    flows = {k: float(zone_flows_m3h.get(k, 0.0)) for k in ZONE_FLOW_KEYS}
    for k, v in flows.items():
        if not math.isfinite(v):
            raise ValueError(f"zone flow {k} must be finite")

    anchors = _anchor_index()
    prov: Dict[str, Any] = {"anchor_id": anchor_id}
    warnings: list[str] = []

    if anchor_id is None or anchor_id not in anchors:
        prov["status"] = "no_anchor"
        prov["fallback"] = "equal_split_within_zone"
        if anchor_id is not None:
            warnings.append(f"unknown_anchor_id:{anchor_id}")
        outlet_flows = _equal_split_within_zones(flows)
        return PhysicalOutletDistributionResult(
            outlet_flows_m3h=outlet_flows,
            provenance=prov,
            warnings=tuple(warnings),
        )

    anchor = anchors[anchor_id]
    prov.update(
        {
            "status": "anchor_weights",
            "source": anchor.get("source"),
            "page": anchor.get("page"),
            "source_region": anchor.get("source_region"),
            "condition": anchor.get("condition"),
            "confidence": anchor.get("confidence"),
            "reviewer_status": anchor.get("reviewer_status"),
            "measured_sum_percent": anchor.get("measured_sum_percent"),
            "split_rules_applied": list(_SPLIT_OUTLET_ZONES.keys()),
        }
    )
    zone_weights = _zone_weights_from_anchor(anchor)
    outlet_flows: Dict[str, float] = {}

    for zone, zone_total in flows.items():
        weights = zone_weights.get(zone)
        if not weights or zone_total <= 0.0:
            continue
        w_sum = sum(weights.values())
        if w_sum <= 0.0:
            warnings.append(f"zero_weight_zone:{zone}")
            continue
        for outlet_id, w in weights.items():
            share = zone_total * (w / w_sum)
            outlet_flows[outlet_id] = outlet_flows.get(outlet_id, 0.0) + share

    prov["zones_applied"] = sorted(k for k in zone_weights if k in flows and flows[k] > 0.0)
    return PhysicalOutletDistributionResult(
        outlet_flows_m3h=outlet_flows,
        provenance=prov,
        warnings=tuple(warnings),
    )


def _equal_split_within_zones(flows: Mapping[str, float]) -> Dict[str, float]:
    """Fallback: equal split among outlets bound to each zone key."""
    by_zone: Dict[str, list[str]] = {}
    for outlet_id, zone in _OUTLET_TO_ZONE.items():
        by_zone.setdefault(zone, []).append(outlet_id)
    for outlet_id, zones in _SPLIT_OUTLET_ZONES.items():
        for zone, oid in zones:
            by_zone.setdefault(zone, []).append(oid)

    result: Dict[str, float] = {}
    for zone, outlets in by_zone.items():
        total = float(flows.get(zone, 0.0))
        if total <= 0.0 or not outlets:
            continue
        share = total / len(outlets)
        for oid in outlets:
            result[oid] = result.get(oid, 0.0) + share
    return result


def total_physical_flow_m3h(outlet_flows: Mapping[str, float]) -> float:
    return float(sum(outlet_flows.values()))


def total_zone_flow_m3h(zone_flows: Mapping[str, float]) -> float:
    return float(sum(float(zone_flows.get(k, 0.0)) for k in ZONE_FLOW_KEYS))


def zone_flows_from_afe_q_out(q_out: Mapping[str, float]) -> Dict[str, float]:
    """Map AFE_Calc-style branch names to zone_flows keys (diagnostic helper).

    ``q_sdv`` maps to ``row2_sd_face`` only; supply ``row2_face_center`` separately
    when using the physical outlet layer with console anchors.
    """
    aliases = {
        "q_fd_v": "frnt_fd_face",
        "q_fp_v": "frnt_fp_face",
        "q_fd_f": "frnt_fd_foot",
        "q_fp_f": "frnt_fp_foot",
        "q_fd_def": "frnt_fd_def",
        "q_fp_def": "frnt_fp_def",
        "q_fd_sv": "frnt_fd_sv",
        "q_fp_sv": "frnt_fp_sv",
        "q_sdv": "row2_sd_face",
        "q_spv": "row2_sp_face",
        "q_sdf": "row2_sd_foot",
        "q_spf": "row2_sp_foot",
    }
    out: Dict[str, float] = {k: 0.0 for k in ZONE_FLOW_KEYS}
    for src, dst in aliases.items():
        if src in q_out:
            out[dst] = float(q_out[src])
    return out
