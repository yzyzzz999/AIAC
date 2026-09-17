"""Actuator-voltage based front HVAC airflow distribution (TDC-CHTD-ACTUATOR-BASED-AFE).

Uses decode_hvac_actuator_scores + bench anchors. Does not use AC_ModeVentilaPosn.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hvac_sim.afe.actuator_score_decoder import (
    ActuatorEndpoints,
    decode_hvac_actuator_scores,
)
from hvac_sim.chtd.bus_index import U_INDEX
from hvac_sim.validation.airflow_reference_loader import build_mode_distribution_templates

FRONT_DRV_FLOW_KEYS = ("FrntFdvFlow", "FrntFdfFlow", "FrntFdDefFlow")
FRONT_PSG_FLOW_KEYS = ("FrntFpvFlow", "FrntFpfFlow", "FrntFpDefFlow")

BENCH_CONFLICT_L1 = 0.35
DEFROST_LEAK_MIN_SCORE = 0.08


def _side_total(u: np.ndarray, keys: Tuple[str, ...]) -> float:
    return float(sum(float(u[U_INDEX[k]]) for k in keys))


def _bench_front_fractions(mode_key: str) -> Tuple[float, float, float]:
    """Return (face, foot, defrost) front-duct fraction priors from bench anchors."""
    templates = build_mode_distribution_templates()
    mode_map = {
        "face_dominant": "face",
        "foot_dominant": "foot",
        "defrost_dominant": "defrost",
        "mixed": "mixed",
    }
    tpl_key = mode_map.get(mode_key, "mixed")
    fracs = templates.get(tpl_key) or templates.get("face", {})
    face = float(fracs.get("FrntFdvFlow", 0)) + float(fracs.get("FrntFpvFlow", 0))
    foot = float(fracs.get("FrntFdfFlow", 0)) + float(fracs.get("FrntFpfFlow", 0))
    defrost = float(fracs.get("FrntFdDefFlow", 0)) + float(fracs.get("FrntFpDefFlow", 0))
    s = face + foot + defrost
    if s < 1e-9:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return face / s, foot / s, defrost / s


def _normalize_scores(face: float, foot: float, defrost: float) -> Tuple[float, float, float]:
    s = face + foot + defrost
    if s < 1e-9:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return face / s, foot / s, defrost / s


def _blend_weights(
    score_w: Tuple[float, float, float],
    bench_w: Tuple[float, float, float],
    *,
    bench_weight: float = 0.35,
) -> Tuple[float, float, float, float]:
    """Blend score weights with bench prior; return (wf, wft, wd, l1_conflict)."""
    bw = float(np.clip(bench_weight, 0.0, 1.0))
    blended = tuple((1.0 - bw) * s + bw * b for s, b in zip(score_w, bench_w))
    s = sum(blended)
    if s < 1e-9:
        blended = score_w
        s = sum(blended)
    wf, wft, wd = (v / s for v in blended)
    l1 = abs(wf - bench_w[0]) + abs(wft - bench_w[1]) + abs(wd - bench_w[2])
    return wf, wft, wd, l1


def compute_actuator_airflow_distribution(
    face_v: float,
    foot_v: float,
    defrost_v: float,
    *,
    airflow_target: Optional[float] = None,
    endpoints: Optional[ActuatorEndpoints | Mapping[str, Any]] = None,
    bench_blend: float = 0.35,
) -> Dict[str, Any]:
    """Compute front face/foot/defrost flow fractions from actuator voltages."""
    decoded = decode_hvac_actuator_scores(face_v, foot_v, defrost_v, endpoints=endpoints)
    fs = decoded["face_open_score"]
    fts = decoded["foot_open_score"]
    ds = decoded["defrost_open_score"]
    score_w = _normalize_scores(fs, fts, ds)
    bench_w = _bench_front_fractions(str(decoded["dominant_mode"]))
    wf, wft, wd, l1 = _blend_weights(score_w, bench_w, bench_weight=bench_blend)
    confidence = "high" if l1 <= BENCH_CONFLICT_L1 else "low"
    total = float(airflow_target) if airflow_target is not None and math.isfinite(airflow_target) and airflow_target > 0 else None
    half = 0.5 * total if total is not None else None
    row: Dict[str, Any] = {
        **decoded,
        "score_weight_face": score_w[0],
        "score_weight_foot": score_w[1],
        "score_weight_defrost": score_w[2],
        "bench_prior_face": bench_w[0],
        "bench_prior_foot": bench_w[1],
        "bench_prior_defrost": bench_w[2],
        "blend_weight_face": wf,
        "blend_weight_foot": wft,
        "blend_weight_defrost": wd,
        "bench_score_l1": l1,
        "distribution_confidence": confidence,
        "airflow_target_m3h": total,
        "front_face_flow_m3h": (half * wf * 2) if half is not None else None,
        "front_foot_flow_m3h": (half * wft * 2) if half is not None else None,
        "front_defrost_flow_m3h": (half * wd * 2) if half is not None else None,
        "ac_mode_ventila_posn_used": False,
    }
    return row


def apply_voltage_actuator_distribution(
    u: np.ndarray,
    *,
    face_v: float,
    foot_v: float,
    defrost_v: float,
    airflow_target: Optional[float] = None,
    endpoints: Optional[ActuatorEndpoints | Mapping[str, Any]] = None,
    defrost_leakage_only: bool = False,
    defrost_leak_ratio: float = 0.10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply voltage-decoded distribution to u-bus front flow slots."""
    out = u.copy()
    dist = compute_actuator_airflow_distribution(
        face_v, foot_v, defrost_v, airflow_target=airflow_target, endpoints=endpoints
    )
    wf = dist["blend_weight_face"]
    wft = dist["blend_weight_foot"]
    wd = dist["blend_weight_defrost"]

    total_drv = _side_total(out, FRONT_DRV_FLOW_KEYS)
    total_psg = _side_total(out, FRONT_PSG_FLOW_KEYS)
    if airflow_target is not None and math.isfinite(airflow_target) and airflow_target > 0:
        total_drv = 0.5 * float(airflow_target)
        total_psg = 0.5 * float(airflow_target)
        dist["total_flow_source"] = "airflow_target"
    else:
        dist["total_flow_source"] = "u_bus_preserving_side_total"

    out[U_INDEX["FrntFdvFlow"]] = total_drv * wf
    out[U_INDEX["FrntFdfFlow"]] = total_drv * wft
    out[U_INDEX["FrntFdDefFlow"]] = total_drv * wd
    out[U_INDEX["FrntFpvFlow"]] = total_psg * wf
    out[U_INDEX["FrntFpfFlow"]] = total_psg * wft
    out[U_INDEX["FrntFpDefFlow"]] = total_psg * wd

    prov = {
        "distribution_method": "voltage_actuator_scores",
        "status": "redistributed",
        "driver_total_m3h": total_drv,
        "passenger_total_m3h": total_psg,
        **dist,
    }

    if defrost_leakage_only and dist["dominant_mode"] == "face_dominant":
        ds = float(dist["defrost_open_score"])
        if ds >= DEFROST_LEAK_MIN_SCORE:
            leak = float(defrost_leak_ratio) * ds
            for side_keys, def_key in (
                (FRONT_DRV_FLOW_KEYS, "FrntFdDefFlow"),
                (FRONT_PSG_FLOW_KEYS, "FrntFpDefFlow"),
            ):
                side_total = _side_total(out, side_keys)
                out[U_INDEX[def_key]] = side_total * leak
                face_key = side_keys[0]
                out[U_INDEX[face_key]] = max(0.0, float(out[U_INDEX[face_key]]) - side_total * leak * 0.5)
                foot_key = side_keys[1]
                out[U_INDEX[foot_key]] = max(0.0, float(out[U_INDEX[foot_key]]) - side_total * leak * 0.5)
            prov["defrost_leakage_applied"] = leak
        else:
            prov["defrost_leakage_applied"] = 0.0
    return out, prov


def build_distribution_audit_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build per-sample rows from cache-like dicts with actuator voltages."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        row = compute_actuator_airflow_distribution(
            float(rec["face_v"]),
            float(rec["foot_v"]),
            float(rec["defrost_v"]),
            airflow_target=rec.get("airflow_target"),
        )
        row["sample_index"] = rec.get("sample_index")
        row["holdout_index"] = rec.get("holdout_index")
        rows.append(row)
    return rows


def write_distribution_summary_md(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    low = sum(1 for r in rows if r.get("distribution_confidence") == "low")
    modes: Dict[str, int] = {}
    for r in rows:
        m = str(r.get("dominant_mode", "unknown"))
        modes[m] = modes.get(m, 0) + 1
    lines = [
        "# Actuator-based airflow distribution summary",
        "",
        f"- samples: {n}",
        f"- distribution_confidence=low: {low} ({100.0 * low / max(n, 1):.1f}%)",
        "",
        "## Dominant mode counts",
        "",
    ]
    for k, v in sorted(modes.items()):
        lines.append(f"- {k}: {v}")
    lines.extend(
        [
            "",
            "## Semantics",
            "",
            "- Face: high_is_open (4.62V open, ~0.4V closed)",
            "- Foot: low_is_open (low V open, 4.4V closed stall)",
            "- Defrost: low_is_open",
            "- AC_ModeVentilaPosn: **not used**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
