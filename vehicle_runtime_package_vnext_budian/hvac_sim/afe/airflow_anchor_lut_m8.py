"""Limited PDF anchor LUT for M8 front HVAC airflow (not a full operating LUT).

Actuator-first when AC_ModeVentilaPosn invalid. PDF only when consistent with actuator mode.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from hvac_sim.afe.actuator_score_decoder import decode_hvac_actuator_scores
from hvac_sim.afe.airflow_mode_keys import (
    FACE_PURE,
    MIXED,
    actuator_can_compatible,
    anchor_mode_key_from_cn,
    can_mode_to_template_key,
    can_mode_valid,
    resolve_actuator_mode_key,
)
from hvac_sim.afe.calibration_data.loader import load_airflow_distribution_anchors
from hvac_sim.data_ingest.mf4_signal_extractor import ChannelSeries
from hvac_sim.validation.airflow_reference_loader import CHTD_FLOW_SLOTS, build_mode_distribution_templates
from hvac_sim.validation.can_airflow_surrogate import (
    DEFROST_CANDIDATES,
    FRONT_BLOWER_CANDIDATES,
    MF4_AIRFLOW_TARGET_CANDIDATES,
    MF4_REAR_AIRFLOW_TARGET_CANDIDATES,
    MODE_CANDIDATES,
    REAR_BLOWER_CANDIDATES,
    _pick_channel_value,
    _voltage_to_q_m3h,
)
from hvac_sim.validation.chtd_mf4_temperature_compare import _channel_lookup

ACTUATOR_VOLTAGE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "face_v": ("AC_DrvrFaceVentActT", "SIG_AcFrntFaceVentActT"),
    "foot_v": ("AC_DrvrFootVentActT", "SIG_AcFrntFootVentActT"),
    "defrost_v": ("AC_DefrostVentActT", "SIG_AcFrntFdDefActT", "AC_FrontDefrostPosn"),
}

PDF_VOLTAGE_ANCHORS: Tuple[float, ...] = (6.0, 12.0)


def _lr_split_from_outlet(outlet: Mapping[str, float]) -> Tuple[Optional[float], Optional[float], str]:
    drv = sum(
        float(outlet.get(k, 0))
        for k in outlet
        if k.startswith("frnt_") and ("foot_fl" in k or k.endswith("_fl") or "flc" in k or "frc" in k)
    )
    psg = sum(float(outlet.get(k, 0)) for k in outlet if "foot_fr" in k or k.endswith("_fr"))
    total = drv + psg
    if total < 1e-6:
        return None, None, "assumed_50_50"
    return drv / total, psg / total, "pdf_outlet_lr"


def _pick_pdf_anchor(
    mode_key: str,
    blower_v: float,
    *,
    templates: Optional[Mapping[str, Dict[str, float]]] = None,
    anchors_doc: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[float], bool]:
    doc = anchors_doc or load_airflow_distribution_anchors()
    tpl = dict(templates or build_mode_distribution_templates(doc))
    fracs = tpl.get(mode_key)
    if not fracs or sum(float(v) for v in fracs.values()) < 1e-9:
        return None, None, False
    v_anchor = min(PDF_VOLTAGE_ANCHORS, key=lambda v: abs(v - float(blower_v)))
    return dict(fracs), v_anchor, True


def _actuator_score_fractions(
    face_v: float,
    foot_v: float,
    defrost_v: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    dec = decode_hvac_actuator_scores(face_v, foot_v, defrost_v)
    fs, fts, ds = dec["face_open_score"], dec["foot_open_score"], dec["defrost_open_score"]
    s = fs + fts + ds
    if s < 1e-9:
        raise ValueError("actuator scores all zero — cannot derive mode fractions")
    wf, wft, wd = fs / s, fts / s, ds / s
    lr = 0.5
    flows = {
        "FrntFdvFlow": wf * lr,
        "FrntFpvFlow": wf * (1.0 - lr),
        "FrntFdfFlow": wft * lr,
        "FrntFpfFlow": wft * (1.0 - lr),
        "FrntFdDefFlow": wd * lr,
        "FrntFpDefFlow": wd * (1.0 - lr),
        "RearSdvFlow": wf * 0.25,
        "RearSpvFlow": wf * 0.25,
        "RearSdfFlow": wft * 0.25,
        "RearSpfFlow": wft * 0.25,
    }
    t = sum(flows.values()) or 1.0
    flows = {k: v / t for k, v in flows.items()}
    dec["actuator_mode_key"] = resolve_actuator_mode_key(fs, fts, ds)
    return flows, dec


def _interpolate_q_from_blower_v(blower_v: float, *, q_max: float, v_max: float = 12.0) -> float:
    if not math.isfinite(blower_v) or blower_v <= 0:
        return 0.0
    v_lo, v_hi = PDF_VOLTAGE_ANCHORS
    q_lo = _voltage_to_q_m3h(v_lo, q_max=q_max, v_max=v_max)
    q_hi = _voltage_to_q_m3h(v_hi, q_max=q_max, v_max=v_max)
    if blower_v <= v_lo:
        return q_lo * (blower_v / v_lo) if v_lo > 0 else 0.0
    if blower_v >= v_hi:
        return q_hi
    t = (blower_v - v_lo) / (v_hi - v_lo)
    return q_lo + t * (q_hi - q_lo)


def _apply_mode_fractions(
    *,
    face_v: float,
    foot_v: float,
    defrost_v: float,
    mode_val: Optional[float],
    def_val: Optional[float],
    blower_v: float,
    templates: Mapping[str, Dict[str, float]],
    prov: Dict[str, Any],
) -> Dict[str, float]:
    dec = decode_hvac_actuator_scores(face_v, foot_v, defrost_v)
    fs, fts, ds = dec["face_open_score"], dec["foot_open_score"], dec["defrost_open_score"]
    actuator_key = resolve_actuator_mode_key(fs, fts, ds)
    can_ok = can_mode_valid(mode_val)
    can_key = can_mode_to_template_key(mode_val, def_val) if can_ok else None

    prov["actuator_mode_key"] = actuator_key
    prov["can_mode_key"] = can_key
    prov["can_mode_valid"] = can_ok
    prov["face_open_score"] = fs
    prov["foot_open_score"] = fts
    prov["defrost_open_score"] = ds
    prov["dominant_mode"] = dec.get("dominant_mode")
    prov["pdf_anchor_rejected_by_actuator_score"] = False

    pdf_fracs, pdf_v, pdf_ok = _pick_pdf_anchor(actuator_key, blower_v, templates=templates)
    use_pdf = False
    reject_reason = ""

    if pdf_ok and pdf_fracs:
        if not can_ok:
            use_pdf = True
        elif can_key is not None and actuator_can_compatible(actuator_key, can_key):
            use_pdf = True
        else:
            reject_reason = f"can_key={can_key}_conflicts_actuator={actuator_key}"
    elif pdf_ok:
        reject_reason = "empty_pdf_fractions"

    if use_pdf:
        prov["mode_fraction_source"] = f"pdf_anchor:{actuator_key}"
        prov["pdf_anchor_used"] = True
        prov["pdf_voltage_anchor"] = pdf_v
        prov["actuator_score_used"] = False
        return dict(pdf_fracs)

    if pdf_ok and reject_reason:
        prov["pdf_anchor_rejected_by_actuator_score"] = True
        prov["fallback_reason"] = prov.get("fallback_reason") or reject_reason

    slot_fracs, _ = _actuator_score_fractions(face_v, foot_v, defrost_v)
    prov["mode_fraction_source"] = f"actuator_score:{actuator_key}"
    prov["actuator_score_used"] = True
    prov["pdf_anchor_used"] = False
    return slot_fracs


def compute_m8_airflow_at_index(
    channels: Sequence[ChannelSeries],
    times: np.ndarray,
    index: int,
    *,
    reference: Optional[Mapping[str, Any]] = None,
    dt_s: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute CHTD slot flows + full provenance for one MF4 timestep."""
    lookup = _channel_lookup(channels)
    ref = dict(reference or {})
    qmax = ref.get("q_max_m3h", {})
    q_front_max = float(qmax.get("front_hvac_box", 555.0))
    q_rear_max = float(qmax.get("rear_booster", 228.0))
    scale = float(ref.get("mf4_airflow_target_scale", 1.0))
    templates = ref.get("mode_distribution_templates") or build_mode_distribution_templates()

    target_val, target_sig, target_miss = _pick_channel_value(
        lookup, MF4_AIRFLOW_TARGET_CANDIDATES, times, index
    )
    rear_tgt, rear_tgt_sig, _ = _pick_channel_value(
        lookup, MF4_REAR_AIRFLOW_TARGET_CANDIDATES, times, index
    )
    blower_val, blower_sig, blower_miss = _pick_channel_value(
        lookup, FRONT_BLOWER_CANDIDATES, times, index
    )
    rear_blower_val, rear_blower_sig, _ = _pick_channel_value(
        lookup, REAR_BLOWER_CANDIDATES, times, index
    )
    mode_val, mode_sig, mode_miss = _pick_channel_value(lookup, MODE_CANDIDATES, times, index)
    def_val, _, _ = _pick_channel_value(lookup, DEFROST_CANDIDATES, times, index)

    face_v, face_sig, _ = _pick_channel_value(lookup, ACTUATOR_VOLTAGE_CANDIDATES["face_v"], times, index)
    foot_v, foot_sig, _ = _pick_channel_value(lookup, ACTUATOR_VOLTAGE_CANDIDATES["foot_v"], times, index)
    defrost_v, defrost_sig, _ = _pick_channel_value(lookup, ACTUATOR_VOLTAGE_CANDIDATES["defrost_v"], times, index)

    fv = float(face_v if face_v is not None and math.isfinite(face_v) else 4.5)
    ftv = float(foot_v if foot_v is not None and math.isfinite(foot_v) else 4.4)
    dv = float(defrost_v if defrost_v is not None and math.isfinite(defrost_v) else 4.52)

    prov: Dict[str, Any] = {
        "total_flow_source": "",
        "mode_fraction_source": "",
        "pdf_anchor_used": False,
        "pdf_voltage_anchor": None,
        "actuator_score_used": False,
        "pdf_anchor_rejected_by_actuator_score": False,
        "lr_split_source": "assumed_50_50",
        "fallback_used": False,
        "fallback_reason": "",
        "signals_used": {},
        "signals_missing": [],
    }

    if target_sig and target_val is not None and math.isfinite(target_val):
        q_front_total = float(target_val) * scale
        prov["total_flow_source"] = f"mf4_target:{target_sig}"
        prov["signals_used"]["airflow_target"] = target_sig
    else:
        prov["signals_missing"].append("airflow_target")
        bv = float(blower_val or 0.0)
        if blower_sig and "Level" in (blower_sig or ""):
            bv = bv * 1.5
        q_front_total = _interpolate_q_from_blower_v(bv, q_max=q_front_max)
        prov["total_flow_source"] = f"blower_voltage_pdf_interp:{blower_sig or 'missing'}"
        prov["fallback_used"] = bool(target_miss)
        if blower_sig:
            prov["signals_used"]["front_blower"] = blower_sig

    if rear_tgt is not None and math.isfinite(rear_tgt) and rear_tgt_sig:
        q_rear_total = float(rear_tgt) * scale
    else:
        rbv = float(rear_blower_val or 0.0)
        if rear_blower_sig and "Level" in (rear_blower_sig or ""):
            rbv = rbv * 1.5
        q_rear_total = _interpolate_q_from_blower_v(rbv, q_max=q_rear_max)

    if mode_miss or not can_mode_valid(mode_val):
        prov["fallback_reason"] = prov.get("fallback_reason") or "can_mode_invalid_actuator_first"

    slot_fracs = _apply_mode_fractions(
        face_v=fv,
        foot_v=ftv,
        defrost_v=dv,
        mode_val=mode_val,
        def_val=def_val,
        blower_v=float(blower_val or 12.0),
        templates=templates,
        prov=prov,
    )

    if face_sig:
        prov["signals_used"]["face_actuator_v"] = face_sig
    if foot_sig:
        prov["signals_used"]["foot_actuator_v"] = foot_sig
    if defrost_sig:
        prov["signals_used"]["defrost_actuator_v"] = defrost_sig
    if mode_sig:
        prov["signals_used"]["mode"] = mode_sig

    flows: Dict[str, float] = {}
    for slot in CHTD_FLOW_SLOTS:
        frac = float(slot_fracs.get(slot, 0.0))
        flows[slot] = (q_front_total if slot.startswith("Frnt") else q_rear_total) * frac

    prov["q_front_total_m3h"] = q_front_total
    prov["q_rear_total_m3h"] = q_rear_total
    return flows, prov
