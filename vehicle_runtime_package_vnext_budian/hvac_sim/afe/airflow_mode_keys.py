"""Canonical airflow mode template keys (PDF + actuator-first)."""

from __future__ import annotations

from typing import Optional

# PDF / cache template keys
FACE_PURE = "face_pure"
FACE_FOOT = "face_foot"
FOOT = "foot"
DEFROST = "defrost"
FOOT_DEFROST = "foot_defrost"
FACE_DEFROST = "face_defrost"  # PDF anchor for 吹面除霜; actuator may collapse to defrost
MIXED = "mixed"

ALL_TEMPLATE_KEYS = (
    FACE_PURE,
    FACE_FOOT,
    FOOT,
    DEFROST,
    FOOT_DEFROST,
    FACE_DEFROST,
    MIXED,
)

SCORE_HIGH = 0.50
SCORE_LOW = 0.15


def anchor_mode_key_from_cn(hvac_mode_cn: str) -> str:
    """Map PDF hvac_mode_cn to distinct template key (no face/face_foot collision)."""
    mode = str(hvac_mode_cn).lower()
    has_face = "面" in mode
    has_foot = "脚" in mode
    has_def = "除霜" in mode
    if has_def and has_face and has_foot:
        return MIXED
    if has_def and has_face:
        return FACE_DEFROST
    if has_def and has_foot:
        return FOOT_DEFROST
    if has_def:
        return DEFROST
    if has_face and has_foot:
        return FACE_FOOT
    if has_foot:
        return FOOT
    if has_face:
        return FACE_PURE
    return MIXED


def resolve_actuator_mode_key(
    face_score: float,
    foot_score: float,
    defrost_score: float,
    *,
    high: float = SCORE_HIGH,
    low: float = SCORE_LOW,
) -> str:
    """Actuator-first mode key from open scores."""
    fs, fts, ds = float(face_score), float(foot_score), float(defrost_score)
    face_h, face_l = fs >= high, fs < low
    foot_h, foot_l = fts >= high, fts < low
    def_h, def_l = ds >= high, ds < low

    if def_h:
        if face_h and foot_h:
            return MIXED
        if face_h:
            return FACE_DEFROST
        if foot_h:
            return FOOT_DEFROST
        return DEFROST
    if face_h and foot_h:
        return FACE_FOOT
    if face_h and foot_l and def_l:
        return FACE_PURE
    if foot_h and face_l:
        return FOOT
    return MIXED


def can_mode_valid(mode_val: Optional[float]) -> bool:
    if mode_val is None or not (mode_val == mode_val):  # NaN
        return False
    return abs(float(mode_val)) > 0.05


def can_mode_to_template_key(mode_val: Optional[float], defrost_val: Optional[float]) -> str:
    """Map AC_ModeVentilaPosn (+ defrost) to template key when CAN is trusted."""
    if defrost_val is not None and float(defrost_val) > 0.55:
        if mode_val is None or float(mode_val) < 0.5:
            return DEFROST
        m = float(mode_val)
        if m <= 2.5:
            return FACE_DEFROST
        return FOOT_DEFROST
    if mode_val is None:
        return MIXED
    m = float(mode_val)
    if m <= 1.5:
        return FACE_PURE
    if m <= 2.5:
        return FOOT
    if m <= 3.5:
        return FACE_FOOT
    if m >= 4.0:
        return DEFROST
    return MIXED


def actuator_can_compatible(actuator_key: str, can_key: str) -> bool:
    """True if PDF keyed by actuator_key may be used alongside CAN hint."""
    if actuator_key == can_key:
        return True
    aliases = {
        FACE_DEFROST: {DEFROST, FACE_DEFROST},
        FOOT_DEFROST: {DEFROST, FOOT_DEFROST, FOOT},
        FACE_PURE: {FACE_PURE},
        FACE_FOOT: {FACE_FOOT, MIXED},
    }
    return can_key in aliases.get(actuator_key, {actuator_key})
