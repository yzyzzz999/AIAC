"""HVAC actuator voltage → open score decoder (TDC-CHTD-ACTUATOR-BASED-AFE).

Does not use AC_ModeVentilaPosn or ActT as position. Endpoints configurable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

DEFAULT_ACTUATOR_ENDPOINTS: Dict[str, float] = {
    "face_open_v": 4.62,
    "face_closed_v": 0.4,
    "foot_open_v": 2.15,
    "foot_closed_v": 4.4,
    "defrost_open_v": 2.15,
    "defrost_closed_v": 4.52,
}

DOMINANT_MARGIN = 0.12


@dataclass(frozen=True)
class ActuatorEndpoints:
    face_open_v: float = 4.62
    face_closed_v: float = 0.4
    foot_open_v: float = 2.15
    foot_closed_v: float = 4.4
    defrost_open_v: float = 2.15
    defrost_closed_v: float = 4.52

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> ActuatorEndpoints:
        if not raw:
            return cls()
        kw = {k: float(raw[k]) for k in asdict(cls()).keys() if k in raw}
        return cls(**kw)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _open_score_v(
    v: float,
    *,
    open_v: float,
    closed_v: float,
    high_is_open: bool,
) -> float:
    if not math.isfinite(v):
        return 0.0
    if high_is_open:
        lo, hi = closed_v, open_v
        numer = float(v) - lo
    else:
        lo, hi = open_v, closed_v
        numer = hi - float(v)
    if abs(hi - lo) < 1e-6:
        return 0.0
    return _clip01(numer / (hi - lo))


def decode_hvac_actuator_scores(
    face_v: float,
    foot_v: float,
    defrost_v: float,
    *,
    endpoints: Optional[ActuatorEndpoints | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Decode face/foot/defrost voltages to normalized open scores and dominant mode."""
    ep = (
        endpoints
        if isinstance(endpoints, ActuatorEndpoints)
        else ActuatorEndpoints.from_mapping(endpoints)
    )
    face_score = _open_score_v(
        face_v, open_v=ep.face_open_v, closed_v=ep.face_closed_v, high_is_open=True
    )
    foot_score = _open_score_v(
        foot_v, open_v=ep.foot_open_v, closed_v=ep.foot_closed_v, high_is_open=False
    )
    defrost_score = _open_score_v(
        defrost_v,
        open_v=ep.defrost_open_v,
        closed_v=ep.defrost_closed_v,
        high_is_open=False,
    )
    scores = {
        "face": face_score,
        "foot": foot_score,
        "defrost": defrost_score,
    }
    dominant = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    if margin < DOMINANT_MARGIN:
        mode_label = "mixed"
        confidence = float(margin / max(DOMINANT_MARGIN, 1e-6))
    else:
        mode_label = f"{dominant}_dominant"
        confidence = _clip01(margin)

    return {
        "face_open_score": face_score,
        "foot_open_score": foot_score,
        "defrost_open_score": defrost_score,
        "dominant_mode": mode_label,
        "mode_confidence": confidence,
        "ac_mode_ventila_posn_used": False,
        "endpoints": asdict(ep),
        "inputs_v": {"face_v": float(face_v), "foot_v": float(foot_v), "defrost_v": float(defrost_v)},
    }
