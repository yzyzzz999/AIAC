"""PMV applicability diagnostics (ISO 7730 reference-range guardrails).

Evaluates whether Fanger PMV outputs are suitable for comfort interpretation
given environmental inputs. Does **not** modify ``fanger.py`` or
``compute_vehicle_pmv``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Union

ApplicabilityStatus = Literal["within_reference_range", "out_of_reference_range"]

PMV_DISPLAY_MIN = -3.0
PMV_DISPLAY_MAX = 3.0

TA_MIN_C = 10.0
TA_MAX_C = 35.0
TR_MIN_C = 10.0
TR_MAX_C = 40.0
VEL_MAX_M_S = 2.0
RH_MIN_PCT = 0.0
RH_MAX_PCT = 100.0
PMV_REF_MIN = -3.0
PMV_REF_MAX = 3.0


def clamp_pmv_display(pmv: float) -> float:
    """Clamp PMV for display/reporting only (does not alter raw PMV)."""
    return float(max(PMV_DISPLAY_MIN, min(PMV_DISPLAY_MAX, pmv)))


def evaluate_pmv_applicability(
    ta: float,
    tr: float,
    vel: float,
    rh: float,
    met: float,
    clo: float,
    pmv: float,
) -> Dict[str, Union[bool, str, float, List[str]]]:
    """Return applicability metadata for one PMV evaluation.

    ``met`` and ``clo`` are accepted for API symmetry; thresholds currently
    apply to ``ta``, ``tr``, ``vel``, ``rh``, and ``pmv`` only.
    """
    _ = (met, clo)  # reserved for future applicability rules
    reasons: List[str] = []
    pmv_raw = float(pmv)

    if not math.isfinite(ta):
        reasons.append("ta_non_finite")
    elif ta < TA_MIN_C:
        reasons.append(f"ta_below_reference_min_{TA_MIN_C}c")
    elif ta > TA_MAX_C:
        reasons.append(f"ta_above_reference_max_{TA_MAX_C}c")

    if not math.isfinite(tr):
        reasons.append("tr_non_finite")
    elif tr < TR_MIN_C:
        reasons.append(f"tr_below_reference_min_{TR_MIN_C}c")
    elif tr > TR_MAX_C:
        reasons.append(f"tr_above_reference_max_{TR_MAX_C}c")

    if not math.isfinite(vel):
        reasons.append("vel_non_finite")
    elif vel > VEL_MAX_M_S:
        reasons.append(f"vel_above_reference_max_{VEL_MAX_M_S}_m_s")

    if not math.isfinite(rh):
        reasons.append("rh_non_finite")
    elif rh < RH_MIN_PCT:
        reasons.append("rh_below_reference_min_0_pct")
    elif rh > RH_MAX_PCT:
        reasons.append("rh_above_reference_max_100_pct")

    if not math.isfinite(pmv_raw):
        reasons.append("pmv_non_finite")
    elif pmv_raw < PMV_REF_MIN:
        reasons.append(f"pmv_below_reference_range_{PMV_REF_MIN}")
    elif pmv_raw > PMV_REF_MAX:
        reasons.append(f"pmv_above_reference_range_{PMV_REF_MAX}")

    valid = len(reasons) == 0
    applicability: ApplicabilityStatus = (
        "within_reference_range" if valid else "out_of_reference_range"
    )

    return {
        "valid_for_comfort_interpretation": valid,
        "applicability": applicability,
        "reasons": reasons,
        "pmv_raw": pmv_raw,
        "pmv_display": clamp_pmv_display(pmv_raw) if math.isfinite(pmv_raw) else pmv_raw,
    }


def applicability_seat_fields(
    applicability: Dict[str, Any],
) -> Dict[str, Any]:
    """Map ``evaluate_pmv_applicability`` result to preview/replay seat keys."""
    return {
        "pmv_raw": float(applicability["pmv_raw"]),
        "pmv_display": float(applicability["pmv_display"]),
        "valid_for_comfort_interpretation": bool(
            applicability["valid_for_comfort_interpretation"]
        ),
        "pmv_applicability": str(applicability["applicability"]),
        "pmv_applicability_reasons": list(applicability["reasons"]),
    }
