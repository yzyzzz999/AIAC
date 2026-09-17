"""Front defrost supply temperature (TmaDef / FrntDefTmaEst) minimal observer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TmaDefResult:
    """Estimated front defrost duct air temperature."""

    tma_def_c: float
    posn_fdh: Optional[float]
    eva_temp_c: float
    hct_temp_c: Optional[float]
    tmadef_approx: bool
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def estimate_posn_fdh_from_blend(
    blend_request: float,
    method: str = "linear",
    angle_min_deg: float = 0.0,
    angle_max_deg: float = 90.0,
) -> float:
    """Map blend-door request to PosnFdh in [0, 1].

    ``blend_request`` may be:
    - a unitless mix ratio in [0, 1], or
    - a door angle in degrees (``angle_min_deg`` … ``angle_max_deg``).

    Future methods (e.g. ``arcsin``, ``geometric``) can be added without
    changing callers.
    """
    req = float(blend_request)
    if not math.isfinite(req):
        raise ValueError("blend_request must be finite")

    if method == "linear":
        if 0.0 <= req <= 1.0:
            return float(req)
        span = float(angle_max_deg) - float(angle_min_deg)
        if span <= 0.0:
            return 0.0
        ratio = (req - float(angle_min_deg)) / span
        return max(0.0, min(1.0, ratio))

    raise ValueError(f"unsupported PosnFdh mapping method: {method!r}")


def estimate_tma_def(
    eva_temp_c: float,
    hct_temp_c: Optional[float] = None,
    posn_fdh: Optional[float] = None,
    blend_request: Optional[float] = None,
    *,
    fallback: str = "eva",
) -> TmaDefResult:
    """Estimate TmaDef = PosnFdh*Hct + (1-PosnFdh)*EvaT.

    When ``hct_temp_c`` or ``posn_fdh`` is missing, ``fallback='eva'`` returns
    ``eva_temp_c`` and sets ``tmadef_approx=True`` in diagnostics.
    """
    if not math.isfinite(eva_temp_c):
        raise ValueError("eva_temp_c must be finite")
    if fallback not in ("eva",):
        raise ValueError(f"unsupported fallback: {fallback!r}")

    resolved_posn = posn_fdh
    posn_source = "posn_fdh"
    if resolved_posn is None and blend_request is not None:
        resolved_posn = estimate_posn_fdh_from_blend(blend_request)
        posn_source = "blend_request"

    hct = float(hct_temp_c) if hct_temp_c is not None else None
    if hct is not None and not math.isfinite(hct):
        hct = None

    if hct is not None and resolved_posn is not None:
        p = max(0.0, min(1.0, float(resolved_posn)))
        t_def = p * hct + (1.0 - p) * float(eva_temp_c)
        return TmaDefResult(
            tma_def_c=float(t_def),
            posn_fdh=p,
            eva_temp_c=float(eva_temp_c),
            hct_temp_c=hct,
            tmadef_approx=False,
            method="PosnFdh×Hct+(1-PosnFdh)×EvaT",
            diagnostics={
                "posn_source": posn_source,
                "formula": "TmaDef = PosnFdh*Hct + (1-PosnFdh)*EvaT",
            },
        )

    approx_reasons = []
    if hct is None:
        approx_reasons.append("hct_missing")
    if resolved_posn is None:
        approx_reasons.append("posn_fdh_missing")

    return TmaDefResult(
        tma_def_c=float(eva_temp_c),
        posn_fdh=(
            max(0.0, min(1.0, float(resolved_posn)))
            if resolved_posn is not None
            else None
        ),
        eva_temp_c=float(eva_temp_c),
        hct_temp_c=hct,
        tmadef_approx=True,
        method=f"fallback_{fallback}",
        diagnostics={
            "approximate": True,
            "approx_reasons": approx_reasons,
            "fallback": fallback,
        },
    )
