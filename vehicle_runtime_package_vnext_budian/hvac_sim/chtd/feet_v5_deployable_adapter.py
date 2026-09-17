"""Feet V5 deployable submodel adapter — opt-in; does not modify ``thermal.py``."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from hvac_sim.afe.actuator_score_decoder import decode_hvac_actuator_scores
from hvac_sim.chtd.bus_index import U_INDEX, X_INDEX
from hvac_sim.validation.chtd_foot_flow_gating_fix import apply_leakage_ratio
from hvac_sim.validation.chtd_runtime_rollout_compare import foot_open_normalized

FEET_V5_SCHEMA = "feet_v5_deployable_adapter_v1"
OPT_IN_ONLY = True


@dataclass(frozen=True)
class FeetV5Config:
    """Deployable feet effective-model knobs (opt-in preview)."""

    variant_id: str = "baseline"
    k_tma: float = 1.0
    face_leakage_ratio: float = 0.0
    use_face_mode_gate: bool = False
    tau_floor_s: Optional[float] = None
    duct_split_diagnostic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": FEET_V5_SCHEMA,
            "opt_in_only": OPT_IN_ONLY,
            **asdict(self),
        }


def apply_v5_effective_foot_tma(u: np.ndarray, x: np.ndarray, k_tma: float) -> np.ndarray:
    """T_eff = T_cabin + k_tma * (T_supply - T_cabin)."""
    out = u.copy()
    for tma_key, cabin_key in (
        ("FrntFdfTma", "CabinTempFd"),
        ("FrntFpfTma", "CabinTempFp"),
    ):
        cabin = float(x[X_INDEX[cabin_key]])
        raw = float(out[U_INDEX[tma_key]])
        out[U_INDEX[tma_key]] = cabin + float(k_tma) * (raw - cabin)
    return out


def is_face_dominant(face_vent_v: float, foot_vent_v: float) -> bool:
    face_norm = foot_open_normalized(face_vent_v) if math.isfinite(face_vent_v) else 0.0
    foot_norm = foot_open_normalized(foot_vent_v) if math.isfinite(foot_vent_v) else 0.0
    return face_norm >= 0.75 and face_norm > foot_norm + 0.1


def apply_v5_face_mode_foot_gate(
    u: np.ndarray,
    *,
    face_vent_v: float,
    foot_vent_v: float,
    leakage_ratio: float,
) -> tuple[np.ndarray, bool]:
    """Face-high / foot-low: foot_flow = leakage_ratio * total_front_flow."""
    if leakage_ratio <= 0.0 or not is_face_dominant(face_vent_v, foot_vent_v):
        return u, False
    foot_norm = foot_open_normalized(foot_vent_v) if math.isfinite(foot_vent_v) else 0.0
    if foot_norm >= 0.35:
        return u, False
    return apply_leakage_ratio(u, leakage_ratio=leakage_ratio), True


def update_floor_proxy(
    proxy: float,
    feet_model: float,
    dt: float,
    tau_floor_s: Optional[float],
) -> float:
    if tau_floor_s is None or tau_floor_s <= 0.0:
        return feet_model
    alpha = min(1.0, max(dt, 1e-3) / float(tau_floor_s))
    return proxy + alpha * (feet_model - proxy)


__all__ = [
    "FeetV5Config",
    "FEET_V5_SCHEMA",
    "OPT_IN_ONLY",
    "apply_v5_effective_foot_tma",
    "apply_v5_face_mode_foot_gate",
    "is_face_dominant",
    "update_floor_proxy",
]
