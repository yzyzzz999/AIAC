"""Feet V6 floor surface proxy — opt-in diagnostic output only.

Does not modify ``thermal.py`` or CHTD ``FeetTemp`` states. FloorProxy is a
post-rollout lag filter on deployable model temperatures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

FEET_V6_SCHEMA = "feet_v6_floor_proxy_adapter_v1"
FLOOR_PROXY_OUTPUT_ONLY = True
OPT_IN_ONLY = True

TAU_FLOOR_GRID = (300.0, 600.0, 900.0, 1200.0)


@dataclass(frozen=True)
class FloorProxyConfig:
    """Opt-in floor surface temperature proxy (diagnostic / preview)."""

    variant_id: str
    drive_mode: str
    tau_floor_s: float
    w_feet: float = 0.0
    w_cabin: float = 0.0
    w_foot_tma: float = 0.0
    w_console: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": FEET_V6_SCHEMA,
            "opt_in_only": OPT_IN_ONLY,
            "floor_proxy_output_only": FLOOR_PROXY_OUTPUT_ONLY,
            **asdict(self),
        }


def compute_t_drive(
    *,
    feet_air_c: float,
    cabin_c: float,
    foot_tma_c: float,
    console_c: float,
    cfg: FloorProxyConfig,
) -> float:
    """Deployable-only drive temperature for floor proxy lag filter."""
    if cfg.drive_mode == "feet_only":
        return feet_air_c
    if cfg.drive_mode == "cabin_only":
        return cabin_c
    if cfg.drive_mode == "console_only":
        return console_c
    w_sum = cfg.w_feet + cfg.w_cabin + cfg.w_foot_tma + cfg.w_console
    if w_sum <= 0.0:
        return feet_air_c
    return (
        cfg.w_feet * feet_air_c
        + cfg.w_cabin * cabin_c
        + cfg.w_foot_tma * foot_tma_c
        + cfg.w_console * console_c
    ) / w_sum


def rollout_floor_proxy_series(
    times_s: np.ndarray,
    feet_air: np.ndarray,
    cabin: np.ndarray,
    foot_tma: np.ndarray,
    console: np.ndarray,
    cabin_init_c: float,
    cfg: FloorProxyConfig,
) -> np.ndarray:
    """First-order lag: FloorProxy follows T_drive; init from deployable cabin only."""
    n = len(times_s)
    out = np.empty(n, dtype=float)
    proxy = float(cabin_init_c)
    prev_t = float(times_s[0])
    for i in range(n):
        if i > 0:
            dt = max(1e-3, float(times_s[i]) - prev_t)
            prev_t = float(times_s[i])
            t_drv = compute_t_drive(
                feet_air_c=float(feet_air[i - 1]),
                cabin_c=float(cabin[i - 1]),
                foot_tma_c=float(foot_tma[i - 1]),
                console_c=float(console[i - 1]),
                cfg=cfg,
            )
            alpha = min(1.0, dt / float(cfg.tau_floor_s))
            proxy = proxy + alpha * (t_drv - proxy)
        out[i] = proxy
    return out


def build_floor_proxy_grid() -> list[FloorProxyConfig]:
    """≤30 variants: 6 drive modes × 4 tau values."""
    drives: list[tuple[str, float, float, float, float]] = [
        ("feet_only", 1.0, 0.0, 0.0, 0.0),
        ("cabin_only", 0.0, 1.0, 0.0, 0.0),
        ("fc_70_30", 0.7, 0.3, 0.0, 0.0),
        ("fc_50_50", 0.5, 0.5, 0.0, 0.0),
        ("fct_50_30_20", 0.5, 0.3, 0.2, 0.0),
        ("cs_50_50", 0.0, 0.5, 0.0, 0.5),
    ]
    out: list[FloorProxyConfig] = []
    for mode, wf, wc, wt, ws in drives:
        for tau in TAU_FLOOR_GRID:
            vid = f"{mode}|tau={int(tau)}"
            out.append(FloorProxyConfig(
                variant_id=vid,
                drive_mode=mode,
                tau_floor_s=float(tau),
                w_feet=wf,
                w_cabin=wc,
                w_foot_tma=wt,
                w_console=ws,
            ))
    return out


__all__ = [
    "FloorProxyConfig",
    "FEET_V6_SCHEMA",
    "FLOOR_PROXY_OUTPUT_ONLY",
    "TAU_FLOOR_GRID",
    "build_floor_proxy_grid",
    "compute_t_drive",
    "rollout_floor_proxy_series",
]
