"""Two-row vehicle runtime policy for CHTD state and input vectors.

Current product is a two-row vehicle. Third-row (Td/Tp) zones are not fully
modelled, but leaving them at arbitrary values can pollute RoofTemp and other
TRACE couplings. Policy: initialize third-row states to AmbT and zero
third-row vent/human heat flows before each CHTD step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .bus_index import N_U, N_X_STATES, U_INDEX, X_INDEX

POLICY_NAME = "two_row_vehicle_v1"

# Third-row zones (Td/Tp suffix) — not implemented for current vehicle.
THIRD_ROW_STATE_NAMES: Tuple[str, ...] = (
    "HeadTempTd",
    "HeadTempTp",
    "FeetTempTd",
    "FeetTempTp",
    "CabinTempTd",
    "CabinTempTp",
    "WinTempTd",
    "WinTempTp",
)

THIRD_ROW_FLOW_NAMES: Tuple[str, ...] = (
    "RearTdvTma",
    "RearTdfTma",
    "RearTpvTma",
    "RearTpfTma",
    "RearTdvFlow",
    "RearTdfFlow",
    "RearTpvFlow",
    "RearTpfFlow",
    "HumHeadTdPower",
    "HumHeadTpPower",
    "HumFeetTdPower",
    "HumFeetTpPower",
)

# Front + second row states that must not be modified by this policy.
PROTECTED_TRACE_STATE_NAMES: Tuple[str, ...] = (
    "HeadTempFd",
    "HeadTempFp",
    "HeadTempSd",
    "HeadTempSp",
    "FeetTempFd",
    "FeetTempFp",
    "FeetTempSd",
    "FeetTempSp",
)

REAR_FOOT_AIR_SPEED_FIELDS: Tuple[str, ...] = (
    "rear_driver_foot_flow",
    "rear_passenger_foot_flow",
)


def _as_float_vector(arr: np.ndarray, length: int, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {out.shape}")
    return out.copy()


def initialize_missing_rear_row_states(
    x: np.ndarray,
    amb_t: float,
) -> tuple[np.ndarray, List[str]]:
    """Set third-row CHTD states to ambient temperature."""
    x_out = _as_float_vector(x, N_X_STATES, "x")
    initialized: List[str] = []
    for name in THIRD_ROW_STATE_NAMES:
        idx = X_INDEX[name]
        x_out[idx] = float(amb_t)
        initialized.append(name)
    return x_out, initialized


def zero_missing_rear_row_flows(u: np.ndarray) -> tuple[np.ndarray, List[str]]:
    """Zero third-row vent and human-heat inputs on Bus_CHTD_u."""
    u_out = _as_float_vector(u, N_U, "u")
    zeroed: List[str] = []
    for name in THIRD_ROW_FLOW_NAMES:
        idx = U_INDEX[name]
        if u_out[idx] != 0.0:
            zeroed.append(name)
        u_out[idx] = 0.0
    return u_out, zeroed


def apply_two_row_vehicle_policy(
    x: np.ndarray,
    u: np.ndarray,
    *,
    amb_t: float | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply two-row vehicle CHTD policy and return provenance metadata."""
    u_arr = _as_float_vector(u, N_U, "u")
    if amb_t is None:
        amb_t = float(u_arr[U_INDEX["AmbT"]])

    x_before = _as_float_vector(x, N_X_STATES, "x")
    protected_before = {
        name: float(x_before[X_INDEX[name]]) for name in PROTECTED_TRACE_STATE_NAMES
    }

    x_out, initialized_states = initialize_missing_rear_row_states(x_before, amb_t)
    u_out, zeroed_flows = zero_missing_rear_row_flows(u_arr)

    protected_after = {
        name: float(x_out[X_INDEX[name]]) for name in PROTECTED_TRACE_STATE_NAMES
    }

    provenance: Dict[str, Any] = {
        "policy": POLICY_NAME,
        "classification": "two_row_vehicle_not_three_row",
        "amb_t_c": float(amb_t),
        "initialized_third_row_states": initialized_states,
        "zeroed_third_row_flows": zeroed_flows,
        "protected_trace_states_unchanged": {
            name: protected_before[name] == protected_after[name]
            for name in PROTECTED_TRACE_STATE_NAMES
        },
        "rear_foot": {
            "rear_driver_foot_flow": None,
            "rear_passenger_foot_flow": None,
            "status": "missing_measurement",
            "note": (
                "Rear foot outlet flows remain unset (None/0) until bus mapping "
                "and bench measurement close G11; not overridden by CHTD policy."
            ),
        },
        "notes": [
            "Third-row Td/Tp states forced to AmbT before CHTD — avoids RoofTemp pollution",
            "Third-row vent/human heat u-bus signals zeroed for two-row vehicle",
            "Does not modify Fd/Fp/Sd/Sp TRACE state values",
        ],
    }
    return x_out, u_out, provenance
