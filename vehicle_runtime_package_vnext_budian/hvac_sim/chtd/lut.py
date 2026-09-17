"""CHTD general-purpose 1-D LUT wrappers — T4 implementation.

Provides three public functions:
    lookup_1d          — validated linear-interp + clip (Simulink 1-D Lookup Table)
    lookup_chtd_amb    — CHTD-specific wrapper over CHTD_AmbT_X breakpoints
    eval_lut_map       — evaluate ALL *_M fields at a given AmbT; returns dict

Optional helpers (not imported by helpers.py — no circular dependency):
    is_lut_field       — classify a params field as a LUT map
    validate_breakpoints — assert breakpoint array is valid

All functions are stateless and do NOT depend on thermal.py.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Union

import numpy as np


# ── Breakpoint validation ─────────────────────────────────────────────────────

def validate_breakpoints(xp: np.ndarray) -> None:
    """Assert that xp is a valid 1-D strictly-increasing breakpoint vector.

    Raises
    ------
    ValueError : if xp is not 1-D, is empty, or is not strictly increasing.
    """
    xp = np.asarray(xp, dtype=float)
    if xp.ndim != 1:
        raise ValueError(
            f"validate_breakpoints: xp must be 1-D, got ndim={xp.ndim}"
        )
    if xp.size < 2:
        raise ValueError(
            f"validate_breakpoints: xp must have >= 2 elements, got {xp.size}"
        )
    diffs = np.diff(xp)
    if not np.all(diffs > 0):
        raise ValueError(
            "validate_breakpoints: xp must be strictly increasing; "
            f"found non-positive diff(s): {diffs[diffs <= 0]}"
        )


# ── General-purpose 1-D LUT ───────────────────────────────────────────────────

def lookup_1d(
    x: float,
    xp: np.ndarray,
    fp: np.ndarray,
) -> float:
    """General-purpose 1-D LUT: linear interpolation with clip extrapolation.

    Matches the behaviour of a Simulink 1-D Lookup Table block with
    'Linear' interpolation and 'Clip' extrapolation.

    Parameters
    ----------
    x  : query point (scalar)
    xp : breakpoint vector, must be 1-D and strictly increasing
    fp : table values, must be 1-D and same length as xp

    Returns
    -------
    Interpolated scalar float.

    Raises
    ------
    ValueError : if xp/fp fail shape/monotone validation.
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.ndim != 1:
        raise ValueError(
            f"lookup_1d: xp must be 1-D, got ndim={xp.ndim}"
        )
    if fp.ndim != 1:
        raise ValueError(
            f"lookup_1d: fp must be 1-D, got ndim={fp.ndim}"
        )
    if xp.shape != fp.shape:
        raise ValueError(
            f"lookup_1d: xp and fp must have the same shape; "
            f"got xp={xp.shape}, fp={fp.shape}"
        )
    if xp.size < 2:
        raise ValueError(
            f"lookup_1d: xp must have >= 2 elements, got {xp.size}"
        )
    diffs = np.diff(xp)
    if not np.all(diffs > 0):
        raise ValueError(
            f"lookup_1d: xp must be strictly increasing; "
            f"found non-positive diff(s): {diffs[diffs <= 0]}"
        )

    return float(np.interp(float(x), xp, fp))


# ── CHTD ambient-temperature LUT ─────────────────────────────────────────────

def lookup_chtd_amb(
    table: np.ndarray,
    amb_t: float,
    params,
) -> float:
    """Evaluate a CHTD_*_M LUT at the current ambient temperature.

    Thin wrapper around lookup_1d that uses params.CHTD_AmbT_X as
    breakpoints and validates table shape against those breakpoints.

    Consistent with helpers.lookup_ambt (T3) — both produce identical
    results for valid inputs.

    Parameters
    ----------
    table  : 1-D LUT values, shape must match params.CHTD_AmbT_X
    amb_t  : ambient temperature [°C]
    params : CHTDParams instance (provides CHTD_AmbT_X breakpoints)

    Returns
    -------
    Interpolated scalar float.

    Raises
    ------
    ValueError : if table is not an ndarray or shape mismatches CHTD_AmbT_X.
    """
    xp = np.asarray(params.CHTD_AmbT_X, dtype=float)
    if not isinstance(table, np.ndarray):
        raise ValueError(
            f"lookup_chtd_amb: table must be ndarray, got {type(table).__name__}"
        )
    if table.shape != xp.shape:
        raise ValueError(
            f"lookup_chtd_amb: table shape {table.shape} does not match "
            f"CHTD_AmbT_X shape {xp.shape}"
        )
    return lookup_1d(float(amb_t), xp, table)


# ── LUT field classifier ──────────────────────────────────────────────────────

def is_lut_field(name: str, value, suffix: str = "_M") -> bool:
    """Return True if this params field is a LUT map.

    A field is considered a LUT map when:
    - its name ends with `suffix` (default "_M"), AND
    - its value is a numpy ndarray.

    Parameters
    ----------
    name   : field name string
    value  : field value (as read from the params instance)
    suffix : field-name suffix that marks LUT maps (default "_M")

    Returns
    -------
    bool
    """
    return name.endswith(suffix) and isinstance(value, np.ndarray)


# ── Evaluate all LUT maps at a given AmbT ────────────────────────────────────

def eval_lut_map(
    params,
    amb_t: float,
    suffix: str = "_M",
) -> dict:
    """Evaluate all *_M LUT fields at the given ambient temperature.

    Iterates over every dataclass field whose name ends with `suffix` AND
    whose value is a numpy ndarray, evaluates it via lookup_chtd_amb, and
    returns a flat dict mapping field names to their interpolated scalars.

    The params instance is NOT modified.

    Parameters
    ----------
    params : CHTDParams instance
    amb_t  : ambient temperature [°C]
    suffix : field suffix that identifies LUT maps (default "_M")

    Returns
    -------
    dict[str, float] — one entry per LUT field, e.g.
        {"CHTD_FdvFdConvCo_M": 1.0, "CHTD_HoodSolarRadCo_M": 1.0, ...}
    """
    result: dict = {}
    for f in dc_fields(params):
        value = getattr(params, f.name)
        if is_lut_field(f.name, value, suffix):
            result[f.name] = lookup_chtd_amb(value, amb_t, params)
    return result
