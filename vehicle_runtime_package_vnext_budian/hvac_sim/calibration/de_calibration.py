"""Legacy AFE/CHTD calibration via scipy differential_evolution.

Kept for backward compatibility. New PMV calibration should use ``pso.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from ..afe.params import AFEParams
from ..afe.resistance import FlapSet, compute_ress_bus, ress_bus_to_rnorm
from ..afe.fan import FanSet, fan_k_coeffs
from ..afe.flow import FlowInputs, afe_calc
from ..chtd.params import CHTDParams
from ..chtd.thermal import CHTDInputs, simulate_chtd, STATE_NAMES


@dataclass
class CalibResult:
    """Calibration output."""

    best_params: np.ndarray
    best_cost: float
    param_names: List[str]
    history: List[float]
    converged: bool
    raw: OptimizeResult = dataclasses.field(repr=False, default=None)  # type: ignore


def _run_de(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    param_names: List[str],
    popsize: int = 15,
    maxiter: int = 100,
    seed: int = 42,
    workers: int = 1,
    tol: float = 1e-6,
    callback_history: Optional[List[float]] = None,
) -> CalibResult:
    """Run scipy differential_evolution and return a CalibResult."""
    history: List[float] = [] if callback_history is None else callback_history

    def _callback(xk, convergence):
        history.append(float(objective(xk)))

    result: OptimizeResult = differential_evolution(
        func=objective,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        workers=workers,
        tol=tol,
        callback=_callback,
        polish=True,
        updating="deferred" if workers != 1 else "immediate",
    )

    return CalibResult(
        best_params=result.x,
        best_cost=float(result.fun),
        param_names=param_names,
        history=history,
        converged=result.success,
        raw=result,
    )


_AFE_CALIB_FIELDS = [
    "OsaFlapRessCo",
    "RecFlapRessCo",
    "EvaRessCo",
    "FdDefFlapRessCo",
    "FdSvFlapRessCo",
    "FdvFlapRessCo",
    "FdfFlapRessCo",
    "FdHexUpRessCo",
    "FdPassUpRessCo",
    "FpDefFlapRessCo",
    "FpSvFlapRessCo",
    "FpvFlapRessCo",
    "FpfFlapRessCo",
    "FpHexUpRessCo",
    "FpPassUpRessCo",
    "SdvFlapRessCo",
    "SdfFlapRessCo",
    "SdHexRessCo",
    "SdPassRessCo",
    "SpvFlapRessCo",
    "SpfFlapRessCo",
    "SpHexRessCo",
    "SpPassRessCo",
]


@dataclass
class AFEMeasurement:
    """A single test-case measurement for AFE calibration."""

    flaps: FlapSet
    fan_set: FanSet
    n_F1: float
    n_F2: float
    n_S: float
    temp_C: float = 25.0
    dp1: float = 0.0
    Qm_measured: Optional[float] = None
    Q1_measured: Optional[float] = None
    Q1p_measured: Optional[float] = None
    Q2p_measured: Optional[float] = None
    Q_rear_measured: Optional[float] = None
    w_Qm: float = 1.0
    w_Q1: float = 0.0
    w_Q1p: float = 1.0
    w_Q2p: float = 1.0
    w_rear: float = 0.5


def _afe_residuals(
    params_vec: np.ndarray,
    measurements: List[AFEMeasurement],
    base_params: AFEParams,
    calib_fields: List[str],
) -> float:
    p = dataclasses.replace(base_params, **dict(zip(calib_fields, params_vec)))
    total_cost = 0.0
    for m in measurements:
        inp = FlowInputs(
            flaps=m.flaps,
            n_F1=m.n_F1,
            n_F2=m.n_F2,
            n_S=m.n_S,
            dp1=m.dp1,
            temp_C=m.temp_C,
        )
        try:
            res = afe_calc(inp, p, m.fan_set)
        except Exception:
            return 1e10

        if m.Qm_measured is not None:
            total_cost += m.w_Qm * (res.Qm - m.Qm_measured) ** 2
        if m.Q1_measured is not None:
            total_cost += m.w_Q1 * (res.Q1 - m.Q1_measured) ** 2
        if m.Q1p_measured is not None:
            total_cost += m.w_Q1p * (res.Q1_prime - m.Q1p_measured) ** 2
        if m.Q2p_measured is not None:
            total_cost += m.w_Q2p * (res.Q2_prime - m.Q2p_measured) ** 2
        if m.Q_rear_measured is not None:
            total_cost += m.w_rear * (res.Q_rear - m.Q_rear_measured) ** 2

    return total_cost


def calibrate_afe(
    measurements: List[AFEMeasurement],
    base_params: Optional[AFEParams] = None,
    calib_fields: Optional[List[str]] = None,
    lb: float = 0.1,
    ub: float = 5000.0,
    popsize: int = 15,
    maxiter: int = 200,
    workers: int = 1,
    seed: int = 42,
) -> CalibResult:
    if base_params is None:
        base_params = AFEParams()
    if calib_fields is None:
        calib_fields = _AFE_CALIB_FIELDS

    bounds = [(lb, ub)] * len(calib_fields)
    objective = lambda v: _afe_residuals(v, measurements, base_params, calib_fields)

    result = _run_de(
        objective=objective,
        bounds=bounds,
        param_names=calib_fields,
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        workers=workers,
    )

    result.raw_params_obj = dataclasses.replace(
        base_params, **dict(zip(calib_fields, result.best_params))
    )
    return result


_CHTD_CALIB_FIELDS = [
    "HoodMassAtb",
    "CabinFrntMassAtb",
    "ConsoleMassAtb",
    "CabinFdMassAtb",
    "CabinFpMassAtb",
    "CabinSdMassAtb",
    "CabinSpMassAtb",
    "CabinTdMassAtb",
    "CabinTpMassAtb",
    "WinFdMassAtb",
    "WinFpMassAtb",
    "WinSdMassAtb",
    "WinSpMassAtb",
    "WinTdMassAtb",
    "WinTpMassAtb",
    "RoofMassAtb",
    "HoodUAAtb",
    "CabinFrntUAAtb",
    "CabinFdUAAtb",
    "CabinFpUAAtb",
    "CabinSdUAAtb",
    "CabinSpUAAtb",
    "WinFdUAAtb",
    "WinFpUAAtb",
    "WinSdUAAtb",
    "WinSpUAAtb",
    "RoofUAAtb",
]


@dataclass
class CHTDMeasurement:
    """A time-series measurement for CHTD calibration."""

    t_vec: np.ndarray
    inputs_list: List[CHTDInputs]
    T_measured: np.ndarray
    T0: float = 20.0
    state_weights: Optional[np.ndarray] = None


def _chtd_residuals(
    params_vec: np.ndarray,
    measurements: List[CHTDMeasurement],
    base_params: CHTDParams,
    calib_fields: List[str],
) -> float:
    p = dataclasses.replace(base_params, **dict(zip(calib_fields, params_vec)))
    if p.Dt <= 0:
        return 1e10

    total_cost = 0.0
    for m in measurements:
        try:
            T_sim = simulate_chtd(m.t_vec, m.inputs_list, p, T0=m.T0)
        except Exception:
            return 1e10

        diff = T_sim - m.T_measured
        w = m.state_weights if m.state_weights is not None else np.ones(len(STATE_NAMES))
        total_cost += float(np.sum(w * np.sum(diff**2, axis=0)))

    return total_cost


def calibrate_chtd(
    measurements: List[CHTDMeasurement],
    base_params: Optional[CHTDParams] = None,
    calib_fields: Optional[List[str]] = None,
    lb: float = 0.1,
    ub: float = 1e6,
    popsize: int = 15,
    maxiter: int = 200,
    workers: int = 1,
    seed: int = 42,
) -> CalibResult:
    if base_params is None:
        base_params = CHTDParams()
    if calib_fields is None:
        calib_fields = _CHTD_CALIB_FIELDS

    bounds = [(lb, ub)] * len(calib_fields)
    objective = lambda v: _chtd_residuals(v, measurements, base_params, calib_fields)

    result = _run_de(
        objective=objective,
        bounds=bounds,
        param_names=calib_fields,
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        workers=workers,
    )

    result.raw_params_obj = dataclasses.replace(
        base_params, **dict(zip(calib_fields, result.best_params))
    )
    return result
