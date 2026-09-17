"""AFE_Calc airflow solve.

This module mirrors AFE_Calc.slx: it builds Rnorm from the AFE resistance bus,
aggregates the network resistances, solves the four Newton equations, and
distributes the solution into the 22-element Q_Out vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.fluid import air_density
from ..core.solver import NewtonResult, newton_solve
from .fan import FanSet, fan_k_coeffs
from .params import AFEParams
from .resistance import FlapSet, _parallel_r, compute_ress_bus, ress_bus_to_rnorm


def _conductance(r: float) -> float:
    """Pressure-flow conductance used for split ratios: 1/sqrt(R)."""
    if r == 0.0:
        return math.inf
    if not math.isfinite(r) or r < 0.0:
        return 0.0
    return 1.0 / math.sqrt(r)


def _split(total_q: float, *resistances: float) -> list[float]:
    conductances = [_conductance(r) for r in resistances]
    c_sum = sum(conductances)
    if c_sum == 0.0:
        return [0.0 for _ in resistances]
    return [total_q * c / c_sum for c in conductances]


def _r_calc(rnorm: np.ndarray, rho: float, index: int) -> float:
    return float(rnorm[index] / rho)


def calc_R_Fd(rnorm: np.ndarray, rho: float) -> float:
    """Front-driver zone resistance in AFE_Calc.slx."""
    r_up = _parallel_r(_r_calc(rnorm, rho, 8), _r_calc(rnorm, rho, 7))
    r_dn = _parallel_r(
        _r_calc(rnorm, rho, 3),
        _r_calc(rnorm, rho, 4),
        _r_calc(rnorm, rho, 5),
        _r_calc(rnorm, rho, 6),
    )
    return r_up + r_dn


def calc_R_Fp(rnorm: np.ndarray, rho: float) -> float:
    """Front-passenger zone resistance in AFE_Calc.slx."""
    r_up = _parallel_r(_r_calc(rnorm, rho, 14), _r_calc(rnorm, rho, 13))
    r_dn = _parallel_r(
        _r_calc(rnorm, rho, 9),
        _r_calc(rnorm, rho, 10),
        _r_calc(rnorm, rho, 11),
        _r_calc(rnorm, rho, 12),
    )
    return r_up + r_dn


def calc_R_2(rnorm: np.ndarray, rho: float) -> float:
    """Rear combined zone resistance in AFE_Calc.slx."""
    r_up = _parallel_r(
        _r_calc(rnorm, rho, 18),
        _r_calc(rnorm, rho, 17),
        _r_calc(rnorm, rho, 22),
        _r_calc(rnorm, rho, 21),
    )
    r_dn = _parallel_r(
        _r_calc(rnorm, rho, 15),
        _r_calc(rnorm, rho, 16),
        _r_calc(rnorm, rho, 19),
        _r_calc(rnorm, rho, 20),
    )
    return r_up + r_dn


def distribute_flows(
    q_newton: np.ndarray,
    R_Fd: float,
    R_Fp: float,
    rnorm: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Build the 22-element Q_Out vector from the Newton solution."""
    q1, q2, q1p, q2p = [float(v) for v in q_newton]

    q_fd, q_fp = _split(q1p, R_Fd, R_Fp)

    q_fd_def, q_fd_sv, q_fd_v, q_fd_f = _split(
        q_fd,
        _r_calc(rnorm, rho, 3),
        _r_calc(rnorm, rho, 4),
        _r_calc(rnorm, rho, 5),
        _r_calc(rnorm, rho, 6),
    )
    q_fp_def, q_fp_sv, q_fp_v, q_fp_f = _split(
        q_fp,
        _r_calc(rnorm, rho, 9),
        _r_calc(rnorm, rho, 10),
        _r_calc(rnorm, rho, 11),
        _r_calc(rnorm, rho, 12),
    )
    q_sdv, q_sdf, q_spv, q_spf = _split(
        q2p,
        _r_calc(rnorm, rho, 15),
        _r_calc(rnorm, rho, 16),
        _r_calc(rnorm, rho, 19),
        _r_calc(rnorm, rho, 20),
    )
    q_sd = q_sdv + q_sdf
    q_sp = q_spv + q_spf

    return np.array([
        q1 + q2,
        q1,
        q2,
        q1p,
        q2p,
        q_fd,
        q_fp,
        q_sd,
        q_sp,
        q_fd_def,
        q_fd_sv,
        q_fd_v,
        q_fd_f,
        q_fp_def,
        q_fp_sv,
        q_fp_v,
        q_fp_f,
        q_sdv,
        q_sdf,
        q_spv,
        q_spf,
        1.0,
    ], dtype=float)


@dataclass
class FlowInputs:
    """Operating-point inputs for a single AFE_Calc evaluation."""
    flaps: FlapSet
    n_F1: float = 1000.0
    n_F2: float = 1000.0
    n_S: float = 1000.0
    dp1: float = 0.0
    temp_C: float = 25.0
    pressure_Pa: float = 101_325.0
    rh_percent: float = 50.0


@dataclass
class AFECalcResult:
    """Output of a single AFE_Calc evaluation."""
    Q1: float
    Q2: float
    Q1_prime: float
    Q2_prime: float
    Q_rear: float
    Qm: float
    rho: float
    converged: bool
    rnorm: np.ndarray
    Q_out: np.ndarray
    R_Fd: float
    R_Fp: float
    R_2: float
    R1: float
    R2: float
    R: float
    iterations: int


def afe_calc(
    inputs: FlowInputs,
    afe_params: AFEParams,
    fan_set: FanSet,
    x0: Optional[np.ndarray] = None,
) -> AFECalcResult:
    """Run the AFE_Calc airflow calculation."""
    rho = air_density(inputs.temp_C, inputs.pressure_Pa, inputs.rh_percent)

    bus = compute_ress_bus(inputs.flaps, afe_params)
    rnorm = ress_bus_to_rnorm(bus, rho)

    R_osa = _r_calc(rnorm, rho, 0)
    R_rec = _r_calc(rnorm, rho, 1)
    R_Fd = calc_R_Fd(rnorm, rho)
    R_Fp = calc_R_Fp(rnorm, rho)
    R_2 = calc_R_2(rnorm, rho)
    R1 = _parallel_r(R_Fd, R_Fp)
    R2 = R_2
    R = 0.0

    k_F1 = fan_k_coeffs(fan_set.F1, inputs.n_F1, rho)
    k_F2 = fan_k_coeffs(fan_set.F2, inputs.n_F2, rho)
    k_S = fan_k_coeffs(fan_set.S, inputs.n_S, rho)

    if x0 is None:
        x0 = np.array([10.0, 10.0, 10.0, 10.0])

    result: NewtonResult = newton_solve(
        dp1=inputs.dp1,
        R_osa=R_osa,
        R_rec=R_rec,
        R1=R1,
        R2=R2,
        R=R,
        x0=x0,
        k_F1=k_F1,
        k_F2=k_F2,
        k_S=k_S,
    )

    q_newton = np.array([result.Q1, result.Q2, result.Q1_prime, result.Q2_prime])
    if result.converged:
        q_out = distribute_flows(q_newton, R_Fd, R_Fp, rnorm, rho)
    else:
        q_out = np.zeros(22, dtype=float)
        q_out[21] = 1.0

    q_total = float(q_out[0])
    return AFECalcResult(
        Q1=result.Q1,
        Q2=result.Q2,
        Q1_prime=result.Q1_prime,
        Q2_prime=result.Q2_prime,
        Q_rear=result.Q2_prime,
        Qm=q_total * rho,
        rho=rho,
        converged=result.converged,
        rnorm=rnorm,
        Q_out=q_out,
        R_Fd=R_Fd,
        R_Fp=R_Fp,
        R_2=R_2,
        R1=R1,
        R2=R2,
        R=R,
        iterations=result.iterations,
    )
