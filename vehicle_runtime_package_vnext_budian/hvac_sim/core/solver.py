"""Newton-Raphson flow solver.

Direct translation of the MATLAB Function block 'NewtonSolver_Simulink' inside
AFE_Calc.slx/Calc.  Solves the 4-equation nonlinear system for the four flow
rates (Q1, Q2, Q1', Q2') that balance pressure drops across the HVAC network.

Network topology (two fans in parallel, each driving its own zone):
    dp1 = pressure rise available at inlet (OSA duct)
    Fan-1 drives zone-1 (front, R1)
    Fan-2 drives zone-2 (front passenger, shares recirculation path, R2)
    Fan-S drives rear zones (R_sub)

Equations (Q = volumetric flow, sq(q) = q·|q|  to keep sign):
    F1: dp1 - R_Osa·sq(Q1) + dp_F1(Q1) + R_Rec·sq(Q2) - dp_F2(Q2) = 0
    F2: Q1 + Q2 - Q1' - Q2' = 0   (mass balance at junction)
    F3: R1·sq(Q1') - R2·sq(Q2') + dp_S(Q2') = 0
    F4: R·sq(Q1+Q2) + R_Rec·sq(Q2) - dp_F2(Q2) + R1·sq(Q1') = 0

Fan pressure rise polynomial:  dp_fan(Q) = k[0]·Q² + k[1]·Q + k[2]
(k[1] is always 0 for symmetric fans; kept general for extensibility)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class NewtonResult:
    Q1: float
    Q2: float
    Q1_prime: float
    Q2_prime: float
    converged: bool
    iterations: int


def _sq(q: float) -> float:
    """Signed square: q·|q|."""
    return q * abs(q)


def _dsq(q: float) -> float:
    """Derivative of signed square: 2·|q|."""
    return 2.0 * abs(q)


def _fan_dp(q: float, k: np.ndarray) -> Tuple[float, float]:
    """Fan pressure rise and its derivative w.r.t. Q.

    k = [k0, k1, k2]  →  dp = k0·q² + k1·q + k2
    Note: k0 is typically negative (fan stalls at high flow).
    """
    dp = k[0] * q * q + k[1] * q + k[2]
    ddp = 2.0 * k[0] * q + k[1]
    return dp, ddp


def newton_solve(
    dp1: float,
    R_osa: float,
    R_rec: float,
    R1: float,
    R2: float,
    R: float,
    x0: np.ndarray,
    k_F1: np.ndarray,
    k_F2: np.ndarray,
    k_S: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    ls_max: int = 5,
) -> NewtonResult:
    """Solve the 4-equation airflow network for (Q1, Q2, Q1', Q2').

    Parameters
    ----------
    dp1     : pressure available at OSA inlet [Pa]
    R_osa   : OSA duct resistance  [Pa·s²/m⁶]
    R_rec   : recirculation duct resistance
    R1, R2  : zone-1 and zone-2 total resistance
    R       : combined outlet resistance
    x0      : initial guess [Q1, Q2, Q1', Q2']
    k_F1/F2/S : fan polynomial coefficients [k0, k1, k2]
    max_iter, tol, ls_max : solver control
    """
    x = np.array(x0, dtype=float).reshape(4)

    for it in range(1, max_iter + 1):
        q1, q2, q1p, q2p = x

        dp_f1, d_f1 = _fan_dp(q1, k_F1)
        dp_f2, d_f2 = _fan_dp(q2, k_F2)
        dp_s, d_s = _fan_dp(q2p, k_S)

        sq1, sq2, sq1p, sq2p = _sq(q1), _sq(q2), _sq(q1p), _sq(q2p)
        sq12 = _sq(q1 + q2)

        F = np.array([
            dp1 - R_osa * sq1 + dp_f1 + R_rec * sq2 - dp_f2,
            q1 + q2 - q1p - q2p,
            R1 * sq1p - R2 * sq2p + dp_s,
            R * sq12 + R_rec * sq2 - dp_f2 + R1 * sq1p,
        ])

        if np.max(np.abs(F)) < tol:
            return NewtonResult(q1, q2, q1p, q2p, converged=True, iterations=it)

        # Jacobian  ∂F/∂x  (4×4)
        d1 = _dsq(q1)
        d2 = _dsq(q2)
        d1p = _dsq(q1p)
        d2p = _dsq(q2p)
        d12 = _dsq(q1 + q2)

        J = np.array([
            [-R_osa * d1 + d_f1,   R_rec * d2 - d_f2,  0.0,            0.0       ],
            [1.0,                   1.0,                -1.0,           -1.0       ],
            [0.0,                   0.0,                 R1 * d1p,      -R2 * d2p + d_s],
            [R * d12,               R * d12 + R_rec * d2 - d_f2,  R1 * d1p,  0.0 ],
        ])

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break

        # Backtracking line search
        alpha = 1.0
        F_norm = float(np.dot(F, F))
        for _ in range(ls_max):
            x_new = x + alpha * dx
            q1n, q2n, q1pn, q2pn = x_new
            dp_f1n, _ = _fan_dp(q1n, k_F1)
            dp_f2n, _ = _fan_dp(q2n, k_F2)
            dp_sn, _ = _fan_dp(q2pn, k_S)
            F_new = np.array([
                dp1 - R_osa * _sq(q1n) + dp_f1n + R_rec * _sq(q2n) - dp_f2n,
                q1n + q2n - q1pn - q2pn,
                R1 * _sq(q1pn) - R2 * _sq(q2pn) + dp_sn,
                R * _sq(q1n + q2n) + R_rec * _sq(q2n) - dp_f2n + R1 * _sq(q1pn),
            ])
            if float(np.dot(F_new, F_new)) < F_norm:
                break
            alpha *= 0.5
        x = x + alpha * dx

    q1, q2, q1p, q2p = x
    return NewtonResult(q1, q2, q1p, q2p, converged=False, iterations=max_iter)
