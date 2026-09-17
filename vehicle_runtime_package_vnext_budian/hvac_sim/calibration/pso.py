"""Pure-numpy particle swarm optimization for PMV calibration (T9-3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

ObjectiveFn = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass(frozen=True)
class PSOOptions:
    """Tunable PSO hyper-parameters."""

    swarm_size: int = 30
    max_iter: int = 100
    inertia: float = 0.729
    cognitive: float = 1.49445
    social: float = 1.49445
    seed: int = 42
    tol: Optional[float] = None


@dataclass
class PSOResult:
    """PSO minimization result."""

    best_x: np.ndarray
    best_loss: float
    history: List[float]
    n_iter: int
    converged: bool


def _normalize_bounds(bounds: Bounds) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (lower, upper) pairs")
    if np.any(arr[:, 0] > arr[:, 1]):
        raise ValueError("each bound pair must satisfy lower <= upper")
    return arr


def _clamp_positions(positions: np.ndarray, bounds_arr: np.ndarray) -> np.ndarray:
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    return np.clip(positions, lo, hi)


def pso_optimize(
    objective: ObjectiveFn,
    bounds: Bounds,
    options: Optional[Union[PSOOptions, dict]] = None,
) -> PSOResult:
    """Minimize a scalar objective with canonical PSO (pure NumPy).

    Parameters
    ----------
    objective:
        Callable ``f(x) -> float`` where ``x`` is a 1-D parameter vector.
    bounds:
        Per-dimension ``(lower, upper)`` pairs. Positions are clamped each step.
    options:
        ``PSOOptions`` instance or dict with keys matching ``PSOOptions`` fields.
    """
    if options is None:
        opts = PSOOptions()
    elif isinstance(options, dict):
        opts = PSOOptions(**options)
    else:
        opts = options

    if opts.swarm_size < 1:
        raise ValueError("swarm_size must be >= 1")
    if opts.max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    bounds_arr = _normalize_bounds(bounds)
    dim = bounds_arr.shape[0]
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    span = hi - lo

    rng = np.random.default_rng(opts.seed)
    positions = lo + span * rng.random((opts.swarm_size, dim))
    velocities = rng.uniform(-0.1, 0.1, size=(opts.swarm_size, dim)) * span

    personal_best = positions.copy()
    personal_best_loss = np.full(opts.swarm_size, np.inf, dtype=float)

    history: List[float] = []
    best_x = np.full(dim, np.nan, dtype=float)
    best_loss = np.inf
    converged = False
    n_iter = 0

    for iteration in range(opts.max_iter):
        n_iter = iteration + 1
        for particle in range(opts.swarm_size):
            loss = float(objective(positions[particle]))
            if loss < personal_best_loss[particle]:
                personal_best_loss[particle] = loss
                personal_best[particle] = positions[particle].copy()
            if loss < best_loss:
                best_loss = loss
                best_x = positions[particle].copy()

        history.append(float(best_loss))

        if opts.tol is not None and len(history) >= 2:
            if abs(history[-2] - history[-1]) <= opts.tol:
                converged = True
                break

        r1 = rng.random((opts.swarm_size, dim))
        r2 = rng.random((opts.swarm_size, dim))
        velocities = (
            opts.inertia * velocities
            + opts.cognitive * r1 * (personal_best - positions)
            + opts.social * r2 * (best_x - positions)
        )
        positions = _clamp_positions(positions + velocities, bounds_arr)

    if not np.isfinite(best_loss):
        raise ValueError("objective returned non-finite values for all particles")

    return PSOResult(
        best_x=best_x,
        best_loss=float(best_loss),
        history=history,
        n_iter=n_iter,
        converged=converged,
    )
