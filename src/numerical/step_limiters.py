"""
Numerical bounding and line search module for the cycle matching solver.
Prevents unphysical state excursions and ensures residual descent.
Incorporates NaN/Inf trapping for robust mathematical evaluation.
"""

import math
import numpy as np
from typing import Callable, Tuple


def calculate_max_step_fraction(
    x_current: np.ndarray, delta_x: np.ndarray, max_rel_change: float = 0.10
) -> float:
    # Guard against division by zero for states near origin
    x_safe = np.where(np.abs(x_current) < 1e-6, 1e-6, x_current)
    rel_changes = np.abs(delta_x / x_safe)

    max_change_idx = int(np.argmax(rel_changes))
    max_change = rel_changes[max_change_idx]

    if max_change > max_rel_change:
        return float(max_rel_change / max_change)
    return 1.0


def armijo_line_search(
    objective_func: Callable[[np.ndarray], float],
    x_current: np.ndarray,
    delta_x: np.ndarray,
    current_residual_norm: float,
    max_step_fraction: float = 1.0,
    alpha: float = 1e-4,
    tau: float = 0.5,
    max_iters: int = 8,
) -> Tuple[np.ndarray, float]:

    step_size = max_step_fraction

    for _ in range(max_iters):
        x_trial = x_current + step_size * delta_x

        # DO-178C Failsafe: Trap domain errors (NaN/Inf) during trial evaluation
        try:
            trial_residual_norm = objective_func(x_trial)
            if math.isnan(trial_residual_norm) or math.isinf(trial_residual_norm):
                trial_residual_norm = float("inf")
        except Exception:
            trial_residual_norm = float("inf")

        # Armijo condition for sufficient decrease in non-linear systems
        if trial_residual_norm <= current_residual_norm * (1.0 - alpha * step_size):
            return x_trial, trial_residual_norm

        # Backtrack: cut the step size in half
        step_size *= tau

    # If line search fails to find a strict descent, return the smallest evaluated conservative step.
    # The solver will attempt to generate a new gradient from here.
    x_fallback = x_current + step_size * delta_x

    try:
        final_norm = objective_func(x_fallback)
        if math.isnan(final_norm):
            final_norm = float("inf")
    except Exception:
        final_norm = float("inf")

    return x_fallback, final_norm
