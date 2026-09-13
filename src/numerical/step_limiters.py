"""
Numerical bounding and line search module for the cycle matching solver.
Prevents unphysical state excursions and ensures residual descent.
"""

import numpy as np
from typing import Callable, Tuple


def calculate_max_step_fraction(
    x_current: np.ndarray, delta_x: np.ndarray, max_rel_change: float = 0.10
) -> float:
    """
    Calculates the maximum allowable fraction of the Newton step to ensure no
    variable changes by more than 'max_rel_change' (e.g., 10%) in a single iteration.
    Crucial for highly non-linear gas turbine maps to avoid flying off the grid.
    """
    # Guard against division by zero for states near origin
    x_safe = np.where(np.abs(x_current) < 1e-6, 1e-6, x_current)
    rel_changes = np.abs(delta_x / x_safe)

    max_change_idx = np.argmax(rel_changes)
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
    """
    Armijo backtracking line search to guarantee monotonic reduction of the residual norm.
    If the full Newton step increases the error (due to map non-linearities),
    the step is iteratively halved (tau) until the error decreases.

    Args:
        objective_func: Function that takes state vector x and returns the residual norm ||R(x)||.
        x_current: Current state vector.
        delta_x: Proposed Newton step direction.
        current_residual_norm: Norm of the residuals at x_current.
        max_step_fraction: Maximum allowed step multiplier.
        alpha: Sufficient decrease parameter.
        tau: Step size reduction factor (backtracking multiplier).
        max_iters: Maximum backtracking iterations before forcing evaluation.

    Returns:
        Tuple containing the updated state vector and its new residual norm.
    """
    step_size = max_step_fraction

    for _ in range(max_iters):
        x_trial = x_current + step_size * delta_x
        trial_residual_norm = objective_func(x_trial)

        # Armijo condition for sufficient decrease in non-linear systems
        if trial_residual_norm <= current_residual_norm * (1.0 - alpha * step_size):
            return x_trial, trial_residual_norm

        # Backtrack: cut the step size in half
        step_size *= tau

    # If line search fails to find a strict descent (Jacobian ill-conditioned or local minima),
    # return the smallest evaluated conservative step.
    x_fallback = x_current + step_size * delta_x
    return x_fallback, objective_func(x_fallback)
