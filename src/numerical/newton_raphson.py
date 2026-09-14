"""
Newton-Raphson numerical solver with real-time telemetry logging per iteration.
"""

import numpy as np
from typing import Callable, Tuple


class NewtonRaphsonSolver:
    def __init__(
        self,
        max_iters: int = 50,
        tolerance: float = 1e-4,
        fd_eps: float = 1e-3,
        max_step_frac: float = 0.05,
    ):
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.fd_eps = fd_eps
        self.max_step_frac = max_step_frac

    def solve(
        self, eval_func: Callable[[np.ndarray], np.ndarray], x0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        x_current = np.array(x0, dtype=float)
        n = len(x_current)

        for iteration in range(self.max_iters):
            residuals = eval_func(x_current)
            res_norm = float(np.linalg.norm(residuals))

            print(
                f"[NR Iter {iteration:02d}] ||R|| = {res_norm:.6e} | State: {np.array2string(x_current, precision=4)}"
            )

            if res_norm < self.tolerance:
                return x_current, residuals, True

            # Compute Jacobian via finite differences
            J = np.zeros((n, n))
            for j in range(n):
                x_perturbed = x_current.copy()
                eps = max(abs(x_perturbed[j]) * self.fd_eps, 1e-6)
                x_perturbed[j] += eps
                res_perturbed = eval_func(x_perturbed)
                J[:, j] = (res_perturbed - residuals) / eps

            try:
                delta_x = np.linalg.solve(J, -residuals)
            except np.linalg.LinAlgError:
                print(f"[NR Error] Singular Jacobian at iteration {iteration}.")
                return x_current, residuals, False

            # Apply strict step bounding
            x_safe = np.where(np.abs(x_current) < 1e-6, 1e-6, x_current)
            rel_changes = np.abs(delta_x / x_safe)
            max_change = np.max(rel_changes)
            if max_change > self.max_step_frac:
                delta_x *= self.max_step_frac / max_change

            x_current += delta_x

        residuals = eval_func(x_current)
        final_norm = float(np.linalg.norm(residuals))
        success = final_norm < self.tolerance
        return x_current, residuals, success
