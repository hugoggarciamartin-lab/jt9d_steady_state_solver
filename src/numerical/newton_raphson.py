"""
Newton-Raphson numerical solver.
Implements Moore-Penrose pseudo-inverse for robust Gauss-Newton steps
across rank-deficient (choked) aerodynamic boundaries.
"""

import numpy as np
from typing import Callable, Tuple


class NewtonRaphsonSolver:
    def __init__(
        self,
        max_iters: int = 50,
        tolerance: float = 1e-4,
        fd_eps: float = 1e-2,
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

            J = np.zeros((n, n))
            for j in range(n):
                x_perturbed = x_current.copy()
                # Minimum absolute perturbation to prevent floating-point stall
                eps = max(abs(x_perturbed[j]) * self.fd_eps, 1e-4)
                x_perturbed[j] += eps
                res_perturbed = eval_func(x_perturbed)
                J[:, j] = (res_perturbed - residuals) / eps

            try:
                # Moore-Penrose pseudo-inverse handles singular/rank-deficient matrices intrinsically
                delta_x, _, rank, _ = np.linalg.lstsq(J, -residuals, rcond=1e-10)
                if rank < n:
                    print(
                        f"  -> [Rank Deficient] J-Rank: {rank}/{n}. Applying Least-Squares projection."
                    )
            except Exception as e:
                print(f"[NR Error] Algebra engine collapse: {e}")
                return x_current, residuals, False

            # Strict bounds to prevent catastrophic numerical divergence
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
