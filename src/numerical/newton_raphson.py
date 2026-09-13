"""
Multidimensional Newton-Raphson solver for 0D engine cycle matching.
Computes finite-difference Jacobians and iteratively reduces system residuals.
"""

import numpy as np
from typing import Callable, Tuple, List
from src.numerical.step_limiters import calculate_max_step_fraction, armijo_line_search


class NewtonRaphsonSolver:
    """
    Robust implicit root-finding algorithm with Armijo backtracking line search
    and relative finite-difference Jacobian construction.
    """

    def __init__(
        self,
        max_iters: int = 50,
        tolerance: float = 1e-5,
        fd_eps: float = 1e-4,
        max_step_frac: float = 0.15,
    ):
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.fd_eps = fd_eps  # Finite difference perturbation factor
        self.max_step_frac = max_step_frac

        # Diagnostics history
        self.residual_history: List[float] = []

    def _compute_jacobian(
        self,
        eval_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        f0: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the forward finite-difference Jacobian matrix.
        Uses relative perturbation to prevent round-off errors across disparate scales.
        """
        n_vars = len(x0)
        J = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            x_pert = x0.copy()
            # Relative perturbation, safeguarded near zero
            dx = self.fd_eps * max(abs(x0[i]), 1.0)
            x_pert[i] += dx

            f_pert = eval_func(x_pert)

            # Forward difference formulation
            J[:, i] = (f_pert - f0) / dx

        return J

    def solve(
        self, eval_func: Callable[[np.ndarray], np.ndarray], x_guess: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Executes the Newton-Raphson loop to find the root of eval_func(x) = 0.

        Args:
            eval_func: Function accepting state vector x and returning residual vector.
            x_guess: Initial state vector estimate.

        Returns:
            Tuple containing: (converged_state_vector, final_residuals, success_boolean)
        """
        x_current = np.array(x_guess, dtype=np.float64)
        self.residual_history = []

        for iteration in range(self.max_iters):
            residuals = eval_func(x_current)
            res_norm = np.linalg.norm(residuals)
            self.residual_history.append(res_norm)

            if res_norm < self.tolerance:
                return x_current, residuals, True

            # Jacobian computation
            J = self._compute_jacobian(eval_func, x_current, residuals)

            # Linear system resolution: J * delta_x = -residuals
            try:
                delta_x = np.linalg.solve(J, -residuals)
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    f"Jacobian matrix became singular at iteration {iteration}. "
                    "Cycle matching failed. Verify initial guess or physical boundaries."
                )

            # Step fraction limiting (preventing unphysical leaps)
            step_frac = calculate_max_step_fraction(
                x_current, delta_x, self.max_step_frac
            )

            # Objective function for line search (returning scalar norm)
            def obj_func(x: np.ndarray) -> float:
                return float(np.linalg.norm(eval_func(x)))

            # Armijo Backtracking Line Search to guarantee descent
            x_current, new_norm = armijo_line_search(
                objective_func=obj_func,
                x_current=x_current,
                delta_x=delta_x,
                current_residual_norm=res_norm,
                max_step_fraction=step_frac,
            )

        # Max iterations reached without satisfying tolerance
        final_res = eval_func(x_current)
        return x_current, final_res, False
