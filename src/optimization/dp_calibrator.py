"""
Design Point calibration module for 0D steady-state engine simulation.
Solves the inverse problem matching scaled empirical maps to test cell telemetry.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple


class DPCalibrator:
    """
    Constrained optimizer designed to adjust map scaling factors.
    Minimizes the cycle matching cost function while enforcing rigid physical bounds
    on the allowable topology deformation.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds: Tuple[Tuple[float, float], ...],
    ):
        """
        Args:
            objective_func: Callback that takes map scalars, runs the NR solver,
                            and returns the Mahalanobis cost scalar.
            bounds: Tuple of (min, max) limits for each scaling factor to prevent
                    unphysical map distortion.
        """
        self.objective_func = objective_func
        self.bounds = bounds

    def calibrate(self, initial_scalars: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        """
        Executes the bounded quasi-Newton optimization (L-BFGS-B).

        Args:
            initial_scalars: Initial guess array for [s_W, s_eff, s_PR] across components.

        Returns:
            Tuple containing: (optimized_scalars, success_boolean, exit_message)
        """
        # Ensure initial guess complies with bounds before starting
        x0 = np.clip(
            initial_scalars, [b[0] for b in self.bounds], [b[1] for b in self.bounds]
        )

        result = minimize(
            fun=self.objective_func,
            x0=x0,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={
                "ftol": 1e-6,  # Strict function tolerance for DO-178C precision
                "gtol": 1e-5,  # Gradient norm tolerance
                "maxiter": 150,
                "maxfun": 500,
            },
        )

        if not result.success:
            # Optimizador falló: devolver los mejores escalares encontrados pero marcar como fallo
            return result.x, False, f"DP Calibration failed: {result.message}"

        return result.x, True, "DP Calibration converged successfully."
