"""
Cost function module for Design Point calibration.
Computes the weighted Mahalanobis distance between model outputs and test cell telemetry.
"""

import numpy as np
from typing import Dict


class CycleCostFunction:
    """
    Evaluates the objective function J(u) for the inverse calibration problem.
    Applies inverse-variance weighting to normalize disparate physical scales (e.g., Thrust vs TSFC).
    """

    def __init__(self, target_data: Dict[str, float], weights: Dict[str, float]):
        """
        Args:
            target_data: The ground-truth certification telemetry (e.g., {'Fn': 205000, 'EGT': 850}).
            weights: The relative importance of each metric (e.g., {'Fn': 1.0, 'EGT': 5.0}).
        """
        self.target_data = target_data
        self.weights = weights

        # Validate that all targets have corresponding weights
        for key in self.target_data:
            if key not in self.weights:
                raise KeyError(
                    f"Falta el peso de ponderación (weight) para la métrica objetivo '{key}'. "
                    "El optimizador requiere parámetros estrictamente definidos."
                )

    def evaluate(self, model_output: Dict[str, float]) -> float:
        """
        Computes the weighted sum of squared relative errors.

        Args:
            model_output: The simulated cycle parameters from the current NR solver iteration.

        Returns:
            J: The scalar objective function value.
        """
        j_cost = 0.0

        for key, target_val in self.target_data.items():
            if key not in model_output:
                raise ValueError(
                    f"La métrica '{key}' no fue generada por la salida del modelo 0D. "
                    "Verifica el mapeo de variables de la planta propulsiva."
                )

            model_val = model_output[key]

            # Defensive guard against division by zero in target data
            safe_target = max(abs(target_val), 1e-6)

            # Relative error squared and weighted
            rel_error = (model_val - target_val) / safe_target
            j_cost += self.weights[key] * (rel_error**2)

        return float(j_cost)
