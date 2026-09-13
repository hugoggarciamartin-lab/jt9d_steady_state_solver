"""
Automated validation suite for Design Point thermodynamic convergence.
Ensures compliance by verifying residual tolerances and data integrity
before off-design map scaling and operating line execution.
"""

import pytest
import numpy as np
from pathlib import Path

# Path resolution for the artifact
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "output" / "logs" / "solver_history.npz"


@pytest.fixture
def dp_artifact():
    """Loads the DP solver history artifact, ensuring it exists."""
    assert ARTIFACT_PATH.exists(), (
        f"DP artifact missing at {ARTIFACT_PATH}. Run DP calibration first."
    )
    return np.load(ARTIFACT_PATH, allow_pickle=True)


def test_dp_solver_success_flag(dp_artifact):
    """Validates that the Newton-Raphson solver explicitly reported convergence."""
    success = dp_artifact["success"]
    assert success, "DP Calibration failed to converge. Off-design execution is unsafe."


def test_thermodynamic_residuals_tolerance(dp_artifact):
    """
    Mathematically verifies that all physical residuals (Fn, EGT) are strictly
    below the 1e-5 DO-178C threshold.
    """
    residuals = dp_artifact["residuals"]
    max_residual = np.max(np.abs(residuals))
    tolerance = 1e-5

    assert max_residual <= tolerance, (
        f"Thermodynamic discontinuity detected. Max residual {max_residual:.2e} "
        f"exceeds strict aerospace tolerance of {tolerance:.2e}."
    )


def test_optimization_vector_integrity(dp_artifact):
    """
    Ensures the 'u' vector was correctly populated without NaN or Inf values.
    """
    assert "u_vector" in dp_artifact, (
        "Optimization vector 'u' is missing from the artifact."
    )

    u_vector = dp_artifact["u_vector"].item()
    for key, value in u_vector.items():
        assert not np.isnan(value), f"Parameter {key} in u_vector is NaN."
        assert not np.isinf(value), f"Parameter {key} in u_vector is Infinite."


def test_map_scaling_factors_validity(dp_artifact):
    """
    Verifies that map scaling factors (SF) are physically realistic.
    Pressure ratios (s_PR) are given a wider tolerance [0.2, 3.0] to accommodate
    cross-generational map scaling (e.g., using modern maps for legacy engines).
    Mass flow and efficiency remain strictly bounded [0.5, 2.0].
    """
    assert "scaling_factors" in dp_artifact, (
        "Scaling factors are missing from the artifact."
    )

    scaling_factors = dp_artifact["scaling_factors"].item()
    for key, value in scaling_factors.items():
        if "s_PR" in key:
            assert 0.2 <= value <= 3.0, (
                f"PR Scaling factor {key} = {value:.3f} is outside cross-generational bounds [0.2, 3.0]. "
                "The base map topology is fundamentally incompatible with the target cycle."
            )
        else:
            assert 0.5 <= value <= 2.0, (
                f"Scaling factor {key} = {value:.3f} is outside physical bounds [0.5, 2.0]. "
                "The baseline DP thermodynamics deviate too far from the raw maps."
            )


def test_dp_magnitudes_extraction(dp_artifact):
    """Verifies that the core thermodynamic physical states were saved."""
    assert "dp_magnitudes" in dp_artifact, "DP magnitudes dictionary is missing."

    dp_mags = dp_artifact["dp_magnitudes"].item()
    required_keys = [
        "PR_fan",
        "PR_lpc",
        "PR_hpc",
        "PR_hpt",
        "PR_lpt",
        "W2_physical",
        "Tt4_physical",
    ]

    for key in required_keys:
        assert key in dp_mags, f"Critical state {key} is missing from DP magnitudes."
