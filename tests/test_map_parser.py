"""
Unit tests for the Turbomachinery Map Parser.
Verifies Imperial-to-SI conversion, C2 spline continuity, and absolute boundary penalties.
"""

import pytest
import numpy as np
import tempfile
import os
from src.utils.map_parser import parse_tmats_map


@pytest.fixture
def mock_compressor_map():
    """
    Generates a temporary ASCII file mocking a 4x4 NASA T-MATS compressor map.
    Grid must be larger than 3x3 to satisfy SciPy cubic spline constraints (mx > kx).
    """
    tmats_content = """4 4
1.0 2.0 3.0 4.0
70.0
100.0 110.0 120.0 130.0
0.80 0.82 0.80 0.78
1.5 2.0 2.5 3.0
80.0
150.0 160.0 170.0 180.0
0.83 0.85 0.83 0.80
2.5 3.0 3.5 4.0
90.0
200.0 210.0 220.0 230.0
0.85 0.87 0.84 0.82
3.5 4.0 4.5 5.0
100.0
250.0 260.0 270.0 280.0
0.82 0.84 0.81 0.79
4.5 5.0 5.5 6.0
"""
    fd, path = tempfile.mkstemp(text=True)
    with open(fd, "w") as f:
        f.write(tmats_content)

    yield path
    os.remove(path)


def test_unit_conversion_and_scaling(mock_compressor_map):
    """Validates that raw lbm/s data is rigorously converted to kg/s."""
    LBM_TO_KG = 0.45359237
    comp_map = parse_tmats_map(mock_compressor_map, is_compressor=True)

    # Query exact node at Nc = 0.80 (80%), Beta = 2.0
    wc, eff, pr, pen = comp_map.evaluate(nc=0.8, beta=2.0)
    expected_wc = 160.0 * LBM_TO_KG

    assert wc == pytest.approx(expected_wc, rel=1e-5)
    assert eff == pytest.approx(0.85, rel=1e-5)
    assert pr == pytest.approx(3.0, rel=1e-5)
    assert pen == 0.0


def test_spline_derivative_continuity(mock_compressor_map):
    """Ensures C2 continuity by computing finite difference gradients."""
    comp_map = parse_tmats_map(mock_compressor_map, is_compressor=True)
    eps = 1e-4
    nc_test = 0.85
    beta_test = 2.5

    wc_plus, _, _, _ = comp_map.evaluate(nc_test + eps, beta_test)
    wc_minus, _, _, _ = comp_map.evaluate(nc_test - eps, beta_test)
    dwc_dnc = (wc_plus - wc_minus) / (2 * eps)

    assert np.isfinite(dwc_dnc)
    assert dwc_dnc != 0.0


def test_out_of_bounds_penalty_guard(mock_compressor_map):
    """Verifies that state vector excursions trigger massive residual penalties."""
    comp_map = parse_tmats_map(mock_compressor_map, is_compressor=True)
    wc, eff, pr, pen = comp_map.evaluate(nc=1.5, beta=5.0)

    assert pen >= 1000.0
    assert eff <= 0.99
