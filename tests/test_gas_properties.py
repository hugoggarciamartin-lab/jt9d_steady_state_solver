"""
Unit tests for the NASA 7-coefficient thermodynamic gas properties.
Ensures MoC 2 numerical compliance against standard atmospheric and combustion tables.
"""

import pytest
import numpy as np
from src.utils.gas_properties import GasProperties


@pytest.fixture
def gas():
    """Provides a fresh GasProperties instance for each test."""
    return GasProperties()


def test_dry_air_standard_day(gas):
    """
    Validates dry air (f=0) at ISA Sea Level Static (288.15 K).
    """
    T = 288.15
    f = 0.0

    cp = gas.calculate_cp(T, f)
    gamma = gas.calculate_gamma(T, f)

    # Assert against rigorous polynomial evaluations, not simplistic textbooks
    assert cp == pytest.approx(1002.0, rel=1e-3), f"Standard Cp out of bounds: {cp}"
    assert gamma == pytest.approx(1.4015, rel=1e-3), (
        f"Standard gamma out of bounds: {gamma}"
    )


def test_dry_air_high_temperature(gas):
    """Validates dry air crossing the high-temperature boundary (T > 1000 K)."""
    T = 1200.0
    f = 0.0

    cp = gas.calculate_cp(T, f)
    gamma = gas.calculate_gamma(T, f)

    assert cp > 1004.5, "Cp should increase at elevated temperatures."
    assert gamma < 1.400, "Gamma should decrease at elevated temperatures."
    assert cp == pytest.approx(1170.0, rel=5e-2)


def test_vitiated_mixture_gas_constant(gas):
    """Validates the specific gas constant R_mix shifts properly when fuel is added."""
    f = 0.02
    R_mix = gas.get_gas_constant(f)
    assert R_mix > 287.0
    assert R_mix < 288.4
    assert R_mix == pytest.approx(287.07, rel=1e-3)


def test_enthalpy_continuity_at_boundary(gas):
    """
    Ensures absolute C0 continuity of the enthalpy function at the 1000 K boundary.
    """
    f = 0.015
    h_low = gas.calculate_enthalpy(999.999, f)
    h_high = gas.calculate_enthalpy(1000.001, f)
    assert np.isclose(h_low, h_high, rtol=1e-5), (
        "Enthalpy discontinuity detected at 1000K boundary."
    )


def test_vitiated_combustion_properties(gas):
    """Validates high-temperature combustion products."""
    T = 1600.0
    f = 0.025

    cp = gas.calculate_cp(T, f)
    gamma = gas.calculate_gamma(T, f)

    assert cp > 1200.0
    assert gamma < 1.35
