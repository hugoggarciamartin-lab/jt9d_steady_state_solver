"""
Custom exception hierarchy for the 0D Steady-State Aerothermodynamic Solver.
Designed to enforce MoC Safety envelopes.
"""


class EngineSimulationError(Exception):
    """
    Root base for all engine simulation exceptions.
    Ensures that any solver-specific failure can be caught uniformly
    """

    pass


# Numerical and Solver Exceptions


class NumericalError(EngineSimulationError):
    """
    Base class for algorithmic and mathematical solver failures
    """

    pass


class SingularJacobianError(NumericalError):
    """
    Raised when the finite-difference Jacobian determinant approaches
    to zero or condition number is too high
    """

    pass


class ConvergenceTimeOutError(NumericalError):
    """
    Used when Newton-Raphson solver hits the maximun iteration limit
    without reaching resiual tolerance
    """

    pass


# Physical & Aerothermal Boundary Exceptions


class PhysicalBoundaryError(EngineSimulationError):
    """
    Base class for any state vector outside thermodynamic or geometric boundaries.

    """

    pass


class MapOutOfBounds(PhysicalBoundaryError):
    """
    Raised when interpolating turbomachinery maps beyond allowable extrapolation limits at sub-idle reigimes.
    """

    pass


class CombustorFlameout(PhysicalBoundaryError):
    """
    Raised when fuel-to-air ratio breaches LDO or RBO Limits.
    """

    pass


class SurgeMarginViolationError(PhysicalBoundaryError):
    """
    Raised when compression stage operates below its minimum surge margin threshold.
    """

    pass


# Thermo-mechanical and Structural Limits


class ThermoMechanicalLimitError(PhysicalBoundaryError):
    """
    Base class for methallurgical and structural envelope breaches.
    """

    pass


class TurbineOverheatError(ThermoMechanicalLimitError):
    """
    Raised when critical gas temperatures (Tt4 or EGT) exceed metallurgical limits.
    """

    pass


class CasingOverpressureError(ThermoMechanicalLimitError):
    """
    Raised when internal pressures (Pt3) exceed casing limits.
    """

    pass


class RotorOverspeedError(ThermoMechanicalLimitError):
    """
    Raised when LP or HP physical shaft speeds exceed certified mechanical redlines.
    """

    pass
