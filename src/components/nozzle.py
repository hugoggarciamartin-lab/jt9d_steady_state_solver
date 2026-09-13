"""
Nozzle component module for 0D steady-state engine simulation.
Models convergent nozzles (core A8 and bypass A18), handling sonic choking
and dynamic discharge coefficients.
"""

import math
from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class NozzleOutput:
    W_calc: float  # Mass flow calculated from nozzle throat physics [kg/s]
    Fg: float  # Gross thrust [N]
    V_exit: float  # Exit velocity [m/s]
    P_exit: float  # Static pressure at the exit plane [Pa]
    T_exit: float  # Static temperature at the exit plane [K]
    Mach: float  # Exit Mach number [-]


class Nozzle:
    """
    Aerothermodynamic Convergent Nozzle component.
    Evaluates critical pressure ratio, choking boundaries, and thrust generation.
    """

    def __init__(self, gas_props: GasProperties):
        self.gas = gas_props

    def _get_cd(self, npr: float) -> float:
        """
        Dynamic Discharge Coefficient Cd as a function of Nozzle Pressure Ratio (NPR).
        Uses a standard empirical curve for convergent aerospace nozzles.
        """
        if npr < 1.1:
            return 0.85
        elif npr > 2.5:
            return 0.99
        else:
            # Simple linear blend for the operational transition
            return 0.85 + (0.99 - 0.85) * ((npr - 1.1) / (2.5 - 1.1))

    def calculate(
        self,
        Pt_in: float,
        Tt_in: float,
        W_in: float,
        f_in: float,
        P_amb: float,
        A_throat: float,
    ) -> NozzleOutput:
        """
        Executes the forward aerothermodynamic pass for the nozzle.

        Args:
            Pt_in: Inlet total pressure [Pa]
            Tt_in: Inlet total temperature [K]
            W_in: Actual physical mass flow arriving at the nozzle [kg/s]
            f_in: Fuel-to-air ratio of the gas [-]
            P_amb: Ambient static pressure [Pa]
            A_throat: Geometric throat area [m^2]
        """
        gamma = self.gas.calculate_gamma(Tt_in, f_in)
        R_gas = self.gas.get_gas_constant(f_in)

        NPR = Pt_in / P_amb

        # Critical Nozzle Pressure Ratio for sonic flow (M=1)
        critical_pr = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))

        is_choked = NPR >= critical_pr

        if is_choked:
            Mach = 1.0
            P_exit = Pt_in / critical_pr
            T_exit = Tt_in / ((gamma + 1.0) / 2.0)
        else:
            # Unchoked: Exit static pressure equals ambient
            # Guard against unphysical expansion (NPR < 1)
            if NPR <= 1.0:
                return NozzleOutput(0.0, 0.0, 0.0, P_amb, Tt_in, 0.0)

            P_exit = P_amb
            Mach = math.sqrt(
                (2.0 / (gamma - 1.0)) * ((NPR ** ((gamma - 1.0) / gamma)) - 1.0)
            )
            T_exit = Tt_in / (1.0 + ((gamma - 1.0) / 2.0) * Mach**2)

        # Gas dynamics at the exit plane
        density_exit = P_exit / (R_gas * T_exit)
        V_exit = Mach * math.sqrt(gamma * R_gas * T_exit)

        # Dynamic discharge coefficient
        Cd = self._get_cd(NPR)
        A_effective = A_throat * Cd

        # Calculated mass flow based on local thermodynamics and area
        # The Newton-Raphson solver must drive (W_in - W_calc) to 0.
        W_calc = density_exit * V_exit * A_effective

        # Gross Thrust = Momentum Thrust + Pressure Thrust
        # We use W_in to respect the upstream mass continuity during solver transients
        momentum_thrust = W_in * V_exit
        pressure_thrust = A_effective * (P_exit - P_amb)

        Fg = momentum_thrust + pressure_thrust

        return NozzleOutput(
            W_calc=W_calc, Fg=Fg, V_exit=V_exit, P_exit=P_exit, T_exit=T_exit, Mach=Mach
        )
