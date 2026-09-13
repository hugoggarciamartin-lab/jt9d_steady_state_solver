"""
Nozzle component module for 0D steady-state engine simulation.
Models convergent nozzles (core A8 and bypass A18), handling sonic choking
and dynamic discharge coefficients based on boundary layer scaling.
"""

import math
from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class NozzleOutput:
    W_calc: float
    Fg: float
    V_exit: float
    P_exit: float
    T_exit: float
    Mach: float


class Nozzle:
    def __init__(self, gas_props: GasProperties):
        self.gas = gas_props

    def calculate(
        self,
        Pt_in: float,
        Tt_in: float,
        W_in: float,
        f_in: float,
        P_amb: float,
        A_throat: float,
        Cd_DP: float = 0.98,
        NPR_DP: float = 2.0,
        k_noz: float = 0.045,
    ) -> NozzleOutput:

        gamma = self.gas.calculate_gamma(Tt_in, f_in)
        R_gas = self.gas.get_gas_constant(f_in)

        NPR = Pt_in / P_amb
        critical_pr = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))
        is_choked = NPR >= critical_pr

        if is_choked:
            Mach = 1.0
            P_exit = Pt_in / critical_pr
            T_exit = Tt_in / ((gamma + 1.0) / 2.0)
        else:
            if NPR <= 1.0:
                return NozzleOutput(0.0, 0.0, 0.0, P_amb, Tt_in, 0.0)

            P_exit = P_amb
            Mach = math.sqrt(
                (2.0 / (gamma - 1.0)) * ((NPR ** ((gamma - 1.0) / gamma)) - 1.0)
            )
            T_exit = Tt_in / (1.0 + ((gamma - 1.0) / 2.0) * Mach**2)

        density_exit = P_exit / (R_gas * T_exit)
        V_exit = Mach * math.sqrt(gamma * R_gas * T_exit)

        # Dynamic Boundary Layer Scaling Law for Discharge Coefficient
        Cd = Cd_DP * (1.0 - k_noz * ((1.0 / NPR) - (1.0 / NPR_DP)))
        A_effective = A_throat * Cd

        W_calc = density_exit * V_exit * A_effective
        momentum_thrust = W_in * V_exit
        pressure_thrust = A_effective * (P_exit - P_amb)
        Fg = momentum_thrust + pressure_thrust

        return NozzleOutput(
            W_calc=W_calc, Fg=Fg, V_exit=V_exit, P_exit=P_exit, T_exit=T_exit, Mach=Mach
        )
