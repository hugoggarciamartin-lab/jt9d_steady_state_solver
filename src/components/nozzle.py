"""
Nozzle component module for 0D steady-state engine simulation.
Computes analytical gas dynamics, choking conditions (NPR), and dynamic Cd
with strict bounds enforcement [0, 1].
"""

import numpy as np
from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class NozzleOutput:
    W_calc: float
    Fg: float
    V_exit: float
    M_exit: float
    Cd: float
    P_stat: float
    T_stat: float


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
        k_noz: float = 0.0,
    ) -> NozzleOutput:

        Pt_eff = Pt_in * (1.0 - k_noz)
        NPR = Pt_eff / P_amb

        # Absolute numerical fallback for sub-atmospheric pressure drops
        if NPR <= 1.001:
            return NozzleOutput(
                W_calc=1e-4,
                Fg=0.0,
                V_exit=0.0,
                M_exit=0.0,
                Cd=0.98,
                P_stat=P_amb,
                T_stat=Tt_in,
            )

        # Dynamic specific heat calculations based on actual mixture
        cp = self.gas.calculate_cp(Tt_in, f_in)
        R_gas = 287.05
        cv = cp - R_gas
        gamma = cp / cv

        PR_crit = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))

        # Analytical baseline for Cd dependent on NPR, strictly bounded between 0 and 1
        Cd_raw = 0.985 - 0.04 * np.exp(-2.0 * (NPR - 1.0))
        Cd_analytical = float(np.clip(Cd_raw, 0.01, 0.999))

        if NPR < PR_crit:
            # Subsonic flow (Unchoked)
            M_exit = np.sqrt(
                (2.0 / (gamma - 1.0)) * (NPR ** ((gamma - 1.0) / gamma) - 1.0)
            )
            T_stat = Tt_in / (1.0 + ((gamma - 1.0) / 2.0) * M_exit**2)
            P_stat = P_amb
        else:
            # Sonic flow (Choked)
            M_exit = 1.0
            T_stat = Tt_in * (2.0 / (gamma + 1.0))
            P_stat = Pt_eff / PR_crit

        V_exit = M_exit * np.sqrt(gamma * R_gas * T_stat)
        rho_exit = P_stat / (R_gas * T_stat)

        A_eff = A_throat * Cd_analytical
        W_calc = rho_exit * A_eff * V_exit

        # Gross thrust = Momentum thrust + Pressure thrust
        Fg = (W_calc * V_exit) + ((P_stat - P_amb) * A_eff)

        return NozzleOutput(
            W_calc=W_calc,
            Fg=Fg,
            V_exit=V_exit,
            M_exit=M_exit,
            Cd=Cd_analytical,
            P_stat=P_stat,
            T_stat=T_stat,
        )
