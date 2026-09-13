"""
Unified Fan component module for 0D steady-state engine simulation.
Processes inlet conditions and map coordinates to output separated core and bypass streams,
accounting for radial pressure distortion.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from src.utils.map_parser import TurbomachineryMap
from src.utils.gas_properties import GasProperties


@dataclass
class FanOutput:
    W_fan: float  # Total physical mass flow [kg/s]
    W_core: float  # Core physical mass flow (Station 21) [kg/s]
    W_bypass: float  # Bypass physical mass flow (Station 13) [kg/s]
    Pt13: float  # Bypass total pressure [Pa]
    Tt13: float  # Bypass total temperature [K]
    Pt21: float  # Core total pressure [Pa]
    Tt21: float  # Core total temperature [K]
    power_req: float  # Mechanical power required [W]
    penalty: float  # Boundary violation penalty for the optimizer


class Fan:
    """
    Aerothermodynamic Fan component.
    Models a unified fan stage splitting flow based on Bypass Ratio, applying
    a radial scalar to differentiate root (core) and tip (bypass) pressure ratios.
    """

    T_REF = 288.15
    P_REF = 101325.0

    def __init__(self, fan_map: TurbomachineryMap, gas_props: GasProperties):
        self.fan_map = fan_map
        self.gas = gas_props

    def _solve_temperature_from_enthalpy(
        self, h_target: float, T_guess: float
    ) -> float:
        """Local Newton-Raphson solver to reverse-lookup temperature from sensible enthalpy."""
        T_iter = T_guess
        for _ in range(10):
            h_current = self.gas.calculate_enthalpy(T_iter, f=0.0)
            residual = h_current - h_target

            if abs(residual) < 1e-3:
                return T_iter

            cp_current = self.gas.calculate_cp(T_iter, f=0.0)
            T_iter -= residual / cp_current

        return T_iter

    def calculate(
        self,
        Pt2: float,
        Tt2: float,
        N1: float,
        N1_design: float,
        beta: float,
        BPR: float,
        sigma: float = 1.0,
    ) -> FanOutput:
        """
        Executes the forward aerothermodynamic pass.

        Args:
            Pt2: Inlet total pressure [Pa]
            Tt2: Inlet total temperature [K]
            N1: Physical LP shaft speed [rpm]
            N1_design: Reference LP shaft speed for scaling [rpm]
            beta: Map operating line auxiliary coordinate [-]
            BPR: Bypass Ratio (W_bypass / W_core) [-]
            sigma: Radial pressure scalar (attenuates pressure rise at the root) [-]
        """
        theta_t2 = Tt2 / self.T_REF
        delta_t2 = Pt2 / self.P_REF
        Nc_fan = (N1 / N1_design) / (theta_t2**0.5)

        Wc, eff, pr_map, penalty = self.fan_map.evaluate(Nc_fan, beta)

        W_fan = Wc * (delta_t2) / (theta_t2**0.5)
        W_core = W_fan / (1.0 + BPR)
        W_bypass = W_fan * (BPR / (1.0 + BPR))

        # Radial distortion: apply sigma only to the pressure rise to prevent unphysical vacuums
        pr_byp = pr_map
        pr_core = 1.0 + (pr_map - 1.0) * sigma

        gamma_in = self.gas.calculate_gamma(Tt2, f=0.0)
        h_in = self.gas.calculate_enthalpy(Tt2, f=0.0)

        # --- Bypass Stream (Tip) Thermodynamics ---
        Pt13 = Pt2 * pr_byp
        Tt13_is = Tt2 * (pr_byp ** ((gamma_in - 1.0) / gamma_in))
        h13_is = self.gas.calculate_enthalpy(Tt13_is, f=0.0)

        delta_h_byp_ideal = h13_is - h_in
        delta_h_byp_real = delta_h_byp_ideal / eff
        h13_real = h_in + delta_h_byp_real

        T_guess_13 = Tt2 + (Tt13_is - Tt2) / eff
        Tt13 = self._solve_temperature_from_enthalpy(h13_real, T_guess_13)

        # --- Core Stream (Root) Thermodynamics ---
        Pt21 = Pt2 * pr_core
        Tt21_is = Tt2 * (pr_core ** ((gamma_in - 1.0) / gamma_in))
        h21_is = self.gas.calculate_enthalpy(Tt21_is, f=0.0)

        delta_h_core_ideal = h21_is - h_in
        delta_h_core_real = delta_h_core_ideal / eff
        h21_real = h_in + delta_h_core_real

        T_guess_21 = Tt2 + (Tt21_is - Tt2) / eff
        Tt21 = self._solve_temperature_from_enthalpy(h21_real, T_guess_21)

        # Mechanical Power Requirement is mass-weighted by stream
        power_req = (W_bypass * delta_h_byp_real) + (W_core * delta_h_core_real)

        return FanOutput(
            W_fan=W_fan,
            W_core=W_core,
            W_bypass=W_bypass,
            Pt13=Pt13,
            Tt13=Tt13,
            Pt21=Pt21,
            Tt21=Tt21,
            power_req=power_req,
            penalty=penalty,
        )
