"""
Compressor component module for 0D steady-state engine simulation.
Evaluates LPC and HPC aerodynamics, enthalpy jumps, and bleed mass extraction.
"""

from typing import Tuple
from dataclasses import dataclass
from src.utils.map_parser import TurbomachineryMap
from src.utils.gas_properties import GasProperties


@dataclass
class CompressorOutput:
    W_in: float  # Physical mass flow demanded at inlet [kg/s]
    W_out: float  # Physical mass flow delivered to next station [kg/s]
    W_bleed: float  # Extracted bleed mass flow [kg/s]
    Pt_out: float  # Discharge total pressure [Pa]
    Tt_out: float  # Discharge total temperature [K]
    power_req: float  # Mechanical power required [W]
    penalty: float  # Boundary violation penalty for the optimizer


class Compressor:
    """
    Aerothermodynamic Compressor component (LPC/HPC).
    Models compression using NASA polynomials and handles fractional bleed extraction.
    """

    T_REF = 288.15
    P_REF = 101325.0

    def __init__(self, comp_map: TurbomachineryMap, gas_props: GasProperties):
        self.comp_map = comp_map
        self.gas = gas_props

    def _solve_temperature_from_enthalpy(
        self, h_target: float, T_guess: float
    ) -> float:
        """Local Newton-Raphson to reverse-lookup temperature from sensible enthalpy."""
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
        Pt_in: float,
        Tt_in: float,
        N_mech: float,
        N_design: float,
        beta: float,
        bleed_fraction: float = 0.0,
    ) -> CompressorOutput:
        """
        Executes the forward aerothermodynamic pass.

        Args:
            Pt_in: Inlet total pressure [Pa]
            Tt_in: Inlet total temperature [K]
            N_mech: Physical shaft speed [rpm]
            N_design: Reference shaft speed [rpm]
            beta: Map operating line auxiliary coordinate [-]
            bleed_fraction: Fraction of inlet mass flow extracted at discharge [-]
        """
        # Corrected Speed Calculation
        theta = Tt_in / self.T_REF
        delta = Pt_in / self.P_REF
        Nc = (N_mech / N_design) / (theta**0.5)

        # Map Interpolation
        Wc, eff, pr, penalty = self.comp_map.evaluate(Nc, beta)

        # Physical Flow Deprojection
        W_in = Wc * delta / (theta**0.5)

        # Isentropic Compression
        Pt_out = Pt_in * pr
        gamma_in = self.gas.calculate_gamma(Tt_in, f=0.0)

        Tt_out_is = Tt_in * (pr ** ((gamma_in - 1.0) / gamma_in))

        # Enthalpy Balance
        h_in = self.gas.calculate_enthalpy(Tt_in, f=0.0)
        h_out_is = self.gas.calculate_enthalpy(Tt_out_is, f=0.0)

        delta_h_ideal = h_out_is - h_in
        delta_h_real = delta_h_ideal / eff
        h_out_real = h_in + delta_h_real

        # Real exit temperature inversion
        T_guess = Tt_in + (Tt_out_is - Tt_in) / eff
        Tt_out = self._solve_temperature_from_enthalpy(h_out_real, T_guess)

        # Mass Bleed Extraction
        W_bleed = W_in * bleed_fraction
        W_out = W_in - W_bleed

        # Mechanical Power Requirement (calculated on full inlet mass)
        power_req = W_in * delta_h_real

        return CompressorOutput(
            W_in=W_in,
            W_out=W_out,
            W_bleed=W_bleed,
            Pt_out=Pt_out,
            Tt_out=Tt_out,
            power_req=power_req,
            penalty=penalty,
        )
