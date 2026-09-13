"""
Thermodynamic gas properties module using NASA 7-coefficient polynomials.
Evaluates semi-perfect gas properties for unvitiated air and vitiated combustion products.
"""

import numpy as np


class GasProperties:
    def __init__(self):
        # Universal and specific gas constants [J/(kg K)]
        self.R_univ = 8314.462618
        self.Mw_air = 28.96546
        self.R_air = self.R_univ / self.Mw_air

        # NASA 7-Coefficient Polynomials for Dry Air (f=0)
        self.air_low_T = np.array(
            [
                3.56839620e00,
                -6.78729429e-04,
                1.55371476e-06,
                -3.29937060e-10,
                -4.66395387e-13,
                -1.06234659e03,
                4.36391065e00,
            ]
        )
        self.air_high_T = np.array(
            [
                3.08792717e00,
                1.24597184e-03,
                -4.23718945e-07,
                6.74774789e-11,
                -3.97076972e-15,
                -9.95262755e02,
                3.28816480e00,
            ]
        )

        # NASA 7-Coefficient Polynomials for stoichiometric Jet-A products (approximate)
        self.prod_low_T = np.array(
            [
                3.84439000e00,
                1.14486000e-03,
                1.48833000e-06,
                -1.68064000e-09,
                5.09348000e-13,
                -1.20000000e04,
                4.20000000e00,
            ]
        )
        self.prod_high_T = np.array(
            [
                3.40000000e00,
                1.50000000e-03,
                -4.50000000e-07,
                6.50000000e-11,
                -3.50000000e-15,
                -1.20000000e04,
                3.50000000e00,
            ]
        )

        # Enforce exact C0 continuity at the 1000 K boundary by shifting the a6 integration constant
        h_rt_air_low = self._evaluate_h_RT(1000.0, self.air_low_T)
        h_rt_air_high = self._evaluate_h_RT(1000.0, self.air_high_T)
        self.air_high_T[5] += (h_rt_air_low - h_rt_air_high) * 1000.0

        h_rt_prod_low = self._evaluate_h_RT(1000.0, self.prod_low_T)
        h_rt_prod_high = self._evaluate_h_RT(1000.0, self.prod_high_T)
        self.prod_high_T[5] += (h_rt_prod_low - h_rt_prod_high) * 1000.0

    def get_gas_constant(self, f: float) -> float:
        """Calculates the specific gas constant R_mix [J/(kg K)] for the vitiated mixture."""
        R_prod = 288.3
        return (self.R_air + f * R_prod) / (1.0 + f)

    def _evaluate_cp_R(self, T: float, coeffs: np.ndarray) -> float:
        """Evaluates Cp/R using NASA polynomials."""
        a1, a2, a3, a4, a5, _, _ = coeffs
        return a1 + a2 * T + a3 * T**2 + a4 * T**3 + a5 * T**4

    def _evaluate_h_RT(self, T: float, coeffs: np.ndarray) -> float:
        """Evaluates h/(RT) using NASA polynomials."""
        a1, a2, a3, a4, a5, a6, _ = coeffs
        return (
            a1
            + (a2 / 2) * T
            + (a3 / 3) * T**2
            + (a4 / 4) * T**3
            + (a5 / 5) * T**4
            + a6 / T
        )

    def calculate_cp(self, T: float, f: float = 0.0) -> float:
        """Calculates the specific heat capacity Cp [J/(kg K)] of the gas mixture."""
        if T <= 1000.0:
            cp_R_air = self._evaluate_cp_R(T, self.air_low_T)
            cp_R_prod = self._evaluate_cp_R(T, self.prod_low_T)
        else:
            cp_R_air = self._evaluate_cp_R(T, self.air_high_T)
            cp_R_prod = self._evaluate_cp_R(T, self.prod_high_T)

        cp_air = cp_R_air * self.R_air

        if f == 0.0:
            return cp_air

        cp_prod = cp_R_prod * 288.3
        return (cp_air + f * cp_prod) / (1.0 + f)

    def calculate_enthalpy(self, T: float, f: float = 0.0) -> float:
        """Calculates the specific sensible enthalpy h [J/kg] of the gas mixture."""
        if T <= 1000.0:
            h_RT_air = self._evaluate_h_RT(T, self.air_low_T)
            h_RT_prod = self._evaluate_h_RT(T, self.prod_low_T)
        else:
            h_RT_air = self._evaluate_h_RT(T, self.air_high_T)
            h_RT_prod = self._evaluate_h_RT(T, self.prod_high_T)

        h_air = h_RT_air * self.R_air * T

        if f == 0.0:
            return h_air

        h_prod = h_RT_prod * 288.3 * T
        return (h_air + f * h_prod) / (1.0 + f)

    def calculate_gamma(self, T: float, f: float = 0.0) -> float:
        """Calculates the ratio of specific heats (gamma) for the gas mixture."""
        cp = self.calculate_cp(T, f)
        R_mix = self.get_gas_constant(f)
        return cp / (cp - R_mix)
