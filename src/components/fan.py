"""
Fan component module.
Enforces strict aerodynamic similarity transformations from corrected T-MATS map
parameters to physical station thermodynamics.
"""

import numpy as np
from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class FanOutput:
    W_in: float
    W_fan: float
    Pt13: float
    Tt13: float
    Pt21: float
    Tt21: float
    power_req: float
    SM: float
    penalty: float


class Fan:
    def __init__(self, fan_map, gas_props: GasProperties):
        self.map = fan_map
        self.gas = gas_props

    def calculate(
        self,
        Pt_in: float,
        Tt_in: float,
        N_mech: float,
        N_des: float,
        beta: float,
        BPR: float,
        sigma: float = 1.0,
    ) -> FanOutput:

        # Similarity Parameters (Theta and Delta referenced to ISA SLS)
        theta = Tt_in / 288.15
        delta = Pt_in / 101325.0

        # Corrected speed fraction
        Nc_actual = N_mech / np.sqrt(theta)
        Nc_design = N_des / np.sqrt(1.0)
        speed_param = Nc_actual / Nc_design

        penalty = 0.0

        # Map Evaluation
        try:
            Wc_map, eff_map, PR_map, SM = self.map.evaluate(speed_param, beta)
        except ValueError:
            Wc_map, eff_map, PR_map, SM = 100.0, 0.50, 1.0, 0.0
            penalty += 1e6

        if PR_map < 1.0:
            PR_map = 1.001
            penalty += 1e5
        if eff_map < 0.1:
            eff_map = 0.1
            penalty += 1e5

        # Transform to Physical Mass Flow
        # W_phys = W_corr * (delta / sqrt(theta))
        W_phys = Wc_map * delta / np.sqrt(theta)

        # Thermodynamic State Updates
        Pt_out_byp = Pt_in * PR_map
        Pt_out_core = (
            Pt_in * PR_map * sigma
        )  # Applies root-tip pressure profile distortion

        h_in = self.gas.calculate_enthalpy(Tt_in, 0.0)

        # Isentropic compression
        cp = self.gas.calculate_cp(Tt_in, 0.0)
        gamma = cp / (cp - 287.05)
        Tt_out_ideal = Tt_in * (PR_map ** ((gamma - 1.0) / gamma))
        h_out_ideal = self.gas.calculate_enthalpy(Tt_out_ideal, 0.0)

        # Actual enthalpy and temperature out
        dh_actual = (h_out_ideal - h_in) / eff_map
        h_out_actual = h_in + dh_actual

        # Using specific heat approximation for inverse temperature if inverse enthalpy method is missing
        cp_out = self.gas.calculate_cp(Tt_out_ideal, 0.0)
        Tt_out_actual = Tt_in + (dh_actual / cp_out)

        # Aerodynamic Power Extraction
        power_req = W_phys * dh_actual

        return FanOutput(
            W_in=W_phys,
            W_fan=W_phys,
            Pt13=Pt_out_byp,
            Tt13=Tt_out_actual,
            Pt21=Pt_out_core,
            Tt21=Tt_out_actual,
            power_req=power_req,
            SM=SM,
            penalty=penalty,
        )
