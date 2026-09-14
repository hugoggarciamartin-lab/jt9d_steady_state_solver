"""
Compressor component module.
Enforces physical mass flow transformations and strict bleed extraction thermodynamics.
"""

import numpy as np
from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class CompressorOutput:
    W_in: float
    W_out: float
    W_bleed: float
    Pt_out: float
    Tt_out: float
    power_req: float
    SM: float
    penalty: float


class Compressor:
    def __init__(self, comp_map, gas_props: GasProperties):
        self.map = comp_map
        self.gas = gas_props

    def calculate(
        self,
        Pt_in: float,
        Tt_in: float,
        N_mech: float,
        N_des: float,
        beta: float,
        bleed_fraction: float = 0.0,
    ) -> CompressorOutput:

        theta = Tt_in / 288.15
        delta = Pt_in / 101325.0

        Nc_actual = N_mech / np.sqrt(theta)
        Nc_design = N_des / np.sqrt(1.0)
        speed_param = Nc_actual / Nc_design

        penalty = 0.0

        try:
            Wc_map, eff_map, PR_map, SM = self.map.evaluate(speed_param, beta)
        except ValueError:
            Wc_map, eff_map, PR_map, SM = 50.0, 0.50, 1.0, 0.0
            penalty += 1e6

        if PR_map < 1.0:
            PR_map = 1.001
            penalty += 1e5
        if eff_map < 0.1:
            eff_map = 0.1
            penalty += 1e5

        W_phys_in = Wc_map * delta / np.sqrt(theta)

        W_bleed = W_phys_in * bleed_fraction
        W_phys_out = W_phys_in - W_bleed

        Pt_out = Pt_in * PR_map

        h_in = self.gas.calculate_enthalpy(Tt_in, 0.0)

        cp = self.gas.calculate_cp(Tt_in, 0.0)
        gamma = cp / (cp - 287.05)
        Tt_out_ideal = Tt_in * (PR_map ** ((gamma - 1.0) / gamma))
        h_out_ideal = self.gas.calculate_enthalpy(Tt_out_ideal, 0.0)

        dh_actual = (h_out_ideal - h_in) / eff_map
        h_out_actual = h_in + dh_actual

        cp_out = self.gas.calculate_cp(Tt_out_ideal, 0.0)
        Tt_out_actual = Tt_in + (dh_actual / cp_out)

        power_req = W_phys_in * dh_actual

        return CompressorOutput(
            W_in=W_phys_in,
            W_out=W_phys_out,
            W_bleed=W_bleed,
            Pt_out=Pt_out,
            Tt_out=Tt_out_actual,
            power_req=power_req,
            SM=SM,
            penalty=penalty,
        )
