"""
Turbine component module.
Computes map-based flow capacity, generates mechanical power, and enforces
enthalpy-weighted cooling bleed mixing at the exit station.
"""

import numpy as np
from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class TurbineOutput:
    W_in: float
    W_out: float
    W_capacity: float
    f_out: float
    Pt_out: float
    Tt_out: float
    power_gen: float
    penalty: float


class Turbine:
    def __init__(self, turb_map, gas_props: GasProperties):
        self.map = turb_map
        self.gas = gas_props

    def calculate(
        self,
        Pt_in: float,
        Tt_in: float,
        W_in_actual: float,
        f_in: float,
        N_mech: float,
        N_des: float,
        PR_req: float,
        W_cool: float = 0.0,
        Tt_cool: float = 288.15,
    ) -> TurbineOutput:

        theta = Tt_in / 288.15
        delta = Pt_in / 101325.0

        Nc_actual = N_mech / np.sqrt(theta)
        Nc_design = N_des / np.sqrt(1.0)
        speed_param = Nc_actual / Nc_design

        penalty = 0.0

        try:
            # Map returns corrected flow capacity and efficiency given speed and PR
            Wc_map, eff_map, _ = self.map.evaluate(speed_param, PR_req)
        except ValueError:
            Wc_map, eff_map = 20.0, 0.85
            penalty += 1e6

        if PR_req < 1.001:
            PR_req = 1.001
            penalty += 1e5
        if eff_map < 0.1:
            eff_map = 0.1
            penalty += 1e5

        # De-correct to find aerodynamic flow capacity
        W_capacity = Wc_map * delta / np.sqrt(theta)

        Pt_out = Pt_in / PR_req

        h_in = self.gas.calculate_enthalpy(Tt_in, f_in)

        cp_in = self.gas.calculate_cp(Tt_in, f_in)
        gamma = cp_in / (cp_in - 287.05)
        Tt_out_ideal = Tt_in * ((1.0 / PR_req) ** ((gamma - 1.0) / gamma))
        h_out_ideal = self.gas.calculate_enthalpy(Tt_out_ideal, f_in)

        dh_actual = (h_in - h_out_ideal) * eff_map
        h_out_main = h_in - dh_actual

        # Power generation strictly from the actual mainstream flow
        power_gen = W_in_actual * dh_actual

        # Mix high-pressure cooling bleed at turbine exit
        h_cool = self.gas.calculate_enthalpy(Tt_cool, 0.0)
        W_out = W_in_actual + W_cool
        f_out = (W_in_actual * f_in) / max(W_out, 1e-6)

        h_out_mix = ((W_in_actual * h_out_main) + (W_cool * h_cool)) / max(W_out, 1e-6)

        cp_out = self.gas.calculate_cp(Tt_out_ideal, f_out)
        Tt_out_actual = Tt_in - ((h_in - h_out_mix) / cp_out)

        return TurbineOutput(
            W_in=W_in_actual,
            W_out=W_out,
            W_capacity=W_capacity,
            f_out=f_out,
            Pt_out=Pt_out,
            Tt_out=Tt_out_actual,
            power_gen=power_gen,
            penalty=penalty,
        )
