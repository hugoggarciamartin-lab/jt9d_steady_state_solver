"""
Combustor component module for 0D steady-state engine simulation.
Evaluates exact enthalpy balance for fuel addition and Rayleigh pressure losses.
"""

import numpy as np
from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class CombustorOutput:
    W_out: float
    f: float
    Pt_out: float
    Tt_out: float
    penalty: float


class Combustor:
    def __init__(self, gas_props: GasProperties, k1: float, k2: float):
        self.gas = gas_props
        self.k1 = k1  # Cold loss coefficient (friction)
        self.k2 = k2  # Hot loss coefficient (heat addition)
        self.f_LDO = 0.005  # Lean Die-Out default (overridden by JSON)
        self.f_RBO = 0.060  # Rich Blow-Out default (overridden by JSON)
        self.eta_comb = 0.995  # Combustor efficiency
        self.LHV = 43.1e6  # Lower Heating Value [J/kg]

    def calculate(
        self, Pt_in: float, Tt_in: float, W_in: float, W_f: float
    ) -> CombustorOutput:
        penalty = 0.0

        # 1. Mass continuity and Fuel-to-Air Ratio
        W_in = max(W_in, 1e-4)
        W_out = W_in + W_f
        f = W_f / W_in

        # Flammability limit penalties (soft continuous barrier for DO-178C Jacobian stability)
        if f < self.f_LDO:
            penalty += 1e5 * (self.f_LDO - f) ** 2
        elif f > self.f_RBO:
            penalty += 1e5 * (f - self.f_RBO) ** 2

        # 2. Strict Enthalpy Balance
        h_in = self.gas.calculate_enthalpy(Tt_in, 0.0)
        heat_added = (W_f * self.LHV * self.eta_comb) / W_out
        h_out_target = (W_in * h_in) / W_out + heat_added

        # Fast Newton-Raphson internal root finding for Tt_out
        Tt_out = Tt_in + heat_added / 1150.0
        for _ in range(5):
            cp = self.gas.calculate_cp(Tt_out, f)
            Tt_out -= (self.gas.calculate_enthalpy(Tt_out, f) - h_out_target) / cp

        # 3. Rayleigh Pressure Loss (Momentum + Heat Addition physics)
        flow_parameter = (W_in * np.sqrt(Tt_in)) / Pt_in
        dp_loss = self.k1 * (flow_parameter**2) + self.k2 * (flow_parameter**2) * (
            (Tt_out / Tt_in) - 1.0
        )

        # Absolute clipping to prevent inverted physics during numerical transients
        dp_loss = np.clip(dp_loss, 0.0, 0.20)

        Pt_out = Pt_in * (1.0 - dp_loss)

        return CombustorOutput(W_out, f, Pt_out, Tt_out, penalty)
