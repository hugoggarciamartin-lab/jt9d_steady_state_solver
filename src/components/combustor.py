"""
Combustor component module for 0D steady-state engine simulation.
Solves the vitiated enthalpy balance and enforces flammability limits,
incorporating analytical cold/hot pressure loss factors (PLF).
"""

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
    def __init__(
        self,
        gas_props: GasProperties,
        LHV: float = 43.125e6,  # Jet-A1 standard [J/kg]
        k1: float = 2.0e-4,
        k2: float = 0.8e-4,
        eta_cc: float = 0.995,
    ):
        self.gas = gas_props
        self.LHV = LHV
        self.k1 = k1
        self.k2 = k2
        self.eta_cc = eta_cc
        self.f_LDO = 0.0080
        self.f_RBO = 0.0580

    def _solve_temperature_from_enthalpy(
        self, h_target: float, T_guess: float, f: float
    ) -> float:
        T_iter = T_guess
        for _ in range(15):
            h_current = self.gas.calculate_enthalpy(T_iter, f)
            residual = h_current - h_target
            if abs(residual) < 1e-3:
                return T_iter
            cp_current = self.gas.calculate_cp(T_iter, f)
            T_iter -= residual / cp_current
        return T_iter

    def calculate(
        self, Pt_in: float, Tt_in: float, W_in: float, W_f: float
    ) -> CombustorOutput:
        penalty = 0.0
        k_penalty = 1e6

        f = W_f / W_in
        if f < self.f_LDO:
            penalty += k_penalty * (self.f_LDO - f) ** 2
            f = self.f_LDO
        elif f > self.f_RBO:
            penalty += k_penalty * (f - self.f_RBO) ** 2
            f = self.f_RBO

        W_out = W_in + W_f
        h_in = self.gas.calculate_enthalpy(Tt_in, f=0.0)
        Q_in = (W_in * h_in) + (W_f * self.LHV * self.eta_cc)
        h_out_target = Q_in / W_out

        cp_approx = 1150.0
        T_guess = Tt_in + (W_f * self.LHV * self.eta_cc) / (W_out * cp_approx)
        Tt_out = self._solve_temperature_from_enthalpy(h_out_target, T_guess, f)

        # Analytical Pressure Loss Factor (PLF) evaluating cold and hot momentum losses
        flow_parameter = (W_in * (Tt_in**0.5)) / Pt_in
        dp_loss = self.k1 * (flow_parameter**2) + self.k2 * (flow_parameter**2) * (
            (Tt_out / Tt_in) - 1.0
        )
        Pt_out = Pt_in * (1.0 - dp_loss)

        if Tt_out > 2000.0:
            penalty += k_penalty * (Tt_out - 2000.0) ** 2

        return CombustorOutput(
            W_out=W_out, f=f, Pt_out=Pt_out, Tt_out=Tt_out, penalty=penalty
        )
