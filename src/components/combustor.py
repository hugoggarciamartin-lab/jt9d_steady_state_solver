"""
Combustor component module for 0D steady-state engine simulation.
Solves the vitiated enthalpy balance and enforces flammability limits.
"""

from dataclasses import dataclass
from src.utils.gas_properties import GasProperties


@dataclass
class CombustorOutput:
    W_out: float  # Mass flow exiting the combustor [kg/s]
    f: float  # Fuel-to-air ratio [-]
    Pt_out: float  # Discharge total pressure [Pa]
    Tt_out: float  # Discharge total temperature (TIT / Tt4) [K]
    penalty: float  # Flammability or physical bound violation penalty


class Combustor:
    """
    Aerothermodynamic Combustor component.
    Models energy addition, pressure drop, and vitiated gas properties.
    """

    def __init__(
        self,
        gas_props: GasProperties,
        LHV: float = 43.2e6,
        dP_norm: float = 0.045,
        eta_cc: float = 0.995,
    ):
        self.gas = gas_props
        self.LHV = LHV  # Lower Heating Value of Jet-A [J/kg]
        self.dP_norm = dP_norm  # Fractional total pressure drop [-]
        self.eta_cc = eta_cc  # Combustion efficiency [-]

        # Flammability boundaries (approximate for MoC 2 guards)
        self.f_LDO = 0.005  # Lean Die-Out limit
        self.f_RBO = 0.065  # Rich Blow-Out limit

    def _solve_temperature_from_enthalpy(
        self, h_target: float, T_guess: float, f: float
    ) -> float:
        """
        Local Newton-Raphson solver to reverse-lookup vitiated gas temperature
        from sensible enthalpy.
        """
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
        """
        Executes the forward aerothermodynamic pass for the burner.

        Args:
            Pt_in: Inlet total pressure (Station 3) [Pa]
            Tt_in: Inlet total temperature (Station 3) [K]
            W_in: Air mass flow from compressor discharge [kg/s]
            W_f: Fuel mass flow injected [kg/s]
        """
        penalty = 0.0
        k_penalty = 1e6

        # Fuel-to-Air Ratio (FAR)
        f = W_f / W_in

        # Flammability Guards
        if f < self.f_LDO:
            penalty += k_penalty * (self.f_LDO - f) ** 2
            f = self.f_LDO
        elif f > self.f_RBO:
            penalty += k_penalty * (f - self.f_RBO) ** 2
            f = self.f_RBO

        # Total Mass Flow Continuity
        W_out = W_in + W_f

        # Viscous Pressure Drop
        Pt_out = Pt_in * (1.0 - self.dP_norm)

        # Energy Balance (h_in * W_in + W_f * LHV * eta = h_out * W_out)
        h_in = self.gas.calculate_enthalpy(Tt_in, f=0.0)

        # Total energy entering the control volume [W]
        Q_in = (W_in * h_in) + (W_f * self.LHV * self.eta_cc)

        # Specific sensible enthalpy of the mixture exiting [J/kg]
        h_out_target = Q_in / W_out

        # Temperature Inversion (Estimate delta T roughly first)
        cp_approx = 1150.0  # J/kgK guess for high-temp gas
        T_guess = Tt_in + (W_f * self.LHV * self.eta_cc) / (W_out * cp_approx)

        Tt_out = self._solve_temperature_from_enthalpy(h_out_target, T_guess, f)

        # High Temperature TCDS Limit Guard (e.g. 1800K for generic turbofans)
        if Tt_out > 2000.0:
            penalty += k_penalty * (Tt_out - 2000.0) ** 2

        return CombustorOutput(
            W_out=W_out, f=f, Pt_out=Pt_out, Tt_out=Tt_out, penalty=penalty
        )
