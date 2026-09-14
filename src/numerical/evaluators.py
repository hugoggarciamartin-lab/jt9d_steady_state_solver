"""
Off Design Operating Line Evaluator.
Calculates the steady state thermodynamic residuals for the solver.
"""

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class OffDesignEvaluator:
    def __init__(
        self,
        components: dict,
        ambient_cond: dict,
        engine_specs: dict,
        dp_data: dict,
        W_f_target: float,
    ):
        self.components = components
        self.ambient_cond = ambient_cond
        self.engine_specs = engine_specs
        self.u_vector = dp_data["u_vector"].item()

        self.N1_des = engine_specs["mechanical"]["design_speeds"]["N1_rpm"]
        self.N2_des = engine_specs["mechanical"]["design_speeds"]["N2_rpm"]
        self.A8_m2 = engine_specs["geometry"]["A8_m2"]
        self.A18_m2 = engine_specs["geometry"]["A18_m2"]
        self.W_f_target = W_f_target

        self.Fn = 0.0
        self.SFC = 0.0
        self.EGT = 0.0
        self.PR_fan = 0.0
        self.PR_hpc = 0.0
        self.Wc_fan = 0.0
        self.Wc_hpc = 0.0

    def evaluate(self, state_vector: np.ndarray) -> np.ndarray:
        beta_fan = np.clip(state_vector[0], 0.01, 0.99)
        beta_lpc = np.clip(state_vector[1], 0.01, 0.99)
        beta_hpc = np.clip(state_vector[2], 0.01, 0.99)
        pr_hpt = max(state_vector[3], 1.001)
        pr_lpt = max(state_vector[4], 1.001)
        BPR = max(state_vector[5], 0.1)
        N1 = max(state_vector[6], 100.0)
        N2 = max(state_vector[7], 100.0)

        Pt2 = self.ambient_cond["Pt"]
        Tt2 = self.ambient_cond["Tt"]
        P_amb = self.ambient_cond["P_amb"]

        fan_out = self.components["fan"].calculate(
            Pt2,
            Tt2,
            N1,
            self.N1_des,
            beta_fan,
            BPR=BPR,
            sigma=self.engine_specs["fan_radial_scalar_sigma"],
        )

        W_core_actual = (
            fan_out.W_in / (1.0 + BPR)
            if hasattr(fan_out, "W_in")
            else fan_out.W_fan / (1.0 + BPR)
        )
        W_byp_actual = W_core_actual * BPR

        lpc_out = self.components["lpc"].calculate(
            fan_out.Pt21, fan_out.Tt21, N1, self.N1_des, beta_lpc, bleed_fraction=0.0
        )

        hpc_out = self.components["hpc"].calculate(
            lpc_out.Pt_out,
            lpc_out.Tt_out,
            N2,
            self.N2_des,
            beta_hpc,
            bleed_fraction=self.u_vector["xi_cool"],
        )

        comb_out = self.components["comb"].calculate(
            hpc_out.Pt_out, hpc_out.Tt_out, hpc_out.W_out, self.W_f_target
        )

        hpt_out = self.components["hpt"].calculate(
            comb_out.Pt_out,
            comb_out.Tt_out,
            comb_out.W_out,
            comb_out.f,
            N2,
            self.N2_des,
            pr_hpt,
            W_cool=hpc_out.W_bleed,
            Tt_cool=hpc_out.Tt_out,
        )

        lpt_out = self.components["lpt"].calculate(
            hpt_out.Pt_out,
            hpt_out.Tt_out,
            hpt_out.W_out,
            hpt_out.f_out,
            N1,
            self.N1_des,
            pr_lpt,
            W_cool=0.0,
            Tt_cool=288.15,
        )

        core_noz_out = self.components["core_noz"].calculate(
            lpt_out.Pt_out,
            lpt_out.Tt_out,
            lpt_out.W_out,
            lpt_out.f_out,
            P_amb,
            A_throat=self.A8_m2,
            k_noz=self.u_vector["k_noz"],
        )

        byp_noz_out = self.components["byp_noz"].calculate(
            fan_out.Pt13,
            fan_out.Tt13,
            W_byp_actual,
            0.0,
            P_amb,
            A_throat=self.A18_m2,
            k_noz=self.u_vector["k_noz"],
        )

        hp_balance = self.components["hp_shaft"].calculate(
            hpt_out.power_gen, hpc_out.power_req
        )
        lp_balance = self.components["lp_shaft"].calculate(
            lpt_out.power_gen, fan_out.power_req + lpc_out.power_req
        )

        res_w_lpc = (lpc_out.W_in - W_core_actual) / max(W_core_actual, 1e-6)
        res_w_hpc = (hpc_out.W_in - lpc_out.W_out) / max(lpc_out.W_out, 1e-6)
        res_w_hpt = (hpt_out.W_capacity - comb_out.W_out) / max(comb_out.W_out, 1e-6)
        res_w_lpt = (lpt_out.W_capacity - hpt_out.W_out) / max(hpt_out.W_out, 1e-6)
        res_w_core = (core_noz_out.W_calc - lpt_out.W_out) / max(lpt_out.W_out, 1e-6)
        res_w_byp = (byp_noz_out.W_calc - W_byp_actual) / max(W_byp_actual, 1e-6)
        res_hp_mech = hp_balance.power_net / 1e6
        res_lp_mech = lp_balance.power_net / 1e6

        self.Fn = core_noz_out.Fg + byp_noz_out.Fg
        self.SFC = (self.W_f_target * 3600.0) / max(self.Fn, 1.0)
        self.EGT = lpt_out.Tt_out
        self.PR_fan = fan_out.Pt13 / Pt2
        self.Wc_fan = fan_out.W_fan * np.sqrt(Tt2 / 288.15) / (Pt2 / 101325.0)
        self.PR_hpc = hpc_out.Pt_out / lpc_out.Pt_out
        self.Wc_hpc = (
            hpc_out.W_in
            * np.sqrt(lpc_out.Tt_out / 288.15)
            / (lpc_out.Pt_out / 101325.0)
        )

        penalty = (
            fan_out.penalty
            + hpc_out.penalty
            + comb_out.penalty
            + hpt_out.penalty
            + lpt_out.penalty
        )

        residuals = np.array(
            [
                res_w_lpc,
                res_w_hpc,
                res_w_hpt,
                res_w_lpt,
                res_w_core,
                res_w_byp,
                res_hp_mech,
                res_lp_mech,
            ]
        )
        return residuals * (1.0 + penalty)
