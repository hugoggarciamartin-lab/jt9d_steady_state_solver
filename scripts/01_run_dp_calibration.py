"""
Design Point Analytical Sizing Script.
Calculates the exact thermodynamic baseline using real station telemetry.
Solves a 2x2 Jacobian (W2, Tt4) to strictly match physical nozzle continuity.
Corrected Twin-Spool station bookkeeping (Pt2 -> Pt21 -> Pt25).
"""

import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.gas_properties import GasProperties
from src.utils.map_parser import parse_tmats_map
from src.numerical.newton_raphson import NewtonRaphsonSolver


def load_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


class AnalyticalDPEvaluator:
    def __init__(self, telemetry: dict, engine_specs: dict):
        self.telemetry = telemetry
        self.specs = engine_specs
        self.gas = GasProperties()

        self.Pt2 = self.telemetry["station_telemetry"]["Pt2_Pa"]
        self.Tt2 = self.telemetry["station_telemetry"]["Tt2_K"]
        self.P_amb = self.telemetry["reference_conditions"]["P0_Pa"]

        # Strict station separation for Twin-Spool architecture
        self.sigma = self.specs.get("fan_radial_scalar_sigma", 1.0)
        self.Pt21 = self.telemetry["station_telemetry"]["Pt13_Pa"] * self.sigma
        self.Tt21 = self.telemetry["station_telemetry"]["Tt13_K"]

        self.PR_fan = self.telemetry["station_telemetry"]["Pt13_Pa"] / self.Pt2
        self.PR_lpc = (
            self.telemetry["station_telemetry"]["Pt25_Pa"] / self.Pt21
        )  # CORRECTED: from Pt21
        self.PR_hpc = (
            self.telemetry["station_telemetry"]["Pt3_Pa"]
            / self.telemetry["station_telemetry"]["Pt25_Pa"]
        )

        self.A8 = self.specs["geometry"]["A8_m2"]
        self.A18 = self.specs["geometry"]["A18_m2"]

        self.PR_hpt = 0.0
        self.PR_lpt = 0.0
        self.eta_fan = 0.0
        self.eta_lpc = 0.0
        self.eta_hpc = 0.0
        self.Tt45_converged = 0.0
        self.Pt45_converged = 0.0
        self.Pt4_converged = 0.0
        self.Tt4_converged = 0.0
        self.Fn_converged = 0.0

    def _evaluate_nozzle_mass_flow(
        self, Pt_in: float, Tt_in: float, f_in: float, A_geom: float, k_noz: float
    ) -> float:
        Pt_eff = Pt_in * (1.0 - k_noz)
        NPR = Pt_eff / self.P_amb
        if NPR <= 1.001:
            return 1e-4

        cp = self.gas.calculate_cp(Tt_in, f_in)
        R_gas = 287.05
        gamma = cp / (cp - R_gas)
        PR_crit = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))

        Cd_analytical = 0.985 - 0.04 * np.exp(-2.0 * (NPR - 1.0))

        if NPR < PR_crit:
            M_exit = np.sqrt(
                (2.0 / (gamma - 1.0)) * (NPR ** ((gamma - 1.0) / gamma) - 1.0)
            )
            T_stat = Tt_in / (1.0 + ((gamma - 1.0) / 2.0) * M_exit**2)
            P_stat = self.P_amb
        else:
            M_exit = 1.0
            T_stat = Tt_in * (2.0 / (gamma + 1.0))
            P_stat = Pt_eff / PR_crit

        V_exit = M_exit * np.sqrt(gamma * R_gas * T_stat)
        rho_exit = P_stat / (R_gas * T_stat)
        return rho_exit * (A_geom * Cd_analytical) * V_exit

    def evaluate(self, state_vector: np.ndarray) -> np.ndarray:
        W2, Tt4 = state_vector
        W2 = max(W2, 100.0)
        Tt4 = max(Tt4, 1000.0)

        BPR = self.specs["geometry"].get("BPR_design", 5.0)
        W_core = W2 / (1.0 + BPR)
        W_byp = W2 - W_core

        h2 = self.gas.calculate_enthalpy(self.Tt2, 0.0)

        # Fan operates on full W2
        Tt13_is = self.Tt2 * (self.PR_fan ** (0.4 / 1.4))
        Tt13_actual = self.telemetry["station_telemetry"]["Tt13_K"]
        dh_fan = self.gas.calculate_enthalpy(Tt13_actual, 0.0) - h2
        self.eta_fan = (self.gas.calculate_enthalpy(Tt13_is, 0.0) - h2) / dh_fan

        # LPC operates from Station 21
        Tt25_is = self.Tt21 * (self.PR_lpc ** (0.4 / 1.4))
        Tt25_actual = self.telemetry["station_telemetry"]["Tt25_K"]
        h_21 = self.gas.calculate_enthalpy(self.Tt21, 0.0)
        h_25 = self.gas.calculate_enthalpy(Tt25_actual, 0.0)
        dh_lpc = h_25 - h_21
        self.eta_lpc = (self.gas.calculate_enthalpy(Tt25_is, 0.0) - h_21) / dh_lpc

        # HPC operates from Station 25
        Tt3_is = Tt25_actual * (self.PR_hpc ** (0.4 / 1.4))
        Tt3_actual = self.telemetry["station_telemetry"]["Tt3_K"]
        h_3 = self.gas.calculate_enthalpy(Tt3_actual, 0.0)
        dh_hpc = h_3 - h_25
        self.eta_hpc = (self.gas.calculate_enthalpy(Tt3_is, 0.0) - h_25) / dh_hpc

        cust_bleed = self.specs["bleeds"].get("customer_bleed_fraction", 0.0)
        cool_bleed = self.specs["bleeds"]["cooling_bleed_fraction"]
        W_bleed_cool = W_core * cool_bleed
        W31 = W_core * (1.0 - cust_bleed - cool_bleed)

        Pt3_actual = self.telemetry["station_telemetry"]["Pt3_Pa"]
        k1 = self.specs["flammability_limits"]["combustor_calibration_defaults"]["k1"]
        k2 = self.specs["flammability_limits"]["combustor_calibration_defaults"]["k2"]
        flow_parameter = (W31 * (Tt3_actual**0.5)) / Pt3_actual
        dp_loss = k1 * (flow_parameter**2) + k2 * (flow_parameter**2) * (
            (Tt4 / Tt3_actual) - 1.0
        )
        Pt4 = Pt3_actual * (1.0 - dp_loss)

        W_f = self.telemetry["global_performance"]["W_f_exp_kg_s"]
        W4 = W31 + W_f
        f_4 = W_f / W31

        req_power_hpc = W_core * dh_hpc
        # CORRECTED: LP shaft drives both Bypass and Core portions through the Fan Hub
        req_power_lp = (W2 * dh_fan) + (W_core * dh_lpc)

        dh_hpt = (req_power_hpc / self.specs["mechanical"]["eta_PGB"]) / W4
        h4 = self.gas.calculate_enthalpy(Tt4, f_4)
        Tt45 = Tt4 - (dh_hpt / 1150.0)
        Pt45 = Pt4 * (Tt45 / Tt4) ** (1.33 / 0.33 * (1 / 0.90))

        W45 = W4 + W_bleed_cool
        f_45 = (W4 * f_4) / W45
        Tt45_mix = (W4 * Tt45 + W_bleed_cool * Tt3_actual) / W45

        dh_lpt = (req_power_lp / self.specs["mechanical"]["eta_PGB"]) / W45
        Tt5 = Tt45_mix - (dh_lpt / 1150.0)
        Pt5 = Pt45 * (Tt5 / Tt45_mix) ** (1.33 / 0.33 * (1 / 0.91))

        self.PR_hpt = Pt4 / Pt45
        self.PR_lpt = Pt45 / Pt5
        self.Tt45_converged = Tt45_mix
        self.Pt45_converged = Pt45
        self.Pt4_converged = Pt4
        self.Tt4_converged = Tt4

        k_noz = self.specs["nozzle_calibration_defaults"]["k_noz"]

        W_core_calc = self._evaluate_nozzle_mass_flow(Pt5, Tt5, f_45, self.A8, k_noz)
        W_byp_calc = self._evaluate_nozzle_mass_flow(
            self.telemetry["station_telemetry"]["Pt13_Pa"],
            Tt13_actual,
            0.0,
            self.A18,
            k_noz,
        )

        res_core_mass = (W_core_calc - W45) / max(W45, 1e-6)
        res_byp_mass = (W_byp_calc - W_byp) / max(W_byp, 1e-6)

        return np.array([res_core_mass, res_byp_mass])


def calc_corrected_flow(W: float, Pt: float, Tt: float) -> float:
    theta = Tt / 288.15
    delta = Pt / 101325.0
    return W * np.sqrt(theta) / delta


def main():
    config_dir = project_root / "data" / "config"
    telemetry_dir = project_root / "data" / "telemetry"
    output_dir = project_root / "output" / "logs"
    map_dir = project_root / "data" / "maps"

    telemetry = load_json(telemetry_dir / "dp_test_cell_data.json")
    specs = load_json(config_dir / "engine_specs.json")

    nr_solver = NewtonRaphsonSolver(max_iters=50, tolerance=1e-5, max_step_frac=0.10)
    evaluator = AnalyticalDPEvaluator(telemetry, specs)

    initial_guess = np.array([600.0, 1500.0])
    converged_state, final_residuals, success = nr_solver.solve(
        evaluator.evaluate, initial_guess
    )

    if success:
        W2_conv, Tt4_conv = converged_state
        BPR = specs["geometry"].get("BPR_design", 5.0)
        W_core_conv = W2_conv / (1.0 + BPR)

        cool_bleed = specs["bleeds"]["cooling_bleed_fraction"]
        cust_bleed = specs["bleeds"].get("customer_bleed_fraction", 0.0)

        Pt21 = telemetry["station_telemetry"]["Pt13_Pa"] * specs.get(
            "fan_radial_scalar_sigma", 1.0
        )
        Tt21 = telemetry["station_telemetry"]["Tt13_K"]

        fan_map = parse_tmats_map(str(map_dir / "FAN.map"), True)
        lpc_map = parse_tmats_map(str(map_dir / "LPC.map"), True)
        hpc_map = parse_tmats_map(str(map_dir / "HPC.map"), True)
        hpt_map = parse_tmats_map(str(map_dir / "HPT1.map"), False)
        lpt_map = parse_tmats_map(str(map_dir / "LPT.map"), False)

        beta_fan_dp, beta_lpc_dp, beta_hpc_dp = 0.50, 0.50, 0.50

        Wc_fan_map, eff_fan_map, pr_fan_map, _ = fan_map.evaluate(1.0, beta_fan_dp)
        Wc_lpc_map, eff_lpc_map, pr_lpc_map, _ = lpc_map.evaluate(1.0, beta_lpc_dp)
        Wc_hpc_map, eff_hpc_map, pr_hpc_map, _ = hpc_map.evaluate(1.0, beta_hpc_dp)
        Wc_hpt_map, eff_hpt_map, _ = hpt_map.evaluate(1.0, 3.0)
        Wc_lpt_map, eff_lpt_map, _ = lpt_map.evaluate(1.0, 3.0)

        W31 = W_core_conv * (1.0 - cust_bleed - cool_bleed)
        W_4 = W31 + telemetry["global_performance"]["W_f_exp_kg_s"]
        W_45 = W_4 + (W_core_conv * cool_bleed)

        Wc_fan_analyt = calc_corrected_flow(
            W2_conv,
            telemetry["station_telemetry"]["Pt2_Pa"],
            telemetry["station_telemetry"]["Tt2_K"],
        )
        # CORRECTED: LPC analytical corrected flow strictly uses Station 21
        Wc_lpc_analyt = calc_corrected_flow(W_core_conv, Pt21, Tt21)
        Wc_hpc_analyt = calc_corrected_flow(
            W_core_conv,
            telemetry["station_telemetry"]["Pt25_Pa"],
            telemetry["station_telemetry"]["Tt25_K"],
        )
        Wc_hpt_analyt = calc_corrected_flow(
            W_4, evaluator.Pt4_converged, evaluator.Tt4_converged
        )
        Wc_lpt_analyt = calc_corrected_flow(
            W_45, evaluator.Pt45_converged, evaluator.Tt45_converged
        )

        def clip_sf(val: float) -> float:
            return float(np.clip(val, 0.85, 1.15))

        scaling_factors = {
            "s_W_fan": clip_sf(Wc_fan_analyt / Wc_fan_map),
            "s_PR_fan": clip_sf((evaluator.PR_fan - 1.0) / (pr_fan_map - 1.0)),
            "s_eff_fan": clip_sf(evaluator.eta_fan / eff_fan_map),
            "s_W_lpc": clip_sf(Wc_lpc_analyt / Wc_lpc_map),
            "s_PR_lpc": clip_sf((evaluator.PR_lpc - 1.0) / (pr_lpc_map - 1.0)),
            "s_eff_lpc": clip_sf(evaluator.eta_lpc / eff_lpc_map),
            "s_W_hpc": clip_sf(Wc_hpc_analyt / Wc_hpc_map),
            "s_PR_hpc": clip_sf((evaluator.PR_hpc - 1.0) / (pr_hpc_map - 1.0)),
            "s_eff_hpc": clip_sf(evaluator.eta_hpc / eff_hpc_map),
            "s_W_hpt": clip_sf(Wc_hpt_analyt / Wc_hpt_map),
            "s_PR_hpt": clip_sf((evaluator.PR_hpt - 1.0) / (3.0 - 1.0)),
            "s_eff_hpt": clip_sf(0.90 / eff_hpt_map),
            "s_W_lpt": clip_sf(Wc_lpt_analyt / Wc_lpt_map),
            "s_PR_lpt": clip_sf((evaluator.PR_lpt - 1.0) / (3.0 - 1.0)),
            "s_eff_lpt": clip_sf(0.91 / eff_lpt_map),
        }

        full_dp_state = np.array(
            [
                beta_fan_dp,
                beta_lpc_dp,
                beta_hpc_dp,
                evaluator.PR_hpt,
                evaluator.PR_lpt,
                BPR,
                specs["mechanical"]["design_speeds"]["N1_rpm"],
                specs["mechanical"]["design_speeds"]["N2_rpm"],
            ]
        )

        P_amb = telemetry["reference_conditions"]["P0_Pa"]
        k_noz = specs["nozzle_calibration_defaults"]["k_noz"]
        NPR_core = (evaluator.Pt45_converged * (1.0 - k_noz)) / P_amb
        NPR_byp = (telemetry["station_telemetry"]["Pt13_Pa"] * (1.0 - k_noz)) / P_amb

        Cd_8_real = (
            float(np.clip(0.985 - 0.04 * np.exp(-2.0 * (NPR_core - 1.0)), 0.01, 0.999))
            if NPR_core > 1.0
            else 0.98
        )
        Cd_18_real = (
            float(np.clip(0.985 - 0.04 * np.exp(-2.0 * (NPR_byp - 1.0)), 0.01, 0.999))
            if NPR_byp > 1.0
            else 0.98
        )

        u_vector = {
            "eta_fan": evaluator.eta_fan,
            "eta_lpc": evaluator.eta_lpc,
            "eta_hpc": evaluator.eta_hpc,
            "eta_hpt": 0.90,
            "eta_lpt": 0.91,
            "Cd_8": Cd_8_real,
            "Cd_18": Cd_18_real,
            "xi_cool": cool_bleed,
            "k_noz": k_noz,
            "k1": specs["flammability_limits"]["combustor_calibration_defaults"]["k1"],
            "k2": specs["flammability_limits"]["combustor_calibration_defaults"]["k2"],
        }

        dp_magnitudes = {
            "PR_fan": evaluator.PR_fan,
            "PR_lpc": evaluator.PR_lpc,
            "PR_hpc": evaluator.PR_hpc,
            "PR_hpt": evaluator.PR_hpt,
            "PR_lpt": evaluator.PR_lpt,
            "W2_physical": W2_conv,
            "Tt4_physical": Tt4_conv,
            "W_f_physical": telemetry["global_performance"]["W_f_exp_kg_s"],
        }

        print("\nMap Scaling Factors (SF)")
        for k, v in scaling_factors.items():
            print(f"  {k}: {v:.5f}")

        np.savez(
            output_dir / "solver_history.npz",
            state_vector=full_dp_state,
            residuals=final_residuals,
            u_vector=u_vector,
            dp_magnitudes=dp_magnitudes,
            scaling_factors=scaling_factors,
            success=success,
        )
        print(
            f"\nAnalytical DP Converged. W2: {W2_conv:.2f} kg/s, Tt4: {Tt4_conv:.2f} K"
        )
    else:
        print("\nAnalytical DP Failed to converge.")


if __name__ == "__main__":
    main()
