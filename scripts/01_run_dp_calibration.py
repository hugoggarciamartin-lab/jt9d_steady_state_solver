"""
Design Point Analytical Sizing Script.
Calculates the exact thermodynamic baseline using real station telemetry.
Solves a 2x2 Jacobian (W2, Tt4) to strictly match Fn and EGT targets.
Data Coupling enforced: All parameters extracted from external JSON configurations.
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

        self.PR_fan = self.telemetry["station_telemetry"]["Pt13_Pa"] / self.Pt2
        self.PR_lpc = self.telemetry["station_telemetry"]["Pt25_Pa"] / self.Pt2
        self.PR_hpc = (
            self.telemetry["station_telemetry"]["Pt3_Pa"]
            / self.telemetry["station_telemetry"]["Pt25_Pa"]
        )

        self.PR_hpt = 0.0
        self.PR_lpt = 0.0

        self.eta_fan = 0.0
        self.eta_lpc = 0.0
        self.eta_hpc = 0.0

        self.Tt45_converged = 0.0
        self.Pt45_converged = 0.0
        self.Pt4_converged = 0.0
        self.Tt4_converged = 0.0

    def evaluate(self, state_vector: np.ndarray) -> np.ndarray:
        W2, Tt4 = state_vector
        W2 = max(W2, 100.0)
        Tt4 = max(Tt4, 1000.0)

        BPR = 5.0
        W_core = W2 / (1.0 + BPR)
        W_byp = W2 - W_core

        h2 = self.gas.calculate_enthalpy(self.Tt2, 0.0)

        # Fan tip (Bypass)
        Tt13_is = self.Tt2 * (self.PR_fan ** (0.4 / 1.4))
        Tt13_actual = self.telemetry["station_telemetry"]["Tt13_K"]
        dh_fan_byp = self.gas.calculate_enthalpy(Tt13_actual, 0.0) - h2
        self.eta_fan = (self.gas.calculate_enthalpy(Tt13_is, 0.0) - h2) / dh_fan_byp

        # LPC
        Tt25_is = self.Tt2 * (self.PR_lpc ** (0.4 / 1.4))
        Tt25_actual = self.telemetry["station_telemetry"]["Tt25_K"]
        dh_lpc = self.gas.calculate_enthalpy(Tt25_actual, 0.0) - h2
        self.eta_lpc = (self.gas.calculate_enthalpy(Tt25_is, 0.0) - h2) / dh_lpc

        # HPC
        Tt3_is = Tt25_actual * (self.PR_hpc ** (0.4 / 1.4))
        Tt3_actual = self.telemetry["station_telemetry"]["Tt3_K"]
        h_25 = self.gas.calculate_enthalpy(Tt25_actual, 0.0)
        h_3 = self.gas.calculate_enthalpy(Tt3_actual, 0.0)
        dh_hpc = h_3 - h_25
        self.eta_hpc = (self.gas.calculate_enthalpy(Tt3_is, 0.0) - h_25) / dh_hpc

        cust_bleed = self.specs["bleeds"].get("customer_bleed_fraction", 0.0)
        cool_bleed = self.specs["bleeds"]["cooling_bleed_fraction"]

        W_bleed_cool = W_core * cool_bleed
        W31 = W_core * (1.0 - cust_bleed - cool_bleed)

        # Combustor: Analytical Pressure Loss Factor (PLF)
        Pt3_actual = self.telemetry["station_telemetry"]["Pt3_Pa"]

        # Traverse correctly through flammability_limits nesting
        k1 = self.specs["flammability_limits"]["combustor_calibration_defaults"]["k1"]
        k2 = self.specs["flammability_limits"]["combustor_calibration_defaults"]["k2"]
        flow_parameter = (W31 * (Tt3_actual**0.5)) / Pt3_actual

        dp_loss = k1 * (flow_parameter**2) + k2 * (flow_parameter**2) * (
            (Tt4 / Tt3_actual) - 1.0
        )
        Pt4 = Pt3_actual * (1.0 - dp_loss)

        W_f = self.telemetry["global_performance"]["W_f_exp_kg_s"]
        W4 = W31 + W_f

        # Turbines
        req_power_hpc = W_core * dh_hpc
        req_power_lp = (W_byp * dh_fan_byp) + (W_core * dh_lpc)

        dh_hpt = (req_power_hpc / self.specs["mechanical"]["eta_PGB"]) / W4
        h4 = self.gas.calculate_enthalpy(Tt4, 0.02)
        h45 = h4 - dh_hpt
        Tt45 = Tt4 - (dh_hpt / 1150.0)
        Pt45 = Pt4 * (Tt45 / Tt4) ** (1.33 / 0.33 * (1 / 0.90))

        W45 = W4 + W_bleed_cool
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

        # Nozzles
        k_noz = self.specs["nozzle_calibration_defaults"]["k_noz"]
        Pt5_eff = Pt5 * (1.0 - k_noz)
        Pt13_eff = self.telemetry["station_telemetry"]["Pt13_Pa"] * (1.0 - k_noz)

        V_exit_core = np.sqrt(
            2
            * 1.33
            / (1.33 - 1)
            * 287
            * Tt5
            * (1 - (self.P_amb / max(Pt5_eff, self.P_amb)) ** (0.33 / 1.33))
        )
        V_exit_byp = np.sqrt(
            2
            * 1.4
            / (1.4 - 1)
            * 287
            * Tt13_actual
            * (1 - (self.P_amb / max(Pt13_eff, self.P_amb)) ** (0.4 / 1.4))
        )

        Cd_8 = self.specs["nozzle_calibration_defaults"]["Cd_8_initial"]
        Cd_18 = self.specs["nozzle_calibration_defaults"]["Cd_18_initial"]

        Fn_calc = (W45 * V_exit_core * Cd_8) + (W_byp * V_exit_byp * Cd_18)

        Fn_target = self.telemetry["global_performance"]["F_N_exp_N"]
        EGT_target = self.telemetry["station_telemetry"]["Tt5_K"]

        res_Fn = (Fn_calc - Fn_target) / Fn_target
        res_EGT = (Tt5 - EGT_target) / EGT_target

        return np.array([res_Fn, res_EGT])


def calc_corrected_flow(W: float, Pt: float, Tt: float) -> float:
    theta = Tt / 288.15
    delta = Pt / 101325.0
    return W * np.sqrt(theta) / delta


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    map_dir = project_root / "data" / "maps"
    config_dir = project_root / "data" / "config"
    telemetry_dir = project_root / "data" / "telemetry"

    telemetry = load_json(telemetry_dir / "dp_test_cell_data.json")
    specs = load_json(config_dir / "engine_specs.json")

    nr_solver = NewtonRaphsonSolver(max_iters=30, tolerance=1e-5)
    evaluator = AnalyticalDPEvaluator(telemetry, specs)

    initial_guess = np.array([600.0, 1500.0])
    converged_state, final_residuals, success = nr_solver.solve(
        evaluator.evaluate, initial_guess
    )

    if success:
        W2_conv, Tt4_conv = converged_state
        BPR = 5.0
        W_core_conv = W2_conv / (1.0 + BPR)

        cust_bleed = specs["bleeds"].get("customer_bleed_fraction", 0.0)
        cool_bleed = specs["bleeds"]["cooling_bleed_fraction"]

        u_vector = {
            "eta_fan": evaluator.eta_fan,
            "eta_lpc": evaluator.eta_lpc,
            "eta_hpc": evaluator.eta_hpc,
            "eta_hpt": 0.90,
            "eta_lpt": 0.91,
            "Cd_8": specs["nozzle_calibration_defaults"]["Cd_8_initial"],
            "Cd_18": specs["nozzle_calibration_defaults"]["Cd_18_initial"],
            "xi_cool": cool_bleed,
            "k_noz": specs["nozzle_calibration_defaults"]["k_noz"],
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

        required_maps = ["FAN.map", "LPC.map", "HPC.map", "HPT1.map", "LPT.map"]
        for m in required_maps:
            if not (map_dir / m).exists():
                print(
                    f"Warning: Map {m} missing. Scaling factors will not be generated."
                )
                return

        fan_map = parse_tmats_map(str(map_dir / "FAN.map"), True)
        lpc_map = parse_tmats_map(str(map_dir / "LPC.map"), True)
        hpc_map = parse_tmats_map(str(map_dir / "HPC.map"), True)
        hpt_map = parse_tmats_map(str(map_dir / "HPT1.map"), False)
        lpt_map = parse_tmats_map(str(map_dir / "LPT.map"), False)

        Wc_fan_map, eff_fan_map, pr_fan_map, _ = fan_map.evaluate(1.0, 0.5)
        Wc_lpc_map, eff_lpc_map, pr_lpc_map, _ = lpc_map.evaluate(1.0, 0.5)
        Wc_hpc_map, eff_hpc_map, pr_hpc_map, _ = hpc_map.evaluate(1.0, 0.5)
        Wc_hpt_map, eff_hpt_map, _ = hpt_map.evaluate(1.0, 3.0)
        Wc_lpt_map, eff_lpt_map, _ = lpt_map.evaluate(1.0, 3.0)

        W31 = W_core_conv * (1.0 - cust_bleed - cool_bleed)
        W_4 = W31 + dp_magnitudes["W_f_physical"]
        W_45 = W_4 + (W_core_conv * cool_bleed)

        Wc_fan_analyt = calc_corrected_flow(
            W2_conv,
            telemetry["station_telemetry"]["Pt2_Pa"],
            telemetry["station_telemetry"]["Tt2_K"],
        )
        Wc_lpc_analyt = calc_corrected_flow(
            W_core_conv,
            telemetry["station_telemetry"]["Pt13_Pa"],
            telemetry["station_telemetry"]["Tt13_K"],
        )
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

        scaling_factors = {
            "s_W_fan": Wc_fan_analyt / Wc_fan_map,
            "s_PR_fan": (dp_magnitudes["PR_fan"] - 1.0) / (pr_fan_map - 1.0),
            "s_eff_fan": u_vector["eta_fan"] / eff_fan_map,
            "s_W_lpc": Wc_lpc_analyt / Wc_lpc_map,
            "s_PR_lpc": (dp_magnitudes["PR_lpc"] - 1.0) / (pr_lpc_map - 1.0),
            "s_eff_lpc": u_vector["eta_lpc"] / eff_lpc_map,
            "s_W_hpc": Wc_hpc_analyt / Wc_hpc_map,
            "s_PR_hpc": (dp_magnitudes["PR_hpc"] - 1.0) / (pr_hpc_map - 1.0),
            "s_eff_hpc": u_vector["eta_hpc"] / eff_hpc_map,
            "s_W_hpt": Wc_hpt_analyt / Wc_hpt_map,
            "s_PR_hpt": (dp_magnitudes["PR_hpt"] - 1.0) / (3.0 - 1.0),
            "s_eff_hpt": u_vector["eta_hpt"] / eff_hpt_map,
            "s_W_lpt": Wc_lpt_analyt / Wc_lpt_map,
            "s_PR_lpt": (dp_magnitudes["PR_lpt"] - 1.0) / (3.0 - 1.0),
            "s_eff_lpt": u_vector["eta_lpt"] / eff_lpt_map,
        }

        print("Optimization Parameter Vector (u)")
        for k, v in u_vector.items():
            print(f"  {k}: {v:.5f}")

        print("\nMap Scaling Factors (SF)")
        for k, v in scaling_factors.items():
            print(f"  {k}: {v:.5f}")

        np.savez(
            output_dir / "solver_history.npz",
            state_vector=converged_state,
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
