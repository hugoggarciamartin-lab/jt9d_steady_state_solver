"""
Design Point Analytical Sizing Script.
Bypasses empirical maps to establish the exact thermodynamic baseline.
Solves a 2x2 Jacobian (W2, Tt4) to strictly match Fn and EGT telemetry targets,
and exports the optimization vector u, DP magnitudes, and map scaling factors.
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


def load_configuration(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file missing: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = project_root / "data" / "maps"

    target_data = {"Fn": 205000.0, "EGT": 815.0}  # JT9D-7A Telemetry

    ambient_cond = {"Pt2": 101325.0, "Tt2": 288.15, "P_amb": 101325.0}

    # JT9D-7A Baseline Design Assumptions (No maps required for DP sizing)
    design_assumptions = {
        "BPR": 5.0,
        "PR_fan": 1.5,
        "eta_fan": 0.89,
        "PR_lpc": 1.8,
        "eta_lpc": 0.88,
        "PR_hpc": 10.0,
        "eta_hpc": 0.87,
        "eta_hpt": 0.90,
        "eta_lpt": 0.91,
        "bleed_frac": 0.105,
        "mech_eff": 0.99,
    }

    gas = GasProperties()
    nr_solver = NewtonRaphsonSolver(max_iters=30, tolerance=1e-5)

    def evaluate_analytical_dp(state_vector: np.ndarray) -> np.ndarray:
        W2, Tt4 = state_vector

        W2 = max(W2, 100.0)
        Tt4 = max(Tt4, 1000.0)

        W_core = W2 / (1.0 + design_assumptions["BPR"])
        W_byp = W2 - W_core

        # --- FAN & COMPRESSION ---
        Pt13 = ambient_cond["Pt2"] * design_assumptions["PR_fan"]
        h2 = gas.calculate_enthalpy(ambient_cond["Tt2"], 0.0)

        Tt13_is = ambient_cond["Tt2"] * (design_assumptions["PR_fan"] ** (0.4 / 1.4))
        dh_fan_byp = (gas.calculate_enthalpy(Tt13_is, 0.0) - h2) / design_assumptions[
            "eta_fan"
        ]

        Pt25 = (
            ambient_cond["Pt2"]
            * design_assumptions["PR_fan"]
            * design_assumptions["PR_lpc"]
        )
        Tt25_is = ambient_cond["Tt2"] * ((Pt25 / ambient_cond["Pt2"]) ** (0.4 / 1.4))
        dh_lpc = (gas.calculate_enthalpy(Tt25_is, 0.0) - h2) / design_assumptions[
            "eta_lpc"
        ]
        Tt25 = ambient_cond["Tt2"] + dh_lpc / 1004.0

        # HPC
        Pt3 = Pt25 * design_assumptions["PR_hpc"]
        Tt3_is = Tt25 * (design_assumptions["PR_hpc"] ** (0.4 / 1.4))
        dh_hpc = (
            gas.calculate_enthalpy(Tt3_is, 0.0) - gas.calculate_enthalpy(Tt25, 0.0)
        ) / design_assumptions["eta_hpc"]
        Tt3 = Tt25 + dh_hpc / 1004.0

        W_bleed = W_core * design_assumptions["bleed_frac"]
        W3 = W_core - W_bleed

        # --- COMBUSTOR ---
        Pi_cc = 0.955
        Pt4 = Pt3 * Pi_cc
        W4 = W3

        # --- TURBINES (Work Balance) ---
        req_power_hpc = W_core * dh_hpc
        req_power_lp = (W_byp * dh_fan_byp) + (W_core * dh_lpc)

        dh_hpt = (req_power_hpc / design_assumptions["mech_eff"]) / W4
        h4 = gas.calculate_enthalpy(Tt4, 0.02)
        h45 = h4 - dh_hpt
        Tt45 = Tt4 - (dh_hpt / 1150.0)
        Pt45 = Pt4 * (Tt45 / Tt4) ** (1.33 / 0.33 * (1 / design_assumptions["eta_hpt"]))

        # Cooling Mix
        W45 = W4 + W_bleed
        Tt45_mix = (W4 * Tt45 + W_bleed * Tt3) / W45

        dh_lpt = (req_power_lp / design_assumptions["mech_eff"]) / W45
        Tt5 = Tt45_mix - (dh_lpt / 1150.0)
        Pt5 = Pt45 * (Tt5 / Tt45_mix) ** (
            1.33 / 0.33 * (1 / design_assumptions["eta_lpt"])
        )

        # Store internal DP pressure ratios for evaluation extraction
        evaluate_analytical_dp.PR_hpt = Pt3 / Pt45
        evaluate_analytical_dp.PR_lpt = Pt45 / Pt5

        # --- NOZZLES ---
        V_exit_core = np.sqrt(
            2
            * 1.33
            / (1.33 - 1)
            * 287
            * Tt5
            * (1 - (ambient_cond["P_amb"] / Pt5) ** (0.33 / 1.33))
        )
        V_exit_byp = np.sqrt(
            2
            * 1.4
            / (1.4 - 1)
            * 287
            * ambient_cond["Tt2"]
            * (1 - (ambient_cond["P_amb"] / Pt13) ** (0.4 / 1.4))
        )

        Fn_calc = (W45 * V_exit_core) + (W_byp * V_exit_byp)

        res_Fn = (Fn_calc - target_data["Fn"]) / target_data["Fn"]
        res_EGT = (Tt5 - target_data["EGT"]) / target_data["EGT"]

        return np.array([res_Fn, res_EGT])

    initial_guess = np.array([600.0, 1500.0])

    converged_state, final_residuals, success = nr_solver.solve(
        evaluate_analytical_dp, initial_guess
    )

    print(f"Analytical DP Converged: {success}")
    print(f"Required Inlet Mass Flow (W2): {converged_state[0]:.2f} kg/s")
    print(f"Required Combustor Temp (Tt4): {converged_state[1]:.2f} K")
    print(f"Residuals [Fn, EGT]: {final_residuals}")

    if success:
        W2_conv, Tt4_conv = converged_state
        W_core_conv = W2_conv / (1.0 + design_assumptions["BPR"])

        # Load raw T-MATS maps
        fan_map = parse_tmats_map(str(map_dir / "FAN.map"), True)
        lpc_map = parse_tmats_map(str(map_dir / "LPC.map"), True)
        hpc_map = parse_tmats_map(str(map_dir / "HPC.map"), True)
        hpt_map = parse_tmats_map(str(map_dir / "HPT1.map"), False)
        lpt_map = parse_tmats_map(str(map_dir / "LPT.map"), False)

        # Evaluate maps at nominal DP design points
        _, eff_fan_map, pr_fan_map, _ = fan_map.evaluate(1.0, 0.5)
        _, eff_lpc_map, pr_lpc_map, _ = lpc_map.evaluate(1.0, 0.5)
        _, eff_hpc_map, pr_hpc_map, _ = hpc_map.evaluate(1.0, 0.5)
        _, eff_hpt_map, _ = hpt_map.evaluate(1.0, 3.0)
        _, eff_lpt_map, _ = lpt_map.evaluate(1.0, 3.0)

        # Optimization Parameter Vector u (matching the reference vector layout)
        u_vector = {
            "eta_fan": design_assumptions["eta_fan"],
            "eta_lpc": design_assumptions["eta_lpc"],
            "eta_hpc": design_assumptions["eta_hpc"],
            "eta_hpt": design_assumptions["eta_hpt"],
            "eta_lpt": design_assumptions["eta_lpt"],
            "pi_cc": 0.955,
            "Cd_8": 0.98,
            "Cd_18": 0.98,
            "xi_cool": design_assumptions["bleed_frac"],
        }

        # DP Thermodynamic Magnitudes
        dp_magnitudes = {
            "PR_fan": design_assumptions["PR_fan"],
            "PR_lpc": design_assumptions["PR_lpc"],
            "PR_hpc": design_assumptions["PR_hpc"],
            "PR_hpt": evaluate_analytical_dp.PR_hpt,
            "PR_lpt": evaluate_analytical_dp.PR_lpt,
            "W2_physical": W2_conv,
            "Tt4_physical": Tt4_conv,
        }

        # Map Scaling Factors (SF)
        theta_2 = ambient_cond["Tt2"] / 288.15
        delta_2 = ambient_cond["Pt2"] / 101325.0
        Wc_fan_analyt = W2_conv * np.sqrt(theta_2) / delta_2
        Wc_fan_map, _, _, _ = fan_map.evaluate(1.0, 0.5)

        scaling_factors = {
            "s_W_fan": Wc_fan_analyt / Wc_fan_map,
            "s_PR_fan": (dp_magnitudes["PR_fan"] - 1.0) / (pr_fan_map - 1.0),
            "s_eff_fan": u_vector["eta_fan"] / eff_fan_map,
            "s_eff_lpc": u_vector["eta_lpc"] / eff_lpc_map,
            "s_eff_hpc": u_vector["eta_hpc"] / eff_hpc_map,
            "s_eff_hpt": u_vector["eta_hpt"] / eff_hpt_map,
            "s_eff_lpt": u_vector["eta_lpt"] / eff_lpt_map,
        }

        print("\n--- Optimization Parameter Vector (u) ---")
        for k, v in u_vector.items():
            print(f"  {k}: {v:.4f}")

        print("\n--- DP Thermodynamic Magnitudes ---")
        for k, v in dp_magnitudes.items():
            print(f"  {k}: {v:.4f}")

        print("\n--- Map Scaling Factors (SF) ---")
        for k, v in scaling_factors.items():
            print(f"  {k}: {v:.4f}")

        np.savez(
            output_dir / "solver_history.npz",
            state_vector=converged_state,
            residuals=final_residuals,
            u_vector=u_vector,
            dp_magnitudes=dp_magnitudes,
            scaling_factors=scaling_factors,
            success=success,
        )


if __name__ == "__main__":
    main()
