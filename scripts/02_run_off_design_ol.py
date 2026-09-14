"""
Off Design Operating Line Solver.
Executes the steady state transient sweep strictly along the validation points.
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
from src.components.fan import Fan
from src.components.compressor import Compressor
from src.components.combustor import Combustor
from src.components.turbine import Turbine
from src.components.nozzle import Nozzle
from src.components.shaft import Shaft
from src.numerical.newton_raphson import NewtonRaphsonSolver
from src.numerical.evaluators import OffDesignEvaluator


def load_data(filepath: Path):
    if not filepath.exists():
        raise FileNotFoundError(f"Missing required artifact or config: {filepath}")
    if filepath.suffix == ".json":
        with open(filepath, "r") as f:
            return json.load(f)
    elif filepath.suffix == ".npz":
        return np.load(filepath, allow_pickle=True)
    raise ValueError(f"Unsupported file format: {filepath.suffix}")


def main():
    config_dir = project_root / "data" / "config"
    telemetry_dir = project_root / "data" / "telemetry"
    log_dir = project_root / "output" / "logs"
    map_dir = project_root / "data" / "maps"

    engine_specs = load_data(config_dir / "engine_specs.json")
    validation_data = load_data(telemetry_dir / "off_design_validation.json")
    dp_data = load_data(log_dir / "solver_history.npz")

    sf = dp_data["scaling_factors"].item()
    u_vec = dp_data["u_vector"].item()
    dp_mags = dp_data["dp_magnitudes"].item()

    fan_map = parse_tmats_map(
        str(map_dir / "FAN.map"),
        True,
        sf_w=sf.get("s_W_fan", 1.0),
        sf_eff=sf.get("s_eff_fan", 1.0),
        sf_pr=sf.get("s_PR_fan", 1.0),
    )
    lpc_map = parse_tmats_map(
        str(map_dir / "LPC.map"),
        True,
        sf_w=sf.get("s_W_lpc", 1.0),
        sf_eff=sf.get("s_eff_lpc", 1.0),
        sf_pr=sf.get("s_PR_lpc", 1.0),
    )
    hpc_map = parse_tmats_map(
        str(map_dir / "HPC.map"),
        True,
        sf_w=sf.get("s_W_hpc", 1.0),
        sf_eff=sf.get("s_eff_hpc", 1.0),
        sf_pr=sf.get("s_PR_hpc", 1.0),
    )
    hpt_map = parse_tmats_map(
        str(map_dir / "HPT1.map"),
        False,
        sf_w=sf.get("s_W_hpt", 1.0),
        sf_eff=sf.get("s_eff_hpt", 1.0),
        sf_pr=sf.get("s_PR_hpt", 1.0),
    )
    lpt_map = parse_tmats_map(
        str(map_dir / "LPT.map"),
        False,
        sf_w=sf.get("s_W_lpt", 1.0),
        sf_eff=sf.get("s_eff_lpt", 1.0),
        sf_pr=sf.get("s_PR_lpt", 1.0),
    )

    gas = GasProperties()

    combustor = Combustor(gas_props=gas, k1=u_vec["k1"], k2=u_vec["k2"])
    combustor.f_LDO = engine_specs["flammability_limits"]["f_LDO"]
    combustor.f_RBO = engine_specs["flammability_limits"]["f_RBO"]

    components = {
        "fan": Fan(fan_map, gas),
        "lpc": Compressor(lpc_map, gas),
        "hpc": Compressor(hpc_map, gas),
        "comb": combustor,
        "hpt": Turbine(hpt_map, gas),
        "lpt": Turbine(lpt_map, gas),
        "core_noz": Nozzle(gas),
        "byp_noz": Nozzle(gas),
        "hp_shaft": Shaft(
            power_ext=engine_specs["mechanical"]["power_accessory_base_W"]
        ),
        "lp_shaft": Shaft(),
    }

    ambient_cond = {"Pt": 101325.0, "Tt": 288.15, "P_amb": 101325.0}
    nr_solver = NewtonRaphsonSolver(
        max_iters=50, tolerance=1e-4, fd_eps=1e-3, max_step_frac=0.05
    )
    current_guess = dp_data["state_vector"]

    results = {
        "Rating": [],
        "W_f": [],
        "Fn": [],
        "SFC": [],
        "EGT": [],
        "N1": [],
        "N2": [],
        "beta_fan": [],
        "PR_fan": [],
        "Wc_fan": [],
        "beta_hpc": [],
        "PR_hpc": [],
        "Wc_hpc": [],
    }

    print("Initiating MoC 2 Validation Sweep...\n")
    W_f_DP = dp_mags["W_f_physical"]
    first_target = validation_data["validation_points"][0]["W_f_target_kg_s"]
    ramp_steps = np.linspace(W_f_DP, first_target, 5)

    homotopy_points = [{"rating": "Nominal DP", "W_f_target_kg_s": W_f_DP}]
    for i, wf in enumerate(ramp_steps[1:]):
        homotopy_points.append(
            {"rating": f"Transition Step {i + 1}", "W_f_target_kg_s": wf}
        )

    validation_points = homotopy_points + validation_data["validation_points"][1:]

    for point in validation_points:
        w_f = point["W_f_target_kg_s"]
        rating = point["rating"]

        evaluator = OffDesignEvaluator(
            components, ambient_cond, engine_specs, dp_data, w_f
        )
        converged_state, final_res, success = nr_solver.solve(
            evaluator.evaluate, current_guess
        )

        if not success:
            print(
                f"[FATAL EXCEPTION] {rating}: Solver divergence at W_f = {w_f:.3f} kg/s."
            )
            print(
                f"Final Failed State Vector: {np.array2string(converged_state, precision=4)}"
            )
            print(f"Final Residuals: {np.array2string(final_res, precision=4)}")
            break

        current_guess = converged_state
        results["Rating"].append(rating)
        results["W_f"].append(w_f)
        results["Fn"].append(evaluator.Fn)
        results["SFC"].append(evaluator.SFC)
        results["EGT"].append(evaluator.EGT)
        results["N1"].append(converged_state[6])
        results["N2"].append(converged_state[7])
        results["beta_fan"].append(converged_state[0])
        results["PR_fan"].append(evaluator.PR_fan)
        results["Wc_fan"].append(evaluator.Wc_fan)
        results["beta_hpc"].append(converged_state[2])
        results["PR_hpc"].append(evaluator.PR_hpc)
        results["Wc_hpc"].append(evaluator.Wc_hpc)

        print(
            f"[{rating}] W_f: {w_f:.3f} kg/s | Fn: {evaluator.Fn:.1f} N | SFC: {evaluator.SFC:.4f} | EGT: {evaluator.EGT:.1f} K"
        )

    np.savez(log_dir / "off_design_ol.npz", **results)
    print("\nValidation complete. Matrix artifact saved to off_design_ol.npz.")


if __name__ == "__main__":
    main()
