"""
Off Design Operating Line Solver.
Executes the steady state transient sweep strictly along the validation points.
Implements a dense homotopy ramp and strict fractional step relaxation.
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


def load_data(filepath: Path):
    if not filepath.exists():
        raise FileNotFoundError(f"Missing required artifact or config: {filepath}")
    if filepath.suffix == ".json":
        with open(filepath, "r") as f:
            return json.load(f)
    elif filepath.suffix == ".npz":
        return np.load(filepath, allow_pickle=True)
    raise ValueError(f"Unsupported file format: {filepath.suffix}")


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
        pr_hpt = max(state_vector[3], 1.05)
        pr_lpt = max(state_vector[4], 1.05)
        BPR = max(state_vector[5], 0.1)
        N1 = max(state_vector[6], 500.0)
        N2 = max(state_vector[7], 1000.0)

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

    # Load exact pre-solved 8-variable state vector directly from calibration history
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

    homotopy_points = [{"rating": "DP Anchor (100%)", "W_f_target_kg_s": W_f_DP}]
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
        converged_state, _, success = nr_solver.solve(evaluator.evaluate, current_guess)

        if not success:
            print(
                f"[FAILED] {rating}: Solver divergence at W_f = {w_f:.3f} kg/s. Limit breach likely."
            )
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
