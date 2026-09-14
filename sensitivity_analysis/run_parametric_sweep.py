"""
Parametric Sweep Engine.
Executes multi-dimensional permutations of Mach, Altitude, and Nozzle Areas.
"""

import sys
import json
import numpy as np
from pathlib import Path
import itertools

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.numerical.newton_raphson import NewtonRaphsonSolver
from src.utils.gas_properties import GasProperties
from src.utils.map_parser import parse_tmats_map
from src.components.fan import Fan
from src.components.compressor import Compressor
from src.components.combustor import Combustor
from src.components.turbine import Turbine
from src.components.nozzle import Nozzle
from src.components.shaft import Shaft
from src.numerical.evaluators import OffDesignEvaluator


def get_isa_conditions(alt_m: float, mach: float) -> tuple:
    T0 = 288.15
    P0 = 101325.0
    L = 0.0065
    R = 287.05
    g = 9.80665
    gamma = 1.4

    if alt_m < 11000.0:
        T_amb = T0 - L * alt_m
        P_amb = P0 * (1.0 - (L * alt_m) / T0) ** (g / (L * R))
    else:
        T_amb = 216.65
        P_11k = P0 * (1.0 - (L * 11000.0) / T0) ** (g / (L * R))
        P_amb = P_11k * np.exp((-g / (R * T_amb)) * (alt_m - 11000.0))

    Tt2 = T_amb * (1.0 + ((gamma - 1.0) / 2.0) * mach**2)
    Pt2 = P_amb * (1.0 + ((gamma - 1.0) / 2.0) * mach**2) ** (gamma / (gamma - 1.0))
    return P_amb, Pt2, Tt2


def main():
    config_dir = project_root / "data" / "config"
    log_dir = project_root / "output" / "logs"
    map_dir = project_root / "data" / "maps"
    out_dir = project_root / "output" / "logs"

    dp_data = np.load(log_dir / "solver_history.npz", allow_pickle=True)
    engine_specs = json.load(open(config_dir / "engine_specs.json", "r"))

    if len(dp_data["state_vector"]) == 0:
        print("[FATAL ERROR] DP Baseline missing. Calibrate engine first.")
        sys.exit(1)

    sf = dp_data["scaling_factors"].item()
    u_vec = dp_data["u_vector"].item()
    gas = GasProperties()

    components = {
        "fan": Fan(
            parse_tmats_map(
                str(map_dir / "FAN.map"),
                True,
                sf_w=sf.get("s_W_fan", 1),
                sf_pr=sf.get("s_PR_fan", 1),
            ),
            gas,
        ),
        "lpc": Compressor(
            parse_tmats_map(
                str(map_dir / "LPC.map"),
                True,
                sf_w=sf.get("s_W_lpc", 1),
                sf_pr=sf.get("s_PR_lpc", 1),
            ),
            gas,
        ),
        "hpc": Compressor(
            parse_tmats_map(
                str(map_dir / "HPC.map"),
                True,
                sf_w=sf.get("s_W_hpc", 1),
                sf_pr=sf.get("s_PR_hpc", 1),
            ),
            gas,
        ),
        "comb": Combustor(gas, k1=u_vec["k1"], k2=u_vec["k2"]),
        "hpt": Turbine(
            parse_tmats_map(
                str(map_dir / "HPT1.map"),
                False,
                sf_w=sf.get("s_W_hpt", 1),
                sf_pr=sf.get("s_PR_hpt", 1),
            ),
            gas,
        ),
        "lpt": Turbine(
            parse_tmats_map(
                str(map_dir / "LPT.map"),
                False,
                sf_w=sf.get("s_W_lpt", 1),
                sf_pr=sf.get("s_PR_lpt", 1),
            ),
            gas,
        ),
        "core_noz": Nozzle(gas),
        "byp_noz": Nozzle(gas),
        "hp_shaft": Shaft(
            power_ext=engine_specs["mechanical"]["power_accessory_base_W"]
        ),
        "lp_shaft": Shaft(),
    }

    nr_solver = NewtonRaphsonSolver(
        max_iters=40, tolerance=1e-4, fd_eps=1e-2, max_step_frac=0.05
    )
    base_state = dp_data["state_vector"]
    base_Wf = dp_data["dp_magnitudes"].item()["W_f_physical"]

    altitudes = np.array([0.0, 5000.0, 10000.0])
    machs = np.array([0.0, 0.4, 0.8])
    a8_variations = np.array([0.95, 1.0, 1.05])

    results = {
        "Alt": [],
        "Mach": [],
        "A8_mod": [],
        "Fn": [],
        "SFC": [],
        "EGT": [],
        "success": [],
    }
    print("Initiating CS-E 500 Parametric Operability Sweep...")

    for alt, mach, a8_mod in itertools.product(altitudes, machs, a8_variations):
        P_amb, Pt2, Tt2 = get_isa_conditions(alt, mach)
        ambient_cond = {"Pt": Pt2, "Tt": Tt2, "P_amb": P_amb}

        current_specs = engine_specs.copy()
        current_specs["geometry"]["A8_m2"] *= a8_mod

        evaluator = OffDesignEvaluator(
            components, ambient_cond, current_specs, dp_data, base_Wf
        )
        converged_state, _, success = nr_solver.solve(
            evaluator.evaluate, base_state.copy()
        )

        results["Alt"].append(alt)
        results["Mach"].append(mach)
        results["A8_mod"].append(a8_mod)

        if success:
            results["Fn"].append(evaluator.Fn)
            results["SFC"].append(evaluator.SFC)
            results["EGT"].append(evaluator.EGT)
            results["success"].append(True)
            print(
                f"  [OK] Alt: {alt:5.0f}m | M: {mach:.1f} | A8: {a8_mod * 100:3.0f}% -> Fn: {evaluator.Fn / 1000:.1f}kN"
            )
        else:
            results["Fn"].append(np.nan)
            results["SFC"].append(np.nan)
            results["EGT"].append(np.nan)
            results["success"].append(False)
            print(
                f"  [FAIL] Alt: {alt:5.0f}m | M: {mach:.1f} | A8: {a8_mod * 100:3.0f}% -> Diverged."
            )

    np.savez(out_dir / "sensitivity_sweeps.npz", **results)
    print(
        "\nSweep matrix saved. Note: Failures are padded with NaN to preserve tensor dimensions."
    )


if __name__ == "__main__":
    main()
