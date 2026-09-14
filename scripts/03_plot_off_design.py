"""
Off-Design Operating Line Plotter.
Reconstructs map boundaries via parser queries and overlays the aerodynamic
operating line and global performance metrics against empirical validation targets.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.map_parser import parse_tmats_map


def load_data(filepath: Path):
    if not filepath.exists():
        raise FileNotFoundError(f"Missing artifact: {filepath}")
    return (
        json.load(open(filepath, "r"))
        if filepath.suffix == ".json"
        else np.load(filepath, allow_pickle=True)
    )


def plot_map_background(ax, map_obj, speed_lines, title):
    """Reconstructs the scaled T-MATS map grid by querying the parser."""
    betas = np.linspace(0.01, 0.99, 50)
    surge_line_Wc, surge_line_PR = [], []

    for Nc in speed_lines:
        Wc_line, PR_line = [], []
        for b in betas:
            try:
                Wc, eff, PR, _ = map_obj.evaluate(Nc, b)
                Wc_line.append(Wc)
                PR_line.append(PR)
            except ValueError:
                continue

        if Wc_line:
            ax.plot(Wc_line, PR_line, color="grey", linewidth=0.8, alpha=0.6)
            ax.text(Wc_line[-1], PR_line[-1], f"{Nc:.2f}", fontsize=8, color="dimgrey")
            surge_line_Wc.append(Wc_line[0])
            surge_line_PR.append(PR_line[0])

    if surge_line_Wc:
        ax.plot(
            surge_line_Wc,
            surge_line_PR,
            "r--",
            linewidth=1.5,
            label="Surge Line (Beta bounds)",
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Corrected Mass Flow, Wc [kg/s]")
    ax.set_ylabel("Pressure Ratio, PR [-]")
    ax.grid(True, linestyle=":", alpha=0.7)


def main():
    log_dir = project_root / "output" / "logs"
    map_dir = project_root / "data" / "maps"
    val_dir = project_root / "data" / "telemetry"
    out_dir = project_root / "output" / "plots" / "operating_lines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data Ingestion
    ol_data = load_data(log_dir / "off_design_ol.npz")
    ol_data = load_data(log_dir / "off_design_ol.npz")

    if ol_data["W_f"].size == 0:
        print("[FATAL ERROR] The off_design_ol.npz result matrix is empty.")
        print(
            "The digital twin (script 02) diverged before converging the anchor point."
        )
        sys.exit(1)

    dp_data = load_data(log_dir / "solver_history.npz")
    val_data = load_data(val_dir / "off_design_validation.json")

    sf = dp_data["scaling_factors"].item()

    fan_map = parse_tmats_map(
        str(map_dir / "FAN.map"), True, sf_w=sf["s_W_fan"], sf_pr=sf["s_PR_fan"]
    )
    hpc_map = parse_tmats_map(
        str(map_dir / "HPC.map"), True, sf_w=sf["s_W_hpc"], sf_pr=sf["s_PR_hpc"]
    )

    speed_lines = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05])

    # Plot Turbomachinery Operating Lines
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_map_background(ax1, fan_map, speed_lines, "FAN Operating Line")
    ax1.plot(
        ol_data["Wc_fan"],
        ol_data["PR_fan"],
        "b-o",
        markersize=4,
        linewidth=2,
        label="Steady-State OL",
    )
    ax1.plot(
        ol_data["Wc_fan"][0],
        ol_data["PR_fan"][0],
        "k*",
        markersize=10,
        label="Design Point",
    )
    ax1.legend()

    plot_map_background(ax2, hpc_map, speed_lines, "HPC Operating Line")
    ax2.plot(
        ol_data["Wc_hpc"],
        ol_data["PR_hpc"],
        "g-o",
        markersize=4,
        linewidth=2,
        label="Steady-State OL",
    )
    ax2.plot(
        ol_data["Wc_hpc"][0],
        ol_data["PR_hpc"][0],
        "k*",
        markersize=10,
        label="Design Point",
    )
    ax2.legend()

    plt.tight_layout()
    fig1.savefig(out_dir / "01_turbomachinery_OL.png", dpi=300)

    # Plot Global Performance vs Validation Targets
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    val_Wf = [p["W_f_target_kg_s"] for p in val_data["validation_points"]]
    val_Fn = [p["Fn_target_N"] for p in val_data["validation_points"]]
    val_SFC = [
        p["SFC_target_g_kNs"] for p in val_data["validation_points"]
    ]  # Assuming g/kN.s

    ax3.plot(
        ol_data["W_f"],
        ol_data["Fn"] / 1000.0,
        "b-",
        linewidth=2,
        label="Digital Twin (MoC 2)",
    )
    if val_Fn[0] > 0:
        ax3.scatter(
            val_Wf,
            [f / 1000 for f in val_Fn],
            color="red",
            zorder=5,
            label="Test Cell Targets",
        )
    ax3.set_title("Thrust vs Fuel Flow", fontweight="bold")
    ax3.set_xlabel("Fuel Flow, W_f [kg/s]")
    ax3.set_ylabel("Net Thrust, Fn [kN]")
    ax3.grid(True, linestyle=":")
    ax3.legend()

    # Convert simulated SFC (kg/W.s) to customary g/(kN.s) or similar based on your units
    sim_sfc = ol_data["SFC"] * 1e6
    ax4.plot(
        ol_data["Fn"] / 1000.0, sim_sfc, "g-", linewidth=2, label="Digital Twin (MoC 2)"
    )
    if val_SFC[0] > 0:
        ax4.scatter(
            [f / 1000 for f in val_Fn],
            val_SFC,
            color="red",
            zorder=5,
            label="Test Cell Targets",
        )
    ax4.set_title("TSFC Hook Curve", fontweight="bold")
    ax4.set_xlabel("Net Thrust, Fn [kN]")
    ax4.set_ylabel("TSFC")
    ax4.grid(True, linestyle=":")
    ax4.legend()

    plt.tight_layout()
    fig2.savefig(out_dir / "02_performance_matching.png", dpi=300)

    print(f"Plots successfully exported to: {out_dir}")


if __name__ == "__main__":
    main()
