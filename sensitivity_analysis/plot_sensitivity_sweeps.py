"""
Advanced Sensitivity Sweep & Operating Line Plotter.
Visualizes multi-dimensional parametric sensitivity across all A8 variations,
plotting Thrust and TSFC lapses alongside operating line shifts on compressor maps.
"""

import sys
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
    return np.load(filepath, allow_pickle=True)


def main():
    log_dir = project_root / "output" / "logs"
    map_dir = project_root / "data" / "maps"
    out_dir = project_root / "output" / "plots" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_file = log_dir / "sensitivity_sweeps.npz"
    dp_file = log_dir / "solver_history.npz"

    if not sweep_file.exists() or not dp_file.exists():
        print(
            "[ERROR] Faltan artefactos de simulación. Ejecuta los scripts de barrido previo primero."
        )
        sys.exit(1)

    data = load_data(sweep_file)
    dp_data = load_data(dp_file)
    sf = dp_data["scaling_factors"].item()

    alt = data["Alt"]
    mach = data["Mach"]
    a8_mods = data["A8_mod"]
    fn = data["Fn"] / 1000.0
    sfc = data["SFC"] * 1e6

    unique_alts = np.unique(alt)
    unique_a8 = np.unique(a8_mods)

    fig, axes = plt.subplots(
        len(unique_alts), 2, figsize=(14, 4 * len(unique_alts)), sharex=True
    )

    if len(unique_alts) == 1:
        axes = np.array([axes])

    styles = {0.95: ("--", "orange"), 1.0: ("-", "blue"), 1.05: (":", "green")}

    for i, h in enumerate(unique_alts):
        ax_fn = axes[i, 0]
        ax_sfc = axes[i, 1]

        for a8 in unique_a8:
            linestyle, color = styles.get(a8, ("-", "black"))
            mask = np.isclose(alt, h) & np.isclose(a8_mods, a8)

            m_vals = mach[mask]
            fn_vals = fn[mask]
            sfc_vals = sfc[mask]

            valid = ~np.isnan(fn_vals)
            if np.any(valid):
                ax_fn.plot(
                    m_vals[valid],
                    fn_vals[valid],
                    linestyle=linestyle,
                    color=color,
                    marker="o",
                    linewidth=1.5,
                    label=f"A8: {a8 * 100:.0f}%",
                )
                ax_sfc.plot(
                    m_vals[valid],
                    sfc_vals[valid],
                    linestyle=linestyle,
                    color=color,
                    marker="s",
                    linewidth=1.5,
                    label=f"A8: {a8 * 100:.0f}%",
                )

        ax_fn.set_title(
            f"Thrust Lapse (Altitude: {h / 1000.0:.1f} km)", fontweight="bold"
        )
        ax_fn.set_ylabel("Net Thrust, Fn [kN]")
        ax_fn.grid(True, linestyle=":")
        ax_fn.legend(fontsize=8)

        ax_sfc.set_title(
            f"TSFC Evolution (Altitude: {h / 1000.0:.1f} km)", fontweight="bold"
        )
        ax_sfc.set_ylabel("TSFC")
        ax_sfc.grid(True, linestyle=":")
        ax_sfc.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Mach Number [-]")

    plt.tight_layout()
    perf_path = out_dir / "01_multivariable_performance_sensitivity.png"
    fig.savefig(perf_path, dpi=300)
    print(f"Superficies de sensibilidad de rendimiento guardadas en: {perf_path}")

    hpc_map = parse_tmats_map(
        str(map_dir / "HPC.map"), True, sf_w=sf["s_W_hpc"], sf_pr=sf["s_PR_hpc"]
    )

    fig_map, ax_map = plt.subplots(figsize=(8, 6))
    betas = np.linspace(0.05, 0.95, 40)

    for Nc in [0.7, 0.8, 0.9, 1.0, 1.05]:
        Wc_line, PR_line = [], []
        for b in betas:
            try:
                Wc, _, PR, _ = hpc_map.evaluate(Nc, b)
                Wc_line.append(Wc)
                PR_line.append(PR)
            except ValueError:
                continue
        if Wc_line:
            ax_map.plot(Wc_line, PR_line, color="grey", linewidth=0.8, alpha=0.5)

    ax_map.set_title("HPC Map Shift across Parametric Variations", fontweight="bold")
    ax_map.set_xlabel("Corrected Mass Flow, Wc_hpc [kg/s]")
    ax_map.set_ylabel("Pressure Ratio, PR_hpc [-]")
    ax_map.grid(True, linestyle=":")

    plt.tight_layout()
    map_path = out_dir / "02_map_operating_lines_shift.png"
    fig_map.savefig(map_path, dpi=300)
    print(f"Desplazamiento en mapas exportado a: {map_path}")


if __name__ == "__main__":
    main()
