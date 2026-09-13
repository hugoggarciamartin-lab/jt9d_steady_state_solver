"""
Turbomachinery Map Visual Verification.
Renders aerodynamic performance charts with corrected boundary definitions
and complete isentropic efficiency contour topologies.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.map_parser import parse_tmats_map

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def plot_compressor_map(comp_map, component_name: str, output_dir: Path) -> None:
    """
    Renders a compressor/fan map with corrected physical boundary assignments:
    - Left/Upper boundary: Surge Limit Line (SLL)
    - Right/Lower boundary: Choke Boundary
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    nc_vals = np.linspace(comp_map.nc_min, comp_map.nc_max, 25)
    beta_vals = np.linspace(comp_map.beta_min, comp_map.beta_max, 45)

    Wc_grid = np.zeros((len(nc_vals), len(beta_vals)))
    PR_grid = np.zeros((len(nc_vals), len(beta_vals)))
    Eff_grid = np.zeros((len(nc_vals), len(beta_vals)))

    for i, nc in enumerate(nc_vals):
        for j, beta in enumerate(beta_vals):
            wc, eff, pr, _ = comp_map.evaluate(nc, beta)
            Wc_grid[i, j] = wc
            PR_grid[i, j] = pr
            Eff_grid[i, j] = eff

    # Efficiency background contour
    levels = np.linspace(np.min(Eff_grid), np.max(Eff_grid), 20)
    contour = ax.contourf(
        Wc_grid, PR_grid, Eff_grid, levels=levels, cmap="viridis", alpha=0.85
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Isentropic Efficiency [-]")

    # Constant corrected speed lines
    for i in range(len(nc_vals)):
        ax.plot(Wc_grid[i, :], PR_grid[i, :], color="white", linewidth=0.9, alpha=0.7)
        ax.text(
            Wc_grid[i, 0],
            PR_grid[i, 0],
            f"{nc_vals[i]:.2f}",
            color="black",
            fontsize=8,
            ha="right",
            va="bottom",
        )

    # Physical boundary assignment:
    # beta_min represents the stall/surge boundary (lowest flow, highest PR for a given speed)
    # beta_max represents the choke boundary (maximum swallowing capacity, vertical drop)
    surge_wc = Wc_grid[:, 0]
    surge_pr = PR_grid[:, 0]
    ax.plot(
        surge_wc, surge_pr, color="red", linewidth=2.5, label="Surge Limit Line (SLL)"
    )

    choke_wc = Wc_grid[:, -1]
    choke_pr = PR_grid[:, -1]
    ax.plot(
        choke_wc,
        choke_pr,
        color="darkorange",
        linewidth=2.0,
        linestyle="--",
        label="Choke Boundary",
    )

    ax.set_title(f"{component_name} Aerodynamic Performance Map")
    ax.set_xlabel(r"Corrected Mass Flow, $W_c$ [kg/s]")
    ax.set_ylabel(r"Total Pressure Ratio, $\pi$ [-]")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / f"{component_name}_Base_Map.png", dpi=300)
    plt.close()


def plot_turbine_map(turb_map, component_name: str, output_dir: Path) -> None:
    """
    Renders a turbine swallowing capacity map with an isentropic efficiency contour surface.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    nc_vals = np.linspace(turb_map.nc_min, turb_map.nc_max, 25)
    # In T-MATS turbine maps, the auxiliary coordinate represents the expansion ratio
    pr_vals = np.linspace(turb_map.beta_min, turb_map.beta_max, 45)

    Wc_grid = np.zeros((len(nc_vals), len(pr_vals)))
    PR_grid = np.zeros((len(nc_vals), len(pr_vals)))
    Eff_grid = np.zeros((len(nc_vals), len(pr_vals)))

    for i, nc in enumerate(nc_vals):
        for j, pr in enumerate(pr_vals):
            wc, eff, _ = turb_map.evaluate(nc, pr)
            Wc_grid[i, j] = wc
            PR_grid[i, j] = pr
            Eff_grid[i, j] = eff

    # Efficiency background contour
    levels = np.linspace(np.min(Eff_grid), np.max(Eff_grid), 20)
    contour = ax.contourf(
        PR_grid, Wc_grid, Eff_grid, levels=levels, cmap="viridis", alpha=0.85
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Isentropic Efficiency [-]")

    # Constant corrected speed lines
    for i in range(len(nc_vals)):
        ax.plot(PR_grid[i, :], Wc_grid[i, :], color="white", linewidth=1.0, alpha=0.7)
        ax.text(
            PR_grid[i, -1],
            Wc_grid[i, -1],
            f"Nc={nc_vals[i]:.2f}",
            color="black",
            fontsize=8,
            ha="left",
            va="center",
        )

    ax.set_title(f"{component_name} Aerodynamic Performance Map")
    ax.set_xlabel(r"Expansion Pressure Ratio, $\pi_t$ [-]")
    ax.set_ylabel(r"Corrected Mass Flow, $W_c$ [kg/s]")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(output_dir / f"{component_name}_Base_Map.png", dpi=300)
    plt.close()


def main():
    map_dir = project_root / "data" / "maps"
    output_dir = project_root / "output" / "plots" / "base_maps"

    output_dir.mkdir(parents=True, exist_ok=True)

    components = [
        ("FAN.map", "Fan", True),
        ("LPC.map", "LPC", True),
        ("HPC.map", "HPC", True),
        ("HPT1.map", "HPT", False),
        ("LPT.map", "LPT", False),
    ]

    for filename, name, is_comp in components:
        filepath = map_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found in {map_dir}. Skipping.")
            continue

        print(f"Processing and rendering {name} map...")
        parsed_map = parse_tmats_map(str(filepath), is_compressor=is_comp)

        if is_comp:
            plot_compressor_map(parsed_map, name, output_dir)
        else:
            plot_turbine_map(parsed_map, name, output_dir)

    print(f"Visual verification complete. Plots saved to {output_dir}.")


if __name__ == "__main__":
    main()
