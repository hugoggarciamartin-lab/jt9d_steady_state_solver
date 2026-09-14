"""
End-to-End Simulation Pipeline Orchestrator.
Sequentially executes design point calibration, off-design operating lines,
parametric sensitivity sweeps, and professional plotting suites with clear paths output.
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path):
    print(f"\n========================================================")
    print(f"Executing: {script_path.name}")
    print(f"Path: {script_path}")
    print(f"========================================================")

    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)

    if result.returncode != 0:
        print(
            f"\n[FATAL ERROR] Script {script_path.name} failed with exit code {result.returncode}."
        )
        sys.exit(result.returncode)
    else:
        print(f"\n[SUCCESS] {script_path.name} completed successfully.")


def main():
    project_root = Path(__file__).resolve().parent
    scripts_dir = project_root / "scripts"
    sensitivity_dir = project_root / "sensitivity_analysis"

    log_dir = project_root / "output" / "logs"
    plot_dir = project_root / "output" / "plots"

    pipeline_steps = [
        scripts_dir / "01_run_dp_calibration.py",
        scripts_dir / "02_run_off_design_ol.py",
        scripts_dir / "03_plot_off_design.py",
        sensitivity_dir / "run_parametric_sweeps.py",
        sensitivity_dir / "plot_sensitivity_sweeps.py",
    ]

    print("Initiating End-to-End Digital Twin Pipeline Orchestrator (JT9D MoC 2)...")

    for script in pipeline_steps:
        if not script.exists():
            print(f"[ERROR] Required script not found: {script}")
            sys.exit(1)
        run_script(script)

    print("\n========================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY. ARTIFACTS SUMMARY:")
    print("========================================================")
    print(f"  -> Logs and .npz matrices saved in: {log_dir.resolve()}")
    print(f"     - solver_history.npz")
    print(f"     - off_design_ol.npz")
    print(f"     - sensitivity_sweeps.npz")
    print(f"  -> Plots and envelopes saved in: {plot_dir.resolve()}")
    print(
        f"     - operating_lines/ (01_turbomachinery_OL.png, 02_performance_matching.png)"
    )
    print(
        f"     - sensitivity/ (01_multivariable_performance_sensitivity.png, 02_map_operating_lines_shift.png)"
    )
    print("========================================================")


if __name__ == "__main__":
    main()
