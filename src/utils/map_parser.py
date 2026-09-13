"""
Turbomachinery map parser and interpolator.
Handles T-MATS ASCII ingestion, legacy Fortran fixed-width token separation,
Imperial to SI conversion, scaling, and C2-continuous splines.
"""

import re
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import RectBivariateSpline


class TurbomachineryMap:
    """
    Evaluator for component maps utilizing C2-continuous bicubic splines.
    Incorporates out-of-bounds quadratic penalty guards to protect solver Jacobians.
    """

    def __init__(
        self,
        nc_grid: np.ndarray,
        beta_grid: np.ndarray,
        wc_data: np.ndarray,
        eff_data: np.ndarray,
        pr_data: Optional[np.ndarray] = None,
    ):
        # Grid boundaries
        self.nc_min, self.nc_max = float(nc_grid[0]), float(nc_grid[-1])
        self.beta_min, self.beta_max = float(beta_grid[0]), float(beta_grid[-1])

        # Pre-compile C2-continuous bicubic splines (kx=3, ky=3)
        self.spline_wc = RectBivariateSpline(nc_grid, beta_grid, wc_data, kx=3, ky=3)
        self.spline_eff = RectBivariateSpline(nc_grid, beta_grid, eff_data, kx=3, ky=3)

        self.spline_pr: Optional[RectBivariateSpline] = None
        if pr_data is not None:
            self.spline_pr = RectBivariateSpline(
                nc_grid, beta_grid, pr_data, kx=3, ky=3
            )

    def evaluate(self, nc: float, beta: float) -> Tuple[float, ...]:
        """
        Evaluates the map. Returns (Wc, Eff, PR, penalty) for compressors,
        or (Wc, Eff, penalty) for turbines.
        """
        penalty = 0.0
        k_penalty = 1e4

        nc_eval = nc
        beta_eval = beta

        # Physical domain guards (clamp and accumulate penalty)
        if nc < self.nc_min:
            penalty += k_penalty * (self.nc_min - nc) ** 2
            nc_eval = self.nc_min
        elif nc > self.nc_max:
            penalty += k_penalty * (nc - self.nc_max) ** 2
            nc_eval = self.nc_max

        if beta < self.beta_min:
            penalty += k_penalty * (self.beta_min - beta) ** 2
            beta_eval = self.beta_min
        elif beta > self.beta_max:
            penalty += k_penalty * (beta - self.beta_max) ** 2
            beta_eval = self.beta_max

        # Query splines
        wc = float(self.spline_wc(nc_eval, beta_eval, grid=False))
        eff = float(self.spline_eff(nc_eval, beta_eval, grid=False))

        # Efficiency physically bounded
        if eff <= 0.0 or eff > 1.0:
            penalty += 1e5
            eff = max(0.01, min(0.99, eff))

        if self.spline_pr is not None:
            pr = float(self.spline_pr(nc_eval, beta_eval, grid=False))
            if pr < 1.0:
                penalty += 1e5
                pr = 1.01
            return wc, eff, pr, penalty

        return wc, eff, penalty


def parse_tmats_map(
    filepath: str,
    is_compressor: bool,
    sf_w: float = 1.0,
    sf_eff: float = 1.0,
    sf_pr: float = 1.0,
) -> TurbomachineryMap:
    """
    Parses a NASA T-MATS ASCII map using a robust flattened stream protocol.
    Applies regex sanitization for legacy Fortran fixed-width truncations,
    0.45359237 kg/lbm conversion, and Design Point scaling factors.
    """
    with open(filepath, "r") as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("!")
        ]

    dims = lines[0].split()
    n_cols, n_speeds = int(dims[0]), int(dims[1])

    raw_data = []
    for line_num, line in enumerate(lines[1:], start=2):
        clean_line = line.split("!")[0].split("#")[0]

        # 1. Separar floats pegados por el signo menos (ej: "1.5-2.0" -> "1.5 -2.0")
        clean_line = re.sub(r"(?<![eE\s])-", " -", clean_line)

        # 2. Separar floats decimales fusionados por falta de ancho (ej: "0.91370.9113" -> "0.9137 0.9113")
        # Busca una parte decimal seguida inmediatamente por un dígito y otro punto decimal.
        clean_line = re.sub(r"(\.\d+?)(?=\d\.)", r"\1 ", clean_line)

        for token in clean_line.split():
            try:
                raw_data.append(float(token))
            except ValueError:
                raise ValueError(
                    f"Artefacto corrupto imposible de parsear en la línea {line_num}: '{token}'. "
                    f"Archivo: {filepath}"
                )

    nc_grid = np.zeros(n_speeds)
    wc_data = np.zeros((n_speeds, n_cols))
    eff_data = np.zeros((n_speeds, n_cols))
    pr_data: Optional[np.ndarray] = (
        np.zeros((n_speeds, n_cols)) if is_compressor else None
    )

    lbm_to_kg = 0.45359237
    ptr = 0

    beta_grid = np.array(raw_data[ptr : ptr + n_cols], dtype=np.float64)
    ptr += n_cols

    for i in range(n_speeds):
        nc_grid[i] = raw_data[ptr] / 100.0
        ptr += 1

        wc_data[i, :] = (
            np.array(raw_data[ptr : ptr + n_cols], dtype=np.float64) * lbm_to_kg * sf_w
        )
        ptr += n_cols

        eff_data[i, :] = (
            np.array(raw_data[ptr : ptr + n_cols], dtype=np.float64) * sf_eff
        )
        ptr += n_cols

        if is_compressor and pr_data is not None:
            raw_pr = np.array(raw_data[ptr : ptr + n_cols], dtype=np.float64)
            pr_data[i, :] = 1.0 + (raw_pr - 1.0) * sf_pr
            ptr += n_cols

    return TurbomachineryMap(nc_grid, beta_grid, wc_data, eff_data, pr_data)
