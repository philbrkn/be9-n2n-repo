from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import openmc

from utils.config import REPO_ROOT


def list_sweep_points(outputs: Path, pattern: str) -> List[Path]:
    pts = sorted(outputs.glob(pattern))
    pts = [p for p in pts if p.is_dir() and (p / "rep_0000").exists()]
    if not pts:
        raise FileNotFoundError(
            f"No sweep points found under {outputs} matching '{pattern}'"
        )
    return pts


_num_re = re.compile(r"_(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:[A-Za-z]+)?$")


def point_x(p: Path) -> float:
    m = _num_re.search(p.name)
    if not m:
        raise ValueError(f"Could not parse sweep value from: {p.name}")
    return float(m.group(1))


def main() -> None:
    OUTPUT_ROOT = REPO_ROOT / "outputs"

    FIG_DIR = REPO_ROOT / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # PATTERN = "test_n2n_scale_*"
    PATTERN = "test_ddx_scale_*"

    # PARAMETERS
    points = list_sweep_points(OUTPUT_ROOT, PATTERN)
    pairs = sorted(((point_x(p), p) for p in points), key=lambda t: t[0])

    # xs_arr = np.array([x for x, _ in pairs], dtype=float)
    # FOR DDXS:
    xs_arr = np.array([1.31734911, 1.15930768, 1.0, 0.84667103, 0.69024445])
    points = [p for _, p in pairs]

    ml_arr = []
    ml_err_arr = []
    for p in points:
        rep_dirs = sorted([p for p in p.glob("rep_*") if p.is_dir()])
        if not rep_dirs:
            raise FileNotFoundError(f"No replicate dirs found in {p}")

        rep_dir = rep_dirs[0]
        sp = openmc.StatePoint(str(rep_dir / "statepoint.30.h5"))
        # Retrieve the ML tally
        tally_ml = sp.get_tally(name="leakage_multiplication")
        df_ml = tally_ml.get_pandas_dataframe()

        # Extract the mean scores (normalized per source neutron)
        n2n_score = df_ml.loc[df_ml["score"] == "(n,2n)", "mean"].values[0]
        abs_score = df_ml.loc[df_ml["score"] == "absorption", "mean"].values[0]
        n2n_err = df_ml.loc[df_ml["score"] == "(n,2n)", "std. dev."].values[0]
        abs_err = df_ml.loc[df_ml["score"] == "absorption", "std. dev."].values[0]

        # calculate integral leakage multiplication
        M_L = 1.0 + n2n_score - abs_score
        M_L_err = np.sqrt(n2n_err**2 + abs_err**2)  # uncorrelated sum
        # print(f"Be ML for {p}: {M_L:.5f}")

        ml_arr.append(M_L)
        ml_err_arr.append(M_L_err)

    print(xs_arr, ml_arr, ml_err_arr)
    # Fit a line to get sensitivity
    weights = 1.0 / np.array(ml_err_arr) ** 2
    coeffs, cov = np.polyfit(xs_arr, ml_arr, 1, w=np.sqrt(weights), cov=True)
    slope = coeffs[0]
    slope_err = np.sqrt(cov[0, 0])

    # Sensitivity: (dr/r) / (dXS/XS) at nominal
    # choose the middle one:
    i0 = int(np.argmin(np.abs(xs_arr - 1.0)))
    r_nominal = ml_arr[i0]
    # sem_nominal = det_sem[i0]
    x0 = xs_arr[i0]
    S = slope * x0 / r_nominal

    S_err = np.abs(S) * np.sqrt(
        (slope_err / slope) ** 2 + (ml_err_arr[i0] / r_nominal) ** 2
    )
    print(f"    Sensitivity coefficient  = {S:.3f} +/- {S_err:.3f}")
    # print(f"Interpretation: 1% change in (n,2n) causes {S:.2f}% change in r[{idx}]")

    precision = ml_err_arr[i0] / ml_arr[i0]
    constraint = 1 / S * precision
    print(f"    Precision: {precision * 100:.2f}%")
    print(f"    Constraint {constraint * 100:.4f}%")
    residuals = np.array(ml_arr) - np.polyval(coeffs, xs_arr)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(ml_arr) - np.mean(ml_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f"    R² = {r_squared:.4f}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
