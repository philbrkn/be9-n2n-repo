import time
from pathlib import Path

import numpy as np

from bootstrap import analyze_with_bootstrap_parallel
from utils.analyze_sr import SRParams


def cov_frac_from_labels(r_boot, det, labels):
    cols = []
    for lab in labels:
        if lab == "det":
            x = det
        else:  # "r0","r1","r2",...
            k = int(lab[1:])
            x = r_boot[:, k]
        cols.append((x - x.mean()) / x.mean())
    X = np.column_stack(cols)  # (B, n_obs)
    return np.cov(X, rowvar=False)  # (n_obs, n_obs)


def calculate_covariance(col_track_file, labels=("r1", "det"), use_cache=True):
    GATE, PREDELAY, DELAY = 28e-6, 4e-6, 1000e-6
    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)

    if use_cache and cache.exists():
        z = np.load(cache)
        r_boot, det = z["r_boot"], z["det"]
    else:
        t0 = time.perf_counter()
        r_boot, det = analyze_with_bootstrap_parallel(
            col_track_file,
            sr,
            n_bootstrap=500,
            segment_duration=1.0,
            seed=12345,
            n_workers=32,
            chunk_size=1,
            return_samples=True,
        )
        print(f"Bootstrapping took {time.perf_counter() - t0:.3f}s")
        np.savez_compressed(cache, r_boot=r_boot, det=det)

    return cov_frac_from_labels(r_boot, det, labels)


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    output_root = base_dir / "outputs"
    cache = base_dir / "cache_nominal.npz"
    col_track_file = (
        output_root
        / "be_rad_n2n_sweep_7.0"
        / "n2n_scale_1.00"
        / "rep_0000"
        / "collision_track.h5"
    )

    # S = {
    #     "sigma_n2n": {"r0": -0.027, "r1": 0.504, "r2": 1.189, "det": 0.627},
    #     "ddx": {"r0": -0.026, "r1": 0.478, "r2": 1.358, "det": 0.143},
    # }
    # for 7 cm:
    S = {
        "sigma_n2n": {"r0": -0.031, "r1": 0.522, "r2": 1.544, "det": 0.650},
        "ddx": {"r0": -0.022, "r1": 0.359, "r2": 1.460, "det": 0.043},
    }

    def build_J(labels, params=("sigma_n2n", "ddx"), S=S):
        # rows = params, cols = observables in `labels`
        return np.array([[S[p][lab] for lab in labels] for p in params], dtype=float)

    labels = ("r0", "r1", "r2", "det")  # or ("r1","det")
    Sigma = calculate_covariance(col_track_file, labels=labels, use_cache=True)
    print("Sigma", Sigma)
    J4 = build_J(labels)
    F = J4 @ np.linalg.inv(Sigma) @ J4.T
    Cov_params = np.linalg.inv(F)
    print("Covariance params", Cov_params)
    rho = Cov_params[0, 1] / np.sqrt(Cov_params[0, 0] * Cov_params[1, 1])
    print(f"rho {rho:.4f}")

    labels = ("r0", "r1", "det")  # or ("r1","det")
    J2 = build_J(labels)
    Sigma = calculate_covariance(col_track_file, labels=labels, use_cache=True)
    print("Sigma", Sigma)
    F = J2 @ np.linalg.inv(Sigma) @ J2.T
    Cov_params = np.linalg.inv(F)
    print("Covariance params", Cov_params)
    rho = Cov_params[0, 1] / np.sqrt(Cov_params[0, 0] * Cov_params[1, 1])
    print(f"rho {rho:.4f}")
