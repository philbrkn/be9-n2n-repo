# analyze_sweep_n2n_collision_files.py
# results of sweep_n2n_scale.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_replicates import ReplicateConfig, analyze_independent_replicates

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"

    # --- user parameters ---
    N_REPS = 10
    N_PARTICLES = 1_000_000

    GATE = 32e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    # scales = [0.0003, 0.0004, 0.0005, 0.0006, 0.0007]  # g/cm³

    # Replicate runner config
    cfg = ReplicateConfig(
        n_replicates=N_REPS,
        particles_per_rep=N_PARTICLES,
        gate=GATE,
        predelay=PREDELAY,
        delay=DELAY,
        rate=3e4,
        base_seed=12345,  # fix for comparability; set None for random
    )

    results = {
        "scale": [],
        "detections_mean": [],
        "detections_std": [],
        "r_mean": [],
        "r_sem": [],
    }

    for scale in scales:
        # Create a per-scale folder so scale points don't overwrite each other
        scale_dir = output_root / f"scale_{scale:.2f}"
        scale_dir.mkdir(parents=True, exist_ok=True)

        # Run replicates into outputs/scale_1.10/rep_XXXX/
        r_mean, r_std, r_sem, all_r, all_det = analyze_independent_replicates(
            input_dir=input_dir,
            output_root=scale_dir,
            cfg=cfg,
        )

        results["scale"].append(scale)
        results["detections_mean"].append(float(np.mean(all_det)))
        results["detections_std"].append(float(np.std(all_det)))
        results["r_mean"].append(r_mean.copy())
        results["r_sem"].append(r_sem.copy())

        print(
            f"[scale {scale:.2f}] mean detections: {np.mean(all_det):.1f}  std: {np.std(all_det):.1f}"
        )

    # === PLOTTING ===
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    scales_arr = np.array(results["scale"], dtype=float)

    # Plot r[0], r[1], r[2]
    for idx, ax in enumerate(axes.flat[:3]):
        means = [r[idx] for r in results["r_mean"]]
        sems = [r[idx] for r in results["r_sem"]]

        ax.errorbar(
            scales, means, yerr=sems, fmt="o-", capsize=5, capthick=2, markersize=8
        )
        ax.set_xlabel("(n,2n) Scale Factor")
        ax.set_ylabel(f"r[{idx}]")
        ax.set_title(f"r[{idx}] vs (n,2n) Cross Section")
        ax.grid(True, alpha=0.3)

    # Plot detections
    ax = axes[1, 1]
    ax.errorbar(
        scales,
        results["detections_mean"],
        yerr=np.array(results["detections_std"]) / np.sqrt(N_REPS),  # SEM
        fmt="s-",
        capsize=5,
        capthick=2,
        markersize=8,
        color="green",
    )
    ax.set_xlabel("(n,2n) Scale Factor")
    ax.set_ylabel("Detections per 100k source")
    ax.set_title("Detection Count vs (n,2n) Cross Section")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/sensitivity_results.png", dpi=300)

    print("\nPlot saved to figures/sensitivity_results.png")

    # === SENSITIVITY ANALYSIS === #
    r1_means = [r[1] for r in results["r_mean"]]
    r1_sems = [r[1] for r in results["r_sem"]]

    # Fit a line to get sensitivity
    coeffs = np.polyfit(scales, r1_means, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Sensitivity: (dr/r) / (dXS/XS) at nominal
    r_nominal = r1_means[2]  # this is where scale=1.0
    sem_nominal = r1_sems[2]
    S = slope * scales[2] / r_nominal  # dr/dXS normalized

    print(f"Sensitivity coefficient S = {S:.3f}")
    print(f"Interpretation: 1% change in (n,2n) causes {S:.2f}% change in r[1]")
    # invert for constraint:
    # if we measure r[1] with precision delr[1] we constrain XS(n,2n) to:
    # deltaXS / XS = (1/S) * (deltar[1]/r[1])
    precision = sem_nominal / r_nominal
    constraint = 1 / S * precision
    print(
        f"A shift register measurement of r[1] with {precision * 100:.2f}% precision"
        f" (achievable with {N_REPS} replicates of {N_PARTICLES} particles) "
        f"can constrain the Be-9 (n,2n) cross section to ±{constraint * 100:.4f}%."
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        scales,
        r1_means,
        yerr=r1_sems,
        fmt="o",
        capsize=5,
        markersize=8,
        color="blue",
        label="Simulation data",
        zorder=3,
    )
    ax.plot(
        scales,
        np.polyval(coeffs, scales),
        "r--",
        linewidth=2,
        label=f"Linear fit (slope={slope:.4f})",
        zorder=2,
    )
    # INVERSE BAND  #
    # Horizontal band: r[1] measurement with uncertainty
    r_low = r_nominal - sem_nominal
    r_high = r_nominal + sem_nominal
    # Map to scale via inverse: scale = (r[1] - intercept) / slope
    scale_low = (r_low - intercept) / slope
    scale_high = (r_high - intercept) / slope
    # Shade the horizontal band (measurement uncertainty on r[1])
    ax.axhspan(
        r_low,
        r_high,
        alpha=0.3,
        color="green",
        label=f"r[1] = {r_nominal:.4f} ± {sem_nominal:.4f}",
    )
    # Shade the vertical band (implied constraint on scale)
    ax.axvspan(
        scale_low,
        scale_high,
        alpha=0.2,
        color="orange",
        label=f"Scale = 1.00 ± {abs(scale_high - 1.0):.3f}",
    )

    ax.set_xlabel("(n,2n) Cross Section Scale Factor", fontsize=12)
    ax.set_ylabel("r[1] (Doubles Rate)", fontsize=12)
    ax.set_title(
        "Sensitivity of Doubles Rate to Be-9 (n,2n) Cross Section", fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/r1_sensitivity.png", dpi=300)
    plt.show()
