# sweep_He_density.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.data

from run_replicates import ReplicateConfig, run_independent_replicates

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
    RATE = 3e4

    densities = [0.0003, 0.0004, 0.0005, 0.0006, 0.0007]  # g/cm³

    results = {
        "scale": [],
        "detections_mean": [],
        "detections_std": [],
        "r_mean": [],
        "r_sem": [],
    }

    for density in densities:
        openmc.reset_auto_ids()

        # Replicate runner config
        cfg = ReplicateConfig(
            n_replicates=N_REPS,
            particles_per_rep=N_PARTICLES,
            gate=GATE,
            predelay=PREDELAY,
            delay=DELAY,
            rate=RATE,
            base_seed=123456,  # fix for comparability; set None for random
            he_density=density,
        )

        # Run replicates into outputs/scale_1.10/rep_XXXX/
        r_mean, r_std, r_sem, all_r, all_det = run_independent_replicates(
            input_dir=input_dir,
            output_root=output_root,
            cfg=cfg,
        )

        results["scale"].append(density)
        results["detections_mean"].append(float(np.mean(all_det)))
        results["detections_std"].append(float(np.std(all_det)))
        results["r_mean"].append(r_mean.copy())
        results["r_sem"].append(r_sem.copy())

        print(
            f"[scale {density:.2f}] mean detections: {np.mean(all_det):.1f}  std: {np.std(all_det):.1f}"
        )

    # === PLOTTING ===
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    scales_arr = np.array(results["scale"], dtype=float)

    # Plot r[0], r[1], r[2]
    for idx, ax in enumerate(axes.flat[:3]):
        means = [r[idx] for r in results["r_mean"]]
        sems = [r[idx] for r in results["r_sem"]]

        ax.errorbar(
            densities,
            means,
            yerr=sems,
            fmt="o-",
            capsize=5,
            capthick=2,
            markersize=8,
        )
        ax.set_xlabel("he densities")
        ax.set_ylabel(f"r[{idx}]")
        ax.set_title(f"r[{idx}] vs densities")
        ax.grid(True, alpha=0.3)

    # Plot detections
    ax = axes[1, 1]
    ax.errorbar(
        densities,
        results["detections_mean"],
        yerr=np.array(results["detections_std"]) / np.sqrt(N_REPS),  # SEM
        fmt="s-",
        capsize=5,
        capthick=2,
        markersize=8,
        color="green",
    )

    ax.set_xlabel("he densities")
    ax.set_ylabel("Detections per 100k source")
    ax.set_title("Detection Count vs source rate")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/he_density_sensitivity_results.png", dpi=300)

    print("\nPlot saved to figures/sensitivity_results.png")
