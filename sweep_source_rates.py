from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_replicates import ReplicateConfig
from utils.sweep import run_sweep

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"

    base_cfg = ReplicateConfig(
        n_replicates=1,
        particles_per_rep=100_000_000,
        gate=28e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
        base_seed=123456,
    )

    source_rates = [1e4, 3e4, 5e4, 1e5, 2e5, 3e5]

    results = run_sweep(
        input_dir=input_dir,
        output_root=output_root,
        base_cfg=base_cfg,
        param_name="rate",
        values=source_rates,
        label_fmt="rate_{:.0f}",
    )

    # plotting (unchanged logic, just read from results["values"] etc.)
    vals = np.array(results["values"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for idx, ax in enumerate(axes.flat[:4]):
        means = [r[idx] if len(r) > idx else np.nan for r in results["r_mean"]]
        sems = [s[idx] if len(s) > idx else np.nan for s in results["r_sem"]]

        ax.errorbar(
            vals, means, yerr=sems, fmt="o-", capsize=5, capthick=2, markersize=8
        )
        ax.set_xlabel("Source rate (neutrons / s)")
        ax.set_ylabel(f"r[{idx}]")
        ax.grid(True, alpha=0.3)

    # ax = axes[1, 1]
    # ax.errorbar(
    #     vals,
    #     results["detections_mean"],
    #     yerr=np.array(results["detections_std"]) / np.sqrt(base_cfg.n_replicates),
    #     fmt="s-",
    #     capsize=5,
    #     capthick=2,
    #     markersize=8,
    # )
    # ax.set_xlabel("Be radius (cm)")
    # ax.set_ylabel("Detections per replicate")
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/source_rate_sensitivity_results.png", dpi=300)
