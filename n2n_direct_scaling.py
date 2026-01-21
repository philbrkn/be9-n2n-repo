import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.data

from run_independent_sims import run_independent_replicates

N_REPS = 10
N_PARTICLES = 100000
# SR parameters
GATE = 32e-6
PREDELAY = 4e-6
DELAY = 1000e-6

be9_path = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
xml_path = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"

base_dir = Path(__file__).parent.resolve()
input_dir = base_dir / "inputs"
output_dir = base_dir / "outputs"
scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
# scale_factors = [0.8]

# Store results for plotting
results = {
    "scale": [],
    "detections_mean": [],
    "detections_std": [],
    "r_mean": [],
    "r_sem": [],
}

for scale in scale_factors:
    be9 = openmc.data.IncidentNeutron.from_hdf5(be9_path)
    # We iterate over the temperature dictionaries in the cross section data
    for temp in be9.reactions[16].xs:
        be9.reactions[16].xs[temp].y *= scale
    # Export modified hdf5
    h5_path = input_dir / f"Be9_scaled_{scale}.h5"
    be9.export_to_hdf5(h5_path, mode="w")
    library = openmc.data.DataLibrary.from_xml(xml_path)
    # if Be9 already exists, we should remove the old reference first
    library.remove_by_material("Be9")
    library.register_file(h5_path.resolve())

    scaled_XS_path = f"{input_dir}/cross_sections_scaled.xml"
    library.export_to_xml(scaled_XS_path)
    os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(scaled_XS_path)

    # RUN IT:
    r_mean, r_std, r_sem, all_r, all_det = run_independent_replicates(
        n_replicates=N_REPS,
        particles_per_rep=N_PARTICLES,
        gate=GATE,
        predelay=PREDELAY,
        delay=DELAY,
        input_path=input_dir,
        output_path=output_dir,
        # base_seed=None,
        # base_seed=12346,
    )

    # Store results
    results["scale"].append(scale)
    results["detections_mean"].append(np.mean(all_det))
    results["detections_std"].append(np.std(all_det))
    results["r_mean"].append(r_mean.copy())
    results["r_sem"].append(r_sem.copy())

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({N_REPS} independent replicates) for scale {scale}")
    print("=" * 60)
    print(f"Mean detections per run: {np.mean(all_det):.0f} +/- {np.std(all_det):.0f}")
    print(f"\n{'idx':>3} | {'mean':>12} | {'std':>12} | {'SEM':>12}")
    print("-" * 50)
    for i in range(min(5, len(r_mean))):
        print(f"{i:>3} | {r_mean[i]:>12.6f} | {r_std[i]:>12.6f} | {r_sem[i]:>12.6f}")


# === PLOTTING ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

scales = np.array(results["scale"])

# Plot r[0], r[1], r[2]
for idx, ax in enumerate(axes.flat[:3]):
    means = [r[idx] for r in results["r_mean"]]
    sems = [r[idx] for r in results["r_sem"]]

    ax.errorbar(scales, means, yerr=sems, fmt="o-", capsize=5, capthick=2, markersize=8)
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
