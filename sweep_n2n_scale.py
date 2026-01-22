# sweep_n2n_scale.py

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.data

from run_replicates import ReplicateConfig, run_independent_replicates


def scale_be9_n2n(
    be9_path: str, xml_path: str, out_xml_path: Path, scaled_h5_path: Path, scale: float
) -> None:
    be9 = openmc.data.IncidentNeutron.from_hdf5(be9_path)
    for temp in be9.reactions[16].xs:
        be9.reactions[16].xs[temp].y *= scale
    be9.export_to_hdf5(scaled_h5_path, mode="w")
    library = openmc.data.DataLibrary.from_xml(xml_path)
    library.remove_by_material("Be9")
    library.register_file(scaled_h5_path.resolve())
    library.export_to_xml(str(out_xml_path))


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"

    be9_path = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
    xml_path = (
        "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
    )

    # --- user parameters ---
    N_REPS = 10
    N_PARTICLES = 1_000_000

    GATE = 32e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]

    # Replicate runner config
    cfg = ReplicateConfig(
        n_replicates=N_REPS,
        particles_per_rep=N_PARTICLES,
        gate=GATE,
        predelay=PREDELAY,
        delay=DELAY,
        rate=3e4,
        base_seed=123456,  # fix for comparability; set None for random
    )

    results = {
        "scale": [],
        "detections_mean": [],
        "detections_std": [],
        "r_mean": [],
        "r_sem": [],
    }

    for scale in scales:
        openmc.reset_auto_ids()

        # Create a per-scale folder so scale points don't overwrite each other
        scale_dir = output_root / f"scale_{scale:.2f}"
        scale_dir.mkdir(parents=True, exist_ok=True)

        scaled_h5_path = input_dir / f"Be9_scaled_{scale:.2f}.h5"
        scaled_xml_path = input_dir / "cross_sections_scaled.xml"

        scale_be9_n2n(
            be9_path=be9_path,
            xml_path=xml_path,
            out_xml_path=scaled_xml_path,
            scaled_h5_path=scaled_h5_path,
            scale=scale,
        )

        os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(str(scaled_xml_path))

        # Run replicates into outputs/scale_1.10/rep_XXXX/
        r_mean, r_std, r_sem, all_r, all_det = run_independent_replicates(
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
