import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc

from run_replicates import ReplicateConfig, run_independent_replicates

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,  # better for multi-panel
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "figure.dpi": 300,
    }
)

if __name__ == "__main__":
    BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
    XML_PATH = (
        "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
    )
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs" / "tendl"

    tendl_h5_path = Path("Be9_TENDL_raw.h5")
    out_xml_path = Path("cross_section_TENDL.xml")

    library = openmc.data.DataLibrary.from_xml(XML_PATH)
    library.remove_by_material("Be9")
    library.register_file(tendl_h5_path.resolve())
    library.export_to_xml(str(out_xml_path))

    os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(str(out_xml_path))

    cfg = ReplicateConfig(
        n_replicates=1,
        particles_per_rep=100_000_000,
        base_seed=12346,
        gate=28e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
    )

    r_mean, r_std, r_sem, all_r, all_det = run_independent_replicates(
        input_dir=input_dir,
        output_root=output_root,
        cfg=cfg,
        log=True,
    )

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({cfg.n_replicates} independent replicates)")
    print("=" * 60)
    print(f"Mean detections per run: {np.mean(all_det):.0f} +/- {np.std(all_det):.0f}")
    print(f"\n{'idx':>3} | {'mean':>12} | {'std':>12} | {'SEM':>12}")
    print("-" * 50)
    for i in range(min(5, len(r_mean))):
        print(f"{i:>3} | {r_mean[i]:>12.6f} | {r_std[i]:>12.6f} | {r_sem[i]:>12.6f}")
