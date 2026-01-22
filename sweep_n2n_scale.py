# sweep_n2n_scale.py
# Your (n,2n) scaling study, but using the refactored replicate runner.
# Only includes definitions / structure; your scaling logic is retained.

import os
from pathlib import Path

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

    scales = [0.8, 0.9, 1.0, 1.1, 1.2]

    cfg = ReplicateConfig(
        n_replicates=10,
        particles_per_rep=100_000,
        base_seed=12345,  # keep fixed for comparability; or None for random
        gate=32e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
    )

    results = {}

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

        results[scale] = {
            "detections_mean": float(np.mean(all_det)),
            "detections_std": float(np.std(all_det)),
            "r_mean": r_mean,
            "r_sem": r_sem,
        }

        print(
            f"[scale {scale:.2f}] mean detections: {np.mean(all_det):.1f}  std: {np.std(all_det):.1f}"
        )
