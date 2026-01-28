import os
from pathlib import Path

import openmc

from run_replicates import ReplicateConfig
from utils.sweep import run_sweep

BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
XML_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"


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


def setup_scaled_xs(scale: float, input_dir: Path, scale_dir: Path) -> None:
    # Put scaled files under THIS sweep point so nothing collides
    xs_dir = scale_dir / "xs"
    xs_dir.mkdir(parents=True, exist_ok=True)

    scaled_h5_path = xs_dir / f"Be9_scaled_{scale:.2f}.h5"
    scaled_xml_path = xs_dir / "cross_sections_scaled.xml"

    scale_be9_n2n(
        be9_path=BE9_PATH,
        xml_path=XML_PATH,
        out_xml_path=scaled_xml_path,
        scaled_h5_path=scaled_h5_path,
        scale=scale,
    )

    os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(str(scaled_xml_path))


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"

    PARTICLES_PER_REP = 100_000_000
    base_cfg = ReplicateConfig(
        n_replicates=1,
        particles_per_rep=PARTICLES_PER_REP,
        gate=85e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
        base_seed=123456,
        max_collisions=0.4 * PARTICLES_PER_REP,
    )

    scales = [0.8, 0.9, 1.0, 1.1, 1.2]

    results = run_sweep(
        input_dir=input_dir,
        output_root=output_root,
        base_cfg=base_cfg,
        values=scales,
        label_fmt="n2n_scale_{:.2f}",
        setup_hook=setup_scaled_xs,
        param_name=None,
    )
