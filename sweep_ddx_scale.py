import os
from pathlib import Path

import numpy as np
import openmc

from run_replicates import ReplicateConfig
from utils.sweep import run_sweep

BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
XML_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"


def perturb_tabular_correlated_energy_pdf(
    file6,  # openmc.data.CorrelatedAngleEnergy
    E_CUT=1.8e6,  # eV (1 MeV)
    delta=0.01,  # probability mass to transfer; +: high->low, -: low->high
    e_idx=None,  # if None, apply to all incident energies; else only one index
    enforce_p0_zero=False,  # optional: force first bin density to 0 when adding-to-low
):
    """
    Perturb only p(E'|E) for MF=6 tabular-correlated distributions (histogram assumed).
    Leaves p(mu|E,E') unchanged.

    Mechanism:
      - Move |DELTA| probability mass from src region to dst region in outgoing energy:
          src = E'>=E_CUT if DELTA>0 else E'<E_CUT
          dst = E'<E_CUT  if DELTA>0 else E'>=E_CUT
      - Remove mass proportionally from src bins (by bin probability mass).
      - Add mass to dst using tapered weights:
          * if adding to low:  w ~ Emid (keeps p near 0 at E'=0)
          * if adding to high: w ~ (Emax-Emid) (keeps p near 0 at E'≈Emax)
      - Renormalize tabular PDF/CDF.
    """
    idxs = range(len(file6.energy)) if e_idx is None else [e_idx]

    for k in idxs:
        tab = file6.energy_out[k]  # openmc.stats.Tabular(E', p(E'|E))
        E = tab.x.copy()
        p = tab.p.copy()

        # Histogram integration convention in OpenMC:
        # mass = sum_{i=0..N-2} p[i] * (E[i+1]-E[i])
        dE = np.diff(E)  # len N-1
        Emid = 0.5 * (E[:-1] + E[1:])  # len N-1
        Emax = E[-1]

        # Define src/dst on bins (midpoints), not edges
        low = Emid < E_CUT
        high = ~low

        # delta = float(DELTA)
        mag = abs(delta)

        src = high if delta > 0 else low
        dst = low if delta > 0 else high
        if not np.any(src) or not np.any(dst):
            continue

        # Bin probability masses
        mass_bins = p[:-1] * dE
        M_src = mass_bins[src].sum()
        if M_src <= 0:
            continue

        take = min(mag, M_src)  # take is absolute probability mass
        if take <= 0:
            continue

        p_new = p.copy()

        # Remove from src proportionally (in mass)
        scale = 1.0 - take / M_src
        p_new[:-1][src] *= scale

        add_density = take / dE[dst].sum()
        p_new[:-1][dst] += add_density

        # Rebuild Tabular, normalize, and store back
        new_tab = openmc.stats.Tabular(
            E, p_new, interpolation=tab.interpolation, ignore_negative=True
        )
        new_tab.normalize()
        new_tab.c = new_tab.cdf()
        file6.energy_out[k] = new_tab

        # f_highbefore = mass_bins[Emid >= E_CUT].sum()  # already normalized
        # mass_bins = p_new[:-1] * dE
        # f_highafter = mass_bins[Emid >= E_CUT].sum()  # already normalized
        # print(
        #     f"{file6.energy[k] / 1e6:.2f}MeV fhighbefore {f_highbefore:.3f} fhighafter = {f_highafter:.3f}"
        # )

    return file6


# save #
# set file6.x = perturebd
# then use to_hdf5
def scale_be9_ddx(
    be9_path: str, xml_path: str, out_xml_path: Path, scaled_h5_path: Path, scale: float
) -> None:
    be9 = openmc.data.IncidentNeutron.from_hdf5(be9_path)
    # for temp in be9.reactions[16].xs:
    # be9.reactions[16].xs[temp].y *= scale
    rx = be9.reactions[16]
    file6 = rx.products[0].distribution[0]  # openmc.data.CorrelatedAngleEnergy type

    file6 = perturb_tabular_correlated_energy_pdf(
        file6,
        E_CUT=1.8e6,  # 1 MeV
        delta=scale,  # move 5% probability mass low->high (negative)
        e_idx=None,
    )

    rx.products[0].distribution[0] = file6

    be9.export_to_hdf5(scaled_h5_path, mode="w")
    library = openmc.data.DataLibrary.from_xml(xml_path)
    library.remove_by_material("Be9")
    library.register_file(scaled_h5_path.resolve())
    library.export_to_xml(str(out_xml_path))


def setup_scaled_ddx(scale: float, scale_dir: Path) -> None:
    # Put scaled files under THIS sweep point so nothing collides
    xs_dir = scale_dir / "xs"
    xs_dir.mkdir(parents=True, exist_ok=True)

    scaled_h5_path = xs_dir / f"Be9_scaled_{scale:.2f}.h5"
    scaled_xml_path = xs_dir / "cross_sections_scaled.xml"

    scale_be9_ddx(
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
    output_root = base_dir / "outputs" / "10e6_particles"

    PARTICLES_PER_REP = 10_000_000
    base_cfg = ReplicateConfig(
        n_replicates=1,
        particles_per_rep=PARTICLES_PER_REP,
        gate=28e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
        base_seed=123456,
        max_collisions=0.5 * PARTICLES_PER_REP,
    )

    scales = [-0.2, -0.1, 0.0, 0.1, 0.2]

    # for scale in scales:
    #     print(f"scale {scale}")
    #     scale_be9_ddx(
    #         be9_path=BE9_PATH,
    #         xml_path=XML_PATH,
    #         out_xml_path=Path("test.xml"),
    #         scaled_h5_path=Path("test.h5"),
    #         scale=scale,
    #     )

    results = run_sweep(
        input_dir=input_dir,
        output_root=output_root,
        base_cfg=base_cfg,
        values=scales,
        label_fmt="ddx_scale_{:.2f}",
        setup_hook=setup_scaled_ddx,
        param_name=None,
    )
