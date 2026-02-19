import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc.data
import openmc.stats
from matplotlib.colors import LogNorm
from perkins_openmc import create_and_run_model


def scale_be9_ddx_mu(
    be9_path: str, xml_path: str, out_xml_path: Path, scaled_h5_path: Path, scale: float
) -> None:
    be9 = openmc.data.IncidentNeutron.from_hdf5(be9_path)
    # for temp in be9.reactions[16].xs:
    # be9.reactions[16].xs[temp].y *= scale
    rx = be9.reactions[16]
    file6 = rx.products[0].distribution[0]  # openmc.data.CorrelatedAngleEnergy type

    for e_idx, energy in enumerate(file6.energy):
        for j, mu_tab in enumerate(file6.mu[e_idx]):
            mu = mu_tab.x
            p_mu = mu_tab.p
            # Tilt angular distribution: more forward or more backward
            p_mu_new = p_mu * (1 + scale * mu)  # positive scale = more forward peaked
            new_mu = openmc.stats.Tabular(
                mu, p_mu_new, interpolation=mu_tab.interpolation, ignore_negative=True
            )
            new_mu.normalize()
            new_mu.c = new_mu.cdf()
            file6.mu[e_idx][j] = new_mu
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

    scale_be9_ddx_mu(
        be9_path=BE9_PATH,
        xml_path=XML_PATH,
        out_xml_path=scaled_xml_path,
        scaled_h5_path=scaled_h5_path,
        scale=scale,
    )

    os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(str(scaled_xml_path))


def plot_mu_perturbation_heatmaps(be9_path, E_incident_MeV, scale, out_path):
    be9 = openmc.data.IncidentNeutron.from_hdf5(be9_path)
    rx = be9.reactions[16]
    file6 = rx.products[0].distribution[0]

    e_idx = np.argmin(np.abs(file6.energy - E_incident_MeV * 1e6))
    E_in = file6.energy[e_idx]

    temp = list(rx.xs.keys())[0]
    n2n_sigma = rx.xs[temp](E_incident_MeV * 1e6)

    pE_tab = file6.energy_out[e_idx]
    E_out = pE_tab.x
    pE = pE_tab.p
    mu_grid = np.linspace(-1, 1, 201)

    # Original joint density
    P_orig = np.zeros((mu_grid.size, E_out.size))
    for j, mu_tab in enumerate(file6.mu[e_idx]):
        P_orig[:, j] = pE[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)

    # Perturbed joint density (mu tilt applied to angular PDFs)
    P_pert = np.zeros((mu_grid.size, E_out.size))
    for j, mu_tab in enumerate(file6.mu[e_idx]):
        mu = mu_tab.x
        p_mu = mu_tab.p
        p_mu_new = p_mu * (1 + scale * mu)
        p_mu_new = np.clip(p_mu_new, 0, None)
        # Normalize using histogram rule
        norm = np.sum(p_mu_new[: mu.size - 1] * np.diff(mu))
        if norm > 0:
            p_mu_new = p_mu_new / norm
        P_pert[:, j] = pE[j] * np.interp(mu_grid, mu, p_mu_new)

    # Convert to cross-section units
    P_orig_xs = P_orig * n2n_sigma * 1e9 / (2 * np.pi)
    P_pert_xs = P_pert * n2n_sigma * 1e9 / (2 * np.pi)

    # Relative difference
    threshold = 0.01 * P_orig_xs.max()  # 1% of peak value
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(
            P_orig_xs > threshold,
            (P_pert_xs - P_orig_xs) / P_orig_xs * 100,
            np.nan,
        )

    extent = [E_out[0] / 1e6, E_out[-1] / 1e6, -1, 1]
    vmin_log = 0.1
    vmax_log = max(P_orig_xs.max(), P_pert_xs.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(
        P_orig_xs,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=LogNorm(vmin=vmin_log, vmax=vmax_log),
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("E' [MeV]")
    axes[0].set_ylabel("μ")
    fig.colorbar(im0, ax=axes[0], label="mb/sr/MeV")

    im1 = axes[1].imshow(
        P_pert_xs,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=LogNorm(vmin=vmin_log, vmax=vmax_log),
    )
    axes[1].set_title(f"Perturbed (mu tilt, scale={scale})")
    axes[1].set_xlabel("E' [MeV]")
    axes[1].set_ylabel("μ")
    fig.colorbar(im1, ax=axes[1], label="mb/sr/MeV")

    vlim = np.nanmax(np.abs(rel_diff))
    vlim = min(vlim, 50)  # cap at 50% to see the real structure
    im2 = axes[2].imshow(
        rel_diff,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
    )
    axes[2].set_title("Relative difference [%]")
    axes[2].set_xlabel("E' [MeV]")
    axes[2].set_ylabel("μ")
    fig.colorbar(im2, ax=axes[2], label="%")

    fig.suptitle(f"E = {E_in / 1e6:.1f} MeV, mu tilt scale = {scale}")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300)


if __name__ == "__main__":
    BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
    XML_PATH = (
        "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
    )
    base_dir = Path(__file__).parent.resolve()

    RUN__UNPERTURBED = False
    RUN__SCALE = False

    INCIDENT_ENERGY = 14.1  # MeV
    ANGLE = 60  # degrees
    D_ANGLE = 1  # degrees
    THICKNESS = 0.1  # cm

    E_MIN = 0
    E_MAX = 9
    ENERGY_BINS = np.arange(E_MIN, E_MAX, 0.1)

    N_BATCHES = 50
    N_PARTICLES = 20_000_000

    run_dir = base_dir / f"run_{INCIDENT_ENERGY:.2f}MeV_{ANGLE:.1f}deg"

    if RUN__UNPERTURBED:
        print("Running unperturbed.")
        create_and_run_model(
            E_incident_MeV=INCIDENT_ENERGY,
            angle_deg=ANGLE,
            angle_width=D_ANGLE,
            thickness=THICKNESS,
            energy_bins_MeV=ENERGY_BINS,
            run_dir=run_dir,
            batches=N_BATCHES,
            particles=N_PARTICLES,
        )

    scale_dir = Path("perkins/scale_mu")
    scale = 0.50
    setup_scaled_ddx(scale, scale_dir)
    if RUN__SCALE:
        print("Running perturbed.")
        create_and_run_model(
            E_incident_MeV=INCIDENT_ENERGY,
            angle_deg=ANGLE,
            angle_width=D_ANGLE,
            thickness=THICKNESS,
            energy_bins_MeV=ENERGY_BINS,
            run_dir=scale_dir,
            batches=N_BATCHES,
            particles=N_PARTICLES,
        )

    energy = 14.1
    plot_mu_perturbation_heatmaps(BE9_PATH, energy, scale, scale_dir)
