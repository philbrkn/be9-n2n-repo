import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc.data
import openmc.stats
from matplotlib.colors import LogNorm
from perkins_openmc import convert_to_mb_sr_mev, create_and_run_model, get_Nareal

BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"


def perturb_tabular_correlated_energy_pdf(
    file6,  # openmc.data.CorrelatedAngleEnergy
    E_CUT=1.0e6,  # eV (1 MeV)
    DELTA=0.01,  # probability mass to transfer; +: high->low, -: low->high
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

        delta = float(DELTA)
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
        E_CUT=1.0e6,  # 1 MeV
        DELTA=scale,  # move 5% probability mass low->high (negative)
        e_idx=None,
    )

    rx.products[0].distribution[0] = file6

    be9.export_to_hdf5(scaled_h5_path, mode="w")
    library = openmc.data.DataLibrary.from_xml(xml_path)
    library.remove_by_material("Be9")
    library.register_file(scaled_h5_path.resolve())
    library.export_to_xml(str(out_xml_path))

    # # === PLOT CHECK === #
    # # just the pdf and relative error
    # plt.figure(figsize=(8, 5))
    # plt.subplot(1, 2, 1)
    # plt.semilogy(E / 1e6, p, label="original")
    # plt.semilogy(E / 1e6, p_new, label="perturbed", ls="--")
    # plt.xlabel("E' [MeV]")
    # plt.ylabel("p(E'|E) [per MeV]")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(E / 1e6, (p_new - p) / p * 100)
    # plt.xlabel("E' [MeV]")
    # plt.ylabel("Relative change [%]")
    # plt.axhline(0, color="k", lw=0.5)
    # plt.title("Effect of tilt perturbation")
    # plt.tight_layout()
    # plt.grid(True, alpha=0.3)
    #
    # plt.savefig(str(Path("perkins") / "pdf_perturbation_comparison.png"), dpi=300)
    #


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


def process_thin_slab_perturbed(
    perturbed_dir: Path,
    unperturbed_dir: Path,
    thickness: float,
    angle: float,
    angle_width: float,
    batches: int,
    E_incident_MeV: float,
    scale: float,
    be9_path: str,
    e_min: float = 0,
    e_max: float = 5,
):
    N_A = get_Nareal(thickness)

    # Load MC results
    sp_pert = openmc.StatePoint(str(perturbed_dir / f"statepoint.{batches}.h5"))
    sp_unpert = openmc.StatePoint(str(unperturbed_dir / f"statepoint.{batches}.h5"))

    df_pert = convert_to_mb_sr_mev(
        sp_pert.get_tally(name="ddx_tally").get_pandas_dataframe(),
        N_A,
        angle,
        angle_width,
    )
    df_unpert = convert_to_mb_sr_mev(
        sp_unpert.get_tally(name="ddx_tally").get_pandas_dataframe(),
        N_A,
        angle,
        angle_width,
    )

    p = df_pert[df_pert["score"] == "current"]
    u = df_unpert[df_unpert["score"] == "current"]

    E_mc = (
        0.5
        * (u["energyout low [eV]"].to_numpy() + u["energyout high [eV]"].to_numpy())
        / 1e6
    )
    y_pert = p["ddx"].to_numpy()
    y_unpert = u["ddx"].to_numpy()

    # ENDF prediction of the perturbation
    be9 = openmc.data.IncidentNeutron.from_hdf5(be9_path)
    rx = be9.reactions[16]
    file6 = rx.products[0].distribution[0]
    e_idx = np.argmin(np.abs(file6.energy - E_incident_MeV * 1e6))

    tab = file6.energy_out[e_idx]
    E = tab.x
    p_orig = tab.p

    E_mean = np.average(E, weights=p_orig)
    E_range = E[-1] - E[0]
    p_new = p_orig * (1 - scale * (E - E_mean) / E_range)

    def histogram_integral(p, x):
        return np.sum(p[: x.size - 1] * np.diff(x))

    p_new = p_new / histogram_integral(p_new, E)

    # Get angular part
    mu0 = np.cos(np.deg2rad(angle))
    Ej = tab.x
    pMu_at_Ej = np.array(
        [
            np.interp(mu0, file6.mu[e_idx][j].x, file6.mu[e_idx][j].p)
            for j in range(len(Ej))
        ]
    )

    temp = list(rx.xs.keys())[0]
    n2n_sigma = rx.xs[temp](E_incident_MeV * 1e6)

    Y_orig = n2n_sigma * p_orig * pMu_at_Ej / np.pi * 1e9
    Y_pert_endf = n2n_sigma * p_new * pMu_at_Ej / np.pi * 1e9

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: spectra
    axes[0].semilogy(E_mc, y_unpert, label="MC unperturbed")
    axes[0].semilogy(E_mc, y_pert, ls="--", label="MC perturbed")
    axes[0].set_xlim(e_min, e_max)
    axes[0].set_ylim(1e-2, 1e3)
    axes[0].set_xlabel("E' [MeV]")
    axes[0].set_ylabel("ddx [mb/sr/MeV]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: MC relative difference
    yerr_pert = p["ddx_std_dev"].to_numpy()
    yerr_unpert = u["ddx_std_dev"].to_numpy()

    # Propagated relative uncertainty on the ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        mc_rel = np.where(y_unpert > 0, (y_pert - y_unpert) / y_unpert * 100, 0.0)
        mc_rel_err = np.where(
            y_unpert > 0,
            np.sqrt(yerr_pert**2 + yerr_unpert**2) / y_unpert * 100,
            0.0,
        )
    axes[1].plot(E_mc, mc_rel, label="MC")
    axes[1].fill_between(
        E_mc, mc_rel - mc_rel_err, mc_rel + mc_rel_err, alpha=0.3, label="±1σ"
    )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlim(e_min, e_max)
    axes[1].set_xlabel("E' [MeV]")
    axes[1].set_ylabel("Relative change [%]")
    axes[1].set_title("MC difference")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: ENDF predicted relative difference
    with np.errstate(divide="ignore", invalid="ignore"):
        endf_rel = np.where(Y_orig > 0, (Y_pert_endf - Y_orig) / Y_orig * 100, 0.0)
    axes[2].plot(E / 1e6, endf_rel, label="ENDF prediction")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xlim(e_min, e_max)
    axes[2].set_xlabel("E' [MeV]")
    axes[2].set_ylabel("Relative change [%]")
    axes[2].set_title("ENDF predicted difference")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(perturbed_dir / "ddx_perturbation_comparison.png"), dpi=300)


def plot_mu_perturbation_heatmaps(be9_path, E_incident_MeV, scale):
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

    E_CUT = 1.5e6  # 1 MeV
    DELTA = 0.1  # 1% probability mass transfer
    tab = file6.energy_out[e_idx]
    E = tab.x.copy()
    p = tab.p.copy()
    high = E[:-1] > E_CUT  # note: histogram p has length N-1 effectively
    low = E[:-1] < E_CUT
    Emid = 0.5 * (E[:-1] + E[1:])  # length N-1
    dE = np.diff(E)  # eV
    mass_bins = p[:-1] * dE  # probability mass per bin (dimensionless)
    M_high = mass_bins[high].sum()
    take = min(DELTA, M_high)
    # print(M_high, DELTA)
    if take > 0 and low.any():
        # remove 'take' uniformly from E' > 1 MeV
        scale_high = 1.0 - take / M_high
        p_new = p.copy()
        p_new[:-1][high] *= scale_high

        # add take into low bins uniformly by mass
        # M_low = mass_bins[low].sum()
        # # simplest: distribute mass uniformly across low bins by mass capacity (dE)
        # add_density = take / dE[low].sum()
        # p_new[:-1][low] += add_density
        w = Emid[low].copy()
        w[0:1] = 0.0  # optional: force the very first low bin to get nothing
        W = np.sum(w * dE[low])  # normalize in "mass space"
        if W > 0:
            # added density per bin so that sum(add_density_i * dE_i) = take
            add_density = take * w / W
            p_new[:-1][low] += add_density

        # renormalize
        new_tab = openmc.stats.Tabular(
            E, p_new, interpolation=tab.interpolation, ignore_negative=True
        )
        new_tab.normalize()
        new_tab.c = new_tab.cdf()
        file6.energy_out[e_idx] = new_tab
    pE_new = file6.energy_out[e_idx].p
    # Perturbed joint density (mu tilt applied to angular PDFs)
    P_pert = np.zeros((mu_grid.size, E_out.size))
    for j, mu_tab in enumerate(file6.mu[e_idx]):
        P_pert[:, j] = pE_new[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)

    # Convert to cross-section units
    P_orig_xs = P_orig * n2n_sigma * 1e9 / (1 * np.pi)
    P_pert_xs = P_pert * n2n_sigma * 1e9 / (1 * np.pi)

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
    plt.savefig("perkins/heat_map_pertubation_comparison.png", dpi=300)


if __name__ == "__main__":
    BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
    XML_PATH = (
        "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
    )
    base_dir = Path(__file__).parent.resolve()

    RUN__UNPERTURBED = False
    RUN__SCALE = True

    INCIDENT_ENERGY = 9  # MeV
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

    scale_dir = Path("perkins/scale")
    scale = 0.05
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

    # process_thin_slab_perturbed(
    #     scale_dir,
    #     run_dir,
    #     thickness=THICKNESS,
    #     angle=ANGLE,
    #     angle_width=D_ANGLE,
    #     batches=N_BATCHES,
    #     E_incident_MeV=INCIDENT_ENERGY,
    #     scale=scale,
    #     be9_path=BE9_PATH,
    #     e_min=E_MIN,
    #     e_max=E_MAX,
    # )
    energy = 14.1
    plot_mu_perturbation_heatmaps(BE9_PATH, energy, scale)
