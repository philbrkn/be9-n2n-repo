from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc


def create_and_run_model(
    E_incident_MeV,
    angle_deg,
    angle_width,
    thickness,
    energy_bins_MeV,
    run_dir,
    batches,
    particles,
):
    "thin Be target for extracting secondary neutron spectra"

    # materials
    be = openmc.Material(name="beryllium")
    be.add_nuclide("Be9", 1.0)
    be.set_density("g/cm3", 1.85)
    mats = openmc.Materials([be])

    # --- Geometry: very thin Be slab in vacuum ---
    # Beam travels +z. Slab centered at z=0. Thickness t.
    radius = 5.0  # cm

    z_min = -thickness / 2
    z_max = thickness / 2

    slab_region = (
        +openmc.ZPlane(z_min)
        & -openmc.ZPlane(z_max)
        & +openmc.XPlane(-radius)
        & -openmc.XPlane(radius)
        & +openmc.YPlane(-radius)
        & -openmc.YPlane(radius)
    )
    slab_cell = openmc.Cell(name="be_slab", fill=be, region=slab_region)

    # Large vacuum sphere for surface counting
    sphere_surf = openmc.Sphere(r=100.0, boundary_type="vacuum")
    vac_region = -sphere_surf & ~slab_region
    vac_cell = openmc.Cell(name="vacuum", region=vac_region)

    geom = openmc.Geometry([slab_cell, vac_cell])

    # --- Settings: pencil beam, monoenergetic ---
    settings = openmc.Settings()
    settings.batches = batches
    settings.inactive = 0
    settings.particles = particles
    settings.run_mode = "fixed source"

    # Pencil beam source at origin, directed +Z
    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0.0, 0.0, z_min - 0.1))  # start just before slab
    src.angle = openmc.stats.Monodirectional((0.0, 0.0, 1.0))  # along +z
    src.energy = openmc.stats.Discrete([E_incident_MeV * 1e6], [1.0])
    settings.source = src

    # --- Filters ---
    # Set energy bins for the output spectrum (secondary neutrons)
    energy_filter = openmc.EnergyoutFilter([e * 1e6 for e in energy_bins_MeV])

    # Angular bins for surface Tally
    theta_rad = np.deg2rad(angle_deg)
    delta_rad = np.deg2rad(angle_width / 2)
    polar_filter = openmc.PolarFilter([theta_rad - delta_rad, theta_rad + delta_rad])

    surf_filter = openmc.SurfaceFilter(sphere_surf)

    # cell_filter = openmc.CellFilter(slab_cell)
    # mu_low = np.cos(theta_rad + delta_rad)
    # mu_high = np.cos(theta_rad - delta_rad)
    # mu_filter = openmc.MuFilter([mu_low, mu_high])

    # --- Tallies ---
    tallies = openmc.Tallies()

    # captures all secondary neutrons. since bins go to 2.2 MeV, the elastic peak is
    # automatically excluded
    tally = openmc.Tally(name="ddx_tally")
    tally.filters = [surf_filter, polar_filter, energy_filter]
    tally.scores = ["current"]  # "current" is the number of particles crossing

    # tally.filters = [cell_filter, mu_filter, energy_filter]
    # tally.scores = ["nu-scatter"]
    # tally.scores = ["(n,2n)"]
    tallies.append(tally)

    # Tally for neutrons that scattered EXACTLY once (the pure signal)
    single_scatter_filter = openmc.CollisionFilter([1])
    tally_single = openmc.Tally(name="single_scatter")
    tally_single.filters = [
        # cell_filter,
        # mu_filter,
        surf_filter,
        polar_filter,
        energy_filter,
        single_scatter_filter,
    ]
    # tally_single.scores = ["nu-scatter"]
    tally_single.scores = ["current"]
    tallies.append(tally_single)

    # Tally for neutrons that scattered MORE than once (the noise)
    multiple_scatter_filter = openmc.CollisionFilter([2, 3, 4, 5])
    tally_multiple = openmc.Tally(name="multiple_scatter")
    tally_multiple.filters = [
        # cell_filter,
        # mu_filter,
        surf_filter,
        polar_filter,
        energy_filter,
        multiple_scatter_filter,
    ]
    # tally_multiple.scores = ["nu-scatter"]
    tally_multiple.scores = ["current"]
    tallies.append(tally_multiple)

    # Total (n,2n) reaction rate in the slab - no angle/energy filters
    cell_filter = openmc.CellFilter(slab_cell)
    tally_rxn = openmc.Tally(name="n2n_rate")
    tally_rxn.filters = [cell_filter]
    tally_rxn.scores = ["(n,2n)"]
    tallies.append(tally_rxn)

    # --- Model export ---
    model = openmc.Model(
        geometry=geom, materials=mats, settings=settings, tallies=tallies
    )

    # Clean up previous run
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    model.export_to_xml(run_path)

    # Run OpenMC
    openmc.run(cwd=run_path, output=False)


def get_Nareal(thickness, density=1.85, molarmass=9.0122):
    """
    returns in atoms / barns
    """
    avogrado = 6.02214e23  # atoms/mol
    return density * thickness * 1e-24 * avogrado / molarmass


def convert_to_mb_sr_mev(df, n_areal, angle, angle_width):
    # solid angle = 2 pi delta_mu
    theta_low = np.deg2rad(angle - angle_width / 2)
    theta_high = np.deg2rad(angle + angle_width / 2)
    delta_omega = 2 * np.pi * (np.cos(theta_low) - np.cos(theta_high))
    # energy bin width in meV
    df["dE"] = (df["energyout high [eV]"] - df["energyout low [eV]"]) / 1e6

    df["ddx"] = (df["mean"] / (n_areal * delta_omega * df["dE"])) * 1000
    df["ddx_std_dev"] = (df["std. dev."] / (n_areal * delta_omega * df["dE"])) * 1000
    return df


def plot_multiple_scatter(sp, path, angle, angle_width, Nareal):
    # 2. Single vs multiple scatter comparison
    t_single = sp.get_tally(name="single_scatter")
    t_multi = sp.get_tally(name="multiple_scatter")
    # Mean values
    single_mean = t_single.mean
    multi_mean = t_multi.mean
    # Sum over all bins (energy × angle × etc.)
    single_total = np.sum(single_mean)
    multi_total = np.sum(multi_mean)
    print(f"Single-scatter total: {single_total:.3e}")
    print(f"Multiple-scatter total: {multi_total:.3e}")
    if single_total > 0:
        print(f"Ratio (multi / single): {multi_total / single_total:.4f}")

    df_s = convert_to_mb_sr_mev(
        t_single.get_pandas_dataframe(), Nareal, angle, angle_width
    )
    df_m = convert_to_mb_sr_mev(
        t_multi.get_pandas_dataframe(), Nareal, angle, angle_width
    )
    s = (
        df_s[df_s["score"] == "current"]
        .groupby(["energyout low [eV]", "energyout high [eV]"], as_index=False)
        .agg({"ddx": "sum"})
    )
    m = (
        df_m[df_m["score"] == "current"]
        .groupby(["energyout low [eV]", "energyout high [eV]"], as_index=False)
        .agg({"ddx": "sum"})
    )
    E_s = (
        0.5
        * (s["energyout low [eV]"].to_numpy() + s["energyout high [eV]"].to_numpy())
        / 1e6
    )
    plt.figure()
    plt.semilogy(E_s, s["ddx"].to_numpy(), label="single scatter")
    plt.semilogy(E_s, m["ddx"].to_numpy(), label="multiple scatter")
    plt.legend()
    # plt.title("Single vs multiple scatter")
    plt.grid()
    plt.savefig(str(path / "scatter_breakdown.png"), dpi=300)


def post_process(
    run_dir, thickness, angle, angle_width, batches, E_incident_MeV, e_min, e_max
):
    sp = openmc.StatePoint(str(run_dir / f"statepoint.{batches}.h5"))
    t_ddx = sp.get_tally(name="ddx_tally")
    df = t_ddx.get_pandas_dataframe()

    N_A = get_Nareal(thickness)
    df_final = convert_to_mb_sr_mev(df, N_A, angle, angle_width)
    plot_comparison(
        df_final,
        path=str(run_dir / "plot_mbsrMeV.png"),
        E_incident_MeV=E_incident_MeV,
        angle=angle,
        e_min=e_min,
        e_max=e_max,
    )

    plot_multiple_scatter(sp, run_dir, angle, angle_width, N_A)

    t_rxn = sp.get_tally(name="n2n_rate")
    rate = t_rxn.mean.flat[0]
    # Should be close to N_areal * sigma_n2n(E) where sigma_n2n ~ 0.5 b at 14 MeV
    print(f"(n,2n) reaction rate per source: {rate:.4e}")
    print(f"Expected (N_A * ~0.5b): {N_A * 0.5:.4e}")


def broaden_variable_gaussian(E_cent, y, frac_sigma):
    """
    Broaden spectrum y(E) by energy-dependent Gaussian with 1σ width:
    sigma_E(E) = frac_sigma(E) * E.
    Preserves total area approximately.
    """
    E = E_cent
    y = np.asarray(y, float)

    yb = np.zeros_like(y)
    for j, Ej in enumerate(E):
        sig = max(frac_sigma(Ej) * Ej, 1e-6)  # MeV
        w = np.exp(-0.5 * ((E - Ej) / sig) ** 2)
        w /= w.sum()  # normalize discrete kernel
        yb += y[j] * w
    return yb


def frac_sigma(E):
    """
    Paper resolution model (assumed 1σ fractional)
    """
    # return 0.03 * np.sqrt(max(E, 0.0))
    fwhm_frac = 0.03 * np.sqrt(max(E, 0.0))
    return fwhm_frac / 2.355


def plot_comparison(
    df_res, path, E_incident_MeV=None, angle=None, e_min=None, e_max=None
):
    plt.figure(figsize=(5, 8))

    # total = df_res[df_res["score"] == "nu-scatter"].copy()
    # total = df_res[df_res["score"] == "(n,2n)"].copy()
    total = df_res[df_res["score"] == "current"].copy()

    E_low = total["energyout low [eV]"].to_numpy()
    E_high = total["energyout high [eV]"].to_numpy()
    E_cent = 0.5 * (E_low + E_high) / 1e6  # MeV

    y = total["ddx"].to_numpy()  # mb/sr/MeV
    yerr = total["ddx_std_dev"].to_numpy()  # mb/sr/MeV

    y_b = broaden_variable_gaussian(E_cent, y, frac_sigma)

    plt.plot(E_cent, y_b, "-", label="MC (broadened)")
    plt.errorbar(E_cent, y, yerr=yerr, fmt="k.", ms=3, alpha=0.4, label="MC (raw, ±1σ)")

    # if E_incident_MeV is not None or angle is not None:
    #     e, y, yerr = get_data(E_incident_MeV, angle, source="auto")
    #
    #     mean_color = "#C1403D"  # Muted red
    #     if yerr is not None:
    #         plt.errorbar(
    #             e,
    #             y,
    #             yerr=yerr,
    #             fmt="s",
    #             color=mean_color,
    #             ms=5,
    #             capsize=2,
    #             label="Baba et al. (EXFOR)",
    #             alpha=0.9,
    #         )
    #     else:
    #         plt.plot(e, y, "-", label="ENDF/B-VIII.1", color=mean_color)

    # plt.xlim(0, 4)
    plt.xlim(e_min, e_max)
    plt.ylim(1e-1, 1e3)
    plt.yscale("log")
    plt.xlabel("Secondary Neutron Energy [MeV]")
    plt.ylabel("Cross Section [mb/sr/MeV]")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=300)


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    INCIDENT_ENERGY = 5  # MeV
    ANGLE = 60.0  # degrees
    D_ANGLE = 1  # degrees
    THICKNESS = 0.1  # cm

    E_MIN = 0
    E_MAX = 2.5
    ENERGY_BINS = np.linspace(E_MIN, E_MAX, 500)

    N_BATCHES = 10
    N_PARTICLES = 10_000_000

    run_dir = base_dir / f"run_{INCIDENT_ENERGY}MeV_{ANGLE}deg"

    Nareal = get_Nareal(thickness=THICKNESS)

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
    post_process(
        run_dir, THICKNESS, ANGLE, D_ANGLE, N_BATCHES, INCIDENT_ENERGY, E_MIN, E_MAX
    )
