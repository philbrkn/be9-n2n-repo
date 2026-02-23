import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.data as data
from matplotlib.colors import LogNorm

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


def _kalbach_pdf(mu, a, r):
    # mu: (nmu,), a,r: (nEout,)
    mu = mu[:, None]
    a = np.asarray(a)[None, :]
    r = np.asarray(r)[None, :]

    print("a range:", a.min(), a.max())
    print("r range:", r.min(), r.max())
    out = np.empty((mu.shape[0], a.shape[1]), float)

    small = np.abs(a) < 1e-12
    out[:, small[0]] = 0.5

    if np.any(~small):
        a2 = a[:, ~small[0]]
        r2 = r[:, ~small[0]]
        out[:, ~small[0]] = (a2 / (2.0 * np.sinh(a2))) * (
            np.cosh(a2 * mu) + r2 * np.sinh(a2 * mu)
        )

    # normalize (safety)
    out /= np.trapz(out, mu[:, 0], axis=0)[None, :]
    return out


def _pick_temp(rx):
    # pick the first available temperature key in the xs dict
    # (works for '294K', '293.6K', '0K', etc.)
    temp = next(iter(rx.xs.keys()))
    print(f"using temp{temp}")
    return temp


def _ddx_heatmap(be9, E_incident_MeV=14.1, mu_grid=None, temp="294K"):
    """
    Returns:
      E_out_MeV (nEout,), mu_grid (nmu,), P (nmu,nEout)
    where P = sigma(E)*p(E'|E)*p(mu|E,E') in arbitrary units
    """
    if mu_grid is None:
        mu_grid = np.linspace(-1, 1, 201)

    rx = be9.reactions[16]

    if temp is None:
        temp = _pick_temp(rx)

    sigma = rx.xs[temp](E_incident_MeV * 1e6)

    dist = rx.products[0].distribution[0]  # KalbachMann or CorrelatedAngleEnergy

    # choose nearest incident-energy grid point
    Egrid = np.asarray(dist.energy) / 1e6
    i = int(np.argmin(np.abs(Egrid - E_incident_MeV)))

    # outgoing energy spectrum
    pE_tab = dist.energy_out[i]
    E_out_eV = np.asarray(pE_tab.x)
    pE_per_eV = np.asarray(pE_tab.p)  # pdf in 1/eV
    E_out_MeV = E_out_eV / 1e6

    P = np.zeros((mu_grid.size, E_out_MeV.size), dtype=float)

    # ENDF-style correlated table
    if hasattr(dist, "mu") and dist.mu is not None:
        for j, mu_tab in enumerate(dist.mu[i]):
            P[:, j] = pE_per_eV[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)

    else:
        # Tabulated1D objects: evaluate on E_out grid (eV)
        r_fun = dist.precompound[i]  # Tabulated1D of r(E')
        a_fun = dist.slope[i]  # Tabulated1D of a(E')
        r = np.array([r_fun(E) for E in E_out_eV])
        a = np.array([a_fun(E) for E in E_out_eV])

        P_mu = _kalbach_pdf(mu_grid, a, r)  # (nmu,nEout)
        P = pE_per_eV[None, :] * P_mu

    # convert 1/eV -> 1/MeV so plotting is nicer; include sigma just for weighting
    P = P * 1e9 * sigma / np.pi
    return E_out_MeV, mu_grid, P, temp


def plot_triptych(endf_be9_h5, tendl_be9, E_incident_MeV=14.1):
    mu = np.linspace(-1, 1, 201)

    endf_be9 = openmc.data.IncidentNeutron.from_hdf5(endf_be9_h5)
    tendl_be9 = openmc.data.IncidentNeutron.from_hdf5(tendl_be9)
    Eo1, mu, P1, t1 = _ddx_heatmap(endf_be9, E_incident_MeV, mu_grid=mu, temp="294K")
    Eo2, mu, P2, t2 = _ddx_heatmap(tendl_be9, E_incident_MeV, mu_grid=mu, temp="294K")

    # interpolate TENDL onto ENDF E' grid if needed
    if (Eo1.size != Eo2.size) or (np.max(np.abs(Eo1 - Eo2)) > 1e-12):
        P2i = np.zeros_like(P1)
        for k in range(P1.shape[0]):
            P2i[k, :] = np.interp(Eo1, Eo2, P2[k, :], left=np.nan, right=np.nan)
        P2 = P2i
        Eo = Eo1
    else:
        Eo = Eo1

    extent = [Eo[0], Eo[-1], -1, 1]
    vmax = np.nanmax([np.nanmax(P1), np.nanmax(P2)])
    vmin = max(vmax * 1e-6, 1e-30)

    fig, ax = plt.subplots(1, 3, figsize=(8.27, 2.3), constrained_layout=True)

    im0 = ax[0].imshow(
        P1,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax[0].set_title(f"ENDF/B-VIII.0 ({t1})")
    ax[0].set_xlabel("E' [MeV]")
    ax[0].set_ylabel("μ")
    ax[0].grid(False)

    im1 = ax[1].imshow(
        P2,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax[1].set_title(f"ENDF/B-VII.1 ({t2})")
    ax[1].set_xlabel("E' [MeV]")
    ax[1].set_ylabel("μ")
    ax[1].grid(False)

    thr = 0.01 * np.nanmax(P1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(P1 > thr, (P2 - P1) / P1 * 100.0, np.nan)

    vlim = np.nanmax(np.abs(rel))
    vlim = min(vlim if np.isfinite(vlim) else 50.0, 50.0)

    im2 = ax[2].imshow(
        rel,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
    )
    ax[2].set_title("ENDF/B-VII.1 − ENDF/B-VIII.0 [%]")
    ax[2].set_xlabel("E' [MeV]")
    ax[2].set_ylabel("μ")
    ax[2].grid(False)

    fig.colorbar(
        im0,
        ax=ax[:2],
        fraction=0.046,
        pad=0.02,
        label=r"$\mathrm{[MeV^{-1}\,sr^{-1}]}$",
    )
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02, label="%")
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    ax[2].set_box_aspect(1)

    # or save:
    # fig.savefig("tendl/tendl_endf_ddx_triptych.png", dpi=300)
    fig.savefig("endf7vs8", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    tendl_endf = "n_004-Be-9_0425.dat"

    h5_path = "Be9_TENDL_raw.h5"
    be9_km = data.IncidentNeutron.from_hdf5(h5_path)
    rx = be9_km.reactions[16]
    # for temp in be9_km.reactions[16].xs:
    #     print(temp)
    sigma = rx.xs["294K"](14.1 * 1e6)
    dist = rx.products[0].distribution[0]  # CorrelatedAngleEnergy
    # print(type(dist))
    # print(dist.__dict__.keys())
    # print(dir(dist))
    # be9_km.export_to_hdf5("Be9_TENDL_raw.h5")

    BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
    BE9_PATH_ENDF7 = "ENDFB-7.1-NNDC_Be9.h5"
    plot_triptych(BE9_PATH, BE9_PATH_ENDF7, E_incident_MeV=14.1)
