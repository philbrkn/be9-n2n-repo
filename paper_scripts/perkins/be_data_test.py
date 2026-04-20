import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openmc.data
from exfor_data import get_data
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

# Load the Be9 data from your environment's library
# Ensure your OPENMC_CROSS_SECTIONS env variable is set
# lib = openmc.data.IncidentNeutron.from_hdf5(
#     openmc.config["cross_sections"].find_nuclide("Be9")
# )
# print(f"Reactions for {lib.name}:")
# for mt, reaction in lib.reactions.items():
#     print(f"MT {mt}: {reaction.description}")
#
BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"

# Actual ENDF Data:
E_incident_MeV = 9
ANGLE = 60

result = get_data(E_incident_MeV, ANGLE, source="endf")
e_endf, y_endf, _ = result
# ANGLE = 55

# === Load the evaluation === #
be9 = openmc.data.IncidentNeutron.from_hdf5(BE9_PATH)
rx = be9.reactions[16]
prod = rx.products[0]
# openmc.data.AngleEnergy type
dist = prod.distribution
# openmc.data.CorrelatedAngleEnergy type
file6 = dist[0]
# has parameters: breakpoints. interpolation, energy, energy_out, mu
# energy: iterable of float
# energy_out iterable of openmc.stats.Univariate.tabular, distribution of outgoing energies for each incoming
# energy_out[i] represents p(E'|E_i)
# mu[i][j] represents p(mu|E_i, E'_j)
# mu iterable of iterable of openmc.stats.Univariate.tabular, distribution of scattering cosine for each incoming/outoing energy

# for i, tab in enumerate(file6.energy_out):
#     integral = np.trapz(tab.p, tab.x)
#     print(
#         f"E={file6.energy[i] / 1e6:.2f} MeV: trapz integral={integral:.4f} | .integral {file6.energy_out[i].integral()}"
#     )

print(prod.yield_)
# print energies and indices:
print(", ".join(f"{i}:{e / 1e6:g}" for i, e in enumerate(file6.energy)))
# 5 is index 5

# === convert to double differential === #
e_idx = np.where(file6.energy == E_incident_MeV * 1e6)[0][0]

# 1) build outgoing energy grid
Eprime_eV = e_endf * 1e6

# 2) evaluate p(E'|Ei) from Tabular (per eV)
pE_njoy = file6.energy_out[e_idx]
print(f"interpolation method: {pE_njoy.interpolation}")
pE = np.interp(Eprime_eV, pE_njoy.x, pE_njoy.p)

# 3) evaluate p(mu|Ei, E') from mu tables
mu0 = np.cos(np.deg2rad(ANGLE))
Ej = pE_njoy.x  # outgoing-energy grid used by the correlated distribution (eV)
# For each Ej, evaluate the angular density at mu0
pMu_at_Ej = np.array(
    [np.interp(mu0, file6.mu[e_idx][j].x, file6.mu[e_idx][j].p) for j in range(len(Ej))]
)
# Interpolate that evaluated density to requested Eprime grid
pMu = np.interp(Eprime_eV, Ej, pMu_at_Ej)  # per unit mu
# 5) convert to per sr
pOmega = pMu / (1 * np.pi)

# how do i choose temp?
# for temp in be9.reactions[16].xs:
#     print(temp)
temp = "294K"
n2n_sigma = rx.xs[temp](E_incident_MeV * 1e6)

Y = n2n_sigma * pE * pOmega * 1e9

# === PLOT 1: DDX RECONSTRUCTION VS ENDF === #
mean_color = "#C1403D"  # Muted red
plt.figure(figsize=(5, 8))
plt.plot(e_endf, Y, linestyle="-", label="OpenMC converted")
plt.plot(e_endf, y_endf, "-", label="ENDF/B-VIII.1", color=mean_color)
plt.ylim(1e-1, 1e3)
plt.yscale("log")
plt.xlabel("Secondary Neutron Energy [MeV]")
plt.ylabel("Cross Section [mb/sr/MeV]")
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig("perkins/debug_endf_data.png", dpi=300)

# === PLOT 1.5: DDX PERTURBED === #
E_in = file6.energy[e_idx]
tab = file6.energy_out[e_idx]  # openmc.stats.Tabular
E = tab.x
p = tab.p
E_mean = np.average(E, weights=p)
E_range = E[-1] - E[0]
p_new = p * (1 - 0.1 * (E - E_mean) / E_range)
p_new = p_new * (np.trapezoid(p, E) / np.trapezoid(p_new, E))
# p_new = np.interp(e_endf, E, p_new)
Y_perb = n2n_sigma * p_new * pMu_at_Ej / np.pi * 1e9
plt.figure(figsize=(5, 8))
plt.subplot(1, 2, 1)
# plt.plot(e_endf, y_endf, "-", label="ENDF/B-VIII.1", color=mean_color)
plt.plot(e_endf, Y, linestyle="-", label="OpenMC converted")
plt.plot(E / 1e6, Y_perb, linestyle="-", label="Perturbed DDX")
plt.ylim(1e-3, 1e3)
# plt.xlim(0, 2.5)
plt.yscale("log")
plt.xlabel("Secondary Neutron Energy [MeV]")
plt.ylabel("Cross Section [mb/sr/MeV]")
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)
# relative error:
plt.subplot(1, 2, 2)
Y_orig = n2n_sigma * p * pMu_at_Ej / np.pi * 1e9
plt.plot(E, (Y_perb - Y_orig) / Y_orig * 100)
plt.xlabel("E' [MeV]")
plt.ylabel("Relative change [%]")
plt.axhline(0, color="k", lw=0.5)
plt.title("Effect of tilt perturbation")
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig("perkins/debug_endf_data_perturbed.png", dpi=300)

# === PLOT 2: Outgoing energy pdf === #
# ---- global style (set once for the whole 3-panel figure) ----
mpl.rcParams.update(
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
        # "mathtext.fontset": "stix",
        # "mathtext.rm": "STIXGeneral",
        # "mathtext.it": "STIXGeneral:italic",
        # "mathtext.bf": "STIXGeneral:bold",
    }
)
data_color = "#2E5090"  # Deep blue
mean_color = "#C1403D"  # Muted red
# tells us: regardless of direction, what's the energy distribution of the secondary neutrons?
# Example: 3 panels across a 7.2 inch wide figure -> ~2.3 inch each
# fig, ax = plt.subplots(figsize=(2.8, 2.8), constrained_layout=True)
fig, axes = plt.subplots(1, 3, figsize=(8.27, 2.5), constrained_layout=True)

i = 16
E_in_MeV = file6.energy[i] * 1e-6
tab = file6.energy_out[i]
x_MeV = tab.x * 1e-6
p_per_MeV = tab.p * 1e6

# --- (a) energy distribution ---
axes[0].plot(x_MeV, p_per_MeV, color=data_color)
axes[0].set_xlabel(r"$E'\ \mathrm{[MeV]}$")
axes[0].set_ylabel(r"$p(E' \mid E)\ \mathrm{[MeV^{-1}]}$")
axes[0].grid(True, alpha=0.1)


# --- (b) mean mu ---
means, E_out_MeV = [], []
for j, mu_tab in enumerate(file6.mu[i]):
    means.append(mu_tab.mean())
    E_out_MeV.append(file6.energy_out[i].x[j] * 1e-6)

axes[1].plot(E_out_MeV, means, color=data_color, marker="o", markersize=2, linewidth=1)
axes[1].set_xlabel(r"$E'\ \mathrm{[MeV]}$")
axes[1].set_ylabel(r"$\langle \mu \rangle$")
axes[1].grid(True, alpha=0.1)

# --- (c) joint density ---
E_in = file6.energy[i]
pE_tab = file6.energy_out[i]
E_out = pE_tab.x
pE = pE_tab.p
mu_grid = np.linspace(-1, 1, 201)

P = np.zeros((mu_grid.size, E_out.size))
for j, mu_tab in enumerate(file6.mu[i]):
    P[:, j] = pE[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)

P_xs = P * n2n_sigma * 1e9 / (1 * np.pi)

im = axes[2].imshow(
    P_xs,
    origin="lower",
    aspect="auto",
    extent=[E_out[0] / 1e6, E_out[-1] / 1e6, -1, 1],
    norm=LogNorm(vmin=0.1, vmax=P_xs.max()),
)
axes[2].set_xlabel(r"$E'\ \mathrm{[MeV]}$")
axes[2].set_ylabel(r"$\mu$")
fig.colorbar(im, ax=axes[2], label=r"$\mathrm{[MeV^{-1}\,sr^{-1}]}$")
axes[2].grid(False)
# make them square:
axes[0].set_box_aspect(1)
axes[1].set_box_aspect(1)
axes[2].set_box_aspect(1)
# plt.savefig("perkins/file6_mean_mu.png", dpi=300)
# fig.savefig("perkins/file6_pdf.png", dpi=300)

# --- subplot labels below x-axis label ---
for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(
        0.5,
        -0.24,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        # fontweight="bold",
    )
fig.savefig("perkins/file6_subplots.pdf", dpi=300)
fig.savefig("perkins/file6_subplots.png", dpi=300)
plt.close(fig)


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
        # Add to dst with tapered weights
        # if np.array_equal(dst, low):
        #     w = Emid[dst].copy()  # ramp-up from 0
        # else:
        #     w = (Emax - Emid[dst]).copy()  # ramp-down to 0 at Emax
        # w = np.clip(w, 0.0, None)
        #
        # # Normalize weights in "mass space" so sum(add_density_i * dE_i) = take
        # W = np.sum(w * dE[dst])
        # if W > 0:
        #     add_density = take * w / W  # units 1/eV
        #     p_new[:-1][dst] += add_density
        # else:
        #     # fallback: uniform by mass capacity
        #     add_density = take / dE[dst].sum()
        #     p_new[:-1][dst] += add_density
        #
        # # Optional: preserve p(0)=0 when adding to low
        # if enforce_p0_zero and delta > 0:
        #     p_new[0] = 0.0

        # Rebuild Tabular, normalize, and store back
        new_tab = openmc.stats.Tabular(
            E, p_new, interpolation=tab.interpolation, ignore_negative=True
        )
        new_tab.normalize()
        new_tab.c = new_tab.cdf()
        file6.energy_out[k] = new_tab

    return file6


e_idx = 16
tab0 = file6.energy_out[e_idx]
E0 = tab0.x.copy()
p0 = tab0.p.copy()

file6 = perturb_tabular_correlated_energy_pdf(
    file6,
    E_CUT=1.0e6,  # 1 MeV
    DELTA=0.2,  # move 5% probability mass low->high (negative)
    e_idx=e_idx,  # just this incident energy for debugging
    enforce_p0_zero=False,
)

p1 = file6.energy_out[e_idx].p.copy()

# Diagnostics: total mass check (histogram)
mass0 = np.sum(p0[:-1] * np.diff(E0))
mass1 = np.sum(p1[:-1] * np.diff(E0))
print("total mass (should be ~1):", mass0, mass1)

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.semilogy(E0 / 1e6, p0 * 1e6, label="original")
plt.semilogy(E0 / 1e6, p1 * 1e6, label="perturbed", ls="--")
plt.xlabel("E' [MeV]")
plt.ylabel("p(E'|E) [per MeV]")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot: absolute difference (recommended)
plt.subplot(1, 2, 2)
plt.plot(E0 / 1e6, (p1 - p0) * 1e6)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("E' [MeV]")
plt.ylabel("Δp(E'|E) [per MeV]")
plt.title("Absolute change")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.subplot(1, 2, 2)
# plt.plot(E / 1e6, (p_new - p) / p * 100)
# plt.xlabel("E' [MeV]")
# plt.ylabel("Relative change [%]")
# plt.axhline(0, color="k", lw=0.5)
# plt.title("Effect of tilt perturbation")
# plt.tight_layout()
# plt.grid(True, alpha=0.3)

plt.savefig("perkins/pdf_perturbation_check.png", dpi=300)


# == PLOT 3: Angular pdfs === #
# each curve shows p(mu|E,E') for  different outgoing E'
# for neutrons that come out at this specific energy, which direction do they go
#  Isotropic would be a flat line.
plt.figure(figsize=(6, 4))
for j in [
    0,
    len(file6.mu[i]) // 4,
    len(file6.mu[i]) // 2,
    3 * len(file6.mu[i]) // 4,
    len(file6.mu[i]) - 1,
]:
    mu_tab = file6.mu[i][j]
    plt.plot(mu_tab.x, mu_tab.p, label=f"j={j}, E'={tab.x[j] / 1e6:g} MeV")
plt.xlabel("μ")
plt.ylabel("p(μ|E,E')")
plt.title(f"Angular PDFs at E={E_in / 1e6:g} MeV")
plt.legend(fontsize=8)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig("perkins/file6_mu_overlays.png", dpi=300)
# plt.figure(figsize=(5, 8))
# j = 9  # outgoing-energy bin/index
# mu_tab = file6.mu[i][j]
# plt.plot(mu_tab.x, mu_tab.p)
# plt.xlabel("Scattering cosine μ")
# plt.ylabel("PDF(μ | E, E')")
# plt.title(f"Angular PDF at E={E_in:.3e} eV, bin {j}")
# plt.tight_layout()
# plt.grid(True, alpha=0.3)
# plt.savefig("perkins/file6_mu_pdf.png", dpi=300)


# === Plot 5: Joint distribution as a heat map === #
# x-axis: outgoing energy, y-axis: scattering cosine, color is prob density.
# bright spots are where most neutrons end up in (energy, angle) space
i = 16
E_in = file6.energy[i]
pE_tab = file6.energy_out[i]
E_out = pE_tab.x  # eV
pE = pE_tab.p  # 1/eV
mu_grid = np.linspace(-1, 1, 201)  # cosine grid
# joint density: p(E',mu|E) = p(E'|E)*p(mu|E,E')
P = np.zeros((mu_grid.size, E_out.size))
for j, mu_tab in enumerate(file6.mu[i]):
    P[:, j] = pE[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)  # 1/eV

P_xs = P * n2n_sigma * 1e9 / (1 * np.pi)  # adjust units as needed
plt.figure(figsize=(6, 4))
plt.imshow(
    P_xs,
    origin="lower",
    aspect="auto",
    extent=[E_out[0] / 1e6, E_out[-1] / 1e6, -1, 1],
    norm=LogNorm(vmin=0.1, vmax=P_xs.max()),
)
plt.xlabel("E' [MeV]")
plt.ylabel("μ")
plt.title(f"Joint density p(E',μ|E) at E={E_in / 1e6:g} MeV")
plt.colorbar(label="density [1/eV]")
plt.tight_layout()
plt.grid(False)
plt.savefig("perkins/file6_joint_heatmap.png", dpi=300)

# === Plot 6: animation === #

fig, ax = plt.subplots(figsize=(6, 4))
mu_grid = np.linspace(-1, 1, 201)
# initialize with first frame
i0 = 0
pE_tab = file6.energy_out[i0]
E_out = pE_tab.x
E_in = file6.energy[i0]
pE = pE_tab.p
P = np.zeros((mu_grid.size, E_out.size))
for j, mu_tab in enumerate(file6.mu[i0]):
    P[:, j] = pE[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)
P_xs = P * rx.xs["294K"](E_in) * 1e9 / (1 * np.pi)
im = ax.imshow(
    P_xs,
    origin="lower",
    aspect="auto",
    extent=[E_out[0] / 1e6, E_out[-1] / 1e6, -1, 1],
    norm=LogNorm(vmin=1e-2, vmax=1e3),
)
cbar = fig.colorbar(im, ax=ax)
ax.set_xlabel("E' [MeV]")
ax.set_ylabel("μ")


def update(frame):
    ax.clear()

    pE_tab = file6.energy_out[frame]
    E_out = pE_tab.x
    E_in = file6.energy[frame]
    pE = pE_tab.p

    P = np.zeros((mu_grid.size, E_out.size))
    for j, mu_tab in enumerate(file6.mu[frame]):
        P[:, j] = pE[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)

    P_xs = P * rx.xs["294K"](E_in) * 1e9 / (1 * np.pi)

    im = ax.imshow(
        P_xs,
        origin="lower",
        aspect="auto",
        extent=[E_out[0] / 1e6, E_out[-1] / 1e6, -1, 1],
        norm=LogNorm(vmin=1e-2, vmax=1e3),
    )

    ax.set_xlabel("E' [MeV]")
    ax.set_ylabel("μ")
    ax.set_title(f"E = {E_in / 1e6:.2f} MeV")

    return (im,)


anim = FuncAnimation(fig, update, frames=len(file6.energy), blit=False)
anim.save("perkins/file6_joint_animation.gif", writer=PillowWriter(fps=2))


indices = [1, 3, 5, 7, 9, 11, 13, 16]
ncols, nrows = 4, 2

fig, axes = plt.subplots(nrows, ncols, figsize=(8.27, 4.5), constrained_layout=True)

# compute global vmax so colorscale is shared
all_P_xs = []
for i in indices:
    E_in = file6.energy[i]
    pE_tab = file6.energy_out[i]
    E_out = pE_tab.x
    pE = pE_tab.p
    mu_grid = np.linspace(-1, 1, 201)
    P = np.zeros((mu_grid.size, E_out.size))
    for j, mu_tab in enumerate(file6.mu[i]):
        P[:, j] = pE[j] * np.interp(mu_grid, mu_tab.x, mu_tab.p)
    all_P_xs.append(P * n2n_sigma * 1e9 / (1 * np.pi))

vmin, vmax = 0.1, max(p.max() for p in all_P_xs)
norm = LogNorm(vmin=vmin, vmax=vmax)

for ax, i, P_xs in zip(axes.flat, indices, all_P_xs):
    E_in_MeV = file6.energy[i] * 1e-6
    pE_tab = file6.energy_out[i]
    E_out = pE_tab.x

    im = ax.imshow(
        P_xs,
        origin="lower",
        aspect="auto",
        extent=[E_out[0] / 1e6, E_out[-1] / 1e6, -1, 1],
        norm=norm,
    )
    ax.set_box_aspect(1)
    ax.grid(False)

    # only left column gets y label
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel(r"$\mu$", fontsize=8)
    else:
        ax.set_yticklabels([])

    # only bottom row gets x label
    # if ax.get_subplotspec().is_last_row():
    ax.set_xlabel(r"$E'\ \mathrm{[MeV]}$", fontsize=8)
    # else:
    #     ax.set_xticklabels([])

    ax.tick_params(labelsize=7)

    # energy annotation top-right
    ax.text(
        0.97,
        0.97,
        rf"$E={E_in_MeV:.2f}\ \mathrm{{MeV}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="white",
    )

# subplot labels bottom-center
labels = list("abcdefgh")
# for ax, label in zip(axes.flat, labels):
#     ax.text(
#         0.5,
#         -0.25,
#         f"({label})",
#         transform=ax.transAxes,
#         ha="center",
#         va="top",
#         fontsize=8,
#     )

# one shared colorbar on the right
fig.colorbar(im, ax=axes, label=r"$\mathrm{[MeV^{-1}\,sr^{-1}]}$", shrink=0.6, pad=0.02)

fig.savefig("perkins/file6_joint_grid.pdf", dpi=300)
fig.savefig("perkins/file6_joint_grid.png", dpi=300)
plt.close(fig)
