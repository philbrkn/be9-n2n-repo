import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

data_color = "#2E5090"
mean_color = "#C1403D"  # Muted red

radii = [7, 8, 9, 10, 11]  # cm

# Fixed outer radius (He-3 at 15 cm, outer at 20 cm)
fixed_outer = {
    "S_r0": [-0.018, -0.020, -0.026, -0.036, -0.044],
    "S_r1": [0.726, 0.538, 0.493, 0.493, 0.458],
    "S_r2": [-2.991, 0.939, 1.440, 1.020, 0.981],
    "S_det": [0.506, 0.571, 0.624, 0.671, 0.699],
    # R² for r1 and det (used to shade reliability)
    "R2_r1": [0.9608, 0.9957, 0.9857, 0.9876, 0.9993],
    "R2_det": [0.9999, 0.9998, 0.9996, 0.9995, 0.9994],
}

# Fixed moderator thickness (HDPE always 3 cm outside Be, He-3/outer scale with Be)
fixed_mod = {
    "S_r0": [-0.032, -0.031, -0.026, -0.025, -0.021],
    "S_r1": [0.564, 0.557, 0.493, 0.508, 0.461],
    "S_r2": [0.947, 1.015, 1.440, 2.097, 1.230],
    "S_det": [0.650, 0.641, 0.624, 0.606, 0.580],
    "R2_r1": [0.9988, 0.9893, 0.9857, 0.9932, 0.9670],
    "R2_det": [0.9999, 0.9997, 0.9996, 0.9994, 0.9992],
}

MARKER_FIXED_OUTER = "o"
MARKER_FIXED_MOD = "s"
# bar plot,  S_r1 and S_det vs Be radius

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.27, 3.5),
    sharex=True,
    constrained_layout=True,
)


ax_r1, ax_det = axes

NOMINAL = 9  # cm


def plot_panel(ax, key, ylabel, ylim=None, legend=False):
    fo = fixed_outer[key]
    fm = fixed_mod[key]

    ax.axvline(NOMINAL, color="gray", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="gray", lw=0.5, ls=":", zorder=0)

    ax.plot(
        radii,
        fo,
        color=mean_color,
        marker=MARKER_FIXED_OUTER,
        ls="--",
        label="Fixed outer radius",
        zorder=3,
        alpha=0.8,
    )
    ax.plot(
        radii,
        fm,
        color=data_color,
        marker=MARKER_FIXED_MOD,
        ls="-",
        label="Fixed mod. thickness",
        zorder=3,
    )

    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if legend:
        ax.legend(frameon=False, loc="best")


# Panel (a): S_r1
plot_panel(
    ax_r1,
    "S_r1",
    r"$S_{r_1}$  (multiplicity sensitivity)",
    ylim=(0.35, 0.85),
    legend=True,
)

# Panel (b): S_det
plot_panel(
    ax_det, "S_det", r"$S_\mathrm{det}$  (count-rate sensitivity)", ylim=(0.45, 0.75)
)

ax_r1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
ax_det.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
ax_r1.set_xlabel(r"Be sphere radius  $r_\mathrm{Be}$  (cm)")
ax_det.set_xlabel(r"Be sphere radius  $r_\mathrm{Be}$  (cm)")

# Nominal annotation (only on top panel)
ax_r1.annotate(
    "nominal\n9 cm",
    xy=(NOMINAL, 0.37),
    xytext=(9.25, 0.39),
    fontsize=8,
    color="gray",
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
)

for ax, label in zip(axes, ["(a)", "(b)"]):
    ax.text(
        0.5,
        -0.18,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        # fontweight="bold",
    )

fig.savefig("figures/be_sweep_sensitivity.pdf", bbox_inches="tight")
fig.savefig("figures/be_sweep_sensitivity.png", dpi=300, bbox_inches="tight")
print("Saved be_sweep_sensitivity.pdf and .png")
