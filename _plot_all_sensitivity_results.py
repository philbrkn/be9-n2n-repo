import matplotlib.pyplot as plt
import numpy as np

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
observables = ["$r_0$", "$r_1$", "$r_2$", "Det. count", "$k_{\\mathrm{eff}}$"]

s_n2n = [0.04, 0.50, 1.20, 0.60, 0.031]
s_ddx = [0.03, 0.55, 1.84, 0.20, None]  # no DDX bar for k_eff

endf_color = "#2E5090"
tendl_color = "#C1403D"

x = np.arange(len(observables))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

# n2n bars for all 5
ax.bar(
    x - width / 2,
    [abs(v) for v in s_n2n],
    width,
    label=r"$S$ to $\sigma_{n,2n}$",
    color=endf_color,
    edgecolor="none",
    alpha=0.85,
)

# DDX bars only for first 4
ax.bar(
    x[:4] + width / 2,
    [abs(v) for v in s_ddx[:4]],
    width,
    label=r"$S$ to DDX",
    color=tendl_color,
    edgecolor="none",
    alpha=0.85,
)


ax.set_ylabel(r"Sensitivity $|S|$")
ax.set_xticks(x)
ax.set_xticklabels(observables)
ax.set_xlim(-0.5, len(observables) - 0.5)
ax.set_ylim(0, 2.2)
ax.legend(frameon=True, fancybox=False, edgecolor="gray", framealpha=0.95)
ax.grid(True, axis="y", linestyle="-", alpha=0.15)
ax.set_axisbelow(True)

fig.savefig("sensitivity_comparison.pdf", dpi=300)
# fig.savefig("sensitivity_comparison.png", dpi=300)
