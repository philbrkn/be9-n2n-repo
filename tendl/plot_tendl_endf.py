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
# Data from your bootstrap results
labels = ["$r_0$ (Singles)", "$r_1$ (Doubles)", "$r_2$ (Triples)"]

# ENDF Nominal values
endf_vals = [0.95131, 0.04733, 0.00137]
endf_errs = [0.00015, 0.00016, 0.00007]

# TENDL values
tendl_vals = [0.96980, 0.03008, 0.00011]
tendl_errs = [0.00014, 0.00016, 0.00007]

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

endf_color = "#2E5090"
tendl_color = "#C1403D"
width = 0.35

rects1 = ax.bar(
    x - width / 2,
    endf_vals,
    width,
    label="ENDF/B-VIII.0",
    yerr=endf_errs,
    capsize=3,
    color=endf_color,
    edgecolor="none",
    alpha=0.85,
)

rects2 = ax.bar(
    x + width / 2,
    tendl_vals,
    width,
    label="TENDL-2025",
    yerr=tendl_errs,
    capsize=3,
    color=tendl_color,
    edgecolor="none",
    alpha=0.85,
)

ax.set_yscale("log")
ax.set_ylabel(r"Shift-register moment $r_k$")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(1e-5, 2.0)
ax.set_xlim(-0.5, len(labels) - 0.5)
ax.legend(frameon=True, fancybox=False, edgecolor="gray", framealpha=0.95)
ax.grid(True, which="both", axis="y", linestyle="-", alpha=0.15)
ax.set_axisbelow(True)

for i in range(len(labels)):
    diff = (tendl_vals[i] - endf_vals[i]) / endf_vals[i] * 100
    y_pos = max(tendl_vals[i], endf_vals[i]) * 1.2
    ax.text(
        x[i],
        y_pos,
        f"{diff:+.0f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
    )


fig.savefig("tendl/multiplicity_sensitivity.pdf", dpi=300)
fig.savefig("tendl/multiplicity_sensitivity.png", dpi=300)
