import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    }
)
data_color = "#2E5090"  # Deep blue
mean_color = "#C1403D"  # Muted red
# Data
scale = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
r1 = np.array([4.219e-02, 4.482e-02, 4.708e-02, 4.952e-02, 5.169e-02])
r1_sem = np.array([1.50e-04, 1.57e-04, 1.64e-04, 1.43e-04, 1.60e-04])
r2 = np.array([1.021e-03, 1.132e-03, 1.367e-03, 1.518e-03, 1.645e-03])
r2_sem = np.array([6.43e-05, 6.10e-05, 6.59e-05, 6.34e-05, 7.30e-05])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# r_1 panel
ax1.errorbar(
    scale, r1, yerr=r1_sem, fmt="o-", capsize=4, markersize=4, color=data_color
)
ax1.set_xlabel("(n,2n) cross section scale factor")
ax1.set_ylabel("$r_1$ (doubles)")
ax1.axvline(1.0, ls="--", color="gray", alpha=0.5)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
ax1.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
ax1.set_axisbelow(True)

# r_2 panel
ax2.errorbar(
    scale, r2, yerr=r2_sem, fmt="o-", capsize=4, markersize=4, color=data_color
)
ax2.set_xlabel("(n,2n) cross section scale factor")
ax2.set_ylabel("$r_2$ (triples)")
ax2.axvline(1.0, ls="--", color="gray", alpha=0.5)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
ax2.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig("n2n_sensitivity_r1_r2.png", dpi=300)
