from pathlib import Path

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
data_color = "#2E5090"  # Deep blue
mean_color = "#C1403D"  # Muted red

base_dir = Path(__file__).parent.resolve()
output_root = base_dir / "outputs"
data = np.load(output_root / "gate_optimization_results.npy", allow_pickle=True)
data1 = np.load(output_root / "gate_optimization_results1.npy", allow_pickle=True)

# Convert the 0-d array back into your list of dictionaries
results = data.tolist()
# results1 = data1.tolist()
# combined_raw = results + results1
# # This keeps the LAST instance of any duplicate gate found.
# merged_dict = {r["gate"]: r for r in combined_raw}
# # Sort by gate width so the plot lines don't zig-zag
# sorted_gates = sorted(merged_dict.keys())
# results = [merged_dict[g] for g in sorted_gates]

# Quick check
print(f"Loaded {len(results)} gate results.")
print(f"First gate width: {results[0]['gate']}")


gates_tau = [r["gate_over_tau"] for r in results]

r0_unc = [(r["r_std"][0] / abs(r["r_full"][0]) * 100) for r in results]
r1_unc = [r["rel_unc_r1"] * 100 for r in results]
r2_unc = [r["rel_unc_r2"] * 100 for r in results]

print(f"minimum for r1 uncertainty {gates_tau[np.argmin(r1_unc)]} {np.min(r1_unc):.2f}")
print(f"minimum for r2 uncertainty {gates_tau[np.argmin(r2_unc)]} {np.min(r2_unc):.2f}")
print(
    f"r1 minimum for r2 uncertainty {gates_tau[np.argmin(r1_unc)]} {r2_unc[np.argmin(r1_unc)]:.2f}"
)


fig, axes = plt.subplots(1, 2, figsize=(8.27, 3.0), constrained_layout=True)

# --- (a) doubles ---
axes[0].plot(gates_tau, r1_unc, "o-", color=data_color, markersize=3)
axes[0].axvline(
    gates_tau[np.argmin(r1_unc)],
    ls="--",
    color=mean_color,
    alpha=0.7,
    label=f"Optimal: {gates_tau[np.argmin(r1_unc)]:.1f}τ",
)
axes[0].set_xlabel(r"Gate width ($\tau$)")
axes[0].set_ylabel(r"Rel. uncertainty on $r_1$ (%)")
axes[0].legend()
axes[0].grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
axes[0].set_axisbelow(True)
axes[0].set_xlim(0, gates_tau[-1] * 1.1)

# --- (b) triples ---
axes[1].plot(gates_tau, r2_unc, "o-", color=data_color, markersize=3)
axes[1].axvline(
    gates_tau[np.argmin(r2_unc)],
    ls="--",
    color=mean_color,
    alpha=0.7,
    label=f"Optimal: {gates_tau[np.argmin(r2_unc)]:.1f}τ",
)
axes[1].set_xlabel(r"Gate width ($\tau$)")
axes[1].set_ylabel(r"Rel. uncertainty on $r_2$ (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
axes[1].set_axisbelow(True)
axes[1].set_xlim(0, gates_tau[-1] * 1.1)

# subplot labels
for ax, label in zip(axes, ["(a)", "(b)"]):
    ax.text(
        0.5, -0.22, label, transform=ax.transAxes, ha="center", va="top", fontsize=9
    )

fig.savefig("gate_optimization.png", dpi=300)

# fig, ax = plt.subplots(figsize=(7, 5))
#
# # ax.plot(gates_tau, r0_unc, "o-", color=data_color, label="Rel. Unc. $r_0$", alpha=1)
# # ax.plot(gates_tau, r1_unc, "o-", color=data_color, label="Rel. Unc. $r_1$", linewidth=2)
# ax.plot(gates_tau, r2_unc, "o-", color=data_color, label="Rel. Unc. $r_2$", alpha=1)
#
# ax.axvline(0.4, ls="--", color=mean_color, alpha=0.5, label="Optimal: 0.4τ")
#
# ax.set_xlabel("Gate width (τ)", fontsize=12)
# ax.set_ylabel("Relative uncertainty on $r_2$ (%)", fontsize=12)
# # ax.set_title("Gate Width Optimization for Be(n,2n) System")
#
# ax.legend()
# ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
# ax.set_axisbelow(True)
# ax.set_xlim(0, gates_tau[-1] * 1.1)
#
# plt.tight_layout()
# plt.savefig("gate_optimization.png", dpi=300, bbox_inches="tight")
