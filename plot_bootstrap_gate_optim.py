from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#4D4D4D",
}

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

base_dir = Path(__file__).parent.resolve()
output_root = base_dir / "outputs"
data = np.load(output_root / "gate_optimization_results.npy", allow_pickle=True)
data1 = np.load(output_root / "gate_optimization_results1.npy", allow_pickle=True)

# Convert the 0-d array back into your list of dictionaries
results = data.tolist()
results1 = data1.tolist()
combined_raw = results + results1
# This keeps the LAST instance of any duplicate gate found.
merged_dict = {r["gate"]: r for r in combined_raw}
# Sort by gate width so the plot lines don't zig-zag
sorted_gates = sorted(merged_dict.keys())
results = [merged_dict[g] for g in sorted_gates]

# Quick check
print(f"Loaded {len(results)} gate results.")
print(f"First gate width: {results[0]['gate']}")

gates_tau = [r["gate_over_tau"] for r in results]

r0_unc = [(r["r_std"][0] / abs(r["r_full"][0]) * 100) for r in results]
r1_unc = [r["rel_unc_r1"] * 100 for r in results]
r2_unc = [r["rel_unc_r2"] * 100 for r in results]

fig, ax = plt.subplots(figsize=(7, 5))

# ax.plot(gates_tau, r0_unc, "o-", color=data_color, label="Rel. Unc. $r_0$", alpha=1)
ax.plot(gates_tau, r1_unc, "o-", color=data_color, label="Rel. Unc. $r_1$", linewidth=2)
# ax.plot(gates_tau, r2_unc, "o-", color=data_color, label="Rel. Unc. $r_2$", alpha=1)

ax.axvline(0.4, ls="--", color=mean_color, alpha=0.5, label="Optimal: 0.4τ")

ax.set_xlabel("Gate width (τ)", fontsize=12)
ax.set_ylabel("Relative uncertainty on $r_1$ (%)", fontsize=12)
# ax.set_title("Gate Width Optimization for Be(n,2n) System")

ax.legend()
ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xlim(0, gates_tau[-1] * 1.1)

plt.tight_layout()
plt.savefig("gate_optimization.png", dpi=300, bbox_inches="tight")
