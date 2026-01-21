import numpy as np
import matplotlib.pyplot as plt

# Your data
scales = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
r1_means = np.array([0.085428, 0.091143, 0.095586, 0.100190, 0.104885])
r1_sems = np.array([0.000402, 0.000462, 0.000731, 0.000683, 0.000271])

# Fit a line to get sensitivity
coeffs = np.polyfit(scales, r1_means, 1)
slope = coeffs[0]
intercept = coeffs[1]

# Sensitivity: (dr/r) / (dσ/σ) at nominal
r_nominal = r1_means[2]  # scale=1.0
S = slope / r_nominal  # dr/dσ normalized

print(f"Sensitivity coefficient S = {S:.3f}")
print(f"Interpretation: 1% change in (n,2n) causes {S:.2f}% change in r[1]")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(scales, r1_means, yerr=r1_sems, fmt='o', capsize=5, markersize=8, label='Data')
ax.plot(scales, np.polyval(coeffs, scales), 'r--', label=f'Linear fit (slope={slope:.4f})')
ax.set_xlabel('(n,2n) Cross Section Scale Factor', fontsize=12)
ax.set_ylabel('r[1] (Doubles Rate)', fontsize=12)
ax.set_title('Sensitivity of Doubles Rate to Be-9 (n,2n) Cross Section', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('r1_sensitivity.png', dpi=150)
plt.show()