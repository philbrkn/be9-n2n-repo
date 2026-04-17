import matplotlib.pyplot as plt
import numpy as np

# Kumabe parameters from Eq (5a,5b) for nucleon emissions
# A_l = 0.0561 + 0.0377*l
# B_l = 47.9 - 27.1*l^{-1/2}

epsilon = np.linspace(0, 20, 200)  # outgoing energy in MeV

fig, ax = plt.subplots()
for l in range(1, 5):  # l = 1,2,3,4 (lmax=4 in Kumabe)
    A_l = 0.0561 + 0.0377 * l
    B_l = 47.9 - 27.1 * l ** (-0.5)
    b_l = (2 * l + 1) / (1 + np.exp(A_l * (B_l - epsilon)))
    ax.plot(epsilon, b_l, label=f"$l={l}$")

ax.set_xlabel("Outgoing energy $\\epsilon$ (MeV)")
ax.set_ylabel("$b_l(\\epsilon)$")
ax.legend()
plt.tight_layout()
plt.savefig("tendl/kumabe_bl.png", dpi=300)
