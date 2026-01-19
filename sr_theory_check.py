import numpy as np

# my measured values
epsilon = 0.3344  # from tally (detection efficiency per leakage neutron)
tau = 42e-6
predelay = 4e-6
gate = 42e-6
n_sources = 10000

# F factor
F = np.exp(-predelay / tau) * (1 - np.exp(-gate / tau))

# Calculate <v(v-1)/2> from your P_s(v) leakage distribution
v = np.array([1, 2, 3, 4, 5, 6, 7])
counts = np.array([4595, 3665, 1065, 529, 103, 38, 4])
P_s = counts / counts.sum()
v_v_minus_1_half = np.sum(v * (v - 1) / 2 * P_s)

# Predicted R
R_predicted = n_sources * F * epsilon**2 * v_v_minus_1_half

print(f"F = {F:.4f}")
print(f"<v(v-1)/2> = {v_v_minus_1_half:.4f}")
print(f"Predicted R = {R_predicted:.0f}")
print("Measured R = 425")
