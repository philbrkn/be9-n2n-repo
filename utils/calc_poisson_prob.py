from math import exp

import numpy as np
from numpy.fft import fft, ifft

# input params
source_rate = 3e4
gate = 28e-6

mean_neutrons = source_rate * gate
print(f"There are {mean_neutrons:.2f} mean source neutrons per gate.")

# calculate the poisson probability of 2 or more source neutrons
prob_of_2 = 1 - mean_neutrons * exp(-mean_neutrons) - exp(-mean_neutrons)
print(
    f"There is a probability of {prob_of_2:.3f} of 2 or more source neutrons in one gate."
)


# safe would be P(n>=2) <0.01


def poisson_compound(r, mu, n_points=None):
    """Poisson-compounded distribution via FFT."""
    if n_points is None:
        mean_burst = np.dot(np.arange(len(r)), r)
        var_burst = np.dot(np.arange(len(r)) ** 2, r) - mean_burst**2
        n_points = int(mu * mean_burst + 10 * np.sqrt(mu * var_burst + 1)) + len(r)
        n_points = max(n_points, 2 * len(r), 64)
        print(f"Using n_points {n_points}.")

    r_padded = np.zeros(n_points)
    r_padded[: len(r)] = r

    R_fft = fft(r_padded)
    G_fft = np.exp(mu * (R_fft - 1))
    p = np.real(ifft(G_fft))
    p = np.clip(p, 0, None)
    p /= p.sum()

    return p


def poisson_decompound(p, mu, n_points=None):
    """Inverse of Poisson compounding via FFT."""
    if n_points is None:
        n_points = max(len(p), 64)

    p_padded = np.zeros(n_points)
    p_padded[: len(p)] = p

    G_fft = fft(p_padded)

    # Invert: R(z) = 1 + log(G(z)) / mu
    # Need to handle log carefully
    R_fft = 1 + np.log(G_fft) / mu

    r = np.real(ifft(R_fft))
    r = np.clip(r, 0, None)
    r /= r.sum()
    return r


measured_r = [0.9513, 0.0473, 0.0013, 0.0001]
r = np.array([9.678e-1, 3.130e-2, 7.1e-4, 1.392e-5])
r /= r.sum()
p_gate = poisson_decompound(r, mean_neutrons)


print("Single-burst r_pred:")
for k in range(len(r)):
    print(f"  r[{k}] = {r[k]:.6f}")

print(f"\nPoisson-decompounded (mu={mean_neutrons:.2f}):")
for m in range(min(10, len(p_gate))):
    if p_gate[m] > 1e-7:
        print(f"  p[{m}] = {p_gate[m]:.6f}")

print("\nMeasured:")
for m in range(len(measured_r)):
    print(f"  r[{m}] = {measured_r[m]:.6f}")
