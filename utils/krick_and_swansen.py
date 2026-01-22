import numpy as np
import pandas as pd


def get_measured_multiplicity_causal(rplusa_dist, a_dist):
    """
    Given pulse triggred distribution p (R+A) and delayed distribution q (A),
    compute real SR multiplicity distrbiution using:
    from Krick and Swansen 84:
        [r_k] = [U_kn]^-1 [p_k]
        U_kn = q_(k-n) if k>=n, else =0
    """
    count_total = np.asarray(rplusa_dist, dtype=float).sum()
    # turn the distribution into probabilities:
    p = np.asarray(rplusa_dist / count_total, dtype=float)
    q = np.asarray(a_dist / count_total, dtype=float)
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    N = len(p)
    r = np.zeros(N, float)
    if q[0] == 0:
        raise ValueError("q[0] must be nonzero for causal deconvolution.")
    for k in range(N):
        s = 0.0
        # sum_{i=1..k} q[i] * r[k-i]
        for i in range(1, k + 1):
            s += q[i] * r[k - i]
        r[k] = (p[k] - s) / q[0]
    return r


def get_measured_multiplicity(rplusa_dist, a_dist):
    """
    Given pulse triggred distribution p (R+A) and delayed distribution q (A),
    compute real SR multiplicity distrbiution using:
    from Krick and Swansen 84:
        [r_k] = [U_kn]^-1 [p_k]
        U_kn = q_(k-n) if k>=n, else =0
    """
    count_total = np.asarray(rplusa_dist, dtype=float).sum()
    # turn the distribution into probabilities:
    p = np.asarray(rplusa_dist / count_total, dtype=float)
    q = np.asarray(a_dist / count_total, dtype=float)
    # Ukn = q_(k-n) shift operator
    N = len(q)
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    lag = k - n
    # where k is greater than n (lower triangular matrix), make it convolution matri
    U = np.where(k >= n, q[lag], 0.0)
    r = np.linalg.solve(U, p)
    return r


def recreate_paper_data():
    # Summary statistics from the top of Table 1
    measurement_metadata = {
        "sample": "531g PuO2 powder (9.2% 240Pu)",
        "measurement_time_s": 76000,
        "predelay_us": 4.5,
        "gate_length_us": 32.0,
        "total_counts": 2226135758,
        "R_plus_A_counts": 2299510651,
        "A_counts": 2086564893,
    }

    # Multiplicity distribution data
    data = {
        "Multiplicity": [0, 1, 2, 3, 4, 5, 6, 7, ">= 8"],
        # "Multiplicity": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "R_plus_A": [
            823605527,
            787936230,
            407018425,
            149965354,
            43919007,
            10827279,
            2327067,
            445039,
            91830,
        ],
        "A": [
            903540582,
            783595250,
            369842869,
            125386442,
            34041080,
            7818722,
            1572969,
            283129,
            54715,
        ],
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    return measurement_metadata, df


if __name__ == "__main__":
    measurement_metadata, df = recreate_paper_data()
    print("--- Measurement Metadata ---")
    for key, value in measurement_metadata.items():
        print(f"{key}: {value}")
    print("\n--- Multiplicity Table ---")
    print(df.to_string(index=False))
    # sum of R_plus_A:
    # print(f"Sum of r+a column should = total counts: {df['R_plus_A'].sum()}")
    # weighted sum of R_plus_A:
    # print(f"Zip sum of r+a : {(df['R_plus_A'] * df['Multiplicity']).sum()}")
    rmeasured = get_measured_multiplicity(df["R_plus_A"], df["A"])
    print(rmeasured)
    rmeasured = get_measured_multiplicity_causal(df["R_plus_A"], df["A"])
    print(rmeasured)
