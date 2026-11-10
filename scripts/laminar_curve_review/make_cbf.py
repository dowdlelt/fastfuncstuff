import sys

import numpy as np

K = 7
dt = 0.01
T = 40
t = np.arange(int(T / dt)) * dt
# depth-varying smooth CBF response: different amplitude/latency per depth, with undershoot
cbf = np.ones((len(t), K))
for k in range(K):
    a = 0.3 + 0.08 * k
    t0 = 2 + 0.15 * k
    s = np.clip(t - t0, 0, None)
    cbf[:, k] = (
        1
        + a * (s**2 * np.exp(-s / 1.2)) / (4 * 1.2**2 * np.exp(-2))
        - 0.05 * a * (s**3 * np.exp(-s / 2.5)) / (27 * 2.5**3 * np.exp(-3))
    )
np.savetxt(sys.argv[1], cbf, fmt="%.17g")
