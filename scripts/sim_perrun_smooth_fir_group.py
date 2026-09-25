"""Group-level (across 'subjects') test of the per-run trend in amp@peak and latency.
Each simulated voxel/ROI is one subject; 20 subjects per group test; 150 group tests."""

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import sim_perrun_smooth_fir as S

N_SUB = 20
for label, kw in [
    ("NULL, noise rises 1x->2x", dict(noise_scale=np.linspace(1, 2, S.R))),
    ("NULL, equal noise", {}),
    ("REAL: amplitude falls 30%", dict(amp_scale=np.linspace(1, 0.7, S.R))),
    (
        "REAL: amplitude falls 30%, noise rises",
        dict(amp_scale=np.linspace(1, 0.7, S.R), noise_scale=np.linspace(1, 2, S.R)),
    ),
]:
    for cnr in (1.0, 2.0):
        ests, truth, kt, _, _ = S.simulate(cnr, seed0=11, **kw)
        pk = int(np.argmax(S.hrf(kt)))
        print(f"\n{label}  CNR {cnr}: group-level rejection (%) [amp@peak | latency]")
        for m in S.METHODS:
            f = S.features(ests[m], kt, pk)
            row = []
            for n in ("amp@peak", "latency"):
                sl, _ = S.slope_test(f[n])
                g = sl[: len(sl) // N_SUB * N_SUB].reshape(-1, N_SUB)
                p = stats.ttest_1samp(g, 0, axis=1).pvalue
                row.append(f"{100 * (p < 0.05).mean():5.1f}")
            print(f"  {m:>11}  {row[0]} | {row[1]}")
