import sys

import nibabel as nib
import numpy as np

d = sys.argv[1]
names = sys.argv[2:]
tr = np.load(f"{d}/truth.npz")
dt = float(tr["dt"])
print(
    "%-8s %3s | lag0 bias pial/mid/deep  | peak est/true pial mid deep | bias-rms  noise-sd  RMSE/knot | hi: (gender-fixate) peak diff est/true pial, deep"
    % ("est", "nk")
)
for name in names:
    res = {}
    for cond in ("fixate_hi", "gender_hi"):
        est = nib.load(f"{d}/est_{name}_iresp_{cond}.nii.gz").get_fdata()[:, :, 0, :]
        nk = est.shape[-1]
        times = np.linspace(0, 14.8, nk)
        h = tr[cond]
        tf = np.arange(h.shape[0]) * dt
        true = np.stack([np.interp(times, tf, h[:, k]) for k in range(7)])
        res[cond] = (est, true)
    est, true = res["fixate_hi"]
    m = est.mean(1)
    pk = true.argmax(1)
    ratio = [m[k, pk[k]] / true[k, pk[k]] for k in (0, 3, 6)]
    dE = (res["gender_hi"][0] - est).mean(1)
    dT = res["gender_hi"][1] - true
    pkd = np.abs(dT).argmax(1)
    print(
        "%-8s %3d | %+.3f %+.3f %+.3f     | %.3f %.3f %.3f           | %.4f    %.4f    %.4f    | %.2f %.2f"
        % (
            name,
            nk,
            m[0, 0],
            m[3, 0],
            m[6, 0],
            *ratio,
            np.sqrt(((m - true) ** 2).mean()),
            est.std(1).mean(),
            np.sqrt(((est - true[:, None]) ** 2).mean()),
            dE[0, pkd[0]] / dT[0, pkd[0]],
            dE[6, pkd[6]] / dT[6, pkd[6]],
        )
    )
