import sys, numpy as np, nibabel as nib

d = sys.argv[1]
tr = np.load(f"{d}/truth.npz")
dt = float(tr["dt"])
print(
    "%-7s %3s | first knot bias pial/mid/deep | peak est/true pial mid deep | bias-rms noise-sd RMSE/knot | roughness(est)/roughness(true)"
    % ("est", "nk")
)
for spec in sys.argv[2:]:
    name, lo, hi = spec.split("@")[0], *map(float, spec.split("@")[1].split(","))
    est = nib.load(f"{d}/est_{name}_iresp_fixate_hi.nii.gz").get_fdata()[:, :, 0, :]
    nk = est.shape[-1]
    times = np.linspace(lo, hi, nk)
    h = tr["fixate_hi"]
    tf = np.arange(h.shape[0]) * dt
    true = np.stack([np.interp(times, tf, h[:, k]) for k in range(7)])
    m = est.mean(1)
    pk = true.argmax(1)
    rough = lambda a: np.abs(np.diff(a, 2, axis=-1)).mean()
    print(
        "%-7s %3d | %+.3f %+.3f %+.3f           | %.3f %.3f %.3f          | %.4f   %.4f   %.4f     | %.2f"
        % (
            name,
            nk,
            *(m[k, 0] - true[k, 0] for k in (0, 3, 6)),
            *(m[k, pk[k]] / true[k, pk[k]] for k in (0, 3, 6)),
            np.sqrt(((m - true) ** 2).mean()),
            est.std(1).mean(),
            np.sqrt(((est - true[:, None]) ** 2).mean()),
            rough(est) / rough(true),
        )
    )
