"""Ground-truth laminar time series on the real sub-3010 ses-fine event timing."""

import csv, glob, os, sys, json, numpy as np, torch, nibabel as nib
from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.laminar.pipeline import reference_p0
from fastfuncstuff.laminar.experiment import faes_priors
from fastfuncstuff.laminar.integrate import integrate, build_input

torch.set_num_threads(2)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
STIM = "/mnt/belegost/Projects/HiHiFaces/fine_outputs/ffs_autoproc_sub-3010_ses-fine_nonlmoco.results/stimuli"
K, N, TR, NT, R, DUR = 7, 3, 1.857, 123, 120, 1.0
dt = 0.05
T = 32.0
spec = ModelSpec(N=N, K=K, n_inputs=2, n_mod=1, TR=dt, dt=dt, p0=reference_p0())


def response(c_scale, B):
    P = {k: v.clone() for k, v in faes_priors(spec, (1, 1, 1)).pE.items()}
    P["C"] = torch.tensor([[0.0, 1.0], [0.0, 1.4], [0.0, 0.9]], dtype=torch.float64) * c_scale
    P["B"] = torch.diag(torch.tensor(B, dtype=torch.float64))[None]
    n = int(T / dt)
    u = build_input(spec, [[0.0], [0.0]], [[DUR], [DUR]], n)
    y = integrate(u, P, spec, n, delay_seconds=0.0)  # (n, K) superficial first, read at i*dt
    return y.numpy(), P


mods = {"fixate": [0, 0, 0], "gender": [0, 0, 0.6], "expres": [0.6, 0, 0]}
amp = {"hi": 1.0, "lo": 0.7}
# scale so the pial peak of fixate_hi is ~1.8 %
y0, _ = response(1.0, mods["fixate"])
scale = 1.8 / y0[:, 0].max()
truth = {}
for task, B in mods.items():
    for lvl, a in amp.items():
        y, P = response(scale * a, B)
        truth[f"{task}_{lvl}"] = y
np.savez(f"{out}/truth.npz", dt=dt, **truth)
print(
    "scale",
    scale,
    {
        k: (round(v[:, 0].max(), 2), round(v[:, -1].max(), 2), round(v[:, 0].min(), 3))
        for k, v in truth.items()
    },
)
tfine = np.arange(int(T / dt)) * dt
rng = np.random.default_rng(1)
sd = np.linspace(0.25, 0.15, K)  # pial noisier, percent per TR for an ROI mean
rho = 0.35
i = 0
for task in ("expres", "fixate", "gender"):
    for r in (1, 2, 3):
        i += 1
        f = f"{STIM}/sub-3010_ses-fine_task-{task}_run-{r}_events.tsv"
        rows = list(csv.DictReader(open(f), delimiter="\t"))
        for row in rows:
            row["onset"] = "%.4f" % ((np.floor(float(row["onset"]) / TR) + 0.5) * TR)
        ts = np.arange(NT) * TR
        sig = np.zeros((NT, K))
        for row in rows:
            h = truth[f"{task}_{row['stim_contrast']}"]
            lag = ts - float(row["onset"])
            for k in range(K):
                sig[:, k] += np.interp(lag, tfine, h[:, k], left=0.0, right=0.0)
        e = rng.standard_normal((NT, K, R))
        for n in range(1, NT):
            e[n] = rho * e[n - 1] + np.sqrt(1 - rho**2) * e[n]
        y = sig[:, :, None] + e * sd[None, :, None]
        drift = rng.standard_normal((1, K, R)) * 0.3 * np.linspace(-1, 1, NT)[:, None, None]
        data = 1000.0 * (1 + (y + drift) / 100.0)
        img = nib.Nifti1Image(
            np.transpose(data, (1, 2, 0))[:, :, None, :].astype(np.float32), np.diag([1, 1, 1, 1.0])
        )
        img.header.set_zooms((1, 1, 1, TR))
        img.header.set_xyzt_units("mm", "sec")
        nib.save(img, f"{out}/task-{task}_run-{r}_bold.nii.gz")
        w = csv.DictWriter(
            open(f"{out}/task-{task}_run-{r}_events.tsv", "w"),
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(rows)
print("done")
