"""Generate ``laminar_dcm_predictive_tones.ipynb``.

The notebook is written from here so the cell text can be reviewed as source and
regenerated after a library change. Run::

    python notebooks/make_laminar_dcm_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------

md("""
# Laminar BOLD DCM — replicating Faes et al. (2026)

Model-based **deveining**: recovering laminar *neuronal* activity from
depth-sampled GE-BOLD by inverting a physiological forward model, rather than by
static spatial deconvolution or vein masking.

This notebook runs the full pipeline of `apply_laminar_BOLD_model.m` on the
authors' published test dataset (Planum Polare, left hemisphere), in PyTorch,
through `fastfuncstuff.laminar`.

**Everything here is verified against the reference MATLAB**, not just
qualitatively similar to it. The state equation, the observation equation and the
free energy all match to machine precision on a fixed oracle
(`tests/test_laminar_forward.py`, `tests/test_laminar_inversion.py`).

### What this does that the reference driver does not

The MATLAB driver ships three steps commented out or cached, each with a note
about how long they take. We do all of them:

| step | driver | here |
|---|---|---|
| depth downsampling | cached in `ds_vox_depth_PP_LH.mat`, loop commented out | recomputed, ~instant |
| voxel PSF estimation | cached in `PP_LH_PSFkernel.mat`, *"takes long!"* | recomputed **per K** |
| vascular resolutions | `Ki = [7, 9]`, *"takes long for each"* | all four, `[7, 9, 10, 11]` |
| model space | 8 models × 2 K = 16 inversions | 8 × 4 = **32** |

The reference also loads one cached kernel (length 7) and reuses it at every K.
Since we estimate the kernel in seconds, each K gets its own.

### Why it is fast enough to bother

One Ito–Taylor integration step costs ~34 ms eager, because PyTorch has no native
forward-mode AD rule for most of these operations and falls back to Python
reference decompositions. Compiled, the dual computation and the unrolled depth
loop fuse into one graph: **245 µs**, a 137× speedup, agreeing with eager to
1e-13. A full integration goes 41 s → 0.28 s.

And because the work is *host-bound*, a batch of 9 costs the same as a batch of
1 — so the Jacobian's `np + 1` probe evaluations go in a single call.
""")

code("""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from fastfuncstuff.laminar import (
    ModelSpec, P0,
    apply_deveining, build_input, deveined_timecourses, deveining_fidelity,
    estimate_depth_psf, fit_model_space, integrate, label_mean,
    laminar_impulse_response, layer_model_names, posterior_model_probabilities,
    static_deveining_matrix, voxels_to_layers,
)

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.grid": True,
    "axes.axisbelow": True, "grid.alpha": 0.3, "figure.facecolor": "white",
})
torch.set_num_threads(max(1, torch.get_num_threads() // 2))
""")

md("""
## Configuration
""")

code("""
CFG = dict(
    data_root=Path("/home/logan/Dropbox/Resources/code/matlab_toolboxes/predictive_tones"),

    # Model sizes. Three neuronal depths is fixed across the literature; the
    # vascular depths are the set Uludag & Havlicek (2021) recommend for it --
    # they show 7-10 BOLD depths are needed to resolve 3 neuronal ones.
    N=3,
    K_values=[7, 9, 10, 11],

    # The contrast. 1 = Comp_H (predictable, baseline), 3 = Oddball_H
    # (mispredicted). Condition *differences* are the right target: the draining
    # bias is largely common to both and partly cancels in the contrast.
    cond_baseline=1,
    cond_test=3,

    TR=1.6,
    dt=0.05,          # microtime bin; TR/dt must be an integer
    gap_size=20,      # white-noise TRs separating the two concatenated ERAs
    seed=20260918,
)

CONDITION_NAMES = ["Comp_H", "Comp_L", "Oddball_H", "Oddball_L", "Unexp_H", "Unexp_L"]
print("contrast:", CONDITION_NAMES[CFG["cond_baseline"] - 1],
      "vs", CONDITION_NAMES[CFG["cond_test"] - 1])
""")

md("""
## 1. Load the example data

`perVoxResp` is the event-related average per condition, per voxel
(6 conditions × 999 voxels × 8 TRs), already normalised to percent signal
change. `EVV2` is the equivolume cortical depth on the 0.4 mm anatomical grid;
`EV_vox_d` is the same thing downsampled onto the 0.8 mm functional grid.
""")

code("""
root = CFG["data_root"]
test_data = sio.loadmat(root / "PP_LH_testData.mat")
layer_dist = sio.loadmat(root / "PP_LH_layer_dist_testData.mat")

per_vox = np.asarray(test_data["perVoxResp"], dtype=float)      # (6, voxels, TRs)
mask_indices = np.asarray(test_data["mask_indices"])            # (voxels, 3)
depth_hi = np.asarray(layer_dist["EVV2"], dtype=float)          # 0.4 mm grid
nx0, ny0, nz0 = (int(layer_dist[k].squeeze()) for k in ("nx0", "ny0", "nz0"))

print(f"{per_vox.shape[1]} voxels x {per_vox.shape[2]} TRs, {per_vox.shape[0]} conditions")
print(f"high-res depth grid {depth_hi.shape}, functional grid ({nx0}, {ny0}, {nz0}) / 2")
""")

md("""
## 2. Depth downsampling — the loop the driver comments out

The reference maps each 0.4 mm anatomical voxel to the 0.8 mm functional voxel
containing it, then averages the depth map within each functional voxel. Its
implementation loops over ~196,000 labels with a boolean mask each time, which is
why the result is cached to `ds_vox_depth_PP_LH.mat` and the loop is commented
out.

Two `bincount` calls do it in one pass. We check against the shipped cache.
""")

code("""
# Label each high-resolution voxel with its functional-grid voxel index.
nxh, nyh, nzh = -(-nx0 // 2), -(-ny0 // 2), -(-nz0 // 2)
labels = np.arange(nxh * nyh * nzh).reshape(nxh, nyh, nzh, order="F")
labels = np.repeat(np.repeat(np.repeat(labels, 2, 0), 2, 1), 2, 2)
labels = labels[: depth_hi.shape[0], : depth_hi.shape[1], : depth_hi.shape[2]]
n_labels = nxh * nyh * nzh

t0 = time.time()
depth_lo = label_mean(depth_hi, labels, n_labels)
print(f"downsampled {n_labels} functional voxels in {time.time() - t0:.2f} s")

cached = np.asarray(sio.loadmat(root / "ds_vox_depth_PP_LH.mat")["EV_vox_d"]).reshape(-1)
both = np.isfinite(depth_lo) & np.isfinite(cached)
print(f"max |ours - shipped cache| = {np.abs(depth_lo[both] - cached[both]).max():.3e}")
print(f"NaN pattern identical: {np.array_equal(np.isnan(depth_lo), np.isnan(cached))}")
""")

md("""
### Select the ROI voxels

Voxels whose depth sits essentially at 0 or 1 are outside the cortical ribbon and
are dropped — they would otherwise anchor the outermost depth bins with partial
volume from WM or CSF.
""")

code("""
vox_label = labels.reshape(-1, order="F")[np.asarray(mask_indices)[:, 2] - 1]
d = depth_lo[vox_label]
keep = (d >= 0.0005) & (d <= 0.9995)
vox_sel = vox_label[keep]
depth_sel = depth_lo[vox_sel]
data_sel = per_vox[:, keep, :]          # (6, n_sel, TRs)

print(f"{keep.sum()} of {keep.size} voxels kept; "
      f"depth range {depth_sel.min():.3f} - {depth_sel.max():.3f}")
""")

md("""
## 3. Voxel → depth sampling

Voxels are not hard-assigned to a depth. Each contributes to every depth with a
bell-shaped weight set by its distance from that depth's centre — at 0.8 mm a
voxel straddles more than one cortical depth, and pretending otherwise
manufactures laminar specificity that the acquisition does not have.
""")

code("""
y_by_k = {
    K: {c: voxels_to_layers(data_sel[c], depth_sel, K) for c in range(6)}
    for K in CFG["K_values"]
}
for K in CFG["K_values"]:
    for c in range(6):
        y_by_k[K][c][0] = 0.0     # the pre-stimulus TR is the baseline by construction
print({K: y_by_k[K][0].shape for K in CFG["K_values"]}, "(TRs, depths)")
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
tr_axis = np.arange(-1, 7)

for c, name in enumerate(CONDITION_NAMES):
    axes[0].plot(tr_axis, data_sel[c].mean(0), lw=2, label=name)
axes[0].set(xlabel="TR", ylabel="signal change (%)", title="ROI-average response")
axes[0].legend(fontsize=8, ncol=2)
axes[0].axvline(0, color="0.4", lw=0.8, ls=":")

K_show = 9
colors = plt.cm.viridis(np.linspace(0, 1, K_show))
for k in range(K_show):
    axes[1].plot(tr_axis, y_by_k[K_show][CFG["cond_test"] - 1][:, k],
                 color=colors[k], lw=1.6)
axes[1].set(xlabel="TR", ylabel="signal change (%)",
            title=f"{CONDITION_NAMES[CFG['cond_test'] - 1]} by depth (K={K_show})")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=axes[1]); cb.set_label("CSF  →  WM")
plt.tight_layout()
""")

md("""
The depth ordering is the whole problem in one picture: the superficial depths
carry several times the amplitude of the deep ones, and almost none of that
gradient is neuronal. It is the ascending vein draining every depth below toward
the pial surface, plus a baseline blood-volume profile that rises toward the
surface. Undoing that is what the generative model is for.
""")

code("""
fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex=True, sharey=True)
for c, ax in enumerate(axes.ravel()):
    for k in range(K_show):
        ax.plot(tr_axis, y_by_k[K_show][c][:, k], color=colors[k], lw=1.3)
    ax.set_title(CONDITION_NAMES[c], fontsize=10)
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
for ax in axes[-1]:
    ax.set_xlabel("TR")
for ax in axes[:, 0]:
    ax.set_ylabel("signal change (%)")
fig.suptitle(f"Depth-resolved event-related averages (K={K_show})", y=1.0)
plt.tight_layout()
""")

md("""
## 4. Voxel point-spread function — the step the driver caches

Synthesise smooth depth profiles on the high-resolution grid, downsample them the
way the functional data was downsampled, and fit the Gaussian blur that explains
the difference. What comes out absorbs voxel size, ROI curvature and the number
of depths at once — which is why the paper describes the kernel as *"adjusted for
the number of vascular depths, the curvature of the ROI, and the voxel size"*
without giving a formula.

The reference computes this once and ships the result (`PP_LH_PSFkernel.mat`,
length 7) with the comment *"save this kernel in case re-running analysis, takes
long!"* — and then reuses that one kernel at every K. We estimate it per K.
""")

code("""
t0 = time.time()
kernels = {
    K: estimate_depth_psf(CFG["N"], K, labels, depth_hi, depth_lo, vox_sel,
                          seed=CFG["seed"])
    for K in CFG["K_values"]
}
print(f"estimated {len(kernels)} kernels in {time.time() - t0:.1f} s")

shipped = np.asarray(sio.loadmat(root / "PP_LH_PSFkernel.mat")["kernel"]).reshape(-1)

fig, ax = plt.subplots(figsize=(6, 3.6))
for K, kern in kernels.items():
    ax.plot(np.linspace(0, 1, 2 * K + 1)[1::2], kern, "o-", lw=1.6, label=f"K={K}")
ax.plot(np.linspace(0, 1, 15)[1::2], shipped, "k--s", lw=1.2, ms=4,
        label="shipped (K=7)", alpha=0.6)
ax.set(xlabel="cortical depth (WM → CSF)", ylabel="weight",
       title="Estimated depth point-spread function")
ax.legend(fontsize=8)
plt.tight_layout()
""")

md("""
## 5. Build the design

The two conditions' event-related averages are concatenated, separated by 20 TRs
of low-amplitude white noise so the fit returns to baseline between them and the
responses do not overlap. **That padding is masked out of the estimation** — it
is there to stop the model bridging the two responses, not to be fitted. This is
the one substantive difference between the reference's inversion and stock SPM.

The driving input is the 4-tone quartet; the modulatory input arrives at the
break of prediction, the 4th tone, 1500 ms after onset. The modulatory column
comes first, and `C` is zero on it, so the modulation acts only through the
neuronal self-connection `σ` — never as extra drive.
""")

code("""
rng = np.random.default_rng(CFG["seed"])
c1, c2 = CFG["cond_baseline"] - 1, CFG["cond_test"] - 1
n_tr_era = y_by_k[CFG["K_values"][0]][c1].shape[0]
ns = 2 * n_tr_era + CFG["gap_size"]

data_for_k, spec_for_k, u_for_k, kernel_for_k = {}, {}, {}, {}
for K in CFG["K_values"]:
    gap = 0.1 * rng.standard_normal((CFG["gap_size"], K))
    y = np.concatenate([y_by_k[K][c1], gap, y_by_k[K][c2]], axis=0)
    data_for_k[K] = torch.tensor(y, dtype=torch.float64)
    spec_for_k[K] = ModelSpec(
        N=CFG["N"], K=K, n_inputs=2, n_mod=1, dt=CFG["dt"], TR=CFG["TR"],
        p0=P0(V0t=3.0, nr=3.0, al_v=0.35, w_v=0.5),
    )
    n_micro = int(round(ns * CFG["TR"] / CFG["dt"]))
    u_for_k[K] = build_input(
        spec_for_k[K],
        onsets=[[47.9], [1.6, 46.4]],       # modulatory first, then driving
        durations=[[0.1], [1.6, 1.6]],
        n_micro=n_micro,
    )
    kernel_for_k[K] = torch.tensor(kernels[K], dtype=torch.float64)

# Fit only the real time points, never the padding.
rows = torch.zeros(ns, dtype=torch.bool)
rows[1:n_tr_era] = True
rows[n_tr_era + CFG["gap_size"] + 1:] = True
print(f"{ns} TRs total, {int(rows.sum())} fitted "
      f"({CFG['gap_size']} noise TRs + 2 baseline TRs excluded)")
""")

code("""
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(np.arange(ns), data_for_k[K_show].numpy(), lw=1, alpha=0.7)
ax.fill_between(np.arange(ns), -1, 3, where=~rows.numpy(),
                color="0.85", zorder=0, label="masked out of the fit")
ax.set(xlabel="TR (concatenated)", ylabel="signal change (%)", ylim=(-1, 3),
       title=f"Model input: {CONDITION_NAMES[c1]} | noise | {CONDITION_NAMES[c2]}")
ax.legend(fontsize=8)
plt.tight_layout()
""")

md("""
## 6. Invert the full model space

Eight hypotheses about **which neuronal depths carry the modulatory effect** —
the null, each depth alone, each pair, and all three. A "model" here is a pattern
of prior *variances* on `B`, not a different set of equations: the same
integration runs for every member, and only which entries are allowed to move
changes.

Times four vascular resolutions. **32 inversions**, where the reference runs 16
and notes that even that "takes long".
""")

code("""
t0 = time.time()
names, averaged, per_k = fit_model_space(
    spec_for_k, data_for_k, u_for_k,
    kernel_for_k=kernel_for_k, rows=rows,
)
elapsed = time.time() - t0
n_fits = len(names) * len(CFG["K_values"])
print(f"{n_fits} inversions in {elapsed / 60:.1f} min ({elapsed / n_fits:.1f} s each)")
""")

md("""
### Free energy per model, per vascular resolution

Free energy is a bound on the log model evidence, so differences are log Bayes
factors: a gap of 3 is already decisive. Running all four K makes it visible
whether a model's advantage is a property of the data or an artefact of one
particular depth sampling.
""")

code("""
F_matrix = np.array([[float(f) for f in a.per_k_F] for a in averaged])

fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(F_matrix - F_matrix.max(axis=0, keepdims=True),
               aspect="auto", cmap="magma")
ax.set_xticks(range(len(CFG["K_values"])),
              [f"K={k}" for k in CFG["K_values"]])
ax.set_yticks(range(len(names)), names)
ax.set_title("Free energy, relative to the best model at each K")
ax.grid(False)
fig.colorbar(im, ax=ax, label="ΔF (nats)")
plt.tight_layout()

for n, row in zip(names, F_matrix):
    print(f"{n:26s} " + "  ".join(f"K={k}: {f:9.2f}" for k, f in zip(CFG["K_values"], row)))
""")

md("""
## 7. Bayesian parameter averaging and model comparison

Parameters are pooled across K by precision-weighted averaging **before** the
models are compared, so the comparison ranks averaged models. Each K's posterior
contributes in proportion to how sharply it determined the parameters.

One note on the reference, reproduced here by default: `spm_dcm_bpa` sets
`BPA = DCM` from the *first* DCM in the list and never touches `BPA.F`, so the F
that gets compared is the first K's, not an accumulation across K — despite the
driver's comment saying otherwise. `bayesian_parameter_average(...,
free_energy="sum")` gives the accumulated version, but neither is a proper joint
log-evidence, since each K is a different sampling of the same voxels rather than
new data.
""")

code("""
F = [float(a.F) for a in averaged]
p_model = posterior_model_probabilities(F).numpy()
best = int(np.argmax(p_model))

fig, ax = plt.subplots(figsize=(7.5, 3.6))
bars = ax.bar(range(len(names)), p_model,
              color=["#c44e52" if i == best else "#4c72b0" for i in range(len(names))])
ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
ax.set(ylabel="posterior probability", title="Model comparison (free energy)")
plt.tight_layout()

print(f"winning model: {names[best]}  (p = {p_model[best]:.4f})\\n")
for n, f, pp in sorted(zip(names, F, p_model), key=lambda t: -t[1]):
    print(f"  {n:26s} F = {f:10.2f}   p = {pp:.4f}")
""")

md("""
## 8. The deveined result

The winning model's `B` is the estimated **modulation of neuronal
self-excitation at each cortical depth** — the quantity the whole pipeline
exists to recover. This is a statement about neurons, with the draining vein
modelled and removed, not about the BOLD depth profile.
""")

code("""
bpa = averaged[best]
B_mean = torch.diagonal(bpa.Ep["B"][0]).numpy()
B_var = torch.diagonal(bpa.Vp["B"][0]).numpy()
B_sd = np.sqrt(np.maximum(B_var, 0))
B_pp = torch.diagonal(bpa.Pp["B"][0]).numpy()
depth_names = ["superficial", "middle", "deep"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].bar(depth_names, B_mean, yerr=1.96 * B_sd, capsize=5, color="#55a868")
axes[0].axhline(0, color="0.3", lw=0.8)
axes[0].set(ylabel="B  (modulation of σ, Hz)",
            title=f"Laminar neuronal modulation — model '{names[best]}'")

for name, m, s, pp in zip(depth_names, B_mean, B_sd, B_pp):
    flag = "" if s > 0 else "   (fixed at zero by this model)"
    print(f"  {name:12s} B = {m:+7.4f} ± {s:.4f}   P(B≠0) = {pp:.4f}{flag}")

hemo = {k: float(bpa.Ep[k]) for k in ("sigma", "s_d", "al_d", "nsig")}
axes[1].bar(list(hemo), list(hemo.values()), color="#8172b2")
axes[1].axhline(0, color="0.3", lw=0.8)
axes[1].set(ylabel="log deviation from prior", title="Estimated nuisance parameters")
plt.tight_layout()
print("\\ns_d is the ascending-vein baseline-CBV slope: the draining-vein "
      "parameter\\nUludag & Havlicek (2021) show is the model's failure mode "
      "when underestimated.")
""")

md("""
## 9. Model fit

Predicted against observed, at the best model, for each depth. The fit is
evaluated only where the mask allows — the padding between the two conditions is
fitted by nothing.
""")

code("""
K_fit = CFG["K_values"][0]
spec = spec_for_k[K_fit]
res_best = per_k[best][0]
y_obs = data_for_k[K_fit].numpy()
y_hat = res_best.y_pred.numpy()

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
slices = [slice(1, n_tr_era), slice(n_tr_era + CFG["gap_size"] + 1, ns)]
titles = [CONDITION_NAMES[c1], CONDITION_NAMES[c2]]
cols = plt.cm.viridis(np.linspace(0, 1, K_fit))
for ax, sl, title in zip(axes, slices, titles):
    for k in range(K_fit):
        ax.plot(y_obs[sl, k], ":o", color=cols[k], lw=1, ms=4, alpha=0.8)
        ax.plot(y_hat[sl, k], "-", color=cols[k], lw=2)
    ax.set(xlabel="TR", title=title)
axes[0].set_ylabel("signal change (%)")
fig.suptitle(f"Data (dotted) vs model fit (solid), K={K_fit}, model '{names[best]}'", y=1.02)
plt.tight_layout()

resid = (y_obs - y_hat)[rows.numpy()]
ss_tot = ((y_obs[rows.numpy()] - y_obs[rows.numpy()].mean()) ** 2).sum()
print(f"variance explained on fitted points: {1 - (resid ** 2).sum() / ss_tot:.4f}")
""")

md("""
## 10. Deveining: what the model says the neurons did

The forward model maps neuronal activity to BOLD. Running it with the estimated
parameters and reading out the *neuronal* states instead of the BOLD gives the
depth-resolved neuronal timecourses — the draining-vein bias removed by
construction rather than by regression.

`deveined_timecourses` is the exact form: it re-integrates the fit. Compare the
measured BOLD curve at each depth against the inferred neuronal one.
""")

code("""
dev = deveined_timecourses(
    spec, res_best.Ep, u_for_k[K_fit], ns,
    y_measured=torch.as_tensor(y_obs), kernel=kernel_for_k[K_fit],
)
xE = dev.neuronal.numpy()
y_hat_dev = dev.y_predicted.numpy()
print(f"neuronal {xE.shape} (TR x neuronal depth), BOLD {y_hat_dev.shape} (TR x vascular depth)")
""")

md("""
### Curves, before and after

The left panel is what the scanner measured at each vascular depth: every curve
carries everything draining through it from below. The right panel is what the
model says the neurons did. The superficial BOLD curve is the one that changes
most, because it is the one most contaminated.
""")

code("""
era = slice(1, n_tr_era)
t_era = np.arange(n_tr_era - 1) * CFG["TR"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
cmap = plt.get_cmap("viridis")
for k in range(K_fit):
    axes[0].plot(t_era, y_obs[era, k], lw=1.8, color=cmap(k / max(K_fit - 1, 1)),
                 label=f"depth {k}" if k in (0, K_fit - 1) else None)
axes[0].set(xlabel="time (s)", ylabel="signal change (%)",
            title=f"Measured BOLD, {K_fit} vascular depths (0 = CSF)")
axes[0].legend(fontsize=8)
axes[0].axhline(0, color="0.7", lw=0.8)

for i, name in enumerate(depth_names):
    axes[1].plot(t_era, xE[era, i], lw=2, label=name)
axes[1].set(xlabel="time (s)", ylabel="excitatory activity (a.u.)",
            title="Deveined laminar neuronal response")
axes[1].legend(fontsize=8)
axes[1].axhline(0, color="0.7", lw=0.8)
plt.tight_layout()
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

peak_bold = y_obs[era].max(axis=0)
axes[0].plot(peak_bold, np.arange(K_fit), "o-", color="#c44e52", lw=2)
axes[0].set(xlabel="peak signal change (%)", ylabel="depth index (0 = CSF)",
            title="Measured BOLD depth profile")
axes[0].invert_yaxis()

peak_neuronal = xE[era].max(axis=0)
axes[1].plot(peak_neuronal, np.arange(spec.N), "o-", color="#55a868", lw=2)
axes[1].set(xlabel="peak excitatory activity (a.u.)", ylabel="neuronal depth",
            title="Inferred neuronal depth profile")
axes[1].set_yticks(range(spec.N))
axes[1].set_yticklabels(depth_names)
axes[1].invert_yaxis()
plt.tight_layout()
""")

md("""
## 11. The deveining operator: a depth x ROI correction you can reuse

The timecourses above are exact but tied to this fit. What a downstream analysis
usually wants is an **operator** — something to multiply into betas, FIR/TENT
curves or event-related averages from any design.

`laminar_impulse_response` gives the full transfer: BOLD at each vascular depth
in response to neuronal drive at each neuronal depth alone. Depth *k* responds to
depth *n* with a **kernel**, not a number — the vein delays and smears — so the
honest object has a time axis.
""")

code("""
ir = laminar_impulse_response(spec, res_best.Ep, n_scans=20, amplitude=1.0)
print(f"transfer {tuple(ir.shape)}  (vascular depth, neuronal depth, TR)")

W, W_inv = static_deveining_matrix(ir, mode="auc")
fid = deveining_fidelity(ir, spec, res_best.Ep, mode="auc")

row_norm = (W.abs() / W.abs().sum(1, keepdim=True)).numpy()
print("\\nRow-normalised forward mixing (rows = vascular depth, 0 = superficial):")
for k in range(K_fit):
    bars = "  ".join(f"{v:5.3f}" for v in row_norm[k])
    print(f"  depth {k}:  {bars}")
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

im = axes[0].imshow(row_norm, cmap="magma", aspect="auto", vmin=0, vmax=1)
axes[0].set(xlabel="neuronal depth", ylabel="vascular depth (0 = CSF)",
            title="Forward mixing W (row-normalised)")
axes[0].set_xticks(range(spec.N))
axes[0].set_xticklabels(depth_names, rotation=20, ha="right")
plt.colorbar(im, ax=axes[0], fraction=0.046)

shift = fid["drainage_shift"].numpy()
axes[1].plot(shift, np.arange(K_fit), "o-", color="#4c72b0", lw=2)
axes[1].set(xlabel="depths deeper than the basis predicts", ylabel="vascular depth",
            title="Vein-attributable contamination")
axes[1].axvline(0, color="0.7", lw=0.8)
axes[1].invert_yaxis()

for k in (0, K_fit // 2, K_fit - 1):
    axes[2].plot(ir[k].sum(0).numpy(), lw=2, label=f"depth {k}")
axes[2].set(xlabel="TR after drive", ylabel="BOLD (a.u.)",
            title="Transit delay and smearing")
axes[2].legend(fontsize=8)
plt.tight_layout()
""")

md("""
**Read the left panel as the justification for all of this.** The matrix is
strictly triangular: the deepest depth draws essentially everything from its own
neurons, while the superficial depth draws roughly half its signal from below.

That is why a *per-depth scalar* is the wrong shape. Rescaling a depth against
itself cannot subtract what drained into it — the fix at the surface is not
"multiply by 0.5", it is "remove the half belonging to the depths beneath". The
K x N matrix expresses that; K scalars cannot.

The middle panel separates the two mechanisms that mix depths. The
neuronal-to-vascular Gaussian basis spreads activity **symmetrically** and is not
contamination at all — it is the depth mapping. Only venous drainage is
**directional**. So the basis is computed explicitly and subtracted: what remains
is how much deeper each depth sources than the mapping alone would give, which
only a vein can cause. It is exactly zero at the deepest depth.
""")

md("""
### Applying it: deveined amplitudes

`apply_deveining` is the whole point — a matrix multiply that turns
depth-resolved measured amplitudes into neuronal-depth ones. It broadcasts over
leading axes, so parcels and timepoints come along for free.
""")

code("""
measured_peaks = torch.as_tensor(y_obs[era].max(axis=0))     # (K,) measured per depth
deveined_peaks = apply_deveining(measured_peaks, W_inv)      # (N,) neuronal depths

print("measured BOLD peak per vascular depth (%):")
print("   " + "  ".join(f"{v:6.3f}" for v in measured_peaks.numpy()))
print("\\ndeveined amplitude per neuronal depth:")
for name, v in zip(depth_names, deveined_peaks.numpy()):
    print(f"   {name:>12s}: {v:8.4f}")

# The same operator applied to a whole stack of parcels x depths would be
# apply_deveining(betas, W_inv) with betas of shape (n_parcels, K).
fake_parcels = measured_peaks.expand(4, K_fit)
print(f"\\nbatched over parcels: {tuple(fake_parcels.shape)} -> "
      f"{tuple(apply_deveining(fake_parcels, W_inv).shape)}")
""")

md("""
**Caveats that must travel with this output.** Deconvolution amplifies noise:
removing the deep contribution means subtracting a *prediction* that carries
posterior uncertainty. A scalar multiplier hides this, scaling signal and noise
together so tSNR looks untouched — the matrix does not. And the operator assumes
amplitudes in the units it was fitted in (percent signal change against the same
baseline), at an effect size near the one it was derived at, since the model is
nonlinear.

See `../../fmri_wiki/concepts/Model-derived static deveining.md`.
""")

md("""
## Notes and caveats

**Identifiability is the open question, not the implementation.** Uludag &
Havlicek (2021) explicitly declined to explore dynamic invertibility and warned
it depends on SNR, ROI size and within-ROI timecourse variability. The neuronal
model is deliberately kept cross-laminar-*uncoupled* because richer models stop
being distinguishable given fMRI noise. Before trusting a layer assignment from
this pipeline on new data, run parameter recovery at the measured tSNR: simulate
superficial-only / middle-only / deep-only modulation and check the winning model
is the true one.

**`s_d` has a known failure direction.** Underestimating the ascending-vein
baseline-CBV slope gives *qualitatively wrong* laminar neuronal profiles;
overestimating it is safe. Here it is estimated rather than fixed, but it is
worth checking against a model-comparison sweep on real data.

**Three reference quirks are reproduced deliberately, not fixed**, because the
published fits were produced with them: the Itô–Taylor correction uses a Jacobian
taken with respect to the exponentiated states (missing a chain-rule factor); the
observation equation scales the ascending vein's intravascular term by the
*venule* intra-to-extravascular ratio; and `spm_int_IT` indexes microtime from 1.
See `../../fmri_wiki/concepts/Laminar DCM parity traps.md`.

**Scaling to parcels or columns.** This ran one ROI. The per-fit cost is
dominated by Python dispatch, not arithmetic, so a batch of parcels costs
close to what one costs — but per-parcel SNR is far below per-ROI SNR, and
identifiability is already marginal at ROI level. Measure the SNR floor by
simulation before assuming patches are reachable.
""")

# ---------------------------------------------------------------------------

nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": (src + "\n").splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for kind, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("laminar_dcm_predictive_tones.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
