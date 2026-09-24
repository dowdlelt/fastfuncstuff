"""Generate ``fir_smoothing_audit.ipynb``.

The notebook is written from here so the cell text can be reviewed as source and
regenerated after a library change. Run::

    python notebooks/make_fir_smoothing_notebook.py
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
# Smooth FIR / TENT under audit

`ffs_deconvolve -tent-smooth` fits the FIR/TENT knots with a roughness penalty,
its strength λ chosen **per voxel** ([[Smooth FIR]] in the wiki):

    min ||y − Xβ − Nγ||² + λ ||Dβ||²

The curves come out dramatically cleaner and the held-out R² goes up. This notebook
checks whether that win is **real, honest, and free** — or what it costs:

1. **Is the held-out R² honest?** With `-tent-smooth loro` the same held-out runs
   *choose* λ and then *score* the winner, so its `_xval_r2` map is optimistically
   biased (the CLI says so). Here every rule is also scored **nested**: λ chosen
   without the run being predicted.
2. **Does smoothing invent HRFs?** A shifted-onset null: same data, same design
   statistics, no real event timing.
3. **What happens to amplitude?** Smoothed peaks are lower. Is that the smoother
   shrinking a true peak, or OLS inflating it (the maximum of a noisy curve is
   biased upward)? Real noise + injected known responses decide it.
4. **What happens to timing?** Latency bias and reliability, per estimator.

Everything uses the library's own design builder, packer and spectral machinery
(`glm/smooth_basis.py`), so what is measured is what `ffs_deconvolve` fits.

**To use other data**: edit the config cell. Any number of conditions works
(`POOL = "all"` collapses them into one); at least two runs are needed for
anything held-out.
""")

code("""
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# ---- data -------------------------------------------------------------------
DATA_DIR = Path(
    "/mnt/belegost/Projects/HiHiFaces/fast_outputs/"
    "ffs_autoproc_nordic3_win5_sub-3010_ses-fast.results"
)
TASK = "expres"
RUNS = [1, 2, 3]
INPUTS = [DATA_DIR / f"stage10.final.ses-fast.task-{TASK}.run-{r}.nii.zst" for r in RUNS]
EVENTS = [DATA_DIR / "stimuli" / f"sub-3010_ses-fast_task-{TASK}_run-{r}_events.tsv" for r in RUNS]
MASK = DATA_DIR / "epi_mask.nii.zst"  # None = every voxel in the volume (slow, mostly air)

# ---- conditions -------------------------------------------------------------
# Third column names the condition; None = the BIDS trial_type column.
EVENT_COLS = ("onset", "duration", "coherence")
EVENT_IGNORE = None  # e.g. ["n/a"]
POOL = None  # e.g. "all": every event one condition (single-condition analysis)

# ---- model: mirrors the ffs_deconvolve defaults -----------------------------
MODEL = "AUTO"  # AUTO | FIR | TENT | CSPLIN  (AUTO: FIR if onsets TR-locked, else TENT)
WINDOW = None  # None = auto from event durations; or (bot, top) seconds for every condition
TENT_N_BASIS = None
POLORT = "A"
DO_SCALE = True
PENALTY = "diff2"  # the -smooth-penalty default
PENALTIES_TO_COMPARE = ["diff1", "diff2", "diff3", "gp:2", "gp:4"]

# ---- evaluation -------------------------------------------------------------
# float64 throughout; "cuda" works too (not MPS: no float64 on Metal).
DEVICE = torch.device("cpu")
CHUNK = 20000  # voxels per chunk; the (chunk, 49 lambdas, K) block is the peak
SEED = 0
SIGNAL_R2 = 0.05  # "signal voxel": nested held-out R² above this for OLS or REML
FOCUS = 0  # condition index used for single-condition plots
N_NOISE_VOX = 3000  # real-noise voxels for the injected-signal test
SYNTH_SNR = None  # injected peak / voxel noise sd; None = 0.5x, 1x, 1.5x the real signal voxels' median
SYNTH_DELAYS = [5.0, 6.0, 7.0]  # SPM canonical "delay" parameter (s)
CLI_PREFIX = DATA_DIR / "new_tent_test" / "express_auto_collapse_smooth-loro_test"  # or None
SAVE_MAPS = None  # e.g. DATA_DIR / "new_tent_test" / "smooth_audit" to write NIfTI maps

rng = np.random.default_rng(SEED)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
# One colour per rule, everywhere.
C = {
    "ols": "#7f7f7f", "reml": "#0072B2", "gcv": "#56B4E9", "loro_nested": "#009E73",
    "loro_biased": "#D55E00", "fixed": "#CC79A7", "raw": "#000000", "truth": "#E69F00",
}
""")

md("""
## 1. Load exactly what `ffs_deconvolve` would fit

Same parser, loader (`-do_scale` → percent signal change), AUTO model rule,
auto window, `-polort A`, microtime offset, and the packed
*shared-task + block-diagonal nuisance* design ([[Block-diagonal nuisance]]).
""")

code('''
from fastfuncstuff.cli_utils import (
    auto_polort, load_and_preprocess_runs, parse_timing_spec, pool_timing,
    resolve_microtime_offset, run_lengths_from_starts,
)
from fastfuncstuff.design.builder import build_per_run_task_designs, pack_for_shared_task_glm
from fastfuncstuff.design.hrf import compute_windows_from_durations
from fastfuncstuff.design.matrices import is_tr_locked

inputs = [str(p) for p in INPUTS]
timing = parse_timing_spec(
    events=[str(p) for p in EVENTS], onsets=None, durations_arg=None, n_runs=len(inputs),
    event_ignore=EVENT_IGNORE, event_cols=tuple(EVENT_COLS) if EVENT_COLS else None,
    input_files=inputs, verbose=False, allow_missing_durations=True,
)
if POOL:
    timing, _note = pool_timing(timing, POOL)
labels = list(timing.condition_labels)
n_cond = len(labels)

t0 = time.perf_counter()
lr = load_and_preprocess_runs(
    input_files=inputs, mask_file=str(MASK) if MASK else None, do_scale=DO_SCALE,
    force_cpu=True, verbose=False, load_threads=len(inputs),
)
tr = float(lr.tr)
run_starts = list(lr.run_starts)
nt = list(run_lengths_from_starts(run_starts, lr.n_timepoints))
R = len(inputs)
assert R >= 2, "held-out evaluation needs at least two runs"
mto = resolve_microtime_offset(None, inputs, tr, verbose=False)

all_on = np.concatenate([np.asarray(o) - mto for c in timing.all_onsets for o in c])
if MODEL == "AUTO":
    model = "FIR" if is_tr_locked(list(all_on), tr, threshold=0.1) else "TENT"
else:
    model = MODEL
windows = [tuple(WINDOW)] * n_cond if WINDOW else compute_windows_from_durations(timing.durations, tr)
fir_window = [float(max(1, round(top / tr))) * tr for _, top in windows] if model == "FIR" else windows
polort = auto_polort(min(nt) * tr, formula="afni") if str(POLORT).upper() == "A" else int(POLORT)

dr = build_per_run_task_designs(
    onsets_per_cond_per_run=timing.all_onsets, n_timepoints_per_run=nt, tr=tr, basis=model,
    condition_labels=labels, fir_window_s=fir_window, tent_n_basis=TENT_N_BASIS,
    microtime_offset=mto, device=torch.device("cpu"),
)
runs = [lr.data[:, s : s + n] for s, n in zip(run_starts, nt)]
packed = pack_for_shared_task_glm(
    runs, dr.per_run, polort, task_column_labels=dr.column_labels,
    drop_empty_nuisance=True, device=torch.device("cpu"),
)
del runs
Y = packed.data_concat  # (V, T) float32, CPU
X = packed.design_concat  # (T, K + nuisance)
K = packed.n_task_cols
nb = list(dr.n_basis_per_condition)
offs = np.cumsum([0] + nb)
knots = [np.asarray(k, dtype=float) for k in dr.lag_times_s]
kdt = [float(np.diff(k)[0]) if len(k) > 1 else tr for k in knots]
V, T = Y.shape
mask3d = lr.mask

def cond(b, c):
    """Condition c's knot values from (..., K) betas."""
    return b[..., offs[c] : offs[c + 1]]

print(f"{V:,} voxels × {T} TRs ({R} runs × {nt}), TR {tr:.3f} s, microtime offset {mto:g} s")
print(f"model {model}, window {windows[0]}, polort {polort}, {n_cond} conditions {labels}")
print(f"knots per condition {nb} at {kdt[0]:.3f} s  →  K = {K} task + {X.shape[1] - K} nuisance columns")
print(f"events per condition: {[sum(len(o) for o in c) for c in timing.all_onsets]}")
print(f"loaded + built in {time.perf_counter() - t0:.1f} s")
''')

md("""
## 2. Why a FIR/TENT fit here needs help: the design

Three things decide how noisy an unpenalized FIR/TENT estimate is:

* **Onset phase within the TR.** TENT interpolates between knots; if onsets sit
  at few sub-TR phases, some knot combinations are barely sampled
  ([[TENT timing and sub-TR resolution]]).
* **Overlap.** With a short ISI every event's response overlaps its neighbours';
  the deconvolution must pull them apart, and the knots of one event trade off
  against the knots of the next.
* **Conditions sharing time.** Two conditions interleaved at a fast rate make
  their knots anti-correlated.

The Gram matrix of the task block (after the drift polynomials are projected out)
summarises all three. Its **eigenvectors are knot patterns**, and the noise variance of
the OLS estimate along each pattern is ∝ 1/eigenvalue. The weakest directions are
the ones that turn a smooth HRF into a zig-zag.
""")

code("""
from fastfuncstuff.design.event_timing import assess_design, design_risk_message
from fastfuncstuff.glm.smooth_basis import _spectrum, penalty_matrix

pen = penalty_matrix(PENALTY, nb, kdt, model in ("TENTzero", "CSPLINzero"))
X64 = X.double().numpy()
xt = X64[:, :K]
nuis = X64[:, K:]
q, sv, _ = np.linalg.svd(nuis[:, np.abs(nuis).sum(0) > 0], full_matrices=False)
q = q[:, sv > sv.max() * 1e-10]
xt_perp = xt - q @ (q.T @ xt)
gram = xt_perp.T @ xt_perp
ev, evec = np.linalg.eigh(gram)

gain = assess_design(
    dr.per_run, nb, timing.all_onsets, nt, tr,
    window_top=max(t for _, t in dr.fir_window_s), microtime_offset=mto, polort=max(polort, 0),
)
print("assess_design:", gain.status, "| worst-direction noise gain vs TR-rounded FIR:",
      f"{gain.amplification:.2f}x")
print(design_risk_message(gain, False, "-tent-smooth") or "  (no design warning)")
print(f"Gram condition number {ev.max() / ev.min():.3g}; "
      f"noise sd along the weakest pattern is {np.sqrt(ev.max() / ev.min()):.1f}x the strongest")

phase = (all_on / tr) % 1.0
isi = np.concatenate([
    np.diff(np.sort(np.concatenate([np.asarray(timing.all_onsets[c][r]) for c in range(n_cond)])))
    for r in range(R)
])
corr = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))

fig, ax = plt.subplots(1, 4, figsize=(17, 3.6), constrained_layout=True)
ax[0].hist(phase, bins=40, color="0.4")
ax[0].set(title="onset phase within the TR", xlabel="fraction of TR", ylabel="events")
ax[1].hist(isi, bins=40, color="0.4")
ax[1].set(title=f"ISI, any event (median {np.median(isi):.2f} s)", xlabel="s")
im = ax[2].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
for o in offs[1:-1]:
    ax[2].axhline(o - 0.5, color="k", lw=0.8); ax[2].axvline(o - 0.5, color="k", lw=0.8)
ax[2].set(title="task-column correlation (after drift)", xlabel="knot column", ylabel="knot column")
fig.colorbar(im, ax=ax[2], shrink=0.8)
ax[3].semilogy(ev[::-1] / ev.max(), "o-", ms=3, color="k")
ax[3].set(title="Gram eigenvalues (relative)", xlabel="pattern (strong → weak)")
plt.show()

fig, ax = plt.subplots(1, 4, figsize=(17, 3.2), constrained_layout=True, sharey=True)
for j, (idx, name) in enumerate([(-1, "strongest"), (2, "3rd weakest"), (1, "2nd weakest"), (0, "weakest")]):
    for c in range(n_cond):
        ax[j].plot(knots[c], cond(evec[:, idx], c), "o-", ms=3, label=labels[c])
    ax[j].axhline(0, color="0.7", lw=0.8)
    ax[j].set(title=f"{name} knot pattern (rel. noise sd {np.sqrt(ev.max() / ev[idx]):.1f})",
              xlabel="lag (s)")
ax[0].legend(fontsize=8)
plt.show()
""")

md("""
### What the penalty does to those directions

The library diagonalises the data term and the penalty together:
`R⁻ᵀ P R⁻¹ = V diag(s) Vᵀ` with `B = A + P = RᵀR`. Each direction `i` has
`s_i ∈ [0,1]`, the share of it that belongs to the penalty rather than to the data. At strength λ
the fit keeps `(1−s_i) / ((1−s_i) + λ s_i)` of that direction. Summed, that is the
**effective number of knots (edf)**. Directions with `s ≈ 0` (smooth patterns the
data pins down) survive any λ; rough, poorly-sampled ones (`s → 1`) go first.
`diff2` leaves a straight line per condition completely free, so at λ → ∞ each
condition's response becomes a line, not zero.
""")

code("""
spec_full = _spectrum(X64, K, pen)
s_full = spec_full.s
log10_grid = np.linspace(-5, 7, 49)
edf_curve = np.array([((1 - s_full) / ((1 - s_full) + 10.0**l * s_full)).sum() for l in log10_grid])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
ax[0].plot(np.sort(s_full), "o", ms=3, color="k")
ax[0].set(title="penalty share s per direction", xlabel="direction (sorted)", ylabel="s")
ax[1].plot(log10_grid, edf_curve, color="k")
ax[1].axhline(K, color="0.7", ls=":")
ax[1].set(title="effective knots vs λ", xlabel="log10 λ (relative)", ylabel="edf", ylim=(0, K + 1))
plt.show()
""")

md("""
## 3. The evaluation engine: every λ rule, scored honestly

Leave-one-run-out ([[LORO cross-validation]]): fit on `R−1` runs, predict the
held-out run, score one COD over the concatenated held-out predictions
(`ffs_deconvolve -save-xval-r2`'s definition). Nuisance is projected
**fold-locally**.

For each outer fold, λ is picked by:

| rule | chooses λ from | honest? |
|---|---|---|
| `ols` | nothing: λ = 1e-6 (unpenalised; verified against the library's plain LORO below) | yes |
| `reml`, `gcv` | the training runs' own fit (what `-tent-smooth reml/gcv -save-xval-r2` reports) | yes |
| `loro_nested` | an **inner** LORO over the training runs only | yes |
| `loro_biased` | the sum of held-out errors over **all** outer folds, including the one scored (what `-tent-smooth loro -save-xval-r2` reports) | **no**: optimistic |
| fixed λ curve | one λ for every voxel (a single number chosen from the whole brain costs ~nothing) | yes |

With 3 runs the inner LORO trains on **one** run, so it tends to over-smooth (less
data wants more smoothing). That makes `loro_nested` a conservative lower bound
for LORO.

Held-out error for every λ is closed form in the fold's shared spectrum
(`||y − M(d⊙z)||²` from `M'y`, `M'M`), so the whole grid costs no refits: one
pass over the brain.
""")

code('''
from fastfuncstuff.glm.smooth_basis import (
    DEFAULT_LOG10_GRID, _criterion, _fold_penalty, _heldout_ss, _refine_log_lambda,
    fit_smooth_basis,
)

LOG10_LAMS = np.asarray(DEFAULT_LOG10_GRID)
LOG_GRID = torch.as_tensor(LOG10_LAMS, dtype=torch.float64, device=DEVICE) * np.log(10.0)
LAMS = torch.exp(LOG_GRID)
OLS_LAM = 1e-6
RULES = ("ols", "reml", "gcv", "loro_nested", "loro_biased")


def _split(design64, penalty, train_runs, test_runs, bounds):
    idx = lambda rs: torch.cat([torch.arange(bounds[r], bounds[r + 1]) for r in rs])
    fp = _fold_penalty(design64, K, penalty, idx(train_runs), idx(test_runs), DEVICE, design64)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=DEVICE)
    return dict(
        train=fp.train.to(DEVICE), test=fp.test.to(DEVICE), q_tr=t(fp.spec.q_nuis),
        g=t(fp.spec.g), s=t(fp.spec.s), n_eff=fp.spec.n_eff, rank_p=fp.spec.rank_penalty,
        q_te=fp.q_test, m=fp.m, m_gram=fp.m_gram,
    )


class Splits:
    """Every outer and inner train/test split, spectra computed once per design."""

    def __init__(self, design, penalty):
        d64 = design.detach().cpu().double().numpy()
        bounds = run_starts + [d64.shape[0]]
        self.outer = [_split(d64, penalty, [q for q in range(R) if q != r], [r], bounds) for r in range(R)]
        self.inner = [
            [_split(d64, penalty, [q for q in range(R) if q not in (r, s)], [s], bounds)
             for s in range(R) if s != r] if R > 2 else []
            for r in range(R)
        ]


def _proj(y, q):
    return y - (y @ q) @ q.T if q.shape[1] else y


def _curves(y, sp):
    """Held-out SS at every λ, plus REML/GCV criteria on the training part."""
    ytr = _proj(y[:, sp["train"]], sp["q_tr"])
    z = ytr @ sp["g"]
    yte = _proj(y[:, sp["test"]], sp["q_te"])
    u = yte @ sp["m"]
    s = sp["s"]
    d = 1.0 / ((1 - s)[None] + LAMS[:, None] * s[None])
    v = d[None] * z[:, None, :]
    ss = (yte * yte).sum(1)[:, None] - 2 * (v * u[:, None]).sum(-1) + ((v @ sp["m_gram"]) * v).sum(-1)
    yytr = (ytr * ytr).sum(1)
    return dict(
        ss=ss, z=z, u=u, yy=(yte * yte).sum(1), s=s, sum=yte.sum(1), n=yte.shape[1], sp=sp,
        reml=_criterion("reml", z * z, yytr, s, LAMS, sp["n_eff"], sp["rank_p"]),
        gcv=_criterion("gcv", z * z, yytr, s, LAMS, sp["n_eff"], sp["rank_p"]),
    )


def _ss_at(c, lam):
    d = 1.0 / ((1 - c["s"])[None] + lam[:, None] * c["s"][None])
    return _heldout_ss(d * c["z"], c["u"], c["yy"], c["sp"]["m_gram"])


def nested_xval(data, splits, rules=RULES, desc="nested LORO"):
    """Held-out COD per voxel for each rule, the chosen log10 λ per fold, and the
    fixed-λ held-out curve (V, L)."""
    n_vox = data.shape[0]
    r2 = {m: np.zeros(n_vox, np.float32) for m in rules}
    lam = {m: np.zeros((n_vox, R), np.float32) for m in rules}
    curve = np.zeros((n_vox, LAMS.numel()), np.float32)
    for a in tqdm(range(0, n_vox, CHUNK), desc=desc, leave=True, disable=n_vox <= CHUNK):
        b = min(a + CHUNK, n_vox)
        y = data[a:b].to(DEVICE, torch.float64)
        co = [_curves(y, sp) for sp in splits.outer]
        n = sum(c["n"] for c in co)
        tot = sum(c["sum"] for c in co)
        sstot = sum(c["yy"] for c in co) - tot * tot / n
        # Constant voxels have nothing to predict: 0, as the library scores them.
        live = sstot > 1e-12 * n
        sstot = sstot.clamp_min(1e-12)
        all_ss = sum(c["ss"] for c in co)
        biased = torch.exp(_refine_log_lambda(all_ss, LOG_GRID))
        acc = {m: torch.zeros(b - a, dtype=torch.float64, device=DEVICE) for m in rules}
        for r, c in enumerate(co):
            pick = {
                "ols": torch.full((b - a,), OLS_LAM, dtype=torch.float64, device=DEVICE),
                "reml": torch.exp(_refine_log_lambda(c["reml"], LOG_GRID)),
                "gcv": torch.exp(_refine_log_lambda(c["gcv"], LOG_GRID)),
                "loro_biased": biased,
            }
            if "loro_nested" in rules:
                inner = sum(_curves(y, sp)["ss"] for sp in splits.inner[r])
                pick["loro_nested"] = torch.exp(_refine_log_lambda(inner, LOG_GRID))
            for m in rules:
                acc[m] += _ss_at(c, pick[m])
                lam[m][a:b, r] = torch.log10(pick[m]).float().cpu().numpy()
        for m in rules:
            r2[m][a:b] = torch.where(live, 1 - acc[m] / sstot, 0.0).float().cpu().numpy()
        curve[a:b] = torch.where(live[:, None], 1 - all_ss / sstot[:, None], 0.0).float().cpu().numpy()
    return r2, lam, curve


def fit_all(data, method, lam=None, design=None, penalty=None, time_index=None):
    """Full-data penalised fit (the betas ffs_deconvolve writes)."""
    d = X if design is None else design
    if time_index is not None:
        d = d[time_index]
    return fit_smooth_basis(
        data, d, K, pen if penalty is None else penalty, method=method, lam=lam,
        device=DEVICE, time_index=time_index,
    )
''')

code("""
t0 = time.perf_counter()
splits = Splits(X, pen)
r2, lam_choice, curve = nested_xval(Y, splits)
print(f"whole-brain nested evaluation: {time.perf_counter() - t0:.1f} s")

# Referee check 1: our "ols" rule must reproduce the library's plain LORO.
from fastfuncstuff.cli_utils import parse_cv_strategy
from fastfuncstuff.glm.xval import compute_xval_r2, generate_cv_splits

lib = compute_xval_r2(
    Y, X, run_starts, list(range(K)), list(range(K, X.shape[1])),
    generate_cv_splits(n_runs=R, strategy=parse_cv_strategy("loro")),
    metric="cod", device=torch.device("cpu"), verbose=False,
)["r2"].numpy()
print(f"ols rule vs glm.xval.compute_xval_r2: max |diff| = {np.abs(lib - r2['ols']).max():.2e}")

# Referee check 2: our loro_biased must reproduce the CLI's -tent-smooth loro map.
if CLI_PREFIX is not None and Path(f"{CLI_PREFIX}_xval_r2.nii.gz").exists():
    from fastfuncstuff.io.afni import load_nifti

    cli = np.asarray(load_nifti(f"{CLI_PREFIX}_xval_r2.nii.gz").get_fdata(), np.float32)
    cli = cli[mask3d] if mask3d is not None else cli.reshape(-1)
    print(f"loro_biased vs CLI {Path(str(CLI_PREFIX)).name}_xval_r2: "
          f"r = {np.corrcoef(cli, r2['loro_biased'])[0, 1]:.5f}, "
          f"median |diff| = {np.median(np.abs(cli - r2['loro_biased'])):.2e}")

SIG = (r2["ols"] > SIGNAL_R2) | (r2["reml"] > SIGNAL_R2)
print(f"signal voxels (OLS or REML nested R² > {SIGNAL_R2}): {SIG.sum():,}")
""")

md("""
The single number that the λ curve is allowed to choose: **one global λ** for every
voxel, picked where the median held-out R² of the signal voxels peaks. Choosing
one scalar from ~10⁵ voxels has negligible selection bias, and it gives a
**linear** estimator: the same filter applied to every voxel. That will matter for
amplitude comparisons later.
""")

code("""
med_curve = np.median(curve[SIG], axis=0)
LAM_STAR = float(10 ** LOG10_LAMS[med_curve.argmax()])
print(f"global λ* = 10^{np.log10(LAM_STAR):.2f}  (edf {np.interp(np.log10(LAM_STAR), log10_grid, edf_curve):.1f} of {K})")

t0 = time.perf_counter()
fits = {
    "ols": fit_all(Y, "fixed", lam=OLS_LAM),
    "reml": fit_all(Y, "reml"),
    "gcv": fit_all(Y, "gcv"),
    "fixed": fit_all(Y, "fixed", lam=LAM_STAR),
    # λ from all held-out runs, refit on all runs: the -tent-smooth loro betas.
    "loro_biased": fit_all(Y, "fixed", lam=torch.as_tensor(10.0 ** lam_choice["loro_biased"][:, 0])),
}
B = {m: f.betas.numpy() for m, f in fits.items()}
print(f"full-data fits: {time.perf_counter() - t0:.1f} s")

# Referee check 3: the λ→0 betas are fit_glm's OLS betas.
from fastfuncstuff.glm.core import fit_glm

probe = np.flatnonzero(SIG)[:500]
ref = fit_glm(Y[probe], X, tr=tr, max_poly_degree=-1, device=torch.device("cpu"), verbose=False)
print(f"OLS betas vs fit_glm: max |diff| = {np.abs(ref.betas[:, :K].numpy() - B['ols'][probe]).max():.2e} "
      f"(beta scale {np.abs(B['ols'][probe]).max():.2f})")
""")

md("""
## 4. Showcase voxels: raw data, fits, the λ path, held-out prediction

Voxels are picked by **OLS** held-out R² quantiles, the default method's own
referee, so the selection does not favour smoothing. One extra voxel is the
opposite case: OLS predicts *nothing* (R² < 0) but REML predicts well. That is where
the question "is it finding structure or inventing it?" bites.

For each voxel:

* **(a)** the **raw event-locked average** (drift removed, no model), against
  the deconvolved curves. With a short ISI the raw average is a smear of
  overlapping responses, and its peak sits well *below* the deconvolved one:
  with events every few seconds the sustained part of the response is part of the
  run mean, and drift removal subtracts it. This is why FIR/TENT exists at all.
* **(b)** the full-data curves by rule, with the **three single-run OLS fits** as
  thin lines: this is how much OLS moves from run to run.
* **(c)** the regularisation path: one curve per λ, rough (yellow) to smooth (blue).
* **(d)** held-out error vs λ per fold, relative to OLS (below 1 = better
  prediction). Dots mark the λ REML picked on each fold's *training* runs.
""")

code('''
def pick_showcase():
    order = {}
    for qn in (0.9995, 0.995, 0.98, 0.95):
        target = np.quantile(r2["ols"], qn)
        order[f"OLS R² q{qn * 100:g}"] = int(np.abs(r2["ols"] - target).argmin())
    rescued = np.flatnonzero(r2["ols"] < 0)
    order["rescued (OLS R² < 0)"] = int(rescued[r2["reml"][rescued].argmax()])
    return order


SHOW = pick_showcase()
for name, v in SHOW.items():
    print(f"{name:24s} voxel {v:7d}   " + "  ".join(f"{m} {r2[m][v]:+.3f}" for m in RULES))

bounds = run_starts + [T]
q_block = [None] * R
for r in range(R):
    blk = X64[bounds[r] : bounds[r + 1], K:]
    blk = blk[:, np.abs(blk).sum(0) > 0]
    u_, s_, _ = np.linalg.svd(blk, full_matrices=False)
    q_block[r] = u_[:, s_ > s_.max() * 1e-10]


def event_locked(v, lags):
    """Drift-removed data sampled at onset + lag, averaged per condition (mean, sem)."""
    out = []
    for c in range(n_cond):
        ep = []
        for r in range(R):
            y = Y[v, bounds[r] : bounds[r + 1]].double().numpy()
            y = y - q_block[r] @ (q_block[r].T @ y)
            t = np.arange(nt[r]) * tr + mto
            for o in np.asarray(timing.all_onsets[c][r]):
                if o + lags[0] >= t[0] and o + lags[-1] <= t[-1]:
                    ep.append(np.interp(o + lags, t, y))
        ep = np.asarray(ep)
        out.append((ep.mean(0), ep.std(0) / np.sqrt(len(ep))))
    return out


def lam_path(v):
    """Full-data knot values at every grid λ for one voxel: (L, K)."""
    y = Y[v].double().numpy()
    y = y - spec_full.q_nuis @ (spec_full.q_nuis.T @ y)
    z = y @ spec_full.g
    d = 1.0 / ((1 - s_full)[None] + (10.0 ** LOG10_LAMS)[:, None] * s_full[None])
    return (d * z) @ spec_full.w.T


single_run_ols = {
    v: [fit_all(Y[[v]], "fixed", lam=OLS_LAM, time_index=torch.arange(bounds[r], bounds[r + 1])).betas[0].numpy()
        for r in range(R)]
    for v in SHOW.values()
}

lags = np.arange(-2.0, windows[0][1] + 3.0, tr / 2)
cmap = plt.get_cmap("viridis_r")
for name, v in SHOW.items():
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.8), constrained_layout=True)
    era = event_locked(v, lags)
    for c in range(n_cond):
        col = f"C{c}"
        ax[0].plot(lags, era[c][0], color=col, lw=1.2, label=f"{labels[c]} raw avg")
        ax[0].fill_between(lags, era[c][0] - era[c][1], era[c][0] + era[c][1], color=col, alpha=0.15)
        ax[0].plot(knots[c], cond(B["reml"][v], c), color=col, lw=2.2, ls="--", label=f"{labels[c]} REML")
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].set(title=f"(a) {name}: raw event-locked avg", xlabel="s from onset", ylabel="% signal")
    ax[0].legend(fontsize=7)

    for c in range(n_cond):
        ls = ["-", "--", ":", "-."][c % 4]
        for rb in single_run_ols[v]:
            ax[1].plot(knots[c], cond(rb, c), color=C["ols"], lw=0.6, alpha=0.6, ls=ls)
        for m in ("ols", "reml", "loro_biased", "fixed"):
            ax[1].plot(knots[c], cond(B[m][v], c), color=C[m], lw=1.8 if m != "ols" else 1.2, ls=ls,
                       label=f"{m}" if c == 0 else None)
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set(title="(b) curves (line style = condition; thin = single-run OLS)", xlabel="lag (s)")
    ax[1].legend(fontsize=7)

    path = lam_path(v)
    for i in range(0, len(LOG10_LAMS), 3):
        ax[2].plot(knots[FOCUS], cond(path[i], FOCUS), color=cmap(i / len(LOG10_LAMS)), lw=1)
    ax[2].plot(knots[FOCUS], cond(B["reml"][v], FOCUS), color="k", lw=2, ls="--", label="REML pick")
    ax[2].set(title=f"(c) λ path, {labels[FOCUS]} (yellow rough → blue smooth)", xlabel="lag (s)")
    ax[2].legend(fontsize=7)

    y1 = Y[[v]].to(DEVICE, torch.float64)
    for r, sp in enumerate(splits.outer):
        cc = _curves(y1, sp)
        rel = (cc["ss"][0] / _ss_at(cc, torch.tensor([OLS_LAM], dtype=torch.float64, device=DEVICE))[0]).cpu().numpy()
        ax[3].plot(LOG10_LAMS, rel, color=f"C{r}", label=f"held-out run {RUNS[r]}")
        ax[3].plot(lam_choice["reml"][v, r], np.interp(lam_choice["reml"][v, r], LOG10_LAMS, rel), "o", color=f"C{r}")
    ax[3].axvline(lam_choice["loro_biased"][v, 0], color=C["loro_biased"], ls="--", lw=1, label="LORO pick (all folds)")
    ax[3].axhline(1, color="0.7", lw=0.8)
    ax[3].set(title="(d) held-out SS ÷ OLS (dot: REML pick)", xlabel="log10 λ", ylabel="relative SS")
    ax[3].legend(fontsize=7)
    plt.show()
''')

md("""
### Predicting a held-out run

What LORO actually scores: the model trained on the *other* runs, predicting this
run's (drift-removed) time series. Grey is data, and the lines are the OLS and REML-λ
predictions. REML's λ is chosen on the training runs only, so this is the nested
comparison. A short window keeps individual events visible.
""")

code("""
def heldout_prediction(v, r, lam):
    sp = splits.outer[r]
    y = Y[[v]].to(DEVICE, torch.float64)
    z = _proj(y[:, sp["train"]], sp["q_tr"]) @ sp["g"]
    yte = _proj(y[:, sp["test"]], sp["q_te"])
    d = 1.0 / ((1 - sp["s"]) + lam * sp["s"])
    return yte[0].cpu().numpy(), (sp["m"] @ (d * z[0])).cpu().numpy()


WIN_S = 80.0
for name, v in list(SHOW.items())[:3] + [list(SHOW.items())[-1]]:
    r = R - 1
    yte, p_ols = heldout_prediction(v, r, OLS_LAM)
    _, p_reml = heldout_prediction(v, r, float(10 ** lam_choice["reml"][v, r]))
    t = np.arange(nt[r]) * tr + mto
    sel = (t >= 30) & (t < 30 + WIN_S)
    r2f = lambda p: 1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(17, 2.8), constrained_layout=True)
    ax.plot(t[sel], yte[sel], color="0.6", lw=1, label="held-out data")
    ax.plot(t[sel], p_ols[sel], color=C["ols"], lw=1.5, ls="--", label=f"OLS  (fold R² {r2f(p_ols):+.3f})")
    ax.plot(t[sel], p_reml[sel], color=C["reml"], lw=2, label=f"REML (fold R² {r2f(p_reml):+.3f})")
    for c in range(n_cond):
        o = np.asarray(timing.all_onsets[c][r])
        o = o[(o >= 30) & (o < 30 + WIN_S)]
        ax.plot(o, np.full(o.size, ax.get_ylim()[0]), "|", color=f"C{c}", ms=12, mew=2)
    ax.set(title=f"{name}: run {RUNS[r]} predicted from the other runs", xlabel="s", ylabel="% signal")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    plt.show()
""")

md("""
## 5. The population: how much better, honestly?

The table is the core result. Compare **`reml`/`gcv`/`loro_nested`** (honest) with
**`ols`** (the default), and see how far **`loro_biased`** (what the CLI's LORO
map shows) sits above the honest LORO.
""")

code("""
def summary_table(r2d, sel, base="ols"):
    rows = []
    for m, v in r2d.items():
        rows.append({
            "rule": m,
            f"median R² (signal vox)": np.median(v[sel]),
            f"mean R² (signal vox)": v[sel].mean(),
            "n > 0.02": int((v > 0.02).sum()),
            "n > 0.05": int((v > 0.05).sum()),
            "n > 0.10": int((v > 0.10).sum()),
            f"% signal vox better than {base}": 100 * (v[sel] > r2d[base][sel]).mean(),
            "median R² (all vox)": np.median(v),
        })
    return pd.DataFrame(rows).set_index("rule")


tab = summary_table(r2, SIG)
display(tab.style.format(precision=4))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
live = (r2["ols"] > -0.05) | (r2["reml"] > 0)
lim = (-0.05, max(0.2, float(np.quantile(r2["loro_biased"], 0.9999))))
hb = ax[0].hexbin(r2["ols"][live], r2["reml"][live], gridsize=90, bins="log", cmap="Greys", extent=lim + lim)
ax[0].plot(lim, lim, color=C["loro_biased"], lw=1)
ax[0].set(title="nested REML vs OLS (held-out R²)", xlabel="OLS", ylabel="REML, λ from training runs")
ax[1].hexbin(r2["reml"][live], r2["loro_biased"][live], gridsize=90, bins="log", cmap="Greys", extent=lim + lim)
ax[1].plot(lim, lim, color=C["loro_biased"], lw=1)
ax[1].set(title="biased LORO (CLI map) vs honest REML", xlabel="REML nested", ylabel="LORO (λ saw the scored run)")
for m in ("reml", "gcv", "loro_nested", "loro_biased"):
    ax[2].hist((r2[m] - r2["ols"])[SIG], bins=120, histtype="step", color=C[m], lw=1.5, label=m,
               range=(-0.03, 0.1))
ax[2].axvline(0, color="k", lw=0.8)
ax[2].set(title="gain over OLS, signal voxels", xlabel="Δ held-out R²", ylabel="voxels")
ax[2].legend()
plt.show()
""")

md("""
## 6. Null check: does smoothing manufacture responses?

Every run's onsets are **cyclically shifted** by a random offset. The event rate, ISI
distribution, condition interleaving and design conditioning are all kept, but the
timing no longer matches the brain. Refit with each rule. Two readouts:

* the **false-positive rate** at each threshold, and each rule's own 99.9th-percentile
  null R², which is the honest threshold to count voxels against;
* whether the curves the smoother returns under the null **look like HRFs**:
  correlation with the canonical response shape, compared with the real design in
  real signal voxels.

The `diff2` penalty has no idea what an HRF is; it only prefers smooth. If
HRF-shaped curves show up with the real timing but not the shifted one, the shape
is coming from the data.

Caveat: at a fast event rate a shifted train still overlaps real events by
chance, so signal voxels leak a little. The false-positive rates are reported
separately inside and outside signal voxels.
""")

code("""
from fastfuncstuff.design.hrf import get_spm_canonical_hrf

null_on = [[None] * R for _ in range(n_cond)]
for r in range(R):
    shift = rng.uniform(0.25, 0.75) * nt[r] * tr
    for c in range(n_cond):
        null_on[c][r] = np.sort((np.asarray(timing.all_onsets[c][r]) + shift) % (nt[r] * tr))
dr_null = build_per_run_task_designs(
    onsets_per_cond_per_run=null_on, n_timepoints_per_run=nt, tr=tr, basis=model,
    condition_labels=labels, fir_window_s=fir_window, tent_n_basis=TENT_N_BASIS,
    microtime_offset=mto, device=torch.device("cpu"),
)
# Only the design is needed: the packed data are the same concatenated runs.
X_null = pack_for_shared_task_glm(
    [torch.zeros((1, n)) for n in nt], dr_null.per_run, polort,
    drop_empty_nuisance=True, device=torch.device("cpu"),
).design_concat
t0 = time.perf_counter()
r2_null, lam_null, _ = nested_xval(Y, Splits(X_null, pen), rules=("ols", "reml", "gcv", "loro_biased"), desc="null")
print(f"null evaluation: {time.perf_counter() - t0:.1f} s")

NULL_THR = {m: float(np.quantile(r2_null[m], 0.999)) for m in r2_null}
rows = []
for m in r2_null:
    rows.append({
        "rule": m,
        "null 99.9th pct": NULL_THR[m],
        "null FPR > 0.02 (signal vox)": (r2_null[m][SIG] > 0.02).mean(),
        "null FPR > 0.02 (other vox)": (r2_null[m][~SIG] > 0.02).mean(),
        "null n > 0.05": int((r2_null[m] > 0.05).sum()),
        "real n > 0.05": int((r2[m] > 0.05).sum()),
        "real n > own null 99.9th": int((r2[m] > NULL_THR[m]).sum()),
    })
display(pd.DataFrame(rows).set_index("rule").style.format(precision=4))
""")

md("""
Under the shifted timing, the best-"predicting" voxels get **straight lines** from REML
(the diff2 null space: λ → ∞), not HRF shapes. The spike in the histogram is a line's
correlation with the template. HRF shapes appear only with the real timing.

The **rescued** voxels (OLS predicts nothing, REML does) are worth a look of their own.
Here most correlate *negatively* with the canonical shape: broad, sustained negative
responses (see the rescued showcase voxel above). The smoother is not inventing them,
since the null gives lines. But what they are (deactivation, draining veins, CSF) is a
separate question, and for de-veining it may be the interesting one.
""")

code("""
def canonical_at_knots(c, delay=6.0):
    dt = 0.01
    h = get_spm_canonical_hrf(microtime_dt=dt, delay=delay, device=torch.device("cpu")).numpy().astype(float)
    dur = max(float(timing.durations[c]), dt)
    resp = np.convolve(h, np.ones(int(round(dur / dt)))) * dt
    tg = np.arange(resp.size) * dt
    return np.interp(knots[c], tg, resp / resp.max()), tg, resp / resp.max()


template = canonical_at_knots(FOCUS)[0]


def shape_r(b):
    b = b - b.mean(1, keepdims=True)
    t_ = template - template.mean()
    return (b @ t_) / np.sqrt((b * b).sum(1) * (t_ @ t_) + 1e-30)


null_reml = fit_all(Y, "reml", design=X_null).betas.numpy()
top_null = np.argsort(r2_null["reml"])[::-1][: max(200, int(SIG.sum() // 5))]
top_real = np.argsort(r2["reml"])[::-1][: top_null.size]
rescued = np.flatnonzero((r2["ols"] < 0) & (r2["reml"] > SIGNAL_R2))

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), constrained_layout=True)
kw = dict(bins=50, range=(-1, 1), histtype="step", lw=1.8, density=True)
ax[0].hist(shape_r(cond(B["reml"][top_real], FOCUS)), color=C["reml"], label=f"real timing, top {top_real.size} vox", **kw)
ax[0].hist(shape_r(cond(null_reml[top_null], FOCUS)), color=C["loro_biased"], label=f"shifted timing, top {top_null.size} vox", **kw)
if rescued.size:
    ax[0].hist(shape_r(cond(B["reml"][rescued], FOCUS)), color=C["loro_nested"],
               label=f"'rescued' vox (OLS<0, REML>{SIGNAL_R2}), n={rescued.size}", **kw)
ax[0].set(title=f"REML curve vs canonical shape ({labels[FOCUS]})", xlabel="shape correlation", ylabel="density")
ax[0].legend(fontsize=8)
ax[1].plot(knots[FOCUS], template, color=C["truth"], lw=2, label="canonical (template)")
for vv, col, lab in ((top_real, C["reml"], "real"), (top_null, C["loro_biased"], "shifted")):
    cur = cond(B["reml"][vv] if lab == "real" else null_reml[vv], FOCUS)
    cur = cur / np.abs(cur).max(1, keepdims=True)
    ax[1].plot(knots[FOCUS], np.median(cur, 0), color=col, lw=2, label=f"{lab}: median normalised curve")
ax[1].axhline(0, color="0.7", lw=0.8)
ax[1].set(title="median shape of the best-predicting voxels", xlabel="lag (s)")
ax[1].legend(fontsize=8)
plt.show()
""")

md("""
## 7. How much smoothing is chosen, and does per-voxel λ earn its keep?

The fixed-λ curve is the honest held-out R² for *one λ applied everywhere*, and it
maps out the bias–variance trade-off directly. Horizontal lines are the per-voxel
rules. If a single global λ matches them, per-voxel selection adds noise without
adding prediction. A single λ also has a simpler interpretation: one linear filter.
""")

code("""
top1 = r2["ols"] > np.quantile(r2["ols"], 0.99)
fig, ax = plt.subplots(1, 3, figsize=(17, 3.9), constrained_layout=True)
for sel, name, ls in ((SIG, "signal voxels", "-"), (top1, "top 1% by OLS", "--")):
    ax[0].plot(LOG10_LAMS, np.median(curve[sel], 0), color="k", ls=ls, label=f"fixed λ, {name}")
    for m in ("ols", "reml", "loro_nested"):
        ax[0].axhline(np.median(r2[m][sel]), color=C[m], ls=ls, lw=1)
ax[0].axvline(np.log10(LAM_STAR), color=C["fixed"], lw=1)
ax2 = ax[0].twinx()
ax2.plot(log10_grid, edf_curve, color="0.75", lw=1)
ax2.set_ylabel("edf", color="0.6")
ax[0].set(title="median held-out R² vs one global λ", xlabel="log10 λ", ylabel="held-out R²")
ax[0].legend(fontsize=8, loc="lower left")
for m in ("reml", "gcv", "loro_nested", "loro_biased"):
    ax[1].hist(lam_choice[m][SIG].ravel(), bins=60, range=(-5, 7), histtype="step", lw=1.5, color=C[m], label=m)
ax[1].axvline(np.log10(LAM_STAR), color=C["fixed"], lw=1, label="global λ*")
ax[1].set(title="per-voxel λ chosen (signal voxels, every fold)", xlabel="log10 λ")
ax[1].legend(fontsize=8)
for m in ("reml", "gcv", "fixed"):
    ax[2].hist(fits[m].edf.numpy()[SIG], bins=60, range=(0, K), histtype="step", lw=1.5, color=C[m], label=m)
ax[2].set(title="effective knots, full-data fit (signal voxels)", xlabel=f"edf (of {K})")
ax[2].legend(fontsize=8)
plt.show()

# Held-out R² of the one global λ, per voxel (column of the curve nearest λ*).
r2["fixed"] = curve[:, int(np.abs(LOG10_LAMS - np.log10(LAM_STAR)).argmin())]
print("λ-spread across folds (REML, signal voxels): median SD of log10 λ =",
      f"{np.median(lam_choice['reml'][SIG].std(1)):.2f} decades")
display(summary_table({m: r2[m] for m in ("ols", "reml", "fixed", "loro_nested")}, SIG).style.format(precision=4))
""")

md("""
## 8. Does the penalty shape matter?

`diffN` penalises N-th differences (null space: polynomials of degree < N);
`gp:L` is a Gaussian-process prior with an `L`-second length scale (full rank,
shrinks toward zero). Scored on the signal voxels plus a random 20k (a whole brain
per shape is just a longer wait).
""")

code("""
sub = np.union1d(np.flatnonzero(SIG), rng.choice(V, min(20000, V), replace=False))
sub_sig = SIG[sub]
rows = []
pen_fits = {}
for spec in PENALTIES_TO_COMPARE:
    p = penalty_matrix(spec, nb, kdt, model in ("TENTzero", "CSPLINzero"))
    rr, _, cc = nested_xval(Y[sub], Splits(X, p), rules=("ols", "reml", "gcv", "loro_nested"), desc=spec)
    best_fixed = np.median(cc[sub_sig], 0).max()
    rows.append({"penalty": spec, **{f"median {m}": np.median(rr[m][sub_sig]) for m in rr},
                 "best global λ": best_fixed, "n reml > 0.05": int((rr["reml"] > 0.05).sum())})
    first = list(SHOW.values())[0]
    pen_fits[spec] = fit_all(Y[[first]], "reml", penalty=p).betas[0].numpy()
display(pd.DataFrame(rows).set_index("penalty").style.format(precision=4))

fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
ax.plot(knots[FOCUS], cond(B["ols"][first], FOCUS), color=C["ols"], lw=1, label="OLS")
for i, (spec, b) in enumerate(pen_fits.items()):
    ax.plot(knots[FOCUS], cond(b, FOCUS), lw=1.8, color=plt.get_cmap("tab10")(i + 1), label=spec)
ax.set(title=f"voxel {first}: REML curve per penalty ({labels[FOCUS]})", xlabel="lag (s)", ylabel="% signal")
ax.legend(fontsize=8)
plt.show()
""")

md("""
## 9. Amplitude: who is biased, OLS or the smoother?

Real data first: smoothed peaks come out lower than OLS peaks. Two explanations:

* **the smoother shrinks a true peak.** A sharp peak is curvature, and `diff2`
  penalises curvature;
* **OLS inflates the peak.** The maximum of *(curve + noise)* over ~18 knots is
  biased upward by roughly the noise sd times the expected maximum of 18 draws,
  and it gets worse as SNR drops.

The area under the curve is a **linear** functional of the knots, so it carries no max-of-noise bias.
If both methods agree on area but differ on peak, the difference is in how
peaked the curve is, not in how much response there is.
""")

code('''
def peak_amp_lat(b, t):
    """Parabolic-interpolated peak value and latency; b (N, n_knots) on uniform knots t."""
    i = b.argmax(1)
    j = i.clip(1, b.shape[1] - 2)
    rows = np.arange(len(b))
    f0, f1, f2 = b[rows, j - 1], b[rows, j], b[rows, j + 1]
    cv = f0 - 2 * f1 + f2
    step = np.where(cv < 0, 0.5 * (f0 - f2) / np.where(cv < 0, cv, -1), 0.0).clip(-1, 1)
    edge = (i == 0) | (i == b.shape[1] - 1)
    step = np.where(edge, 0.0, step)
    at = np.where(edge, i, j)
    val = np.where(edge, b[rows, i], f1 - 0.25 * (f0 - f2) * step)
    return val, t[at] + step * (t[1] - t[0]), edge


def centroid_lat(b, t):
    w = np.clip(b, 0, None)
    return (w * t).sum(1) / np.maximum(w.sum(1), 1e-12)


def halfmax_lat(b, t):
    """First crossing of half the peak on the rising side (linear interpolation)."""
    pk = b.max(1, keepdims=True)
    above = b >= 0.5 * pk
    k = above.argmax(1)
    rows = np.arange(len(b))
    km = (k - 1).clip(0)
    y0, y1 = b[rows, km], b[rows, k]
    frac = np.where(k > 0, (0.5 * pk[:, 0] - y0) / np.where(y1 != y0, y1 - y0, 1), 0.0)
    return t[km] + frac.clip(0, 1) * (t[1] - t[0]) * (k > 0)


def auc(b, dt):
    return b.sum(1) * dt


t_f, dt_f = knots[FOCUS], kdt[FOCUS]
sig_idx = np.flatnonzero(SIG)
pk = {m: peak_amp_lat(cond(B[m][sig_idx], FOCUS), t_f)[0] for m in B}
ar = {m: auc(cond(B[m][sig_idx], FOCUS), dt_f) for m in B}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
lim = (0, float(np.quantile(pk["ols"], 0.995)))
ax[0].hexbin(pk["ols"], pk["reml"], gridsize=70, bins="log", cmap="Greys", extent=lim + lim)
ax[0].plot(lim, lim, color=C["loro_biased"])
ax[0].set(title=f"peak ({labels[FOCUS]}), signal voxels", xlabel="OLS peak (%)", ylabel="REML peak (%)")
lim = tuple(np.quantile(ar["ols"], [0.005, 0.995]))
ax[1].hexbin(ar["ols"], ar["reml"], gridsize=70, bins="log", cmap="Greys", extent=lim + lim)
ax[1].plot(lim, lim, color=C["loro_biased"])
ax[1].set(title="area under the curve", xlabel="OLS AUC (%·s)", ylabel="REML AUC (%·s)")
for m in ("reml", "fixed", "loro_biased"):
    ax[2].hist(pk[m] / pk["ols"], bins=80, range=(0, 1.6), histtype="step", lw=1.5, color=C[m], label=f"{m} peak / OLS peak")
    ax[2].hist(ar[m] / ar["ols"], bins=80, range=(0, 1.6), histtype="step", lw=1.5, ls=":", color=C[m], label=f"{m} AUC / OLS AUC")
ax[2].axvline(1, color="k", lw=0.8)
ax[2].set(title="ratio to OLS", xlabel="ratio")
ax[2].legend(fontsize=7)
plt.show()
print("median peak ratio to OLS:", {m: round(float(np.nanmedian(pk[m] / pk["ols"])), 3) for m in pk if m != "ols"})
print("median AUC  ratio to OLS:", {m: round(float(np.nanmedian(ar[m] / ar["ols"])), 3) for m in ar if m != "ols"})
''')

md("""
### Ground truth: real noise + injected responses

Take voxels where **no rule predicts anything** (every held-out R² < 0), so their
time series are real noise with its real autocorrelation, drift and artefacts. Add a
known response at every real onset: the SPM canonical HRF convolved with each
event's duration, and three latencies. The truth is **not** in the TENT basis, so basis
mismatch is part of the test.

Amplitude is set **per voxel as a multiple of its own noise sd** (peak / residual sd,
"SNR"; conditions scaled 1 → 0.6). The default levels are ½×, 1× and 1.5× the median of the same
ratio measured in the real signal voxels, so the test runs at the SNR the real data
actually has. A fixed %-signal amplitude would not: the noise voxels are noisier
than the signal voxels, so it would test the wrong regime.

Every rule then fits the same series, and bias is measured against the truth sampled at the knots:

* `peak/A`: recovered peak ÷ true peak (1 = unbiased; A = the true peak)
* `AUC ratio`: recovered area ÷ true area over the window
* `proj/A`: least-squares amplitude of the recovered curve on the *true* shape (a linear, noise-robust amplitude)
* `RMSE/A`: whole-curve error
* latency bias and **IQR** for three estimators: peak, centroid, half-max rise; `edge %` is the share of voxels whose peak sits on the first or last knot (a flattened curve)

The median nested held-out R² of the synthetic voxels is printed per level. Compare
it with the real signal voxels.
""")

code('''
def resid_sd(idx, betas=None):
    """Noise sd per voxel: drift-removed series, minus the task fit when given."""
    y = Y[torch.as_tensor(idx)].double().numpy()
    y = y - (y @ q) @ q.T
    if betas is not None:
        y = y - betas @ xt_perp.T
    return np.sqrt((y * y).sum(1) / (T - q.shape[1]))


sd_sig = resid_sd(sig_idx, B["fixed"][sig_idx])
snr_real = peak_amp_lat(cond(B["fixed"][sig_idx], FOCUS), t_f)[0] / sd_sig
levels = SYNTH_SNR or [float(f"{f * np.median(snr_real):.2g}") for f in (0.5, 1.0, 1.5)]
print(f"real signal voxels, {labels[FOCUS]} peak / noise sd (fixed-λ fit): "
      f"quartiles {np.round(np.quantile(snr_real, [0.25, 0.5, 0.75]), 2)}  →  injected SNR levels {levels}")

# Real noise that looks like the signal voxels' noise: no rule predicts anything,
# and the noise sd sits inside the signal voxels' range.
lo_sd, hi_sd = np.quantile(sd_sig, [0.1, 0.9])
cand = np.flatnonzero(np.all(np.stack([r2[m] for m in RULES]) < 0, axis=0))
cand = rng.choice(cand, min(cand.size, 50 * N_NOISE_VOX), replace=False)
sd_cand = resid_sd(cand)
noise_pool = cand[(sd_cand >= lo_sd) & (sd_cand <= hi_sd)]
keep = rng.choice(noise_pool.size, min(N_NOISE_VOX, noise_pool.size), replace=False)
noise_vox = noise_pool[keep]
sd_noise = sd_cand[(sd_cand >= lo_sd) & (sd_cand <= hi_sd)][keep]
Yn = Y[torch.as_tensor(noise_vox)].double()
rel_amp = np.linspace(1.0, 0.6, n_cond) if n_cond > 1 else np.ones(1)
print(f"noise voxels: {noise_vox.size} (sd {lo_sd:.2f}–{hi_sd:.2f} %). Real signal voxels: "
      f"median OLS held-out R² {np.median(r2['ols'][SIG]):.3f}, REML {np.median(r2['reml'][SIG]):.3f}")


def injected(delay, amp):
    dt = 0.01
    h = get_spm_canonical_hrf(microtime_dt=dt, delay=delay, device=torch.device("cpu")).numpy().astype(float)
    out, truth = [], []
    for c in range(n_cond):
        dur = max(float(timing.durations[c]), dt)
        resp = np.convolve(h, np.ones(int(round(dur / dt)))) * dt
        resp = resp / resp.max()
        tg = np.arange(resp.size) * dt
        truth.append((amp * rel_amp[c] * np.interp(knots[c], tg, resp), tg[resp.argmax()]))
    for r in range(R):
        t = np.arange(nt[r]) * tr + mto
        s = np.zeros(nt[r])
        for c in range(n_cond):
            dur = max(float(timing.durations[c]), dt)
            resp = np.convolve(h, np.ones(int(round(dur / dt)))) * dt
            resp = amp * rel_amp[c] * resp / resp.max()
            tg = np.arange(resp.size) * dt
            for o in np.asarray(timing.all_onsets[c][r]):
                s += np.interp(t - o, tg, resp, left=0.0, right=0.0)
        out.append(s)
    return np.concatenate(out), truth


def iqr(x):
    return float(np.subtract(*np.quantile(x, [0.75, 0.25])))


rows, mean_curves, lat_pools = [], {}, {}
syn_splits = splits  # same design, same folds
for delay in SYNTH_DELAYS:
    sig, truth = injected(delay, 1.0)
    for amp in levels:
        scale = amp * sd_noise  # per-voxel peak in % signal
        Ys = (Yn + torch.as_tensor(scale)[:, None] * torch.as_tensor(sig)[None]).float()
        rr, ll, cc = nested_xval(Ys, syn_splits, rules=("ols", "reml", "loro_biased"), desc="synth")
        syn = {
            "ols": fit_all(Ys, "fixed", lam=OLS_LAM),
            "reml": fit_all(Ys, "reml"),
            "gcv": fit_all(Ys, "gcv"),
            "fixed": fit_all(Ys, "fixed", lam=LAM_STAR),
            "loro_biased": fit_all(Ys, "fixed", lam=torch.as_tensor(10.0 ** ll["loro_biased"][:, 0])),
        }
        tk, t_peak = truth[FOCUS]
        tA = rel_amp[FOCUS]  # betas are divided by each voxel's scale below
        true_c = centroid_lat(tk[None], t_f)[0]
        true_h = halfmax_lat(tk[None], t_f)[0]
        for m, f in syn.items():
            b = cond(f.betas.numpy(), FOCUS) / scale[:, None]
            pv, pl, edge = peak_amp_lat(b, t_f)
            cl, hl = centroid_lat(b, t_f), halfmax_lat(b, t_f)
            rows.append({
                "delay": delay, "SNR": amp, "rule": m,
                "xval R² (ols/reml)": f"{np.median(rr['ols']):.3f}/{np.median(rr['reml']):.3f}",
                "RMSE/A": np.sqrt(((b - tk) ** 2).mean(1)).mean() / tA,
                "peak/A": np.median(pv) / tA,
                "AUC ratio": np.median(auc(b, dt_f)) / auc(tk[None], dt_f)[0],
                "proj/A": np.median(b @ tk / (tk @ tk)),
                "peak lat bias": np.median(pl) - t_peak, "peak lat IQR": iqr(pl),
                "centroid bias": np.median(cl) - true_c, "centroid IQR": iqr(cl),
                "halfmax bias": np.median(hl) - true_h, "halfmax IQR": iqr(hl),
                "edge %": 100 * edge.mean(),
            })
            if delay == 6.0:
                mean_curves[(amp, m)] = (np.quantile(b, [0.25, 0.5, 0.75], axis=0), tk)
                lat_pools[(amp, m)] = (pl, cl)

syn_df = pd.DataFrame(rows)
cols = ["RMSE/A", "peak/A", "AUC ratio", "proj/A", "peak lat bias", "peak lat IQR",
        "centroid bias", "centroid IQR", "halfmax bias", "halfmax IQR", "edge %"]
display(
    syn_df.groupby(["SNR", "rule"])[cols].mean().style.format(precision=2)
    .set_caption("averaged over the injected latencies")
)
display(syn_df[["delay", "SNR", "xval R² (ols/reml)"]]
        .drop_duplicates(["delay", "SNR"]).set_index(["delay", "SNR"]))
''')

code("""
rules_syn = ["ols", "reml", "gcv", "fixed", "loro_biased"]
g = syn_df.groupby(["SNR", "rule"])[cols].mean().reset_index()
fig, ax = plt.subplots(1, 4, figsize=(19, 3.8), constrained_layout=True)
for m in rules_syn:
    s = g[g.rule == m]
    ax[0].plot(s.SNR, s["peak/A"], "o-", color=C[m], label=m)
    ax[1].plot(s.SNR, s["proj/A"], "o-", color=C[m])
    ax[2].plot(s.SNR, s["RMSE/A"], "o-", color=C[m])
    ax[3].plot(s.SNR, s["centroid IQR"], "o-", color=C[m])
    ax[3].plot(s.SNR, s["peak lat IQR"], "o:", color=C[m], alpha=0.7)
for a, (title, yl) in zip(ax, [("peak / true peak", "ratio"), ("linear amplitude / true", "ratio"),
                               ("curve RMSE / true peak", "relative error"), ("latency IQR: centroid (—), peak (···)", "s")]):
    a.set(title=title, xlabel="injected SNR (peak / noise sd)", ylabel=yl, xscale="log")
    a.set_xticks(levels, [f"{x:g}" for x in levels])
for a in ax[:2]:
    a.axhline(1, color="k", lw=0.8)
ax[0].legend(fontsize=8)
plt.show()

fig, ax = plt.subplots(1, len(levels), figsize=(5.5 * len(levels), 3.6), constrained_layout=True, squeeze=False)
for j, amp in enumerate(levels):
    for m in ("ols", "reml", "fixed"):
        (q25, q50, q75), tk = mean_curves[(amp, m)]
        ax[0, j].plot(t_f, q50, color=C[m], lw=2, label=m)
        ax[0, j].fill_between(t_f, q25, q75, color=C[m], alpha=0.15)
    ax[0, j].plot(t_f, tk, color=C["truth"], lw=2.5, ls="--", label="truth")
    ax[0, j].set(title=f"SNR {amp:g}, delay 6: median ± IQR over voxels (÷ scale)", xlabel="lag (s)")
ax[0, 0].legend(fontsize=8)
plt.show()
""")

md("""
## 10. Timing and reliability on the real data

No ground truth here, but runs are independent measurements. Each rule is fitted to
**each run alone**, and for every signal voxel we ask how well the runs agree:

* curve correlation between runs (shape reliability)
* SD across runs of the peak / centroid / half-max latency
* coefficient of variation across runs of the peak and of the AUC (amplitude reliability)

Per-voxel λ is re-chosen *per run*, so if REML picks very different λ on different
runs, the amplitude can wobble even though the shape is cleaner. The global λ*
keeps the filter fixed.
""")

code("""
rel_vox = sig_idx if sig_idx.size <= 8000 else rng.choice(sig_idx, 8000, replace=False)
Yr = Y[torch.as_tensor(rel_vox)]
per_run = {}
for m in ("ols", "reml", "gcv", "fixed"):
    bs = []
    for r in range(R):
        ti = torch.arange(bounds[r], bounds[r + 1])
        if m in ("ols", "fixed"):
            f = fit_all(Yr, "fixed", lam=OLS_LAM if m == "ols" else LAM_STAR, time_index=ti)
        else:
            f = fit_all(Yr, m, time_index=ti)
        bs.append(cond(f.betas.numpy(), FOCUS))
    per_run[m] = np.stack(bs)  # (R, N, n_knots)


def pairwise_curve_r(b):
    rs = []
    for i in range(R):
        for j in range(i + 1, R):
            a1 = b[i] - b[i].mean(1, keepdims=True)
            a2 = b[j] - b[j].mean(1, keepdims=True)
            rs.append((a1 * a2).sum(1) / np.sqrt((a1 * a1).sum(1) * (a2 * a2).sum(1) + 1e-30))
    return np.mean(rs, 0)


rows = []
for m, b in per_run.items():
    pv = np.stack([peak_amp_lat(b[r], t_f)[0] for r in range(R)])
    pl = np.stack([peak_amp_lat(b[r], t_f)[1] for r in range(R)])
    cl = np.stack([centroid_lat(b[r], t_f) for r in range(R)])
    hl = np.stack([halfmax_lat(b[r], t_f) for r in range(R)])
    ab = np.stack([auc(b[r], dt_f) for r in range(R)])
    rows.append({
        "rule": m,
        "curve r across runs": np.median(pairwise_curve_r(b)),
        "peak-lat SD (s)": np.median(pl.std(0)),
        "centroid SD (s)": np.median(cl.std(0)),
        "halfmax SD (s)": np.median(hl.std(0)),
        "peak CV": np.median(pv.std(0) / np.abs(pv.mean(0))),
        "AUC CV": np.median(ab.std(0) / np.abs(ab.mean(0))),
    })
display(pd.DataFrame(rows).set_index("rule").style.format(precision=3)
        .set_caption(f"median over {rel_vox.size} signal voxels, {labels[FOCUS]}"))

full_lat = {m: peak_amp_lat(cond(B[m][sig_idx], FOCUS), t_f)[1] for m in ("ols", "reml", "fixed")}
full_cen = {m: centroid_lat(cond(B[m][sig_idx], FOCUS), t_f) for m in ("ols", "reml", "fixed")}
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
for m in ("ols", "reml", "fixed"):
    ax[0].hist(full_lat[m], bins=np.arange(-0.5, t_f[-1] + 1, kdt[FOCUS] / 2), histtype="step", lw=1.5, color=C[m], label=m)
    ax[1].hist(full_cen[m], bins=60, histtype="step", lw=1.5, color=C[m], label=m)
ax[0].set(title="peak latency, full-data fit", xlabel="s")
ax[1].set(title="centroid latency, full-data fit", xlabel="s")
ax[0].legend(fontsize=8)
b = per_run["reml"]
lat1 = centroid_lat(b[0], t_f)
lat_rest = np.mean([centroid_lat(b[r], t_f) for r in range(1, R)], 0)
b0 = per_run["ols"]
ax[2].scatter(centroid_lat(b0[0], t_f), np.mean([centroid_lat(b0[r], t_f) for r in range(1, R)], 0),
              s=2, alpha=0.3, color=C["ols"], label=f"OLS r={np.corrcoef(centroid_lat(b0[0], t_f), np.mean([centroid_lat(b0[r], t_f) for r in range(1, R)], 0))[0, 1]:.2f}")
ax[2].scatter(lat1, lat_rest, s=2, alpha=0.3, color=C["reml"], label=f"REML r={np.corrcoef(lat1, lat_rest)[0, 1]:.2f}")
ax[2].set(title="centroid latency: run 1 vs mean of the others", xlabel=f"run {RUNS[0]} (s)", ylabel="other runs (s)")
ax[2].legend(fontsize=8, markerscale=6)
plt.show()
""")

md("""
## 11. Maps

Slices along the third voxel axis (whatever the orientation: axial for RAS/LPI data, coronal for this RSP data) through the part of the volume with the most signal voxels. Left to right: held-out R²
OLS, held-out R² REML (nested), the gain, the full-data REML log10 λ, and edf
(effective knots).
""")

code("""
def to_vol(v, fill=0.0):
    out = np.full(mask3d.shape if mask3d is not None else lr.volume_shape, fill, np.float32)
    if mask3d is not None:
        out[mask3d] = v
    else:
        out = v.reshape(out.shape)
    return out


vols = {
    "held-out R², OLS": to_vol(r2["ols"], np.nan),
    "held-out R², REML (nested)": to_vol(r2["reml"], np.nan),
    "gain (REML − OLS)": to_vol(r2["reml"] - r2["ols"], np.nan),
    "log10 λ (REML, full data)": to_vol(np.log10(fits["reml"].lam.numpy()), np.nan),
    "edf (REML)": to_vol(fits["reml"].edf.numpy(), np.nan),
}
sigvol = to_vol(SIG.astype(np.float32))  # slices along the third voxel axis, wherever it points
zc = int(sigvol.sum((0, 1)).argmax())
slices = [z for z in range(zc - 6, zc + 7, 3) if 0 <= z < sigvol.shape[2]]
lims = {"held-out R², OLS": (0, 0.2), "held-out R², REML (nested)": (0, 0.2), "gain (REML − OLS)": (-0.05, 0.05),
        "log10 λ (REML, full data)": (-2, 7), "edf (REML)": (0, K)}
cmaps = {"gain (REML − OLS)": "RdBu_r", "log10 λ (REML, full data)": "viridis", "edf (REML)": "magma"}
fig, ax = plt.subplots(len(slices), len(vols), figsize=(3.1 * len(vols), 2.9 * len(slices)), constrained_layout=True)
for i, z in enumerate(slices):
    for j, (name, vol) in enumerate(vols.items()):
        im = ax[i, j].imshow(vol[:, :, z].T, origin="upper", cmap=cmaps.get(name, "hot"),
                             vmin=lims[name][0], vmax=lims[name][1], interpolation="nearest")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(name, fontsize=9)
        if j == 0:
            ax[i, j].set_ylabel(f"k = {z}")
for j in range(len(vols)):
    fig.colorbar(ax[0, j].images[0], ax=ax[:, j], shrink=0.4, location="bottom")
plt.show()

if SAVE_MAPS is not None:
    from fastfuncstuff.io.afni import save_nifti

    Path(SAVE_MAPS).mkdir(parents=True, exist_ok=True)
    for name, v in {"xval_r2_ols": r2["ols"], "xval_r2_reml_nested": r2["reml"],
                    "xval_r2_loro_nested": r2["loro_nested"], "xval_r2_loro_biased": r2["loro_biased"],
                    "xval_r2_fixed_lamstar": r2["fixed"], "null_xval_r2_reml": r2_null["reml"]}.items():
        save_nifti(to_vol(v), Path(SAVE_MAPS) / f"{name}.nii.gz", reference_img=inputs[0])
    print("maps written to", SAVE_MAPS)
""")

md("""
## 12. Summary

The numbers below are computed from whatever data the config cell points at.
""")

code('''
def med(m, sel=SIG):
    return float(np.median(r2[m][sel]))


print(f"""
HELD-OUT PREDICTION (signal voxels, median nested LORO R²)
  OLS {med('ols'):.4f} | REML {med('reml'):.4f} | GCV {med('gcv'):.4f} | LORO nested {med('loro_nested'):.4f} | global λ* {med('fixed'):.4f}
  CLI-style LORO (λ saw the scored run) {med('loro_biased'):.4f}  ← optimistic by {med('loro_biased') - med('loro_nested'):+.4f} vs honest LORO
  REML beats OLS in {100 * (r2['reml'][SIG] > r2['ols'][SIG]).mean():.1f}% of signal voxels
  voxels above their own null 99.9th pct: OLS {(r2['ols'] > NULL_THR['ols']).sum():,}, REML {(r2['reml'] > NULL_THR['reml']).sum():,}

INJECTED TRUTH, per SNR level (mean over latencies; the middle level is the real median SNR)""")
keycols = ["peak/A", "proj/A", "RMSE/A", "peak lat bias", "peak lat IQR", "centroid IQR", "edge %"]
display(syn_df.groupby(["SNR", "rule"])[keycols].mean().unstack("SNR").round(2))
''')

md("""
### Reading the results

**Measured when this notebook was written: sub-3010 `expres`, hi/lo pooled, TENT
0–11.4 s, 2 × 18 knots, 3 runs, TR 0.67 s. Signal voxels = 3,653.**

**Prediction: the gain is real, and the CLI's LORO map overstates it only a little.**

| median held-out R², signal voxels | |
|---|---|
| OLS (default) | 0.066 |
| REML, λ from training runs | 0.085 (better in 98% of signal voxels) |
| LORO nested (λ from inner folds) | 0.084 |
| one global λ = 10^1.25 (edf 11.7 of 36) | **0.086** |
| `-tent-smooth loro` as the CLI reports it | 0.090 (+0.006 selection bias) |

* The honest map to show is `-tent-smooth reml -save-xval-r2`.
* Penalty shape doesn't matter: diff1/2/3, gp:2 and gp:4 all land between 0.084 and 0.087.
* The null behaves: shifted onsets give REML **straight lines**, not HRFs. Smoothing
  widens the null somewhat (99.9th percentile 0.029 vs OLS 0.012), so count voxels
  against each rule's own null.

**Amplitude: "lower smoothed peaks" is mostly OLS inflating its peak.** Injected
truth at the real median SNR (peak/noise sd 0.53):

| | OLS | REML | fixed λ* |
|---|---|---|---|
| peak / true peak | **1.30** | 0.92 | 0.96 |
| linear amplitude (projection) | 0.98 | 0.92 | 0.94 |
| curve RMSE / peak | 0.32 | 0.20 | 0.18 |

The real data agree: REML peaks are 0.83× OLS, and areas match (1.00×). At **half** the
median SNR, per-voxel REML starts flattening weak voxels: peak 0.81, linear
amplitude 0.80, 21% of curves peak at the window edge. A fixed λ holds up there (1.04 / 0.91 / 2.5%).
**Per-voxel λ makes the amplitude bias depend on SNR; a fixed λ does not, or much less.**

**Timing.**

* Smoothing tightens peak latency, with an IQR of 0.81 s against 1.29 s for OLS at median SNR.
  On the real data the run-to-run SD falls from 0.76 to 0.61 s, and the curve correlation
  across runs rises from 0.56 to 0.74 (REML) or 0.77 (fixed λ).
* It costs a **later peak by about 0.2–0.3 s** (OLS about 0).
* The centroid is equally good for every rule. All rules share a −0.2 to −0.3 s centroid bias,
  which comes from the window cutting the tail, not from the smoothing.
* Across runs, the peak-amplitude CV is lowest for OLS (0.26; REML 0.40, fixed 0.31). That is not
  a point for OLS: its peak is mostly the *noise maximum*, always large. The AUC CV is the
  same for every rule (0.32).

### Implications

* **One global λ is the sweet spot**: as predictive as per-voxel λ, a linear estimator
  (the same filter everywhere), and no SNR-dependent flattening. `ffs_deconvolve` has
  no rule for "one λ chosen by LORO over the whole brain" yet. The number is on the
  fixed-λ curve above and can be passed as `-tent-smooth <λ>`.
* **ffs_librarian** SVDs per-voxel FIR curves. OLS curves spend components on zig-zags,
  and smoothing concentrates variance on the HRF manifold. Prefer a fixed λ, because
  per-voxel λ pulls weak voxels toward a common flat shape and could narrow the library's
  latency spread artificially.
* **De-veining / laminar profiles**: SNR changes with depth and near veins. OLS peak
  inflation grows as SNR falls, and per-voxel-λ shrinkage grows as SNR falls, so
  **either can manufacture a depth profile**. Compare depths on linear amplitudes (AUC,
  or projection onto a fixed shape) from a fixed-λ fit. The "rescued" voxels are
  mostly broad *negative* responses and deserve a look in that light.
* **Timing maps**: read latency with the centroid or half-max, not the argmax. Report
  the +0.2–0.3 s peak shift if peaks are used, and use a fixed λ so that SNR-dependent
  flattening can't masquerade as latency.
""")

# ---------------------------------------------------------------------------

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "markdown"
        else {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
        for kind, text in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).with_name("fir_smoothing_audit.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
