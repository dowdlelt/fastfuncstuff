"""Probe: does giving event GROUPS their own library HRF beat one shared HRF?

Not a tool -- a measurement kept so the question can be reopened.  See the wiki
note "Grouped HRFs per event type" for the results and what they mean.

Runs ffs_hrfopt's own front end (events, nuisance, projection, per-curve design
build) and captures the projected data and designs at the point the library
would be scored.  Every combination of one library curve per group (N**G
designs) is then scored from per-run sufficient statistics (X'X, X'y, y'y).
X'y is gathered from G*N per-curve blocks, so only the tiny k x k solves scale
as N**G -- 8000 triplets x 25k voxels is ~50 s on one consumer GPU.

Each set partition of the groups is a model: groups in one block share a curve,
so the one-block partition is the ordinary per-voxel library HRF.  Scoring is
NESTED leave-one-run-out: for each outer run the curve combination is chosen by
inner LORO over the remaining runs, fit on all runs but the outer one and scored
on it.  Plain LORO max-over-combinations is optimistic and the bias grows with
the size of the search, so it must not be used to judge a grouping.

Usage:
    python scripts/probe_grouped_hrf.py OUT.npz 'S=scale;P=_(DP|SP)$;I=_(DI|SI|PI)$' \
        -- <ffs_hrfopt args: -input ... -events ... -mask ... -prefix ... -device ...>

Every condition label must match exactly one group regex.  The npz holds, per
model, the nested held-out SS (ss__<model>) and the chosen combination per outer
fold (sel__<model>, indexing `combos`), plus ss_tot; dR2 = (ss__SHARED - ss__M) / ss_tot.
Memory: the (runs, N**G, K, chunk) X'y tensor is the peak; lower `chunk` for
bigger searches (blend, 40 curves, is 64000 triplets at G=3 -- use coordinate
descent instead of exhaustive search there).
"""

import itertools
import re
import sys

import numpy as np
import torch

from fastfuncstuff.design import hrf_selection as hs

out_path, group_spec = sys.argv[1], sys.argv[2]
assert sys.argv[3] == "--"
hrfopt_argv = sys.argv[4:]
groups = [tuple(g.split("=", 1)) for g in group_spec.split(";")]

captured: dict = {}
_orig_fit = hs.fit_glm_hrf_library_with_xval


def _fit_wrapper(*args, **kwargs):
    captured["labels"] = list(kwargs["condition_labels"])
    captured["library"] = torch.as_tensor(kwargs["hrf_library"]).cpu().numpy()
    captured["run_starts"] = list(kwargs["run_starts"])
    return _orig_fit(*args, **kwargs)


class _CapturedError(Exception):
    pass


def _eval_wrapper(projected_data, projected_designs, run_starts, cv_splits, device, **kw):
    captured["data"] = projected_data
    captured["designs"] = projected_designs
    captured["device"] = device
    raise _CapturedError


hs.fit_glm_hrf_library_with_xval = _fit_wrapper
hs._evaluate_hrfs_batched = _eval_wrapper

from fastfuncstuff.cli import hrfopt  # noqa: E402

sys.argv = ["ffs_hrfopt", *hrfopt_argv]
try:
    hrfopt.main()
except _CapturedError:
    pass

labels = captured["labels"]
lib = captured["library"]
N = lib.shape[0]
designs = captured["designs"][:N]
device = captured["device"]
run_starts = captured["run_starts"]
Y = captured["data"]
n_vox, n_t = Y.shape
K = designs[0].shape[1]
assert K == len(labels)
G = len(groups)
col_group = []
for lab in labels:
    hits = [g for g, (_, rx) in enumerate(groups) if re.search(rx, lab)]
    assert len(hits) == 1, (lab, hits)
    col_group.append(hits[0])
for g, (name, _) in enumerate(groups):
    print(f"group {name}:", [lab for lab, cg in zip(labels, col_group, strict=True) if cg == g])
col_group_t = torch.tensor(col_group, device=device)

wide = torch.float64
F = torch.stack([d.to(device) for d in designs]).to(wide)  # (N, T, K)
combos = torch.tensor(list(itertools.product(range(N), repeat=G)), device=device)  # (P, G)
P = combos.shape[0]
col_curve = combos[:, col_group_t]  # (P, K): curve used by each column
kk = torch.arange(K, device=device)


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1 :]
        yield [[first]] + part


models = []
for part in set_partitions(list(range(G))):
    ok = torch.ones(P, dtype=torch.bool, device=device)
    for block in part:
        for g in block[1:]:
            ok &= combos[:, g] == combos[:, block[0]]
    name = "|".join("".join(groups[g][0] for g in sorted(b)) for b in sorted(part, key=min))
    models.append((name, ok.nonzero().squeeze(1)))
models.sort(key=lambda m: len(m[1]))
print("models:", [(n, len(i)) for n, i in models])

run_ends = run_starts[1:] + [n_t]
R = len(run_starts)
xtx = torch.zeros(R, P, K, K, dtype=wide, device=device)
for r, (s, e) in enumerate(zip(run_starts, run_ends, strict=True)):
    for ps in range(0, P, 1000):
        pe = min(ps + 1000, P)
        X = F[col_curve[ps:pe], s:e, kk].permute(0, 2, 1)  # (p, T_r, K)
        xtx[r, ps:pe] = X.transpose(1, 2) @ X
xtx_tot = xtx.sum(0)


def pinv_train(excluded):
    m = xtx_tot.clone()
    for r in excluded:
        m -= xtx[r]
    return torch.linalg.pinv(m.cpu()).to(device)


pinv_outer = {o: pinv_train([o]) for o in range(R)}
pinv_inner = {(o, k): pinv_train([o, k]) for o in range(R) for k in range(R) if k != o}


def test_ss(pinv, excluded, test, xty, xty_tot, yy):
    """Held-out SS on run `test` of the fit trained on every run not in `excluded`."""
    b = pinv @ (xty_tot - sum(xty[r] for r in excluded))
    return yy[test][None] - 2 * (b * xty[test]).sum(1) + (b * (xtx[test] @ b)).sum(1)


chunk = 200
ss = {name: np.zeros(n_vox) for name, _ in models}
sel = {name: np.zeros((R, n_vox), dtype=np.int32) for name, _ in models}
ss_tot = np.zeros(n_vox)

for cs in range(0, n_vox, chunk):
    ce = min(cs + chunk, n_vox)
    C = ce - cs
    y = Y[cs:ce].to(device).to(wide)
    ss_tot[cs:ce] = ((y - y.mean(1, keepdim=True)) ** 2).sum(1).cpu().numpy()
    Gxy = torch.stack(
        [F[:, s:e].transpose(1, 2) @ y[:, s:e].T for s, e in zip(run_starts, run_ends, strict=True)]
    )  # (R, N, K, C)
    xty = Gxy[:, col_curve, kk]  # (R, P, K, C)
    yy = torch.stack([(y[:, s:e] ** 2).sum(1) for s, e in zip(run_starts, run_ends, strict=True)])
    xty_tot = xty.sum(0)

    cidx = torch.arange(C, device=device)
    for o in range(R):
        outer = test_ss(pinv_outer[o], [o], o, xty, xty_tot, yy)  # (P, C)
        inner = sum(
            test_ss(pinv_inner[(o, k)], [o, k], k, xty, xty_tot, yy) for k in range(R) if k != o
        )
        for name, idx in models:
            best = idx[inner[idx].argmin(0)]
            ss[name][cs:ce] += outer[best, cidx].cpu().numpy()
            sel[name][o, cs:ce] = best.cpu().numpy()
    if (cs // chunk) % 20 == 0:
        print(f"  voxels {ce}/{n_vox}", flush=True)

np.savez_compressed(
    out_path,
    labels=np.array(labels),
    col_group=np.array(col_group),
    group_names=np.array([g[0] for g in groups]),
    combos=combos.cpu().numpy(),
    library=lib,
    ss_tot=ss_tot,
    model_names=np.array([n for n, _ in models]),
    **{f"ss__{n}": v for n, v in ss.items()},
    **{f"sel__{n}": v for n, v in sel.items()},
)
shared = [n for n, i in models if "|" not in n][0]
for name, _ in models:
    d = (ss[shared] - ss[name]) / ss_tot
    print(
        f"{name:8s} nested dR2 vs shared: mean {d.mean():+.5f} median {np.median(d):+.5f} frac>0 {(d > 0).mean():.3f}"
    )
