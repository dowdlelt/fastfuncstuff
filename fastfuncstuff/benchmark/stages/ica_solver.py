"""ICA solver benchmark: isolates FastICA from upstream prep.

Given MELODIC's whitening matrix and concat_data, applies whitening identical
to MELODIC and runs FFS's FastICA solver on top. Compares the resulting IC
spatial maps against MELODIC's. If varnorm/PCA/whitening already matched, this
would be redundant — but with the upstream divergence the trace stage exposes,
this stage tells us whether the *solver* itself agrees when fed identical
inputs.

Pipeline:
    melodic_white (k×T) @ concat_data (T×V) → whitened (k×V)
    FFS FastICA on whitened → unmixing W (k×k)
    components = W @ whitened → (k, V)
    Compare components to MELODIC's melodic_oIC.nii.gz (k spatial maps)

Requires MELODIC --debug --Oall outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import time

from ..runner import BenchmarkContext

name = "ica_solver"
description = "FastICA solver parity given MELODIC's whitened input"


def _ica_tasks(ctx: BenchmarkContext) -> list[str]:
    params = ctx.get_stage_params("ica_solver")
    return params.get("tasks", ctx.task_names())


def _melodic_dir(ctx: BenchmarkContext, dataset: str) -> Path:
    return ctx.melodic_ica_dir / f"all_{dataset}_melodic.ica"


def _solver_dir(ctx: BenchmarkContext, dataset: str) -> Path:
    return ctx.ffs_ica_dir / f"all_{dataset}_solver"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    for dataset in _ica_tasks(ctx):
        md = _melodic_dir(ctx, dataset)
        for f in ["melodic_white", "concat_data.nii.gz", "mask.nii.gz", "melodic_oIC.nii.gz"]:
            if not (md / f).exists():
                missing.append(str(md / f))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    return 0.0


def _run_solver(mel_dir: Path, out_dir: Path, seed: int = 1) -> None:
    """Apply MELODIC whitening, run FFS FastICA, save components and W."""
    import nibabel as nib
    import torch

    from fastfuncstuff.decomposition.ica import FastICA

    out_dir.mkdir(parents=True, exist_ok=True)

    white = np.loadtxt(mel_dir / "melodic_white").astype(np.float64)  # (k, T)
    mask_img = nib.load(str(mel_dir / "mask.nii.gz"))
    mask = mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
    concat_img = nib.load(str(mel_dir / "concat_data.nii.gz"))
    concat_4d = concat_img.get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    if concat_4d.ndim != 4:
        raise ValueError(f"concat_data not 4D: shape={concat_4d.shape}")
    concat_tv = concat_4d[mask].astype(np.float64).T  # (T, V)
    if concat_tv.shape[0] != white.shape[1]:
        raise ValueError(
            f"shape mismatch: white={white.shape}, concat_TV={concat_tv.shape}"
        )

    k = white.shape[0]
    whitened = (white @ concat_tv).astype(np.float32)  # (k, V)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MELODIC's default contrast is FSL --nl=pow3, which has a non-standard
    # update rule (W = 3*E[X*u^2] - E[u]*W). FFS exposes this exactly as
    # fun='pow3' (NOT 'cube' — 'cube' is the textbook FastICA cube g=u^3,
    # different update equation).
    ica = FastICA(
        n_components=k,
        pca_components=None,
        max_iter=5000,
        tol=1e-5,
        fun="pow3",
        random_state=seed,
        whiten=False,
        device=device,
    )
    X = torch.as_tensor(whitened, device=device)
    W, n_iter = ica._fastica(X, n_components=k)
    final_delta = float(getattr(ica, "_final_delta", float("nan")))
    components = (W @ X).cpu().numpy().astype(np.float32)  # (k, V)

    np.save(out_dir / "ic_components.npy", components)
    np.save(out_dir / "unmixing.npy", W.cpu().numpy())
    np.save(out_dir / "whitened_input.npy", whitened)
    (out_dir / "n_iter.txt").write_text(str(int(n_iter)))
    (out_dir / "final_delta.txt").write_text(f"{final_delta:.6e}")


def run_ffs(ctx: BenchmarkContext) -> float:
    total = 0.0
    for dataset in _ica_tasks(ctx):
        out = _solver_dir(ctx, dataset)
        if (out / "ic_components.npy").exists() and not ctx.force_ffs:
            continue
        mel_dir = _melodic_dir(ctx, dataset)
        print(f"  Running: ica_solver {dataset}...")
        t0 = time.monotonic()
        _run_solver(mel_dir, out)
        elapsed = time.monotonic() - t0
        print(f"    elapsed: {elapsed:.2f}s")
        total += elapsed
    return total


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum()))
    if denom < 1e-15:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def _compare(mel_dir: Path, out_dir: Path) -> dict:
    import nibabel as nib

    comp_p = out_dir / "ic_components.npy"
    if not comp_p.exists():
        return {"error": "ic_components.npy missing"}
    ffs_comp = np.load(comp_p)  # (k, V)
    mel_img = nib.load(str(mel_dir / "melodic_oIC.nii.gz"))
    mask = nib.load(str(mel_dir / "mask.nii.gz")).get_fdata() > 0.5  # type: ignore[attr-defined]
    mel_4d = mel_img.get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    if mel_4d.ndim != 4:
        return {"error": f"melodic_oIC not 4D: {mel_4d.shape}"}
    mel_comp = mel_4d[mask].T.astype(np.float64)  # (k_mel, V)

    k_mel, k_ffs = mel_comp.shape[0], ffs_comp.shape[0]
    k = min(k_mel, k_ffs)

    # Build cross-correlation (k_mel × k_ffs) of |corr|.
    a = mel_comp - mel_comp.mean(axis=1, keepdims=True)
    b = ffs_comp.astype(np.float64)
    b = b - b.mean(axis=1, keepdims=True)
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    cross = np.abs(a_n @ b_n.T)  # (k_mel, k_ffs)

    from scipy.optimize import linear_sum_assignment

    cost = 1.0 - cross[:k, :k]
    row, col = linear_sum_assignment(cost)
    matched = cross[row, col]

    delta_p = out_dir / "final_delta.txt"
    n_iter_p = out_dir / "n_iter.txt"
    final_delta = float(delta_p.read_text().strip()) if delta_p.exists() else float("nan")
    n_iter = int(n_iter_p.read_text().strip()) if n_iter_p.exists() else -1

    return {
        "n_matched": int(len(matched)),
        "k_melodic": int(k_mel),
        "k_ffs": int(k_ffs),
        "mean_matched_r": float(matched.mean()),
        "median_matched_r": float(np.median(matched)),
        "min_matched_r": float(matched.min()),
        "frac_above_0.9": float((matched >= 0.9).mean()),
        "frac_above_0.7": float((matched >= 0.7).mean()),
        "n_iter": n_iter,
        "final_delta": final_delta,
    }


def validate(ctx: BenchmarkContext) -> dict:
    results = {}
    for dataset in _ica_tasks(ctx):
        results[dataset] = _compare(
            _melodic_dir(ctx, dataset), _solver_dir(ctx, dataset)
        )

    valid = [r for r in results.values() if "mean_matched_r" in r]
    if not valid:
        return {"passed": False, "summary": "no datasets produced solver output", "per_dataset": results}

    mean_r = float(np.mean([r["mean_matched_r"] for r in valid]))
    passed = mean_r >= 0.85

    parts = [f"mean_matched_r={mean_r:.4f}"]
    for ds, r in results.items():
        if "mean_matched_r" in r:
            parts.append(
                f"{ds}: r={r['mean_matched_r']:.3f} "
                f">0.9={r['frac_above_0.9']:.2f} "
                f"(k={r['k_melodic']}, iter={r['n_iter']}, "
                f"delta={r['final_delta']:.2e})"
            )
        else:
            parts.append(f"{ds}: ERR ({r.get('error', '?')})")

    return {"passed": passed, "summary": ", ".join(parts), "per_dataset": results}
