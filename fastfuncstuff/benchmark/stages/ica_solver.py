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

import time
from pathlib import Path

import numpy as np

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
    """Apply MELODIC whitening, run FFS FastICA from two inits:

    random_init/  : random seed, tests solver basin from cold start.
    mel_init/     : initial W = melodic_unmix, tests whether MELODIC's
                    converged solution is also a fixed point of FFS's
                    pow3 update + symmetric decorrelation. If FFS stays
                    there (n_iter≈1, mean_matched_r≈1.0) the solvers are
                    mathematically equivalent and the only thing that
                    caused divergence in the random-init case was pow3's
                    multi-modal landscape.
    """
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
        raise ValueError(f"shape mismatch: white={white.shape}, concat_TV={concat_tv.shape}")

    k = white.shape[0]
    whitened = (white @ concat_tv).astype(np.float32)  # (k, V)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(whitened, device=device)

    # Load MELODIC's converged unmixing as a candidate W0 in whitened space.
    #   melodic_unmix is (k, T): the FULL unmixing (whitening folded in).
    #     S = melodic_unmix @ data
    #   To recover the pure unmixing in whitened space (k, k):
    #     melodic_unmix = W_pure @ melodic_white, so W_pure = melodic_unmix @ dewhite
    #     since melodic_white @ melodic_dewhite ≈ I_k.
    mel_unmix_p = mel_dir / "melodic_unmix"
    mel_dewhite_p = mel_dir / "melodic_dewhite"
    mel_W = None
    if mel_unmix_p.exists() and mel_dewhite_p.exists():
        mel_unmix = np.loadtxt(mel_unmix_p)  # (k, T)
        mel_dewhite = np.loadtxt(mel_dewhite_p)  # (T, k)
        if mel_unmix.shape[0] == k and mel_dewhite.shape == (mel_unmix.shape[1], k):
            W_pure = mel_unmix @ mel_dewhite  # (k, k)
            mel_W = torch.as_tensor(W_pure.astype(np.float32), device=device)
        else:
            print(
                f"  WARN: shape mismatch unmix={mel_unmix.shape} dewhite={mel_dewhite.shape} "
                f"k={k}; skipping mel_init"
            )
    else:
        print("  WARN: melodic_unmix or melodic_dewhite missing; skipping mel_init")

    def _run_one(sub: str, w_init: torch.Tensor | None) -> None:
        sub_dir = out_dir / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
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
        W, n_iter = ica._fastica(X, n_components=k, w_init=w_init)
        final_delta = float(getattr(ica, "_final_delta", float("nan")))
        components = (W @ X).cpu().numpy().astype(np.float32)
        np.save(sub_dir / "ic_components.npy", components)
        np.save(sub_dir / "unmixing.npy", W.cpu().numpy())
        (sub_dir / "n_iter.txt").write_text(str(int(n_iter)))
        (sub_dir / "final_delta.txt").write_text(f"{final_delta:.6e}")

    np.save(out_dir / "whitened_input.npy", whitened)

    print(f"  [random_init] FastICA from random seed={seed}...")
    _run_one("random_init", None)
    if mel_W is not None:
        print("  [mel_init] FastICA from MELODIC's melodic_unmix...")
        _run_one("mel_init", mel_W)


def run_ffs(ctx: BenchmarkContext) -> float:
    total = 0.0
    for dataset in _ica_tasks(ctx):
        out = _solver_dir(ctx, dataset)
        if (out / "random_init" / "ic_components.npy").exists() and not ctx.force_ffs:
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
    denom = float(np.sqrt((a_c**2).sum() * (b_c**2).sum()))
    if denom < 1e-15:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def _compare_one(mel_dir: Path, sub_dir: Path, mel_comp: np.ndarray) -> dict:
    """Compare components in sub_dir against pre-loaded MELODIC IC maps."""
    comp_p = sub_dir / "ic_components.npy"
    if not comp_p.exists():
        return {"error": f"ic_components.npy missing in {sub_dir.name}"}
    ffs_comp = np.load(comp_p)  # (k, V)

    k_mel, k_ffs = mel_comp.shape[0], ffs_comp.shape[0]
    k = min(k_mel, k_ffs)

    a = mel_comp - mel_comp.mean(axis=1, keepdims=True)
    b = ffs_comp.astype(np.float64)
    b = b - b.mean(axis=1, keepdims=True)
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    cross = np.abs(a_n @ b_n.T)

    from scipy.optimize import linear_sum_assignment

    cost = 1.0 - cross[:k, :k]
    row, col = linear_sum_assignment(cost)
    matched = cross[row, col]

    delta_p = sub_dir / "final_delta.txt"
    n_iter_p = sub_dir / "n_iter.txt"
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


def _compare(mel_dir: Path, out_dir: Path) -> dict:
    import nibabel as nib

    mel_img = nib.load(str(mel_dir / "melodic_oIC.nii.gz"))
    mask = nib.load(str(mel_dir / "mask.nii.gz")).get_fdata() > 0.5  # type: ignore[attr-defined]
    mel_4d = mel_img.get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    if mel_4d.ndim != 4:
        return {"error": f"melodic_oIC not 4D: {mel_4d.shape}"}
    mel_comp = mel_4d[mask].T.astype(np.float64)

    return {
        "random_init": _compare_one(mel_dir, out_dir / "random_init", mel_comp),
        "mel_init": _compare_one(mel_dir, out_dir / "mel_init", mel_comp),
    }


def validate(ctx: BenchmarkContext) -> dict:
    results = {}
    for dataset in _ica_tasks(ctx):
        results[dataset] = _compare(_melodic_dir(ctx, dataset), _solver_dir(ctx, dataset))

    # Pull out per-init means for the headline.
    rand_rs = [
        r["random_init"]["mean_matched_r"]
        for r in results.values()
        if "random_init" in r and "mean_matched_r" in r["random_init"]
    ]
    mel_rs = [
        r["mel_init"]["mean_matched_r"]
        for r in results.values()
        if "mel_init" in r and "mean_matched_r" in r["mel_init"]
    ]
    if not rand_rs:
        return {
            "passed": False,
            "summary": "no datasets produced solver output",
            "per_dataset": results,
        }

    rand_mean = float(np.mean(rand_rs))
    mel_mean = float(np.mean(mel_rs)) if mel_rs else float("nan")
    # Pass if MELODIC-init reproduces MELODIC's solution (proves solver
    # equivalence). The random-init number can be lower without failing —
    # pow3 is multi-modal.
    passed = (mel_mean >= 0.95) if mel_rs else (rand_mean >= 0.85)

    parts = [f"rand_r={rand_mean:.4f}", f"mel_init_r={mel_mean:.4f}"]
    for ds, r in results.items():
        ri = r.get("random_init", {})
        mi = r.get("mel_init", {})
        if "mean_matched_r" in ri:
            parts.append(
                f"{ds}/rand: r={ri['mean_matched_r']:.3f} "
                f">0.9={ri['frac_above_0.9']:.2f} "
                f"(iter={ri['n_iter']}, delta={ri['final_delta']:.1e})"
            )
        if "mean_matched_r" in mi:
            parts.append(
                f"{ds}/mel: r={mi['mean_matched_r']:.3f} "
                f">0.9={mi['frac_above_0.9']:.2f} "
                f"(iter={mi['n_iter']}, delta={mi['final_delta']:.1e})"
            )
        if "error" in r:
            parts.append(f"{ds}: ERR ({r.get('error', '?')})")

    return {"passed": passed, "summary": ", ".join(parts), "per_dataset": results}
