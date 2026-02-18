"""Reusable postprocessing utilities for ICA workflows.

This module centralizes masking, scoring, lag analysis, and plotting helpers
that were previously embedded in CLI scripts.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch

from .ica_tools import apply_high_pass_fft, apply_polort_projection

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - import guard for optional runtime dep
    nib = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def load_run_ortvec_design(
    ortvec_specs: list[list[str]] | None,
    run_idx: int,
    n_timepoints: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, list[str] | None]:
    """Load run-specific nuisance regressors from CLI-style ortvec specs."""
    if not ortvec_specs:
        return None, None

    mats: list[np.ndarray] = []
    labels: list[str] = []
    run_tag = f"{run_idx + 1:03d}"

    for spec in ortvec_specs:
        label = str(spec[0])
        files = spec[1:]
        file_idx = run_idx if len(files) > 1 else 0
        fpath = files[file_idx]

        arr = np.loadtxt(fpath, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim != 2:
            raise ValueError(f"ortvec file must be 1D/2D: {fpath}")

        if arr.shape[0] != n_timepoints and arr.shape[1] == n_timepoints:
            arr = arr.T
        if arr.shape[0] != n_timepoints:
            raise ValueError(
                f"ortvec rows ({arr.shape[0]}) do not match run timepoints ({n_timepoints}) in {fpath}"
            )

        mats.append(arr)
        for col in range(arr.shape[1]):
            labels.append(f"{label}{run_tag}_{col + 1}")

    design_np = np.concatenate(mats, axis=1).astype(np.float32)
    design_tc = torch.as_tensor(design_np, device=device, dtype=torch.float32)
    return design_tc, labels


def expand_mask_file(mask_path: str, shape3d: tuple[int, int, int]) -> list[tuple[str, np.ndarray]]:
    """Expand a spatial mask file into one or more boolean 3D masks.

    Supported formats
    -----------------
    - 3D binary/non-binary: one mask (values > 0)
    - 3D integer labels: one mask per positive label value
    - 4D: one mask per frame (values > 0 in each frame)
    """
    if nib is None:
        raise ImportError("nibabel is required for mask loading")

    img = nib.load(mask_path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    stem = Path(mask_path).name

    masks: list[tuple[str, np.ndarray]] = []
    if data.ndim == 4:
        if tuple(data.shape[:3]) != tuple(shape3d):
            raise ValueError(
                f"Mask {mask_path} has spatial shape {data.shape[:3]} but run shape is {shape3d}"
            )
        for frame_idx in range(data.shape[3]):
            frame_mask = np.isfinite(data[..., frame_idx]) & (data[..., frame_idx] > 0)
            masks.append((f"{stem}:frame{frame_idx + 1:03d}", frame_mask))
        return masks

    if data.ndim != 3:
        raise ValueError(f"Mask {mask_path} must be 3D or 4D, got ndim={data.ndim}")
    if tuple(data.shape) != tuple(shape3d):
        raise ValueError(f"Mask {mask_path} has shape {data.shape} but run shape is {shape3d}")

    finite_data = np.where(np.isfinite(data), data, 0.0)
    pos_vals = finite_data[finite_data > 0]
    if pos_vals.size == 0:
        masks.append((f"{stem}:all", np.zeros(shape3d, dtype=bool)))
        return masks

    rounded = np.rint(pos_vals)
    is_integer_like = np.all(np.abs(pos_vals - rounded) < 1e-5)
    unique_labels = np.unique(rounded.astype(np.int32)) if is_integer_like else np.array([], dtype=np.int32)

    if is_integer_like and unique_labels.size > 1:
        for lbl in unique_labels:
            if lbl <= 0:
                continue
            masks.append((f"{stem}:label{int(lbl)}", np.rint(finite_data).astype(np.int32) == int(lbl)))
    else:
        masks.append((f"{stem}:all", finite_data > 0))
    return masks


def prepare_guidance_masks(
    mask_paths: list[str] | None,
    kind: str,
    shape3d: tuple[int, int, int],
    brain_mask3d: np.ndarray | None,
    n_vox_masked: int,
    verbose: bool = False,
) -> list[dict]:
    """Parse masks and map them into masked voxel index space."""
    if not mask_paths:
        return []

    out: list[dict] = []
    for path in mask_paths:
        expanded = expand_mask_file(path, shape3d)
        for mask_name, raw_mask in expanded:
            m = raw_mask.astype(bool)
            if brain_mask3d is not None:
                m = m & brain_mask3d
                selector = m[brain_mask3d]
            else:
                selector = m.reshape(-1)

            if selector.shape[0] != n_vox_masked:
                raise ValueError(
                    f"Internal mask mapping error for {mask_name}: selector has {selector.shape[0]} "
                    f"voxels, expected {n_vox_masked}"
                )

            n_sel = int(selector.sum())
            if n_sel == 0:
                if verbose:
                    print(f"    Warning: {kind} mask {mask_name} has 0 voxels after brain masking; skipped")
                continue

            out.append(
                {
                    "name": mask_name,
                    "source": path,
                    "selector": selector.astype(bool),
                    "n_voxels": n_sel,
                    "kind": kind,
                }
            )
    return out


def prepare_depth_mask(
    depth_mask_path: str | None,
    shape3d: tuple[int, int, int],
    brain_mask3d: np.ndarray | None,
    n_vox_masked: int,
) -> dict | None:
    """Load depth labels and map each positive integer depth to masked voxel space."""
    if depth_mask_path is None:
        return None
    if nib is None:
        raise ImportError("nibabel is required for depth mask loading")

    img = nib.load(depth_mask_path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"-depth_mask must be 3D integer-labeled image, got ndim={data.ndim}")
    if tuple(data.shape) != tuple(shape3d):
        raise ValueError(
            f"Depth mask {depth_mask_path} has shape {data.shape} but run shape is {shape3d}"
        )

    rounded = np.rint(np.where(np.isfinite(data), data, 0.0)).astype(np.int32)
    labels = [int(v) for v in np.unique(rounded) if int(v) > 0]

    depth_selectors: dict[int, np.ndarray] = {}
    for lbl in labels:
        m = rounded == lbl
        if brain_mask3d is not None:
            m = m & brain_mask3d
            selector = m[brain_mask3d]
        else:
            selector = m.reshape(-1)
        if selector.shape[0] != n_vox_masked:
            raise ValueError(
                f"Internal depth mask mapping error for label {lbl}: {selector.shape[0]} vs {n_vox_masked}"
            )
        if int(selector.sum()) > 0:
            depth_selectors[lbl] = selector.astype(bool)

    return {
        "path": depth_mask_path,
        "labels": sorted(depth_selectors.keys()),
        "selectors": depth_selectors,
    }


def mean_abs_by_selector(comp_kv: np.ndarray, selector_v: np.ndarray) -> np.ndarray:
    if int(selector_v.sum()) == 0:
        return np.zeros(comp_kv.shape[0], dtype=np.float32)
    return np.mean(np.abs(comp_kv[:, selector_v]), axis=1).astype(np.float32)


def mean_z_excess_by_selector(z_kv: np.ndarray, selector_v: np.ndarray, z_thresh: float) -> np.ndarray:
    if int(selector_v.sum()) == 0:
        return np.zeros(z_kv.shape[0], dtype=np.float32)
    z_sel = np.abs(z_kv[:, selector_v])
    return np.mean(np.maximum(z_sel - float(z_thresh), 0.0), axis=1).astype(np.float32)


def normalize_0_1(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if mx <= 1e-8:
        return np.zeros_like(x)
    return x / mx


def best_lag_and_r(
    x_t: np.ndarray,
    y_t: np.ndarray,
    tr: float,
    max_lag_s: float,
) -> tuple[float, float, str]:
    """Estimate lag (seconds) and peak correlation between two timeseries."""
    x = np.asarray(x_t, dtype=np.float64)
    y = np.asarray(y_t, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Lag input timeseries must be 1D and same length")

    try:
        rcor = importlib.import_module("rapidtide.correlate")
        if hasattr(rcor, "quickcorr"):
            maxlag_tr = int(max(1, round(float(max_lag_s) / float(tr))))
            lag_tr, r_val = rcor.quickcorr(x, y, maxlag=maxlag_tr)  # type: ignore[attr-defined]
            return float(lag_tr) * float(tr), float(r_val), "rapidtide.quickcorr"
    except Exception:
        pass

    x = x - np.mean(x)
    y = y - np.mean(y)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0, 0.0, "numpy.xcorr.degenerate"

    xz = x / x_std
    yz = y / y_std

    n_t = xz.shape[0]
    maxlag_tr = int(max(1, round(float(max_lag_s) / float(tr))))
    maxlag_tr = min(maxlag_tr, n_t - 1)

    full = np.correlate(xz, yz, mode="full") / float(n_t)
    lags = np.arange(-n_t + 1, n_t, dtype=np.int32)
    keep = np.abs(lags) <= maxlag_tr
    full_k = full[keep]
    lags_k = lags[keep]

    best_idx = int(np.argmax(full_k))
    lag_tr = int(lags_k[best_idx])
    r_best = float(full_k[best_idx])
    return float(lag_tr) * float(tr), r_best, "numpy.xcorr"


def weighted_depth_timeseries(
    source_vox_t: np.ndarray,
    selector_v: np.ndarray,
    weight_v: np.ndarray,
    min_voxels: int,
) -> tuple[np.ndarray | None, int]:
    use = selector_v & np.isfinite(weight_v) & (weight_v > 0)
    n_use = int(use.sum())
    if n_use < int(min_voxels):
        return None, n_use
    w = np.asarray(weight_v[use], dtype=np.float64)
    if float(np.sum(w)) <= 1e-12:
        return None, n_use
    x = np.asarray(source_vox_t[use, :], dtype=np.float64)
    return np.average(x, axis=0, weights=w), n_use


def save_scree_plot(evr: np.ndarray, out_png: Path, title: str) -> None:
    if plt is None:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(x, evr, lw=1.5)
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2.plot(x, np.clip(cum, 0.0, 1.0), lw=1.5)
    ax2.set_xlabel("Component index")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_depth_lag_plot(
    lag_matrix_kd: np.ndarray,
    depth_labels: list[int],
    out_png: Path,
    title: str,
) -> None:
    if plt is None or lag_matrix_kd.size == 0 or len(depth_labels) == 0:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.nanmax(np.abs(lag_matrix_kd))) if np.isfinite(lag_matrix_kd).any() else 1.0
    vmax = max(vmax, 1e-3)
    fig_w = max(8.0, 0.5 * len(depth_labels) + 4.0)
    fig_h = max(6.0, 0.14 * lag_matrix_kd.shape[0] + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    m = np.where(np.isfinite(lag_matrix_kd), lag_matrix_kd, np.nan)
    im = ax.imshow(m, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Depth label")
    ax.set_ylabel("Component index")
    ax.set_xticks(np.arange(len(depth_labels)))
    ax.set_xticklabels([str(v) for v in depth_labels], rotation=0)
    ax.set_yticks(np.arange(lag_matrix_kd.shape[0]))
    ax.set_yticklabels([str(i + 1) for i in range(lag_matrix_kd.shape[0])], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lag (s) vs reference depth")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_corr_heatmap(corr_kn: np.ndarray, labels: list[str], out_png: Path, title: str) -> None:
    if plt is None or corr_kn.size == 0 or len(labels) == 0:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.max(np.abs(corr_kn))) if corr_kn.size > 0 else 1.0
    vmax = max(vmax, 1e-3)
    fig_w = max(8.0, 0.5 * len(labels) + 5.0)
    fig_h = max(6.0, 0.14 * corr_kn.shape[0] + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr_kn, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Design / ortvec regressor")
    ax.set_ylabel("Component index")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(corr_kn.shape[0]))
    ax.set_yticklabels([str(i + 1) for i in range(corr_kn.shape[0])], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_score_heatmap(
    scores_kn: np.ndarray,
    labels: list[str],
    out_png: Path,
    title: str,
    cmap: str,
) -> None:
    if plt is None or scores_kn.size == 0 or len(labels) == 0:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.max(scores_kn)) if scores_kn.size > 0 else 1.0
    vmax = max(vmax, 1e-4)
    fig_w = max(8.0, 0.45 * len(labels) + 4.0)
    fig_h = max(6.0, 0.14 * scores_kn.shape[0] + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(scores_kn, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Mask")
    ax.set_ylabel("Component index")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(scores_kn.shape[0]))
    ax.set_yticklabels([str(i + 1) for i in range(scores_kn.shape[0])], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def preprocess_design_for_correlation(
    design_tc: torch.Tensor,
    tr: float,
    polort: int,
    high_pass_hz: float | None,
    device: torch.device,
) -> torch.Tensor:
    """Apply the same temporal preprocessing as ICA data before correlations."""
    if design_tc.numel() == 0:
        return design_tc

    reg_ct = design_tc.T
    reg_ct = apply_polort_projection(reg_ct, polort=polort, device=device)
    if high_pass_hz is not None and high_pass_hz > 0:
        reg_ct = apply_high_pass_fft(reg_ct, tr=tr, high_pass_hz=high_pass_hz)
    return reg_ct.T


def component_condition_spectral_correlations(
    mixing_tk: torch.Tensor,
    design_tc: torch.Tensor,
) -> np.ndarray:
    """Compute component-vs-regressor spectral correlations using log rFFT power."""
    x = mixing_tk.detach().cpu().numpy().astype(np.float64)
    y = design_tc.detach().cpu().numpy().astype(np.float64)

    sx = np.abs(np.fft.rfft(x, axis=0)) ** 2
    sy = np.abs(np.fft.rfft(y, axis=0)) ** 2

    if sx.shape[0] > 1:
        sx = sx[1:, :]
        sy = sy[1:, :]

    if sx.shape[0] == 0:
        return np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)

    sx = np.log1p(sx)
    sy = np.log1p(sy)

    sx = sx - sx.mean(axis=0, keepdims=True)
    sy = sy - sy.mean(axis=0, keepdims=True)
    sx_std = np.clip(sx.std(axis=0, keepdims=True), 1e-8, None)
    sy_std = np.clip(sy.std(axis=0, keepdims=True), 1e-8, None)

    sxz = sx / sx_std
    syz = sy / sy_std
    return (sxz.T @ syz) / float(sxz.shape[0])
