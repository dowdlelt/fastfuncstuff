"""GPU-accelerated FDR (Benjamini-Hochberg) utilities and AFNI FDRCURVE writer.

Three layers:

- `stat_to_pvalue` — convert a statistic map to a two-sided p-value, dispatching
  on AFNI sub-brick stat codes (`fitt`, `fift`, `fizt`, ...).
- `fdr_qvalues` — per-voxel BH q-values with optional Storey pi0 correction.
  Mask-aware (zero/NaN voxels excluded), returns NaN outside the mask.
- `compute_fdr_curve` / `add_fdrcurves_to_nifti` — build AFNI-format 101-point
  z(q) lookup tables and inject them into the NIfTI AFNI extension as
  `FDRCURVE_NNNNNN` attributes. Lets AFNI's GUI / `fdrval` read q from a
  threshold without re-running BH.

References
----------
- AFNI `mri_fdrize.c` / `thd_dsetatr.c` (FDRCURVE storage layout)
- Benjamini & Hochberg 1995; Storey 2002 (pi0 estimator)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Two-sided z for q: z = qginv(q/2) where qginv = inverse upper-tail Gaussian.
# Implemented via scipy.stats.norm.isf for the curve build (CPU, 101 points).
try:
    from scipy.special import ndtri
    from scipy.stats import f as _scipy_f
    from scipy.stats import t as _scipy_t
except ImportError as _err:  # pragma: no cover
    raise ImportError("fastfuncstuff.stats.fdr requires scipy") from _err


# ---------------------------------------------------------------------------
# Stat → p-value
# ---------------------------------------------------------------------------

def _torch_t_sf(stat: torch.Tensor, dof: float) -> torch.Tensor:
    """Two-sided survival function for Student-t. CPU fallback to scipy.

    GPU path uses a regularized incomplete-beta proxy that's accurate enough
    for FDR thresholds (~1e-15 floor anyway).
    """
    # Survival of |t| under H0: 2 * P(T > |t|).  Use scipy for correctness.
    if stat.is_cuda:
        # Move to CPU; FDR is run once per stat sub-brick — overhead tiny.
        s_cpu = stat.detach().cpu().numpy()
    else:
        s_cpu = stat.detach().numpy()
    p = 2.0 * _scipy_t.sf(np.abs(s_cpu), df=dof)
    return torch.from_numpy(np.ascontiguousarray(p)).to(stat.device).to(torch.float32)


def _torch_f_sf(stat: torch.Tensor, df_num: float, df_den: float) -> torch.Tensor:
    s_cpu = stat.detach().cpu().numpy() if stat.is_cuda else stat.detach().numpy()
    # F is one-sided (always positive).
    p = _scipy_f.sf(np.maximum(s_cpu, 0.0), dfn=df_num, dfd=df_den)
    return torch.from_numpy(np.ascontiguousarray(p)).to(stat.device).to(torch.float32)


def _torch_z_sf(stat: torch.Tensor) -> torch.Tensor:
    """Two-sided survival for standard normal — pure GPU via erf."""
    return torch.erfc(stat.abs() / float(np.sqrt(2.0)))


def stat_to_pvalue(
    stats: torch.Tensor,
    stat_code: str,
    dof: float | tuple[float, float] | None = None,
) -> torch.Tensor:
    """Two-sided p-value for AFNI stat codes.

    Parameters
    ----------
    stats : Tensor
        Statistic map (any shape).
    stat_code : str
        AFNI code: 'fitt' (Student-t), 'fift' (F-test), 'fizt' (z-score).
    dof : float or (df_num, df_den)
        Degrees of freedom. For 'fift' provide a 2-tuple.
    """
    code = stat_code.lower()
    if code == "fitt":
        if dof is None:
            raise ValueError("stat_code='fitt' requires scalar dof")
        return _torch_t_sf(stats, float(dof if not isinstance(dof, tuple) else dof[0]))
    if code == "fift":
        if not isinstance(dof, tuple) or len(dof) != 2:
            raise ValueError("stat_code='fift' requires dof=(df_num, df_den)")
        return _torch_f_sf(stats, float(dof[0]), float(dof[1]))
    if code == "fizt":
        return _torch_z_sf(stats)
    raise ValueError(f"Unsupported stat_code: {stat_code!r}")


# ---------------------------------------------------------------------------
# BH q-values
# ---------------------------------------------------------------------------

def _storey_pi0(p_sorted: torch.Tensor, lambda_: float = 0.5) -> float:
    """Storey 2002 pi0 estimator at fixed lambda.

    pi0 = #{p > lambda} / ((1 - lambda) * N).  Capped to [0, 1].
    """
    n = p_sorted.numel()
    if n == 0:
        return 1.0
    above = float((p_sorted > lambda_).float().sum().item())
    return float(min(1.0, max(0.0, above / max(1.0 - lambda_, 1e-12) / n)))


@torch.inference_mode()
def fdr_qvalues(
    stats: torch.Tensor,
    *,
    stat_code: str | None = None,
    dof: float | tuple[float, float] | None = None,
    pvalues: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pi0: bool = True,
) -> torch.Tensor:
    """Per-voxel BH FDR q-values (optionally Storey-corrected).

    Either *stat_code+dof* or *pvalues* must be provided. Voxels outside
    *mask* (or zero/NaN if no mask) are returned as NaN.

    Parameters
    ----------
    stats : Tensor
        Statistic map. Required for shape; ignored if *pvalues* provided.
    stat_code, dof : see `stat_to_pvalue`.
    pvalues : Tensor, optional
        Pre-computed two-sided p-values (same shape as *stats*).
    mask : bool Tensor, optional
        Voxels to include. Default: stats != 0 and finite.
    pi0 : bool
        If True, scale q by Storey's pi0 estimate (matches AFNI's m1 spirit).

    Returns
    -------
    q : Tensor (same shape, float32). NaN outside mask.
    """
    if pvalues is None:
        if stat_code is None:
            raise ValueError("Provide either pvalues or stat_code+dof")
        pvalues = stat_to_pvalue(stats, stat_code, dof)
    p = pvalues.float()
    if mask is None:
        mask = torch.isfinite(stats) & (stats != 0)
    mask = mask & torch.isfinite(p)

    flat_p = p.flatten()
    flat_mask = mask.flatten()

    # AFNI's mri_fdrize: floor p>=1e-15, drop p>=0.9999.
    p_in = flat_p[flat_mask].clamp(min=1e-15)
    valid_in_mask = p_in < 0.9999
    p_use = p_in[valid_in_mask]
    n_use = p_use.numel()
    q_out = torch.full_like(flat_p, float("nan"))
    if n_use == 0:
        return q_out.view_as(p)

    # Sort ascending, compute q_i = N * p_i / rank
    sort_p, sort_idx = torch.sort(p_use)
    ranks = torch.arange(1, n_use + 1, device=p_use.device, dtype=p_use.dtype)
    q_sorted = (sort_p * n_use) / ranks
    if pi0:
        q_sorted = q_sorted * _storey_pi0(sort_p)
    # Monotonicity: cummin from the right (largest p downward).
    q_sorted = torch.flip(q_sorted, dims=(0,))
    q_sorted = torch.cummin(q_sorted, dim=0).values
    q_sorted = torch.flip(q_sorted, dims=(0,)).clamp(max=1.0)

    # Unsort back to p_use order, then scatter into the original mask order.
    q_unsorted = torch.empty_like(q_sorted)
    q_unsorted.scatter_(0, sort_idx, q_sorted)

    # Reassemble: q for in-mask voxels with p<0.9999, else NaN (kept as NaN).
    q_in = torch.full_like(p_in, float("nan"))
    q_in[valid_in_mask] = q_unsorted

    q_out[flat_mask] = q_in
    return q_out.view_as(p)


# ---------------------------------------------------------------------------
# AFNI-format FDR curve
# ---------------------------------------------------------------------------

def _q_to_z(q: np.ndarray) -> np.ndarray:
    """AFNI z(q) = qginv(q/2): inverse upper-tail Gaussian of q/2.

    For q very small this gives a large positive z; q=1 → z=0.
    """
    q = np.clip(q, 1e-15, 1.0)
    # ndtri returns the inverse CDF of N(0,1); we need upper-tail inverse.
    # qginv(p) = -ndtri(p)  ⇒  z(q) = -ndtri(q/2)
    return -ndtri(q / 2.0)


def compute_fdr_curve(
    stats: np.ndarray | torch.Tensor,
    stat_code: str,
    dof: float | tuple[float, float],
    mask: np.ndarray | torch.Tensor | None = None,
    n_curve: int = 101,
    pi0: bool = True,
) -> dict:
    """Build a 101-point AFNI FDRCURVE for one stat sub-brick.

    Returns
    -------
    dict with keys:
        x0 : float — minimum statistic in the curve grid
        dx : float — grid spacing
        z  : ndarray (n_curve,) float32 — z(q) at x = x0 + i*dx
    """
    s = stats if isinstance(stats, torch.Tensor) else torch.from_numpy(np.asarray(stats))
    s = s.float()
    if mask is not None:
        m = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(np.asarray(mask))
        m = m.bool()
    else:
        m = torch.isfinite(s) & (s != 0)

    q = fdr_qvalues(s, stat_code=stat_code, dof=dof, mask=m, pi0=pi0)

    s_in = s[m].cpu().numpy().astype(np.float64)
    q_in = q[m].cpu().numpy().astype(np.float64)
    finite = np.isfinite(s_in) & np.isfinite(q_in)
    s_in = s_in[finite]
    q_in = q_in[finite]
    if s_in.size == 0:
        return {"x0": 0.0, "dx": 1.0, "z": np.zeros(n_curve, dtype=np.float32)}

    # Build z(q) for in-mask voxels, sort by stat, interpolate onto grid.
    z_vals = _q_to_z(q_in)
    s_sorted = np.sort(s_in)
    # Make z monotone non-decreasing in |stat|: AFNI sorts by z ascending and
    # uses the upper envelope. For a faithful curve we take cummax of z over
    # increasing |stat|, matching AFNI's "extreme value" reading.
    abs_order = np.argsort(np.abs(s_in), kind="stable")
    s_abs = np.abs(s_in)[abs_order]
    z_abs = np.maximum.accumulate(z_vals[abs_order])
    s_grid_min = float(s_sorted[0])
    s_grid_max = float(s_sorted[-1])
    if s_grid_max <= s_grid_min:
        s_grid_max = s_grid_min + 1.0
    dx = (s_grid_max - s_grid_min) / (n_curve - 1)
    grid = s_grid_min + np.arange(n_curve, dtype=np.float64) * dx

    # Interpolate using |stat| → z mapping (AFNI builds curves in this space).
    z_curve = np.interp(np.abs(grid), s_abs, z_abs, left=0.0, right=z_abs[-1])
    return {
        "x0": float(s_grid_min),
        "dx": float(dx),
        "z": z_curve.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# AFNI XML extension writer
# ---------------------------------------------------------------------------

def add_fdrcurves_to_nifti(
    nifti_path: str | Path,
    curves_by_brick: dict[int, dict],
) -> None:
    """Inject FDRCURVE_NNNNNN floatvec attributes into a NIfTI's AFNI extension.

    Parameters
    ----------
    nifti_path : path
        Existing NIfTI file with an AFNI extension (ecode=4).
    curves_by_brick : dict
        Mapping {sub_brick_index: {'x0': float, 'dx': float, 'z': ndarray[n]}}.
        AFNI uses n=101 points by convention.
    """
    import re

    import nibabel as nib

    path = Path(nifti_path)
    img = nib.load(str(path))
    header = img.header
    extensions = getattr(header, "extensions", None)
    if not extensions:
        raise ValueError(f"{path}: no NIfTI extensions present")

    AFNI_ECODE = 4
    afni_idx = next(
        (i for i, ext in enumerate(extensions) if ext.get_code() == AFNI_ECODE), None
    )
    if afni_idx is None:
        raise ValueError(f"{path}: no AFNI extension (ecode=4)")
    afni_xml = extensions[afni_idx].content.decode("utf-8", errors="replace")

    for brick_idx, curve in sorted(curves_by_brick.items()):
        z = np.asarray(curve["z"], dtype=np.float32)
        floatvec = np.concatenate([[float(curve["x0"]), float(curve["dx"])], z])
        ni_dimen = floatvec.size  # = len(z) + 2
        atr_name = f"FDRCURVE_{brick_idx:06d}"
        # Strip any existing attr with the same name.
        afni_xml = re.sub(
            rf'<AFNI_atr\s[^>]*atr_name="{atr_name}"[^>]*>.*?</AFNI_atr>\s*',
            "",
            afni_xml,
            flags=re.DOTALL,
        )
        body = " ".join(f"{v:.6g}" for v in floatvec)
        new_attr = (
            f'<AFNI_atr ni_type="float" ni_dimen="{ni_dimen}" '
            f'atr_name="{atr_name}">\n  {body}\n</AFNI_atr>\n'
        )
        # Insert before the closing </AFNI_attributes> tag if present, else append.
        if "</AFNI_attributes>" in afni_xml:
            afni_xml = afni_xml.replace("</AFNI_attributes>", new_attr + "</AFNI_attributes>")
        else:
            afni_xml = afni_xml + "\n" + new_attr

    new_ext = nib.nifti1.Nifti1Extension(AFNI_ECODE, afni_xml.encode("utf-8"))
    extensions[afni_idx] = new_ext
    nib.save(img, str(path))


__all__ = [
    "stat_to_pvalue",
    "fdr_qvalues",
    "compute_fdr_curve",
    "add_fdrcurves_to_nifti",
]


# TODO(follow-up): wire `-add_fdr` into ffs_reml and ffs_deconvolve CLIs.
# Plan: after the bucket NIfTI is written and -substatpar is applied, walk the
# stat sub-bricks (we already track (brick_idx, stat_code, dof) tuples in
# glm/outputs.py), call compute_fdr_curve per brick on a CPU/GPU stat tensor,
# then add_fdrcurves_to_nifti(bucket_path, curves). Match AFNI's exact
# `mri_fdrize` algorithm (m1 histogram correction) once we have a side-by-side
# regression check against `3drefit -addFDR` output on a known stat map.
