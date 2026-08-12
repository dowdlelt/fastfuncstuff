#!/usr/bin/env python
"""
ffs_util_concalc — recompute GLM contrasts on top of an existing bucket
without rerunning the GLM.

The math (REML / ARMA(1,1)):

    cᵀβ                                          # contrast estimate (per voxel)
    σ² · cᵀ(XᵀΣ⁻¹X)⁻¹c                          # contrast variance
    t = cᵀβ / √variance

`(XᵀΣ⁻¹X)⁻¹` depends only on the ARMA parameters (a, b), which live on a
small grid (~117 valid pairs in AFNI's default). We therefore compute
``M⁻¹(a, b) = (XᵀR(a,b)⁻¹X)⁻¹`` **once per unique grid bin** and look up
per voxel. This makes concalc essentially free even for huge brains.

Inputs:
  -stats FILE   The original Rbuck (β + stats sub-bricks).
  -rvar FILE    The corresponding *_ffsremlvar* with (a, b, σ) per voxel.
  -spec FILE    The design.spec — its [[contrasts]] block is the source of
                truth. Every contrast in the spec is recomputed; this lets
                you fix a bad contrast and add new ones in one shot.

Outputs:
  default     <stats stem>_concalc.<ext>
  -inplace    overwrite -stats (existing contrast sub-bricks are stripped
              and replaced; β / σ² / F sub-bricks are preserved).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli.design_spec import _do_compile as _design_spec_compile
from fastfuncstuff.cli.design_spec import _resolve_spec_path
from fastfuncstuff.cli_utils import setup_device, spinner
from fastfuncstuff.design.spec import load_spec, resolve_contrast
from fastfuncstuff.glm.arma import build_arma11_covariance
from fastfuncstuff.io.afni import load_nifti, read_afni_design_matrix

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_util_concalc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-stats", required=True, metavar="FILE", help="Original Rbuck output (NIfTI). Read for β."
    )
    p.add_argument(
        "-rvar",
        required=False,
        metavar="FILE",
        help="Companion *_ffsremlvar file from ffs_reml. "
        "Sub-bricks 0/1/3 are a, b, StDev. Required for REML "
        "buckets; pass -ols for OLS buckets instead.",
    )
    p.add_argument(
        "-ols",
        action="store_true",
        help="Treat -stats as an OLS bucket (no Rvar): "
        "one global (XᵀX)⁻¹, σ² derived per voxel from any "
        "existing β/t pair. Mutually exclusive with -rvar.",
    )
    p.add_argument(
        "-fout",
        action="store_true",
        help="Also emit a <contrast>_Fstat sub-brick for every "
        "t-test contrast (F = t², dof=(1, residual)). F-test "
        "contrasts always emit Fstat regardless.",
    )
    p.add_argument(
        "-spec",
        required=True,
        metavar="FILE",
        help="design.spec TOML. Every [[contrasts]] entry is "
        "recomputed. Extension auto-appended (.toml).",
    )
    p.add_argument(
        "-out",
        metavar="FILE",
        default=None,
        help="Output path. Default: <stats stem>_concalc<ext>.",
    )
    p.add_argument(
        "-inplace",
        action="store_true",
        help="Rewrite -stats in place, replacing all contrast "
        "sub-bricks. β / σ² / F sub-bricks are kept.",
    )
    p.add_argument(
        "-mask",
        metavar="FILE",
        help="Optional brain mask. If absent, voxels with "
        "non-finite or all-zero (a, b) are skipped.",
    )
    p.add_argument(
        "-matrix",
        metavar="FILE",
        help="Use an existing xmat instead of compiling -spec. "
        "Must match the original design column order.",
    )
    p.add_argument(
        "-overwrite", action="store_true", help="Allow overwriting an existing output file."
    )
    p.add_argument(
        "-force",
        action="store_true",
        help="Proceed even when sub-bricks would be *dropped* (not just "
        "recomputed). Without this, concalc refuses to write if any "
        "existing sub-brick would be lost — usually a sign the spec's stim "
        "labels don't match the bucket. Recomputing a contrast that already "
        "exists never needs -force.",
    )
    p.add_argument("-device", default=None, help="torch device override (cpu / cuda / mps).")
    p.add_argument(
        "-verb", type=int, default=1, help="0 = quiet, 1 = summary, 2 = per-bin progress."
    )
    return p


# ---------------------------------------------------------------------------
# NIfTI helpers (read β by AFNI sub-brick labels; preserve geometry on write)
# ---------------------------------------------------------------------------


_XML_LABS_RE = re.compile(
    r'atr_name\s*=\s*"BRICK_LABS"[^>]*>\s*"([^"]+)"',
    re.S,
)
_XML_STATAUX_RE = re.compile(
    r'<AFNI_atr[^>]*atr_name\s*=\s*"BRICK_STATAUX"[^>]*>\s*([0-9eE.+\-\s]+?)\s*</AFNI_atr>',
    re.S,
)


# AFNI numeric stat codes (a subset — what fitt/fift map to in BRICK_STATAUX).
_STAT_CODE_TTEST = 3  # `fitt`
_STAT_CODE_FTEST = 4  # `fift`


def _parse_stataux(xml_text: str) -> dict[int, tuple[int, tuple[float, ...]]]:
    """Parse an AFNI ``BRICK_STATAUX`` block into ``{brick_idx: (code, params)}``.

    Returns an empty dict when no STATAUX attribute is present (i.e. the
    bucket carries no per-sub-brick stat metadata).
    """
    m = _XML_STATAUX_RE.search(xml_text)
    if not m:
        return {}
    floats = [float(x) for x in m.group(1).split()]
    out: dict[int, tuple[int, tuple[float, ...]]] = {}
    i = 0
    while i < len(floats):
        if i + 3 > len(floats):
            break
        idx = int(floats[i])
        code = int(floats[i + 1])
        n_par = int(floats[i + 2])
        params = tuple(floats[i + 3 : i + 3 + n_par])
        out[idx] = (code, params)
        i += 3 + n_par
    return out


def _statsym_for(code: int, params: tuple[float, ...]) -> str:
    """Symbolic AFNI label for one sub-brick's stat (``Ttest(1052)`` /
    ``Ftest(4,1052)``). Returns ``"none"`` for non-stat sub-bricks."""
    if code == _STAT_CODE_TTEST and len(params) == 1:
        return f"Ttest({int(params[0])})"
    if code == _STAT_CODE_FTEST and len(params) == 2:
        return f"Ftest({int(params[0])},{int(params[1])})"
    return "none"


def _format_stataux_block(
    stataux: dict[int, tuple[int, tuple[float, ...]]],
    n_sub: int,
) -> tuple[str, str]:
    """Build ``BRICK_STATAUX`` (float list) and ``BRICK_STATSYM`` (string)
    attribute payloads from a per-sub-brick stat-info dict."""
    # STATAUX: flat list of (idx, code, n_par, *params) tuples.
    stataux_floats: list[float] = []
    for idx in sorted(stataux):
        code, params = stataux[idx]
        stataux_floats.extend([float(idx), float(code), float(len(params))])
        stataux_floats.extend(float(p) for p in params)
    # STATSYM: one entry per sub-brick, "none" for non-stat sub-bricks.
    syms = [_statsym_for(*stataux[i]) if i in stataux else "none" for i in range(n_sub)]
    return (
        " " + "\n ".join(f"{v:g}" for v in stataux_floats),
        ";".join(syms),
    )


def _read_brick_labels(img) -> list[str]:
    """Pull sub-brick labels out of an AFNI NIfTI extension. Returns ``[]``
    when the input wasn't written by an AFNI-aware tool.

    Two payload formats appear in the wild:

    1. Legacy plain-text: ``BRICK_LABS=label1~label2~…\\x00``
    2. AFNI XML (current ``3dcopy`` / ``3drefit`` output):
       ``<AFNI_atr ... atr_name="BRICK_LABS" ...> "label1~label2~…" </AFNI_atr>``

    We check both, in that order.
    """
    for ext in img.header.extensions or []:
        if ext.get_code() != 4:
            continue
        content = ext.get_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        # Legacy.
        if "BRICK_LABS=" in content and "atr_name" not in content[:50]:
            label_str = content.split("BRICK_LABS=")[1].split("\x00")[0]
            return label_str.split("~")
        # XML.
        m = _XML_LABS_RE.search(content)
        if m:
            return [s.strip() for s in m.group(1).split("~") if s.strip()]
    return []


def _afni_bucket_extension(
    labels: list[str],
    stataux: dict[int, tuple[int, tuple[float, ...]]] | None = None,
    idcode: str | None = None,
):
    """Like :func:`_afni_brick_labels_extension` but also carries per-sub-brick
    statistical metadata so AFNI viewers can compute p-values live.

    ``stataux`` is ``{brick_idx: (stat_code, params_tuple)}``. Sub-bricks that
    aren't statistics (β coefficients, masks, etc.) are simply absent from
    the dict and end up as ``"none"`` in BRICK_STATSYM.
    """
    import nibabel as nib

    from fastfuncstuff.io.afni import _generate_afni_idcode

    labs = "~".join(labels)
    idc = idcode or _generate_afni_idcode()

    parts = [
        "<?xml version='1.0' ?>\n"
        "<AFNI_attributes\n"
        f'  self_idcode="{idc}"\n'
        '  ni_form="ni_group" >\n',
        "<AFNI_atr\n"
        '  ni_type="String"\n'
        '  ni_dimen="1"\n'
        '  atr_name="BRICK_LABS" >\n'
        f' "{labs}"\n'
        "</AFNI_atr>\n",
    ]

    if stataux:
        floats_str, sym_str = _format_stataux_block(stataux, len(labels))
        n_floats = sum(3 + len(p) for _, p in stataux.values())
        parts.append(
            "<AFNI_atr\n"
            '  ni_type="float"\n'
            f'  ni_dimen="{n_floats}"\n'
            '  atr_name="BRICK_STATAUX" >\n'
            f"{floats_str}\n"
            "</AFNI_atr>\n"
        )
        parts.append(
            "<AFNI_atr\n"
            '  ni_type="String"\n'
            '  ni_dimen="1"\n'
            '  atr_name="BRICK_STATSYM" >\n'
            f' "{sym_str}"\n'
            "</AFNI_atr>\n"
        )

    parts.append("</AFNI_attributes>\n\x00")
    return nib.nifti1.Nifti1Extension(4, "".join(parts).encode("utf-8"))


def _afni_brick_labels_extension(labels: list[str], idcode: str | None = None):
    """Build an AFNI-style NIfTI extension (code=4) carrying BRICK_LABS in
    AFNI's XML form so ``3dinfo`` / ``3drefit`` see the labels round-tripped.

    The legacy ``BRICK_LABS=...~...`` plain-text form is still accepted by
    AFNI, but newer tooling sometimes ignores it. Emitting the XML form
    matches what AFNI itself writes.

    ``self_idcode`` follows AFNI's convention (``AFN_`` + 22 base64-ish chars).
    A unique value is auto-generated when *idcode* is not provided — AFNI
    refuses to track datasets that share an identifier, so this matters as
    soon as the user keeps multiple concalc outputs side-by-side.
    """
    import nibabel as nib

    from fastfuncstuff.io.afni import _generate_afni_idcode

    labs = "~".join(labels)
    idc = idcode or _generate_afni_idcode()
    payload = (
        "<?xml version='1.0' ?>\n"
        "<AFNI_attributes\n"
        f'  self_idcode="{idc}"\n'
        '  ni_form="ni_group" >\n'
        "<AFNI_atr\n"
        '  ni_type="String"\n'
        '  ni_dimen="1"\n'
        '  atr_name="BRICK_LABS" >\n'
        f' "{labs}"\n'
        "</AFNI_atr>\n"
        "</AFNI_attributes>\n"
        "\x00"
    )
    return nib.nifti1.Nifti1Extension(4, payload.encode("utf-8"))


# ---------------------------------------------------------------------------
# Per-bin (X̃ᵀX̃)⁻¹ computation
# ---------------------------------------------------------------------------


def _bin_index(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map per-voxel (a, b) onto compact bin indices.

    Returns
    -------
    bin_idx : np.ndarray, shape (n_voxels,)
        Index into the unique-(a, b) table for each voxel.
    unique_ab : np.ndarray, shape (n_bins, 2)
        The unique (a, b) pairs in the same order bin_idx points at.
    valid : np.ndarray, shape (n_voxels,) bool
        Voxels with finite, in-range parameters.
    """
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 1.0) & (np.abs(b) < 1.0)
    # Round to 3 dp so floating-point jitter doesn't fragment bins.
    ar = np.round(a, 3)
    br = np.round(b, 3)
    unique_ab, inverse = np.unique(
        np.stack([ar, br], axis=1)[valid],
        axis=0,
        return_inverse=True,
    )
    bin_idx = np.full(a.shape, -1, dtype=np.int64)
    bin_idx[valid] = inverse
    return bin_idx, unique_ab, valid


def _xtxinv_per_bin(
    X: torch.Tensor,
    unique_ab: np.ndarray,
    run_starts: list[int] | None,
    device: torch.device,
    verbose: int = 1,
) -> tuple[torch.Tensor, np.ndarray]:
    """For each unique (a, b) pair, compute ``M⁻¹ = (Xᵀ R(a,b)⁻¹ X)⁻¹``.

    Returns
    -------
    M_inv : torch.Tensor, shape (n_bins, n_reg, n_reg)
        Per-bin whitened inverse. Float32 on *device*.
    keep : np.ndarray, shape (n_bins,) bool
        Whether each bin produced a valid covariance (some (a, b) pairs are
        rejected by build_arma11_covariance — e.g. λ < 0).
    """
    n_tp, n_reg = X.shape
    n_bins = unique_ab.shape[0]
    M_inv = torch.zeros((n_bins, n_reg, n_reg), device=device, dtype=torch.float32)
    keep = np.zeros(n_bins, dtype=bool)
    for i, (a, b) in enumerate(unique_ab):
        R = build_arma11_covariance(
            float(a),
            float(b),
            n_tp,
            device=device,
            dtype=torch.float32,
            run_starts=run_starts,
        )
        if R is None:
            continue
        try:
            L = torch.linalg.cholesky(R)
        except RuntimeError:
            continue
        # Whitened design: solve L · X̃ = X (i.e. X̃ = L⁻¹ X). M = X̃ᵀ X̃.
        X_white = torch.linalg.solve_triangular(L, X, upper=False)
        M = X_white.T @ X_white
        try:
            M_inv[i] = torch.linalg.inv(M)
        except RuntimeError:
            continue
        keep[i] = True
        if verbose >= 2 and (i % 20 == 0 or i == n_bins - 1):
            print(f"   bin {i + 1}/{n_bins}  (a={a:+.3f}, b={b:+.3f})", flush=True)
    return M_inv, keep


# ---------------------------------------------------------------------------
# Stats bucket parsing
# ---------------------------------------------------------------------------


_COEF_RE = re.compile(r"^(.+?)#(\d+)_Coef$")
_TSTAT_RE = re.compile(r"^(.+?)#(\d+)_Tstat$")
# Per-stim z-stat, e.g. written by ffs_util_updatedof -numcomps (which turns
# each t into a DoF-adjusted z, giving Coef/Tstat/Zstat triples per stim).
_ZSTAT_RE = re.compile(r"^(.+?)#(\d+)_Zstat$")
# Matches a contrast-style label (no '#N' segment) — i.e. a sub-brick that came
# from a previous GLT computation rather than a per-stim coefficient.
_CONTRAST_COEF_RE = re.compile(r"^(?P<name>.+?)_Coef$")
_CONTRAST_TSTAT_RE = re.compile(r"^(?P<name>.+?)_Tstat$")
_CONTRAST_FSTAT_RE = re.compile(r"^(?P<name>.+?)_Fstat$")
_CONTRAST_ZSTAT_RE = re.compile(r"^(?P<name>.+?)_Zstat$")


def _contrast_base_name(label: str) -> str | None:
    """Strip a contrast-style sub-brick label to its bare contrast name.

    ``face_vs_place#0_Coef`` / ``face_vs_place_Tstat`` → ``face_vs_place``.
    Returns ``None`` for labels that don't carry a Coef/Tstat/Fstat/Zstat/R²
    suffix (i.e. not a contrast output). ``_Zstat`` appears when a DoF tool
    (ffs_util_updatedof -numcomps) has post-processed the bucket. The trailing
    ``#N`` AFNI sub-index — present on ffs_reml-written contrasts but not
    concalc-written ones — is stripped.
    """
    for suf in ("_Coef", "_Tstat", "_Fstat", "_Zstat", "_R2semi", "_R2"):
        if label.endswith(suf):
            return re.sub(r"#\d+$", "", label[: -len(suf)])
    return None


def _unsafe_drops(
    stats_labels: list[str],
    keep_idx: list[int],
    new_contrast_labels: set[str],
) -> list[str]:
    """Labels that would be dropped *without* being recomputed under the same
    name. Dropping a contrast we're about to rebuild is expected (fixing a bad
    contrast); dropping anything else loses data — usually the tripwire for a
    spec whose stim labels don't match the bucket (singular ``face`` vs the
    bucket's ``faces#0_Coef``), which reclassifies every β as a stale contrast.
    """
    kept = set(keep_idx)
    return [
        lbl
        for i, lbl in enumerate(stats_labels)
        if i not in kept and _contrast_base_name(lbl) not in new_contrast_labels
    ]


def _select_non_contrast_subbricks(
    stats_labels: list[str],
    stim_base_labels: list[str],
) -> list[int]:
    """Return the indices of sub-bricks to *keep* when rewriting the bucket.

    A label is treated as an existing contrast (and dropped) when it ends
    in ``_Coef`` / ``_Tstat`` / ``_Fstat`` / ``_Zstat`` AND its name part has
    no ``#N`` segment AND it's not the bucket-wide ``Full_Fstat``. Everything
    else (β, per-stim t/z, Full_Fstat, partial R², drift, anything we don't
    know about) is preserved verbatim.
    """
    keep: list[int] = []
    stim_set = set(stim_base_labels)
    for i, lbl in enumerate(stats_labels):
        if lbl == "Full_Fstat":
            keep.append(i)
            continue
        # Per-stim β / t / z — has '#N' segment, name part is in stim list.
        # (The z brick appears after ffs_util_updatedof -numcomps.)
        m_coef = _COEF_RE.match(lbl)
        m_tstat = _TSTAT_RE.match(lbl)
        m_zstat = _ZSTAT_RE.match(lbl)
        if m_coef and m_coef.group(1) in stim_set:
            keep.append(i)
            continue
        if m_tstat and m_tstat.group(1) in stim_set:
            keep.append(i)
            continue
        if m_zstat and m_zstat.group(1) in stim_set:
            keep.append(i)
            continue
        # No '#N' suffix → likely a previous-contrast Coef/Tstat/Fstat/Zstat. Drop.
        if (
            _CONTRAST_COEF_RE.match(lbl)
            or _CONTRAST_TSTAT_RE.match(lbl)
            or _CONTRAST_FSTAT_RE.match(lbl)
            or _CONTRAST_ZSTAT_RE.match(lbl)
        ):
            continue
        # Unknown shape — keep, on the principle "leave user data alone".
        keep.append(i)
    return keep


def _identify_subbricks(labels: list[str]) -> dict[str, list[int]]:
    """Split sub-brick labels into categories so we know which to keep when
    rewriting the bucket. Returns indices grouped as:

    - ``coef``      β sub-bricks for stim regressors (kept; we need β values)
    - ``tstat``     per-stim t-stats (kept untouched — they were ANY-vs-zero)
    - ``contrast``  existing contrast outputs (Coef + Tstat; *dropped* on rewrite)
    - ``other``     overall Full_Fstat, partial R² etc. (kept verbatim)
    """
    groups: dict[str, list[int]] = {
        "coef": [],
        "tstat": [],
        "contrast": [],
        "other": [],
    }
    # Contrasts are anything whose Coef/Tstat label *doesn't* match a #0
    # suffix on a stim — but in practice everything in the bucket today is
    # either Full_Fstat, `<stim>#N_Coef/Tstat`, or `<contrast>_Coef/Tstat`.
    # We discriminate via the explicit contrast-name list at runtime.
    for i, lbl in enumerate(labels):
        if lbl == "Full_Fstat" or lbl.endswith("_Rsq") or lbl.endswith("_Fstat"):
            groups["other"].append(i)
        elif _COEF_RE.match(lbl):
            groups["coef"].append(i)
        elif _TSTAT_RE.match(lbl):
            groups["tstat"].append(i)
        else:
            groups["other"].append(i)
    return groups


def _beta_per_voxel_by_label(
    bucket_data: np.ndarray,
    labels: list[str],
    column_labels: list[str],
    mask_bool: np.ndarray,
) -> np.ndarray:
    """Build a (n_voxels, n_regressors) β matrix by matching each design
    column label against the bucket's ``<label>#0_Coef`` sub-brick.

    Polynomial / nuisance columns that the bucket doesn't expose get
    ``NaN`` — those columns must be 0 in any sensible contrast (since the
    spec resolver only weights stim labels), so the contrast math sees a
    zero contribution from them regardless.
    """
    n_vox = int(mask_bool.sum())
    n_reg = len(column_labels)
    beta = np.zeros((n_vox, n_reg), dtype=np.float32)

    # Build a label → bucket sub-brick index lookup. The bucket uses
    # `<name>#N_Coef`, design columns use `<name>#N`. We match on the bare
    # name part stripped of #N for stim, and full prefix for IM events.
    coef_idx_by_name: dict[str, int] = {}
    for i, lbl in enumerate(labels):
        m = _COEF_RE.match(lbl)
        if m:
            base = f"{m.group(1)}#{m.group(2)}"
            coef_idx_by_name[base] = i

    for j, col_label in enumerate(column_labels):
        bidx = coef_idx_by_name.get(col_label)
        if bidx is None:
            continue
        vol = bucket_data[..., bidx]
        beta[:, j] = vol[mask_bool].astype(np.float32, copy=False)
    return beta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _derive_ols_sigma2(
    stats_data: np.ndarray,
    stats_labels: list[str],
    column_labels: list[str],
    XtX_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-voxel σ² for an OLS bucket from existing β / t pairs.

    For OLS: t = β / SE and SE² = σ² · [(XᵀX)⁻¹]ᵢᵢ for stim column i.
    Therefore σ² = (β/t)² / [(XᵀX)⁻¹]ᵢᵢ. We average across every stim that
    has both a ``<stim>#N_Coef`` and a ``<stim>#N_Tstat`` sub-brick — this
    is robust against any one stim having a near-zero t.

    Returns
    -------
    sigma2 : np.ndarray, shape (X, Y, Z)
        Per-voxel σ². ``np.nan`` where it couldn't be derived.
    mask : np.ndarray, shape (X, Y, Z) bool
        Voxels where at least one stim yielded a finite, positive σ².
    """
    # Build lookups for β and t by stim column label.
    coef_idx_by_col: dict[str, int] = {}
    tstat_idx_by_col: dict[str, int] = {}
    for i, lbl in enumerate(stats_labels):
        m_c = _COEF_RE.match(lbl)
        m_t = _TSTAT_RE.match(lbl)
        if m_c:
            coef_idx_by_col[f"{m_c.group(1)}#{m_c.group(2)}"] = i
        if m_t:
            tstat_idx_by_col[f"{m_t.group(1)}#{m_t.group(2)}"] = i

    vol_shape = stats_data.shape[:3]
    sigma2_sum = np.zeros(vol_shape, dtype=np.float64)
    sigma2_count = np.zeros(vol_shape, dtype=np.int32)

    for j, col_label in enumerate(column_labels):
        ci = coef_idx_by_col.get(col_label)
        ti = tstat_idx_by_col.get(col_label)
        if ci is None or ti is None:
            continue
        diag = float(XtX_inv[j, j])
        if not np.isfinite(diag) or diag <= 0:
            continue
        beta = stats_data[..., ci].astype(np.float64)
        tstat = stats_data[..., ti].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma2_j = (beta * beta) / (tstat * tstat * diag)
        good = np.isfinite(sigma2_j) & (sigma2_j > 0)
        sigma2_sum[good] += sigma2_j[good]
        sigma2_count[good] += 1

    sigma2 = np.where(
        sigma2_count > 0,
        sigma2_sum / np.maximum(sigma2_count, 1),
        np.nan,
    )
    mask = sigma2_count > 0
    return sigma2.astype(np.float32), mask


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.inplace:
        return Path(args.stats)
    if args.out:
        return Path(args.out)
    src = Path(args.stats)
    name = src.name
    for suffix in (".nii.gz", ".nii.zst", ".nii"):
        if name.endswith(suffix):
            return src.with_name(name[: -len(suffix)] + "_concalc" + suffix)
    return src.with_name(src.stem + "_concalc" + src.suffix)


def _save_bucket(
    out_path: Path,
    data_4d: np.ndarray,
    labels: list[str],
    reference_img,
    stataux: dict[int, tuple[int, tuple[float, ...]]] | None = None,
) -> None:
    """Write a 4D NIfTI with AFNI BRICK_LABS + BRICK_STATAUX + BRICK_STATSYM."""
    import nibabel as nib

    affine = reference_img.affine
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(data_4d.shape)
    new_img = nib.Nifti1Image(data_4d.astype(np.float32), affine, header)
    # Strip pre-existing AFNI extensions; we re-add ours.
    new_img.header.extensions[:] = [
        ext for ext in (new_img.header.extensions or []) if ext.get_code() != 4
    ]
    new_img.header.extensions.append(_afni_bucket_extension(labels, stataux))
    with spinner(f"Writing {out_path.name}"):
        nib.save(new_img, str(out_path))


def main() -> int:
    args = _build_parser().parse_args()

    # ── 0) Mutex: exactly one of -rvar / -ols ─────────────────────────────
    if args.ols and args.rvar:
        print("ERROR: -rvar and -ols are mutually exclusive.", file=sys.stderr)
        return 1
    if not args.ols and not args.rvar:
        print("ERROR: pass -rvar FILE (REML bucket) or -ols (OLS bucket).", file=sys.stderr)
        return 1

    # ── 1) Resolve and compile the spec to get X + column labels ──────────
    spec_path = _resolve_spec_path(args.spec)
    spec = load_spec(spec_path)
    if not spec.contrasts:
        print("ERROR: spec has no [[contrasts]] entries to compute.", file=sys.stderr)
        return 1

    if args.matrix:
        xmat_path = Path(args.matrix)
    else:
        # Compile spec → xmat in a sibling tempdir-ish location so we don't
        # mess up the user's main folder. Stem matches the spec name.
        xmat_path = spec_path.with_name(f"{spec_path.stem}_concalc.xmat.1D")
        if args.verb >= 1:
            print(f"📐 Compiling spec → {xmat_path}", flush=True)
        compile_ns = argparse.Namespace(
            spec=str(spec_path),
            xmat=str(xmat_path),
            verb=0,
            overwrite=True,
        )
        rc = _design_spec_compile(compile_ns)
        if rc != 0:
            return rc

    design_info = read_afni_design_matrix(str(xmat_path))
    X_np = design_info["matrix"].astype(np.float32)
    column_labels = design_info["column_labels"]
    run_starts = list(design_info.get("run_starts") or [0])
    n_tp, _n_reg = X_np.shape
    del n_tp  # consumed via X.shape downstream

    # ── 2) Resolve every contrast against the column labels ─────────────
    # We resolve against the *base* stim labels (without #0) so users can
    # write `+1*face -1*house` even though the column label is `face#0`.
    base_to_full: dict[str, str] = {}
    for lbl in column_labels:
        m = _COEF_RE.match(lbl + "_Coef") or _TSTAT_RE.match(lbl + "_Tstat")
        if m:
            base_to_full[m.group(1)] = lbl
        else:
            base = lbl.split("#")[0] if "#" in lbl else lbl
            base_to_full.setdefault(base, lbl)
    stim_base_labels = sorted({k for k in base_to_full.keys() if not k.startswith("Run#")})

    resolved_rows_per_contrast = []
    for c in spec.contrasts:
        rows = resolve_contrast(c, stim_base_labels)
        # Translate base labels back to the actual column names (face → face#0)
        translated = []
        for row in rows:
            translated.append({base_to_full.get(name, name): val for name, val in row.items()})
        resolved_rows_per_contrast.append((c.label, translated))

    # Resolve each contrast to an (n_rows, n_reg) matrix. n_rows == 1 → t-test,
    # n_rows > 1 → F-test (joint null on all rows).
    from fastfuncstuff.design.builder import glt_rows_to_matrix

    contrast_matrices: list[tuple[str, np.ndarray]] = []
    for label, rows in resolved_rows_per_contrast:
        mat = glt_rows_to_matrix(rows, column_labels, stim_ranges=None)
        contrast_matrices.append((label, mat))
    if not contrast_matrices:
        print("ERROR: no usable contrasts after resolution.", file=sys.stderr)
        return 1

    # ── 3) Output path / overwrite guard ────────────────────────────────
    out_path = _resolve_output_path(args)
    if out_path.exists() and not args.overwrite and not args.inplace:
        print(
            f"ERROR: output {out_path} already exists. "
            "Pass -overwrite to replace, or -inplace to modify the input.",
            file=sys.stderr,
        )
        return 1

    # ── 4) Load Rvar and stats bucket ───────────────────────────────────
    if args.verb >= 1:
        print(f"📥 Reading Rvar : {args.rvar}", flush=True)
    # REML path reads Rvar (a, b, σ); OLS path skips it (covariance is I,
    # σ² is derived per voxel from any existing β/t pair further below).
    if args.ols:
        if args.verb >= 1:
            print("📥 OLS mode (no Rvar)", flush=True)
        a_map = b_map = stdev_map = None
    else:
        if args.verb >= 1:
            print(f"📥 Reading Rvar : {args.rvar}", flush=True)
        with spinner(f"Loading {Path(args.rvar).name}"):
            rvar = load_nifti(args.rvar).get_fdata(dtype=np.float32)
        if rvar.shape[-1] < 4:
            print(
                f"ERROR: Rvar has {rvar.shape[-1]} sub-bricks; expected ≥4 (a, b, lambda, StDev).",
                file=sys.stderr,
            )
            return 1
        a_map = rvar[..., 0]
        b_map = rvar[..., 1]
        stdev_map = rvar[..., 3]

    if args.verb >= 1:
        print(f"📥 Reading stats: {args.stats}", flush=True)
    with spinner(f"Loading {Path(args.stats).name}"):
        stats_img = load_nifti(args.stats)
        stats_data = stats_img.get_fdata(dtype=np.float32)
    stats_labels = _read_brick_labels(stats_img)
    if not stats_labels:
        print(
            "ERROR: stats bucket has no AFNI sub-brick labels; cannot identify β columns.",
            file=sys.stderr,
        )
        return 1

    # Parse the input bucket's stat metadata so we can preserve per-sub-brick
    # dof info on the kept sub-bricks. Empty dict if absent — concalc still
    # writes correct STATAUX for the new contrast sub-bricks below.
    input_stataux: dict[int, tuple[int, tuple[float, ...]]] = {}
    for ext in stats_img.header.extensions or []:
        if ext.get_code() == 4:
            txt = ext.get_content()
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8", errors="ignore")
            input_stataux = _parse_stataux(txt)
            break

    if args.ols:
        vol_shape = stats_data.shape[:3]
    else:
        assert a_map is not None, "a_map is always set when not args.ols"
        vol_shape = a_map.shape
    if not args.ols and stats_data.shape[:3] != vol_shape:
        print(
            f"ERROR: stats and rvar volume shapes disagree "
            f"({stats_data.shape[:3]} vs {vol_shape}).",
            file=sys.stderr,
        )
        return 1

    # ── 5) Mask ─────────────────────────────────────────────────────────
    if args.mask:
        with spinner(f"Loading {Path(args.mask).name}"):
            mask = load_nifti(args.mask).get_fdata().astype(bool)
    elif args.ols:
        # OLS: any voxel with at least one finite β / t pair we can derive
        # σ² from. Refined further below once we pick a reference stim.
        mask = np.ones(vol_shape, dtype=bool)
    else:
        # REML: voxels with non-finite or trivially-zero (a, b) get skipped.
        mask = (
            np.isfinite(a_map)
            & np.isfinite(b_map)
            & np.isfinite(stdev_map)
            & (stdev_map > 0)
            & ((a_map != 0) | (b_map != 0))
        )

    device = setup_device(args.device)
    if args.verb >= 1:
        print(f"🖥️  Device: {device}")

    X = torch.from_numpy(X_np).to(device)

    # ── 6) Build (X̃ᵀX̃)⁻¹ per bin (REML) or a single global (XᵀX)⁻¹ (OLS)
    if args.ols:
        # OLS: covariance is I, so M = XᵀX and M_inv applies to every voxel.
        XtX_inv_global = torch.linalg.inv(X.T @ X)  # (n_reg, n_reg)
        M_inv = XtX_inv_global.unsqueeze(0)  # (1, n_reg, n_reg)
        bin_keep = np.array([True])

        # Derive σ² per voxel from any existing β/t pair. For OLS:
        #   t = β / SE,  SE² = σ² · [(XᵀX)⁻¹]ᵢᵢ
        # so σ² = (β/t)² / [(XᵀX)⁻¹]ᵢᵢ. Average across stims for robustness.
        XtX_inv_np = XtX_inv_global.cpu().numpy()
        sigma2_volume, deriv_mask = _derive_ols_sigma2(
            stats_data,
            stats_labels,
            column_labels,
            XtX_inv_np,
        )
        mask &= deriv_mask
        sigma2_vox = sigma2_volume[mask].astype(np.float32)
        n_vox = int(mask.sum())

        # Single bin → all voxels point at index 0.
        bin_idx_flat = np.zeros(n_vox, dtype=np.int64)
        n_bins = 1
        if args.verb >= 1:
            total = int(np.prod(vol_shape))
            print(
                f"   Valid voxels: {n_vox}/{total} "
                f"({100.0 * n_vox / total:.1f}%) — σ² derived from "
                f"existing β/t pairs"
            )
    else:
        n_vox = int(mask.sum())
        if args.verb >= 1:
            total = int(np.prod(vol_shape))
            print(f"   Valid voxels: {n_vox}/{total} ({100.0 * n_vox / total:.1f}%)")
        a_vox = a_map[mask]
        b_vox = b_map[mask]
        sigma2_vox = (stdev_map[mask].astype(np.float32)) ** 2
        bin_idx_flat, unique_ab, _ = _bin_index(a_vox, b_vox)
        n_bins = unique_ab.shape[0]
        if args.verb >= 1:
            print(f"🧮 Unique (a, b) bins: {n_bins}")
        M_inv, bin_keep = _xtxinv_per_bin(
            X,
            unique_ab,
            run_starts,
            device,
            verbose=args.verb,
        )

    # ── 7) Compute β per voxel by matching design columns to bucket β────
    beta = _beta_per_voxel_by_label(stats_data, stats_labels, column_labels, mask)
    beta_t = torch.from_numpy(beta).to(device)

    # ── 8) Per-contrast: t-test for 1-row, F-test for multi-row ──────────
    # t-test result: (label, "t", coef, tstat)
    # F-test result: (label, "F", fstat)        # AFNI matches: no _Coef sub-brick
    contrast_results: list[tuple] = []
    n_tp_design = X.shape[0]
    dof_residual = max(1, n_tp_design - X.shape[1])
    invalid_voxel = (bin_idx_flat < 0) | (~bin_keep[np.clip(bin_idx_flat, 0, n_bins - 1)])

    for label, c_mat in contrast_matrices:
        n_rows = c_mat.shape[0]
        C = torch.from_numpy(c_mat.astype(np.float32)).to(device)  # (r, n_reg)

        # Cβ for every row, every voxel: (n_vox, r)
        Cb = (beta_t @ C.T).cpu().numpy()

        if n_rows == 1:
            c = C[0]  # (n_reg,)
            cMc_bins = torch.einsum("r,brs,s->b", c, M_inv, c).cpu().numpy()
            cMc_vox = np.where(bin_idx_flat >= 0, cMc_bins[bin_idx_flat], np.nan)
            var = cMc_vox * sigma2_vox
            cb = Cb[:, 0]
            tstat = np.where(
                var > 0,
                cb / np.sqrt(np.clip(var, 1e-20, None)),
                0.0,
            )
            cb[invalid_voxel] = 0.0
            tstat[invalid_voxel] = 0.0
            contrast_results.append(
                (
                    label,
                    "t",
                    cb.astype(np.float32),
                    tstat.astype(np.float32),
                )
            )
            if args.verb >= 1:
                print(
                    f"   {label} (t): cᵀβ peak={np.nanmax(np.abs(cb)):.3f}, "
                    f"|t| peak={np.nanmax(np.abs(tstat)):.3f}"
                )
        else:
            # F-test: q = (Cβ)ᵀ (C M⁻¹ Cᵀ)⁻¹ (Cβ); F = q / (r · σ²).
            # CMC[bin] is (r, r); invert per-bin then apply per voxel.
            CMC_bins = torch.einsum(
                "ir,brs,js->bij",
                C,
                M_inv,
                C,
            )  # (n_bins, r, r)
            try:
                CMC_bins_inv = torch.linalg.inv(CMC_bins)
            except RuntimeError as exc:
                print(
                    f"WARN: contrast {label!r}: CMᵢⱼC inversion failed ({exc}). Skipping.",
                    file=sys.stderr,
                )
                continue
            CMC_inv_np = CMC_bins_inv.cpu().numpy()  # (n_bins, r, r)
            # Per voxel: q = Cb[v]ᵀ · CMC_inv[bin[v]] · Cb[v]
            bin_per_vox = np.clip(bin_idx_flat, 0, n_bins - 1)
            CMC_inv_vox = CMC_inv_np[bin_per_vox]  # (n_vox, r, r)
            q = np.einsum("vi,vij,vj->v", Cb, CMC_inv_vox, Cb)
            fstat = np.where(
                sigma2_vox > 0,
                q / (n_rows * np.clip(sigma2_vox, 1e-20, None)),
                0.0,
            )
            fstat[invalid_voxel] = 0.0
            fstat = np.clip(fstat, 0.0, None)  # F is non-negative
            contrast_results.append((label, "F", fstat.astype(np.float32)))
            if args.verb >= 1:
                print(
                    f"   {label} (F, {n_rows} rows): F peak="
                    f"{np.nanmax(fstat):.3f}, dof=({n_rows}, {dof_residual})"
                )

    # ── 9) Build output bucket ─────────────────────────────────────────
    # Strip every sub-brick whose label is a *contrast* output from a prior
    # concalc / GLM run, so re-running doesn't stack old + new. β / t per
    # stim, Full_Fstat, partial R², and any other non-contrast sub-brick is
    # preserved verbatim.
    keep_idx = _select_non_contrast_subbricks(stats_labels, stim_base_labels)

    # A dropped sub-brick is only *safe* when it's a contrast we're about to
    # recompute under the same name (fixing a bad contrast is expected). Any
    # other drop loses data that can't be regenerated without rerunning the
    # model — refuse unless -force. This is the tripwire for the classic
    # footgun: a spec whose stim labels don't match the bucket (e.g. singular
    # `face` vs the bucket's `faces#0_Coef`) silently reclassifies every β as
    # a stale contrast and strips it.
    new_contrast_labels = {label for label, _ in contrast_matrices}
    unsafe_drops = _unsafe_drops(stats_labels, keep_idx, new_contrast_labels)
    if unsafe_drops and not args.force:
        shown = ", ".join(unsafe_drops[:8]) + (" …" if len(unsafe_drops) > 8 else "")
        print(
            f"ERROR: {len(unsafe_drops)} sub-brick(s) would be dropped without "
            f"being recomputed:\n  {shown}\n"
            "These are not contrasts named in the spec, so they'd be lost "
            "(and with -inplace, permanently). This usually means the spec's "
            "stim labels don't match the bucket — check that the [[events]] "
            "trial_type names line up with the bucket's β labels. Pass -force "
            "to drop them anyway.",
            file=sys.stderr,
        )
        return 1

    out_labels = [stats_labels[i] for i in keep_idx]
    out_subs = [stats_data[..., i] for i in keep_idx]
    n_dropped = stats_data.shape[-1] - len(keep_idx)
    if args.verb >= 1 and n_dropped > 0:
        print(f"   Dropped {n_dropped} existing contrast sub-brick(s)")

    # Remap stat metadata for kept sub-bricks. Anything we drop is silently
    # discarded; anything we keep without a STATAUX entry is treated as a
    # non-stat sub-brick (β, mask, whatever).
    out_stataux: dict[int, tuple[int, tuple[float, ...]]] = {}
    for new_i, old_i in enumerate(keep_idx):
        if old_i in input_stataux:
            out_stataux[new_i] = input_stataux[old_i]

    n_new_subs = 0
    for entry in contrast_results:
        label, kind = entry[0], entry[1]
        if kind == "t":
            _, _, cb_vox, t_vox = entry
            coef_vol = np.zeros(vol_shape, dtype=np.float32)
            tstat_vol = np.zeros(vol_shape, dtype=np.float32)
            coef_vol[mask] = cb_vox
            tstat_vol[mask] = t_vox
            out_subs.append(coef_vol)
            out_labels.append(f"{label}_Coef")
            out_subs.append(tstat_vol)
            out_labels.append(f"{label}_Tstat")
            # -fout: emit F = t² with dof=(1, residual). AFNI's "-fout"
            # behaviour for contrasts. Single coef brick is still kept (you
            # need the magnitude for direction), the F sub-brick lands after.
            if args.fout:
                fstat_vol = tstat_vol * tstat_vol
                out_subs.append(fstat_vol)
                out_labels.append(f"{label}_Fstat")
                out_stataux[len(out_labels) - 1] = (
                    _STAT_CODE_FTEST,
                    (1.0, float(dof_residual)),
                )
                n_new_subs += 1
            # Coef brick has no stat metadata; Tstat brick gets fitt(dof).
            out_stataux[len(out_labels) - 1] = (
                _STAT_CODE_TTEST,
                (float(dof_residual),),
            )
            n_new_subs += 2
        else:  # "F"
            n_rows = entry[2] if len(entry) > 3 else None  # legacy guard
            _, _, fstat_vox = entry
            fstat_vol = np.zeros(vol_shape, dtype=np.float32)
            fstat_vol[mask] = fstat_vox
            out_subs.append(fstat_vol)
            out_labels.append(f"{label}_Fstat")
            # Look up n_rows from the matching contrast matrix.
            f_n_rows = next(
                (cm.shape[0] for lab, cm in contrast_matrices if lab == label),
                1,
            )
            out_stataux[len(out_labels) - 1] = (
                _STAT_CODE_FTEST,
                (float(f_n_rows), float(dof_residual)),
            )
            del n_rows
            n_new_subs += 1

    out_4d = np.stack(out_subs, axis=-1)

    if args.verb >= 1:
        print(
            f"💾 Writing {out_path} ({out_4d.shape[-1]} sub-bricks, +{n_new_subs} new contrast)",
            flush=True,
        )
    _save_bucket(out_path, out_4d, out_labels, stats_img, stataux=out_stataux)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
