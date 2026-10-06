"""MP2RAGE preparation for FreeSurfer — the presurfer workflow on ffs_segment.

presurfer (Kashyap, https://github.com/srikash/presurfer) turns an MP2RAGE UNI/INV2 pair
into a FreeSurfer-ready image with SPM Segment and voxel arithmetic:

1. **MPRAGEise** — ``UNI · scale01(INV2_biascorrected)``. UNI is a ratio image, so the
   noise outside the head is amplified to full brightness ("salt and pepper"); INV2 is a
   plain magnitude image that is dark there, so multiplying by it suppresses the
   background while leaving tissue contrast alone.
2. **stripmask** from the INV2 segmentation — ``1 - ((c3+c4+c5+c6) > 0.5)``: everything
   not confidently non-GM/WM.
3. **brainmask** ``(c1+c2+c3) > 0.3`` and **WMmask** ``c2 > 0.5`` from the segmentation of
   the MPRAGEised UNI.

The arithmetic is reproduced here exactly (the ``_raw`` outputs). :func:`clean_mask`
then adds what presurfer leaves to hand editing: dropping voxels the warped template says
cannot be brain (the eyes, which INV2's proton-density-like contrast can label as
tissue), an optional opening to cut thin bridges, the largest connected component, and
filling enclosed holes (presurfer's stripmask excludes CSF, so ventricles labelled CSF
become holes).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def scale_unit(image: Tensor, mode: str = "minmax", *, robust_pct: float = 99.9) -> Tensor:
    """Rescale an image to ``[0, 1]``.

    ``"minmax"`` is MATLAB's ``mat2gray`` (presurfer's choice). One hot voxel sets the
    maximum, so ``"robust"`` uses the ``robust_pct`` percentile of the nonzero voxels as
    the top instead (and 0 as the bottom), clipping above it.
    """
    image = image.to(torch.float32)
    finite = torch.isfinite(image)
    vals = image[finite]
    if mode == "minmax":
        lo, hi = float(vals.min()), float(vals.max())
    elif mode == "robust":
        nz = vals[vals != 0]
        lo = 0.0
        # torch.quantile caps its input size; a strided subsample is plenty for a percentile
        step = max(1, nz.numel() // 4_000_000)
        hi = float(torch.quantile(nz[::step], robust_pct / 100.0)) if nz.numel() else 1.0
    else:
        raise ValueError(f"scale_unit mode must be 'minmax' or 'robust', got {mode!r}")
    if hi <= lo:
        return torch.zeros_like(image)
    out = ((image - lo) / (hi - lo)).clamp(0.0, 1.0)
    return torch.where(finite, out, torch.zeros_like(out))


def mprageise(uni: Tensor, inv2_corrected: Tensor, mode: str = "minmax") -> Tensor:
    """``UNI · scale01(INV2)`` — suppress UNI's amplified background noise."""
    return uni.to(torch.float32) * scale_unit(inv2_corrected, mode)


def class_sum_mask(posteriors: Tensor, classes: tuple[int, ...], thresh: float) -> Tensor:
    """``Σ_{k∈classes} c_k > thresh`` as a bool mask (0-based class indices)."""
    return posteriors[list(classes)].sum(dim=0) > thresh


def strip_mask(posteriors: Tensor, keep: tuple[int, ...] = (0, 1), thresh: float = 0.5) -> Tensor:
    """presurfer's stripmask ``1 - (Σ_{k∉keep} c_k > thresh)``.

    Written as a complement over the *other* classes, as presurfer does, rather than
    ``Σ_keep c_k ≥ 1 - thresh``: the two differ wherever the classes do not sum to 1.
    """
    others = tuple(k for k in range(posteriors.shape[0]) if k not in keep)
    return ~class_sum_mask(posteriors, others, thresh)


def clean_mask(
    mask: Tensor,
    *,
    brain_prior: Tensor | None = None,
    prior_thresh: float = 0.05,
    open_radius: int = 0,
    largest_cluster: bool = True,
    fill_holes: bool = True,
) -> tuple[Tensor, Tensor]:
    """Remove non-brain islands from a tissue mask, in a fixed order.

    1. **Prior gate** — drop voxels whose warped template brain probability
       (``brain_prior``, GM+WM+CSF of the TPM pulled into this space by the fit) is at or
       below ``prior_thresh``. The eyes have zero brain prior, so this removes them
       however they are connected — which plain cluster-keeping cannot do when an optic
       nerve or orbital fat bridges them to the frontal lobe.
    2. **Opening** (``open_radius`` > 0) — erode, keep the largest component, dilate back
       (``open_radius + 1``) inside the gated mask: cuts bridges thinner than
       ~``2·open_radius`` voxels.
    3. **Largest 6-connected component.**
    4. **Fill enclosed holes.**

    Returns:
        ``(cleaned, removed)`` — bool masks; ``removed`` is what steps 1–3 took away
        (a QC map: what would have had to be hand-edited).
    """
    from scipy import ndimage

    m = mask.detach().cpu().numpy().astype(bool)
    original = m.copy()
    if brain_prior is not None:
        m &= brain_prior.detach().cpu().numpy() > prior_thresh
    six = ndimage.generate_binary_structure(3, 1)
    if open_radius > 0 and m.any():
        core = ndimage.binary_erosion(m, six, iterations=open_radius)
        core = _largest_component(core, six)
        if core.any():
            # one step further than the erosion: a 6-conn erode/dilate pair alone shaves
            # the corners off a convex surface; the extra step restores them (inside the
            # mask) while only re-growing a 1-voxel stub into a cut bridge
            m &= ndimage.binary_dilation(core, six, iterations=open_radius + 1)
    if largest_cluster:
        m = _largest_component(m, six)
    removed = original & ~m
    if fill_holes:
        m = ndimage.binary_fill_holes(m)
    dev = mask.device
    return torch.from_numpy(m).to(dev), torch.from_numpy(removed).to(dev)


def _largest_component(m: np.ndarray, structure: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, n = ndimage.label(m, structure=structure)
    if n <= 1:
        return m
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def warped_brain_prior(
    log_prior: Tensor,
    tpm_affine: Tensor,
    fit: dict,
    out_shape: tuple[int, int, int],
    classes: tuple[int, ...] = (0, 1, 2),
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Template brain probability (Σ of ``classes``) pulled into the input grid by ``fit``."""
    from .segment import cast_template_to_input

    prior = torch.exp(log_prior[list(classes)]).sum(dim=0)
    out = cast_template_to_input(
        prior, tpm_affine, tpm_affine, fit, out_shape, kernel="linear", device=device
    )
    return out[0] if out.ndim == 4 else out


def segment_image(
    volume: Tensor,
    subj_affine: Tensor,
    log_prior: Tensor,
    tpm_affine: Tensor,
    bg_low: Tensor,
    bg_high: Tensor,
    *,
    affreg: str = "mni",
    dither: float = 0.0,
    mrf: float = 1.0,
    cleanup: int = 1,
    device: torch.device | str = "cpu",
    verbose: bool = True,
    **fit_kwargs,
) -> tuple[dict, dict]:
    """``ffs_segment`` without the I/O: TPM affine → fit → full-resolution apply.

    ``fit_kwargs`` go to :func:`segment.fit_segment` (``ngaus``, ``biasreg``, ``reg``,
    ``samp``, …). Returns ``(fit, out)`` — ``out`` from :func:`segment.segment_apply`
    (``posteriors``, ``corrected``, ``bias``).
    """
    from .segment import fit_segment, segment_apply
    from .segment_affine import affine_to_tpm

    world = np.eye(4)
    if affreg != "off":
        world = affine_to_tpm(
            volume,
            np.asarray(subj_affine.cpu(), dtype=np.float64),
            log_prior,
            np.asarray(tpm_affine.cpu(), dtype=np.float64),
            bg_low,
            bg_high,
            samp=float(fit_kwargs.get("samp", 3.0)),
            fwhm=float(fit_kwargs.get("fwhm", 0.0)),
            regtype=affreg,
            dither=dither,
            verbose=verbose,
        )
    fit = fit_segment(
        volume,
        subj_affine,
        log_prior,
        tpm_affine,
        bg_low,
        bg_high,
        torch.as_tensor(world, dtype=torch.float64, device=device),
        dither=dither,
        device=device,
        verbose=verbose,
        **fit_kwargs,
    )
    out = segment_apply(
        volume, log_prior, bg_low, bg_high, fit, mrf=mrf, cleanup=cleanup,
        device=device, verbose=verbose,
    )  # fmt: skip
    return fit, out
