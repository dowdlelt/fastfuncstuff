"""
Cortical-depth sampling and the voxel point-spread function.

Getting from voxels to the ``K`` depth timecourses the generative model expects,
and estimating how much the finite voxel grid blurs a depth profile on the way.

FFS does not build equivolume depth surfaces -- that is a surface pipeline's job
(LayNii and friends). What arrives here is a per-voxel *normalised cortical
depth* map, 0 at the white-matter boundary and 1 at the pial surface, plus the
voxel timecourses. Depth files, column files and layer files being separate
aligned volumes is the expected input shape.

Two of the three functions here are the ones the reference driver comments out
with "this takes time so might consider saving and then loading from file". They
are label-wise means and a two-parameter fit; vectorised, they are instant.

Ported from ``BOLD_voxels2layers_flipdata.m``, ``BOLD_estimate_laminar_PSF_3D.m``
and ``ConvKernel.m``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize


def depth_bin_centres(K: int) -> np.ndarray:
    """Midpoints of K equal depth bins on [0, 1]. WM at 0, CSF at 1."""
    return np.linspace(0, 1, 2 * K + 1)[1::2]


def _bell_weights(depth: np.ndarray, K: int) -> np.ndarray:
    """Soft voxel-to-depth assignment, ``(n_voxels, K)``.

    Each voxel contributes to every depth with a bell-shaped weight set by its
    distance from that depth's centre -- Faes et al.'s "normalized equally-spaced
    bell-shaped weighted probability maps". Voxels are *not* hard-assigned, which
    matters because a 0.8 mm voxel straddles more than one cortical depth.

    The bins are padded with one phantom depth at each end before normalising, so
    voxels beyond the outermost centres shed weight instead of piling onto the
    edge depths.
    """
    centres = depth_bin_centres(K)
    step = centres[1] - centres[0]
    extended = np.concatenate([[centres[0] - step], centres, [centres[-1] + step]])
    # The cube makes the bell narrow enough that a voxel is effectively shared
    # between at most two or three depths.
    H = np.exp(-((len(extended) * (depth[:, None] - extended[None, :])) ** 2)) ** 3
    H = H / H.sum(axis=1, keepdims=True)
    return H[:, 1:-1]


def voxels_to_layers(data: np.ndarray, depth: np.ndarray, K: int) -> np.ndarray:
    """Average voxel timecourses into K depth timecourses.

    Parameters
    ----------
    data : ``(n_voxels, n_time)``
    depth : ``(n_voxels,)`` normalised cortical depth, 0 = WM, 1 = CSF.

    Returns
    -------
    ``(n_time, K)`` with **depth 0 superficial** (closest to CSF), matching the
    model's convention and the reference's ``fliplr``.
    """
    if data.shape[0] != depth.shape[0]:
        raise ValueError(f"{data.shape[0]} voxels of data vs {depth.shape[0]} depths")
    order = np.argsort(depth, kind="stable")
    H = _bell_weights(depth[order], K)
    # Second normalisation, down the voxel axis: each depth's timecourse is a
    # weighted *mean* over voxels, not a weighted sum, so depths sampled by more
    # voxels are not scaled up.
    Hs = H / H.sum(axis=0, keepdims=True)
    y = (Hs.T @ data[order]).T
    return y[:, ::-1]


def label_mean(values: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """NaN-ignoring mean of ``values`` within each integer label.

    The reference loops over labels with a boolean mask each time, which is why
    it is precomputed and cached to a ``.mat``. Two ``bincount`` calls do it in
    one pass.
    """
    flat_v = np.asarray(values, dtype=float).reshape(-1)
    flat_l = np.asarray(labels).reshape(-1)
    good = np.isfinite(flat_v)
    total = np.bincount(flat_l[good], weights=flat_v[good], minlength=n_labels)
    count = np.bincount(flat_l[good], minlength=n_labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out[:n_labels]


def _conv_kernel_cost(
    par: np.ndarray, y: np.ndarray, inp: np.ndarray, dist: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Residual of a Gaussian blur explaining voxel-sampled depth profiles.

    ``par`` is (amplitude, variance). The model is: take the true high-resolution
    depth profile, convolve it along depth with a Gaussian, and it should look
    like what the voxel grid actually measured.
    """
    sp = np.linspace(0, 1, 1000)
    kernel = 1.0 / np.sqrt(2 * np.pi * par[1]) * np.exp(-((sp - 0.5) ** 2) / (2 * par[1]))
    kernel = kernel / kernel.sum()

    on_sp = PchipInterpolator(dist, inp, axis=0, extrapolate=True)(sp)
    # conv2(..., 'same') down the depth axis, column by column.
    L = len(kernel)
    start = L // 2
    pred = np.stack(
        [
            np.convolve(on_sp[:, j], kernel, mode="full")[start : start + len(sp)]
            for j in range(on_sp.shape[1])
        ],
        axis=1,
    )
    pred = np.stack(
        [np.interp(dist, sp, pred[:, j], left=np.nan, right=np.nan) for j in range(pred.shape[1])],
        axis=1,
    )
    pred = par[0] * pred
    err = float(np.linalg.norm(np.nan_to_num(pred).ravel() - y.ravel()))
    return err, kernel / kernel.max(), sp


def estimate_depth_psf(
    N: int,
    K: int,
    labels: np.ndarray,
    depth_hi: np.ndarray,
    depth_lo: np.ndarray,
    voxel_index: np.ndarray,
    *,
    n_profiles: int = 10,
    curvature_sd: float = 0.06,
    seed: int | None = 0,
    centres: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the depth point-spread function of this ROI's voxel grid.

    The idea: synthesise smooth depth profiles on the *high-resolution*
    anatomical grid, downsample them the way the functional data was
    downsampled, and fit the Gaussian blur that explains the difference. What
    comes out absorbs the voxel size, the ROI's curvature and the number of
    depths at once -- which is why the paper describes the kernel as "adjusted
    for the number of vascular depths, the curvature of the ROI, and the voxel
    size" without giving a formula.

    Parameters
    ----------
    labels : high-resolution voxel -> functional voxel index map (0-based).
    depth_hi : normalised depth on the high-resolution grid, same shape.
    depth_lo : normalised depth per functional voxel.
    voxel_index : functional voxel indices of the selected ROI voxels.
    seed : the reference draws ``n_profiles`` curvature centres at random, so
        the kernel is stochastic. Seeded here; pass ``None`` to match the
        reference's unseeded behaviour.
    centres : supply the curvature centres directly instead of drawing them.
        Used to check parity against a MATLAB run, whose RNG differs.

    Returns
    -------
    ``(K,)`` kernel, normalised to sum to 1, for :func:`apply_depth_psf`.
    """
    if centres is None:
        rng = np.random.default_rng(seed)
        centres = 0.5 + curvature_sd * rng.standard_normal(n_profiles)
    centres = np.asarray(centres, dtype=float).reshape(-1)
    n_profiles = centres.size
    nsig = (1.0 / (4 * (N + 1))) ** 2

    flat_hi = np.asarray(depth_hi, dtype=float).reshape(-1)
    profiles = (
        1.0
        / np.sqrt(2 * np.pi * nsig)
        * np.exp(-((flat_hi[:, None] - centres[None, :]) ** 2) / (2 * nsig))
    )

    n_labels = int(np.asarray(labels).max()) + 1
    downsampled = np.stack(
        [label_mean(profiles[:, j], labels, n_labels) for j in range(n_profiles)], axis=1
    )

    sel_depth = np.asarray(depth_lo)[voxel_index]
    order = np.argsort(sel_depth, kind="stable")
    dist = sel_depth[order]
    measured = downsampled[voxel_index][order]

    # The same profiles at the same depths but *without* voxel averaging: the
    # unblurred reference the blur has to explain.
    uniq_depth, first = np.unique(flat_hi, return_index=True)
    truth = np.stack(
        [
            np.interp(dist, uniq_depth, profiles[first, j], left=np.nan, right=np.nan)
            for j in range(n_profiles)
        ],
        axis=1,
    )
    truth = np.nan_to_num(truth)

    fit = minimize(
        lambda p: _conv_kernel_cost(p, measured, truth, dist)[0],
        np.array([1.0, 0.1]),
        method="Nelder-Mead",
    )
    _, kernel, sp = _conv_kernel_cost(fit.x, measured, truth, dist)

    at_depths = PchipInterpolator(sp, kernel, extrapolate=True)(depth_bin_centres(K))
    return at_depths / at_depths.sum()
