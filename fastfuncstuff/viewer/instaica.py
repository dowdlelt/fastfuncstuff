"""The InstaICA engine: spatial ICA as a rotation of an InstaPCA decomposition.

MELODIC's recipe -- reduce the run to its leading temporal dimensions, whiten,
find the rotation that makes the spatial maps most non-Gaussian -- without
touching the data a second time. InstaPCA has already prepared every voxel
(drift out, unit length) and projected it onto the leading eigenvectors ``u``:
``P = c u`` is ``(V, K)``, and its columns, scaled to unit variance, *are* the
whitened spatial components. FastICA runs on those ``K`` rows of ``V`` samples
(milliseconds at K = 30), and everything else is ``K x K`` algebra:

* the data is ``c^T ~ u D Z`` with ``Z`` the whitened components and ``D`` their
  scales; ICA writes ``Z = W^T S``, so the IC time courses are ``A = u D W^T``;
* each voxel's correlation with an IC's time course is ``c a / |a|`` --
  ``P D W^T`` with columns normalised -- the same amplitude-free map InstaPCA
  draws, now for an IC;
* amplitude is that correlation times InstaPCA's per-voxel scale, unchanged;
* the z-map is the spatial source ``S`` itself, z-scored: MELODIC's picture.

So a decomposition costs one InstaPCA plus a few milliseconds, and the review
loop -- labels, the noise ortvec, Denoise's "ICA noise" -- is the same code.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from fastfuncstuff.utils import accum_dtype
from fastfuncstuff.viewer.instapca import Decomposition


def rotate_to_ica(
    pca: Decomposition,
    *,
    seed: int = 0,
    max_iter: int = 1000,
    fun: str = "logcosh",
    device: torch.device | None = None,
) -> Decomposition:
    """Spatial ICA of ``pca``'s components: same span, rotated to independence.

    Every ``(V, K)`` pass runs on ``device`` in float32 with float64 accumulators for the
    reductions.  At V = 4.2M voxels the host float64 version spent ~3 s of a 4.7 s call in
    numpy (centre, std, transpose copy, ``W @ z``, ``corr @ D W^T``, cubes, ``argsort``,
    ``astype`` -- each a pass over ~1 GB) around a FastICA that is itself on the GPU.
    """
    from fastfuncstuff.decomposition.ica import fastica_batch

    dev = device or torch.device("cpu")
    n_vox, k = pca.correlation.shape
    if k < 2:
        raise ValueError("ICA needs at least two dimensions")
    f64 = accum_dtype(dev)  # two-pass reductions below; Metal has no float64

    corr = torch.as_tensor(np.ascontiguousarray(pca.correlation), dtype=torch.float32).to(dev)
    # Whitened spatial components: each projection column at unit variance.
    centred = corr - corr.mean(0, dtype=f64).float()
    d = (centred.square().sum(0, dtype=f64) / n_vox).sqrt()  # population std, as numpy's
    d = torch.where(d > 0, d, torch.ones_like(d))
    z = (centred / d.float()).T.contiguous()  # (K, V)
    del centred
    # K rows of V samples: at K = 30 and V ~ 2e5 the FastICA iterations are
    # real arithmetic, so they go where the decomposition went.
    w, _, converged = fastica_batch(z, 1, k, fun=fun, max_iter=max_iter, tol=1e-4, base_seed=seed)
    W = w[0].to(f64)  # (K, K), orthogonal unmixing
    s_src = W.float() @ z  # (K, V) spatial sources
    del z

    # The time courses: A = u D W^T. ``pca.timecourses`` is u * sqrt(T) (unit
    # std, sign-fixed); the constant and the signs cancel in normalisation.
    d_np = d.cpu().numpy()
    W_np = W.cpu().numpy()
    tc = np.asarray(pca.timecourses, dtype=np.float64) @ np.diag(d_np) @ W_np.T  # (T, K)
    # Each voxel's correlation with each IC time course: P D W^T over |a|.
    mix = torch.as_tensor(np.diag(d_np) @ W_np.T, dtype=torch.float32, device=dev)
    norms = np.linalg.norm(tc, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    inv_norm = torch.as_tensor(
        np.sqrt(tc.shape[0]) / norms, dtype=torch.float32, device=dev
    )  # 1 / (|a| / sqrt(T))
    ic_corr = (corr @ mix) * inv_norm  # (V, K)
    del corr
    # The sign convention InstaPCA and MELODIC share: heavy tail positive.
    flip_np = np.where(s_src.pow(3).sum(1, dtype=f64).cpu().numpy() < 0, -1.0, 1.0)
    flip = torch.as_tensor(flip_np, dtype=torch.float32, device=dev)
    s_src = s_src * flip[:, None]
    tc *= flip_np[None, :]
    ic_corr = ic_corr * flip[None, :]
    tc = (tc - tc.mean(0)) / np.where(tc.std(0) > 0, tc.std(0), 1.0)

    # Ordered by how much of the prepared variance each explains, as PCs are.
    energy = ic_corr.square().sum(0, dtype=f64).cpu().numpy()
    order_np = np.argsort(-energy)
    total = float(pca.explained.sum()) if pca.explained.size else 1.0
    explained = energy[order_np] / max(energy.sum(), 1e-30) * total
    order = torch.as_tensor(order_np, device=dev)

    zmap = s_src[order].T  # (V, K)
    z_mean = zmap.mean(0, dtype=f64).float()
    z_std = ((zmap - z_mean).square().sum(0, dtype=f64) / n_vox).sqrt()
    zmap = (zmap - z_mean) / torch.where(z_std > 0, z_std, torch.ones_like(z_std)).float()
    corr_out = ic_corr[:, order]
    scale = (
        torch.as_tensor(pca.scale, dtype=torch.float32, device=dev)
        if pca.scale is not None
        else torch.zeros(n_vox, dtype=torch.float32, device=dev)
    )
    return replace(
        pca,
        timecourses=tc[:, order_np],
        correlation=corr_out.cpu().numpy(),
        amplitude=(corr_out * scale[:, None]).cpu().numpy(),
        explained=np.asarray(explained, dtype=np.float64),
        zmap=zmap.cpu().numpy(),
        kind="IC",
        # Usual when some kept dimensions are noise: a Gaussian direction has
        # no preferred rotation. Said, not hidden -- fewer dims or another seed.
        note=""
        if bool(converged[0])
        else f"FastICA did not settle in {max_iter} iterations; fewer # dims or another seed",
    )
