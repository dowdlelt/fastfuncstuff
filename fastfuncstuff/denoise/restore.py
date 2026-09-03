"""Put a wrongly-removed component back into a denoised series.

A patch-based denoiser decides what to discard from a singular-value spectrum, and
nothing in that decision knows what the signal is. When a component that carried real
response ends up in what was removed, the fix is not to denoise less everywhere -- that
is what a threshold cap like ``ffs_nordic -retain_dof`` does, and it gives back noise in
every patch to rescue signal in a few. It is to give back **that component**, where it
lives, and nothing else.

The input is a decomposition of the REMOVED field (``ffs_nordic -save_task_loss`` writes
it; ``ffs_ica`` decomposes it), and the operation is rank-1 per component::

    restored = denoised + sum_c  gamma_c * a_c (x) s_c

``a_c`` is the component's time course and ``s_c`` its spatial map, so what returns is
confined to where the component actually is -- a voxel the map does not touch gets
nothing back, which is the whole point of choosing this over regressing the time course
into every voxel.

``gamma_c`` is not optional bookkeeping. ICA maps and mixing columns carry the scale of
whatever whitening and variance normalisation the decomposition applied, not the units
of the series, so adding ``a_c (x) s_c`` straight onto a magnitude image is off by an
arbitrary factor. The scale comes from the removed field itself: shape from the
decomposition, amplitude from the data.

The fit is **joint over the selected components**, not one projection each. ICA leaves
components only approximately independent, and a marginal
``<L, a_c (x) s_c> / ||a_c (x) s_c||^2`` then credits each one with whatever its
neighbours explain too -- restoring a set of three planted terms that way misses their
amplitudes outright. The joint solve is closed-form and costs nothing, because the Gram
of rank-1 terms factorises::

    G[c, d] = <a_c (x) s_c, a_d (x) s_d> = (s_c . s_d) (a_c . a_d)
    b[c]    = <L, a_c (x) s_c>           = s_c . (L a_c)
    gamma   = G^-1 b

so it is an ``n_selected``-square system whatever the image size. It reduces to the
marginal answer exactly when the selected components are orthogonal.

Degrees of freedom stay countable: each restored component returns exactly one, which is
what makes this compatible with a NORDIC-aware DoF adjustment (see ``ffs_util_updatedof``
/ ``ffs_reml -adjust_dof``) rather than leaving the statistics in an unknown state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RestoreResult:
    """The restored series plus what it cost."""

    restored: torch.Tensor  # (nx, ny, nz, T)
    gammas: np.ndarray  # (n_selected,) amplitude fitted per component
    var_returned: np.ndarray  # (n_selected,) share of the removed field's variance
    # The joint total, which is NOT the sum of the per-component shares unless the
    # selected components happen to be orthogonal.
    var_returned_total: float
    indices: list[int]
    dof_returned: int


def restore_components(
    denoised: torch.Tensor,
    removed: torch.Tensor,
    maps_kv: torch.Tensor,
    mixing_tk: torch.Tensor,
    indices: list[int],
) -> RestoreResult:
    """``denoised + sum_c gamma_c * a_c (x) s_c``, amplitudes fitted to ``removed``.

    ``maps_kv`` is (K, V) and ``mixing_tk`` is (T, K), the shapes ``ffs_ica`` writes.
    ``denoised`` and ``removed`` are 4-D and must share a grid and frame count.
    """
    if denoised.shape != removed.shape:
        raise ValueError(
            f"denoised {tuple(denoised.shape)} and removed {tuple(removed.shape)} "
            "must have the same shape"
        )
    n_t = denoised.shape[3]
    if mixing_tk.shape[0] != n_t:
        raise ValueError(
            f"time courses have {mixing_tk.shape[0]} rows but the series has {n_t} frames"
        )
    n_vox = int(np.prod(denoised.shape[:3]))
    if maps_kv.shape[1] != n_vox:
        raise ValueError(f"maps carry {maps_kv.shape[1]} voxels, the series has {n_vox}")
    if maps_kv.shape[0] != mixing_tk.shape[1]:
        raise ValueError(f"{maps_kv.shape[0]} maps against {mixing_tk.shape[1]} time courses")
    bad = [i for i in indices if not 0 <= i < maps_kv.shape[0]]
    if bad:
        raise ValueError(f"component index out of range for {maps_kv.shape[0]} components: {bad}")

    lost = removed.reshape(n_vox, n_t).double()
    total_ss = float((lost**2).sum())
    a_sel = mixing_tk[:, indices].double()  # (T, n_sel)
    s_sel = maps_kv[indices].double()  # (n_sel, V)
    zero = [
        indices[i]
        for i in range(len(indices))
        if float(a_sel[:, i] @ a_sel[:, i]) <= 0 or float(s_sel[i] @ s_sel[i]) <= 0
    ]
    if zero:
        raise ValueError(f"component(s) {zero} have a zero time course or a zero map")

    gram = (s_sel @ s_sel.T) * (a_sel.T @ a_sel)  # (n_sel, n_sel)
    rhs = (s_sel * (lost @ a_sel).T).sum(dim=1)  # (n_sel,)
    # lstsq rather than solve: two components can be near-duplicates after an ICA that
    # split one source, and a singular Gram should give the minimum-norm answer instead
    # of a crash the user cannot act on.
    gammas = torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution.squeeze(1)

    out = denoised.reshape(n_vox, n_t).clone().double()
    added = (s_sel * gammas.unsqueeze(1)).T @ a_sel.T  # (V, T)
    out += added
    shares = np.asarray(
        [
            float(g) ** 2
            * float(s_sel[i] @ s_sel[i])
            * float(a_sel[:, i] @ a_sel[:, i])
            / max(total_ss, 1e-30)
            for i, g in enumerate(gammas)
        ],
        dtype=np.float64,
    )
    return RestoreResult(
        restored=out.reshape(denoised.shape).to(denoised.dtype),
        gammas=gammas.cpu().numpy().astype(np.float64),
        var_returned=shares,
        var_returned_total=float((added**2).sum()) / max(total_ss, 1e-30),
        indices=list(indices),
        # One per component, by construction: a rank-1 term costs one direction of the
        # time axis, whatever its spatial extent.
        dof_returned=len(indices),
    )
