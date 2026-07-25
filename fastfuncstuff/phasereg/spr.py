"""Source-localized phase regressor (sPR) — Vu & Gallant 2015.

Standard phase regression assumes a voxel containing a large vein has high
*phase* fSNR.  It often does not.  A vein roughly the size of a voxel places
the voxel across the whole off-resonance dipole lobe, so the sampled field
offsets are nearly symmetric about 0 Hz and cancel; intravascular phase
contributes little because venous T2* is short relative to parenchyma at high
field.  The result is a voxel with large task-correlated *magnitude* change and
almost no task-correlated phase — slope ~ 0, and the vein survives PR intact.
This is exactly the "implausibly large magnitude change in a dark voxel that
phase regression cannot see" failure mode at 7T.

Voxels *adjacent* to such a vein have the opposite profile: they sample one
polarity of the dipole field, so phase fSNR is high while magnitude fSNR is
low.  sPR borrows the phase regressor from the neighbour whose phase best
tracks the target voxel's magnitude:

    S_i^sPR(t) = S_i^m(t) - corr(S_i^m, S_{k*}^p) * S_{k*}^p(t)      (Eq. 12)

    k* = argmax_k |corr(S_i^m, S_k^p)|,   k in N(i)

where S^m, S^p are z-scored and N(i) is voxel i plus its six face-adjacent
neighbours.  Because OLS is scale-equivariant, the z-scored slope corr(.,.) is
identical to the raw-unit OLS slope of magnitude on donor phase; this module
therefore only selects the donor, and the existing regression path in
``phasereg.core`` produces Vu's estimator verbatim when run with
``regression="ols"``.

Note the loss function matters as much as the donor.  Vu & Gallant adopt plain
L_OLS rather than Menon's chi-squared (errors-in-variables) loss, because the
b1 term in the chi-squared denominator biases b1 upward and over-suppresses —
Nencka & Rowe 2007.  See ../fmri_wiki/concepts/Source-localized phase regression.md.
"""

from __future__ import annotations

import torch

# Offsets defining the donor search neighbourhood.  Vu & Gallant use the
# 6-connected (face-adjacent) set and call it "the smallest, most logical
# increment" over PR's one-voxel neighbourhood; the larger sets are offered
# because they explicitly leave neighbourhood size as an open question.
_FACE = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
_EDGE = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if abs(dx) + abs(dy) + abs(dz) == 2
]
_CORNER = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if abs(dx) + abs(dy) + abs(dz) == 3
]

NEIGHBOURHOODS = {
    6: _FACE,
    18: _FACE + _EDGE,
    26: _FACE + _EDGE + _CORNER,
}


def build_neighbor_index(
    volume_shape: tuple[int, int, int],
    mask_flat: torch.Tensor | None,
    connectivity: int = 6,
) -> torch.Tensor:
    """Map each in-mask voxel to its in-mask spatial neighbours.

    Parameters
    ----------
    volume_shape : (nx, ny, nz)
        Shape of the 3-D grid the voxels were flattened from (C order).
    mask_flat : Tensor (nx*ny*nz,) bool, or None
        Which volume voxels are present in the data array, in flattened order.
        None means every voxel is present.
    connectivity : {6, 18, 26}
        Donor search neighbourhood.  6 (face-adjacent) is Vu & Gallant's.

    Returns
    -------
    neighbors : Tensor (n_voxels, 1 + connectivity), int64
        Column 0 is the voxel itself; remaining columns are neighbour indices
        *into the masked voxel array*.  Neighbours that fall outside the volume
        or outside the mask are set to the voxel's own index, making them
        no-op candidates that can never beat the self-donor spuriously.
    """
    if connectivity not in NEIGHBOURHOODS:
        raise ValueError(
            f"connectivity must be one of {sorted(NEIGHBOURHOODS)}, got {connectivity}"
        )

    nx, ny, nz = volume_shape
    n_all = nx * ny * nz

    if mask_flat is None:
        mask_bool = torch.ones(n_all, dtype=torch.bool)
    else:
        mask_bool = torch.as_tensor(mask_flat).reshape(-1).bool()
        if mask_bool.numel() != n_all:
            raise ValueError(
                f"mask has {mask_bool.numel()} elements but volume_shape implies {n_all}"
            )

    n_vox = int(mask_bool.sum().item())

    # position[x,y,z] -> index into the masked array, or -1 if not in mask.
    position = torch.full((n_all,), -1, dtype=torch.int64)
    position[mask_bool] = torch.arange(n_vox, dtype=torch.int64)
    position = position.reshape(nx, ny, nz)

    self_idx = torch.arange(n_vox, dtype=torch.int64)
    offsets = NEIGHBOURHOODS[connectivity]
    columns = [self_idx]

    for dx, dy, dz in offsets:
        # Shift the whole position volume by (dx,dy,dz), padding with -1 so
        # voxels at the volume face see "no neighbour" rather than wrapping.
        shifted = torch.full((nx, ny, nz), -1, dtype=torch.int64)
        sx_dst = slice(max(dx, 0), nx + min(dx, 0))
        sy_dst = slice(max(dy, 0), ny + min(dy, 0))
        sz_dst = slice(max(dz, 0), nz + min(dz, 0))
        sx_src = slice(max(-dx, 0), nx + min(-dx, 0))
        sy_src = slice(max(-dy, 0), ny + min(-dy, 0))
        sz_src = slice(max(-dz, 0), nz + min(-dz, 0))
        shifted[sx_dst, sy_dst, sz_dst] = position[sx_src, sy_src, sz_src]

        neigh = shifted.reshape(-1)[mask_bool]
        columns.append(torch.where(neigh >= 0, neigh, self_idx))

    return torch.stack(columns, dim=1)


def select_phase_donor(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    neighbors: torch.Tensor,
    device: torch.device | str = "cpu",
    chunk_size: int = 50000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick, per voxel, the neighbour whose phase best tracks its magnitude.

    Implements the k* selection of Vu & Gallant 2015 Eq. 12.

    Parameters
    ----------
    magnitude, phase : Tensor (n_timepoints, n_voxels)
        Series the donor is chosen on.  Pass the task-residualised, filtered
        series that the slope will subsequently be fit on.
    neighbors : Tensor (n_voxels, K) int64
        From :func:`build_neighbor_index`.
    device : torch.device or str
    chunk_size : int
        Voxels per chunk.

    Returns
    -------
    donor : Tensor (n_voxels,) int64
        Index of the chosen donor voxel (== self where the voxel's own phase
        wins, which is standard PR).
    donor_corr : Tensor (n_voxels,)
        Signed corr(S_i^m, S_{k*}^p) at the chosen donor.  This is the z-scored
        sPR slope; the raw-unit slope is recovered by the OLS fit downstream.
    """
    device = torch.device(device)
    n_vox = magnitude.shape[1]

    # Correlation is scale-free, so normalise once and reuse across chunks.
    # Phase is normalised over ALL voxels (not just the chunk) because donors
    # can lie outside the chunk being processed.
    pha_c = phase - phase.mean(dim=0, keepdim=True)
    pha_n = pha_c / pha_c.norm(dim=0, keepdim=True).clamp(min=1e-12)

    donor = torch.zeros(n_vox, dtype=torch.int64)
    donor_corr = torch.zeros(n_vox)

    for start in range(0, n_vox, chunk_size):
        end = min(start + chunk_size, n_vox)

        mag_c = magnitude[:, start:end].to(device)
        mag_c = mag_c - mag_c.mean(dim=0, keepdim=True)
        mag_n = mag_c / mag_c.norm(dim=0, keepdim=True).clamp(min=1e-12)

        neigh_chunk = neighbors[start:end].to(device)  # (c, K)
        best_abs = torch.full((end - start,), -1.0, device=device)
        best_val = torch.zeros(end - start, device=device)
        best_idx = neigh_chunk[:, 0].clone()

        for k in range(neigh_chunk.shape[1]):
            idx = neigh_chunk[:, k]
            donor_pha = pha_n[:, idx.cpu()].to(device)  # (n_tp, c)
            corr = (mag_n * donor_pha).sum(dim=0)
            better = corr.abs() > best_abs
            best_abs = torch.where(better, corr.abs(), best_abs)
            best_val = torch.where(better, corr, best_val)
            best_idx = torch.where(better, idx, best_idx)

        donor[start:end] = best_idx.cpu()
        donor_corr[start:end] = best_val.cpu()

    del pha_n, pha_c
    return donor, donor_corr
