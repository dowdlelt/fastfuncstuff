"""MotSim: motion-simulation nuisance regressors ([[Patriat 2017]]).

The standard motion nuisance model regresses the realignment *parameters*, which
assumes the signal change they cause is a linear function of them. At a curved
intensity edge it is not. MotSim instead models the signal changes themselves:
move one acquired volume by the inverse of the estimated motion, and every
fluctuation in the resulting series is caused by motion and nothing else. Its
temporal PCs are the regressors.

Three variants, named as in the paper:
  - forward:  the simulated series (MotSim). No second registration pass.
  - backward: that series re-registered (MotSimReg) — what a real correction
              leaves behind: interpolation error and motion-estimation error.
  - both:     forward and backward spatially concatenated, then PCA.

The count is not sacred. The paper used 12 only to match the regressor count of
the 6-params-plus-derivatives model it was competing with, and says outright that
the right number depends on run length and motion level, so a spec may ask for a
variance fraction instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from .affine import dicom_matrix_to_voxel, params_to_matrix

# The paper's model names, and what a bare -motsim means.
MOTSIM_VARIANTS = ("forward", "backward", "both")
DEFAULT_MOTSIM_VARIANT = "both"
DEFAULT_MOTSIM_NPCS = 12


@dataclass(frozen=True)
class MotSimSpec:
    """A parsed ``-motsim`` spec: which simulation, and how many components.

    ``n_pcs`` is an int (exactly that many) or a float in (0, 1) (however many
    reach that fraction of the simulated series' variance).
    """

    variant: str = DEFAULT_MOTSIM_VARIANT
    n_pcs: int | float = DEFAULT_MOTSIM_NPCS

    def __str__(self) -> str:
        return f"{self.variant},{self.n_pcs:g}"

    @property
    def needs_backward(self) -> bool:
        return self.variant in ("backward", "both")


def parse_motsim_spec(text: str | None) -> MotSimSpec:
    """Parse ``MODE[,N]`` into a :class:`MotSimSpec`.

    ``N`` is an integer PC count, or a fraction in (0, 1) meaning "enough PCs to
    reach that much of the simulated series' variance". An empty/None spec is the
    paper's headline model, 12Both.

    >>> parse_motsim_spec("both,12")
    MotSimSpec(variant='both', n_pcs=12)
    >>> parse_motsim_spec("forward,0.95")
    MotSimSpec(variant='forward', n_pcs=0.95)
    """
    if text is None or not text.strip():
        return MotSimSpec()

    parts = [p.strip() for p in text.split(",")]
    if len(parts) > 2:
        raise ValueError(f"-motsim takes MODE[,N], got {text!r}")

    variant = parts[0].lower()
    if variant not in MOTSIM_VARIANTS:
        raise ValueError(
            f"-motsim mode must be one of {'/'.join(MOTSIM_VARIANTS)}, got {parts[0]!r}"
        )

    if len(parts) == 1:
        return MotSimSpec(variant=variant)

    raw = parts[1]
    try:
        value: int | float = int(raw)
    except ValueError:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(
                f"-motsim component count must be an integer or a fraction, got {raw!r}"
            ) from None
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"-motsim fractional count must be in (0, 1) — a whole number of "
                f"components is written without a decimal point, got {raw!r}"
            ) from None
    else:
        if value < 1:
            raise ValueError(f"-motsim component count must be >= 1, got {raw!r}")

    return MotSimSpec(variant=variant, n_pcs=value)


@dataclass
class MotSimResult:
    """Regressors plus everything worth writing out or asserting on."""

    pcs: Tensor  # (nt, k) unit-variance, demeaned
    var_explained: Tensor  # (k,) fraction of the SIMULATED series' variance
    spec: MotSimSpec
    mask: Tensor  # (nz, ny, nx) bool — the dilated mask the PCA ran in
    forward: Tensor | None = None  # (nt, nz, ny, nx) MotSim, CPU
    backward: Tensor | None = None  # (nt, nz, ny, nx) MotSimReg, CPU


def load_motion_1d(path: str) -> np.ndarray:
    """Load 6-column .1D motion file (roll pitch yaw dS dL dP).

    Returns (nt, 6) DICOM params [dx, dy, dz, rz, rx, ry] matching
    the internal convention used by ffs_moco.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 6:
                raise ValueError(f"Expected 6 columns in {path}, got {len(vals)}")
            roll, pitch, yaw, dS, dL, dP = vals[:6]
            # Reverse AFNI mapping: rz=-roll, rx=pitch, ry=yaw, dz=-dS, dx=dL, dy=dP
            rows.append([dL, dP, -dS, -roll, pitch, yaw])
    return np.array(rows, dtype=np.float64)


def load_dfile(path: str) -> np.ndarray:
    """Load 9-column dfile (vol# roll pitch yaw dS dL dP rms_bef rms_aft).

    Returns (nt, 6) DICOM params [dx, dy, dz, rz, rx, ry].
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 7:
                raise ValueError(f"Expected >= 7 columns in dfile {path}, got {len(vals)}")
            # Columns: vol# roll pitch yaw dS dL dP [rms_bef rms_aft]
            _, roll, pitch, yaw, dS, dL, dP = vals[:7]
            # Reverse AFNI mapping (same as .1D)
            rows.append([dL, dP, -dS, -roll, pitch, yaw])
    return np.array(rows, dtype=np.float64)


def params_to_voxel_matrices(
    params_dicom: np.ndarray,
    nifti_affine: np.ndarray,
) -> Tensor:
    """Convert (nt, 6) DICOM rigid params to (nt, 4, 4) voxel-space matrices.

    Builds 4x4 DICOM-space matrices from [dx, dy, dz, rz, rx, ry],
    then converts to voxel index space using the NIfTI affine.
    """
    nt = params_dicom.shape[0]
    matrices_vox = torch.zeros(nt, 4, 4, dtype=torch.float32)

    for t in range(nt):
        # Build full 12-param vector: [dx,dy,dz, rz,rx,ry, sx,sy,sz, shyx,shzx,shzy]
        p12 = torch.zeros(12, dtype=torch.float32)
        p12[:6] = torch.from_numpy(params_dicom[t].astype(np.float32))
        p12[6:9] = 1.0  # scales = identity

        M_dicom = params_to_matrix(p12)
        M_vox = dicom_matrix_to_voxel(M_dicom, nifti_affine, nifti_affine)
        matrices_vox[t] = M_vox

    return matrices_vox


def build_motsim_mask(
    reference: Tensor,
    mask: Tensor | None = None,
    dilate: int = 2,
    device: torch.device | None = None,
) -> Tensor:
    """The PCA mask: brain, dilated outward.

    The dilation is not incidental. The paper's whole-brain average gain over the
    standard model is only 4.1%; the gain is concentrated at the brain *edge*,
    which is exactly where a tight mask throws it away. A caller-supplied mask is
    dilated too — a mask drawn for some other purpose is not drawn for this one.

    Args:
        reference: (nz, ny, nx) volume the mask is derived from when none is given.
        mask: optional (nz, ny, nx) mask to use instead of an automask.
        dilate: outward dilation in voxels (the paper's value is 2).
        device: device to build on; defaults to the reference's.
    """
    from .mask import _dilate_6conn, automask

    dev = device if device is not None else reference.device
    if mask is None:
        # dilate_extra is automask's own outward dilation, so the paper's mask is
        # one call rather than a threshold rolled by hand.
        return automask(reference.to(dev), dilate_extra=dilate, device=dev)
    return _dilate_6conn(mask.to(dev).bool(), iterations=dilate)


def _sim_moco_config(parent, interp: str | None = None, verb: int = 0):
    """A MocoConfig for the simulated data, derived from the real correction's.

    Everything about *how the volumes are registered and resampled* is inherited,
    so the backward model is the residual of the correction that actually ran
    rather than of some cheaper one. Everything about *what else that pass did* is
    cleared: the simulated series has no slice-timing structure to unwind, no
    outlier voxels to reweight against, and no injected weights to reuse.
    """
    overrides: dict = dict(
        base_index=0,
        skip_resample=False,
        verb=verb,
        slice_times=None,
        st_tr=None,
        st_tzero=None,
        reweight=False,
        weight_override=None,
        derivs_override=None,
    )
    if interp is not None:
        overrides["interp"] = interp
        overrides["final_interp"] = interp
    return replace(parent, **overrides)


def run_forward_sim(
    reference: Tensor,
    matrices_vox: Tensor,
    device: torch.device,
    interp: str = "cubic",
    verb: int = 1,
    config=None,
) -> Tensor:
    """Build the MotSim series: the reference, moved by the inverse motion.

    ``matrices_vox[t]`` is moco's own transform, which maps *base* voxel
    coordinates to *acquired* ones — resampling a volume through it undoes that
    volume's motion. So the volume the scanner would have acquired with the head
    at pose t is the reference resampled through its inverse, and motion-correcting
    the result recovers the matrices we started from (``test_forward_sim_roundtrip``).

    Args:
        reference: (nz, ny, nx) reference EPI.
        matrices_vox: (nt, 4, 4) voxel-space registration matrices from ffs_moco.
        device: torch device.
        interp: interpolation method; ignored when ``config`` is given.
        verb: verbosity.
        config: MocoConfig whose resampling settings (final_interp, use_shear,
            memory debug) to reuse. Built from ``interp`` when None.

    Returns:
        (nt, nz, ny, nx) forward simulation on the CPU.
    """
    from .ffs_moco import MocoConfig, resample_timeseries

    nt = matrices_vox.shape[0]
    # verb-1: the progress bar below is this pass's own reporting; moco's
    # "Resampling batch size" line belongs to a correction, not to a simulation.
    if config is not None:
        cfg = _sim_moco_config(config, verb=max(verb - 1, 0))
    else:
        cfg = MocoConfig(final_interp=interp, device=str(device), verb=max(verb - 1, 0))

    inv = torch.linalg.inv(matrices_vox.double()).to(matrices_vox.dtype)

    # One volume resampled nt ways: expand rather than repeat, so the source costs
    # nothing, and let the batched Pass-2 resampler do the chunking and (on CUDA)
    # the fused shear kernel.
    sources = reference.unsqueeze(0).expand(nt, *reference.shape)
    forward_sim, _ = resample_timeseries(
        sources,
        inv,
        cfg,
        device,
        disable_pbar=(verb < 1 or nt < 32),
    )

    if verb >= 1:
        print(f"  MotSim (forward): {nt} volumes, {cfg.final_interp} interp")

    return forward_sim


def run_backward_sim(
    forward_sim: Tensor,
    reference: Tensor,
    device: torch.device,
    interp: str = "cubic",
    verb: int = 1,
    config=None,
    header_info: dict | None = None,
) -> Tensor:
    """Build MotSimReg: the MotSim series put back through motion correction.

    The registration is **re-estimated** from the simulated data, not obtained by
    inverting the transform that created it — inverting would recover the reference
    exactly and leave nothing to model. What survives is interpolation error plus
    the error of estimating motion from low-resolution EPI.

    ``config`` should be the same MocoConfig the real correction ran under. A
    cheaper one (fewer iterations, coarser interpolation) makes this series the
    residual of a correction nobody performed.

    Args:
        forward_sim: (nt, nz, ny, nx) MotSim series.
        reference: (nz, ny, nx) reference EPI — the base to register back to.
        device: torch device.
        interp: interpolation method; ignored when ``config`` is given.
        verb: verbosity.
        config: MocoConfig of the real correction.
        header_info: NIfTI header dict, passed through to moco.

    Returns:
        (nt, nz, ny, nx) backward simulation on the CPU.
    """
    from .ffs_moco import MocoConfig, moco

    if config is not None:
        cfg = _sim_moco_config(config, verb=max(verb - 1, 0))
    else:
        # The shear resampler has no linear kernel; cubic is its floor.
        moco_interp = interp if interp != "linear" else "cubic"
        cfg = MocoConfig(
            base_index=0,
            interp=moco_interp,
            final_interp=moco_interp,
            compile=False,
            device=str(device),
            verb=max(verb - 1, 0),
        )

    if verb >= 1:
        print(f"  MotSimReg (backward): re-registering {forward_sim.shape[0]} volumes...")

    # The reference is the base, handed over externally so no volume of the
    # simulated series is copied verbatim in its place.
    result = moco(forward_sim, cfg, header_info=header_info, base_vol=reference)
    return result.aligned


def extract_pcs(
    data_4d: Tensor,
    mask: Tensor,
    n_pcs: int | float,
    verb: int = 1,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Temporal PCs of masked 4D data, as unit-variance demeaned regressors.

    Args:
        data_4d: (nt, nz, ny, nx) simulation data.
        mask: (nz, ny, nx) boolean mask.
        n_pcs: int (exactly that many) or float in (0, 1) (that much variance).
        verb: verbosity.
        device: device for the SVD; defaults to the data's.

    Returns:
        (pcs, var_explained): pcs is (nt, k), var_explained is (k,).
    """
    from fastfuncstuff.decomposition.pca import PCA

    nt = data_4d.shape[0]
    dev = device if device is not None else data_4d.device

    # Centring costs a rank, so nt-1 components is all there is to have.
    n_req = min(n_pcs, nt - 1) if isinstance(n_pcs, int) else n_pcs

    mask_flat = mask.reshape(-1).to(data_4d.device)
    mat = data_4d.reshape(nt, -1)[:, mask_flat].float()

    pca = PCA(n_components=n_req, device=dev)
    scores = pca.fit_transform(mat)  # (nt, k)

    # Demean explicitly: PCA's fit_transform re-reads the caller's tensor, so the
    # scores carry a DC offset whenever the fit ran on a different device than the
    # input. A nuisance regressor with an arbitrary constant in it is at best
    # redundant with the polynomial baseline and at worst collinear with it.
    scores = scores - scores.mean(dim=0, keepdim=True)
    sc_std = scores.std(dim=0, keepdim=True).clamp(min=1e-10)
    pcs = (scores / sc_std).cpu()

    ratios = pca.explained_variance_ratio_
    assert ratios is not None  # set by fit()
    var_explained = ratios.cpu()

    if verb >= 2:
        cumvar = var_explained.cumsum(0)
        print(
            f"  PC variance explained (cumulative): "
            f"{', '.join(f'{v * 100:.1f}%' for v in cumvar.tolist())}"
        )

    return pcs, var_explained


def motsim_regressors(
    reference: Tensor,
    matrices_vox: Tensor,
    spec: MotSimSpec,
    device: torch.device,
    *,
    config=None,
    interp: str = "cubic",
    mask: Tensor | None = None,
    dilate: int = 2,
    header_info: dict | None = None,
    keep_sims: bool = False,
    verb: int = 1,
) -> MotSimResult:
    """Forward sim → (optional) backward sim → masked temporal PCA.

    The one entry point ffs_motsim and ffs_moco both call, so the regressors do
    not depend on which tool produced them.

    Args:
        reference: (nz, ny, nx) reference EPI (the registration base).
        matrices_vox: (nt, 4, 4) voxel-space transforms from ffs_moco.
        spec: which variant and how many components.
        device: torch device.
        config: MocoConfig of the real correction; its resampling settings drive
            the forward sim and its full settings the backward re-registration.
        interp: fallback interpolation when ``config`` is None.
        mask: optional brain mask; an automask is derived from the reference
            otherwise. Either way it is dilated by ``dilate``.
        dilate: outward mask dilation in voxels.
        header_info: NIfTI header dict for the backward pass.
        keep_sims: retain the simulated 4D series on the result (for -save_sim).
        verb: verbosity.
    """
    brain = build_motsim_mask(reference, mask=mask, dilate=dilate, device=device).cpu()

    forward = run_forward_sim(
        reference, matrices_vox, device, interp=interp, verb=verb, config=config
    )

    backward = None
    if spec.needs_backward:
        backward = run_backward_sim(
            forward,
            reference,
            device,
            interp=interp,
            verb=verb,
            config=config,
            header_info=header_info,
        )

    if spec.variant == "forward":
        pca_input, pca_mask = forward, brain
    elif spec.variant == "backward":
        pca_input, pca_mask = backward, brain
    else:
        # Spatial concatenation, as in the paper: one PCA over both series at once,
        # so the components are whatever explains the most across the pair.
        assert backward is not None
        pca_input = torch.cat([forward, backward], dim=1)
        pca_mask = expand_mask_both(brain)

    pcs, var_explained = extract_pcs(pca_input, pca_mask, spec.n_pcs, verb=verb, device=device)

    return MotSimResult(
        pcs=pcs,
        var_explained=var_explained,
        spec=spec,
        mask=brain,
        forward=forward if keep_sims else None,
        backward=backward if keep_sims else None,
    )


def save_1d(
    pcs: Tensor,
    var_explained: Tensor,
    path: str,
    variant: str,
    n_vols: int,
) -> None:
    """Write PCs as AFNI-style .1D file."""
    n_pcs = pcs.shape[1]
    cum = float(var_explained.sum()) * 100.0
    with open(path, "w") as f:
        f.write("# MotSim regressors (Patriat, Reynolds & Birn 2017, NeuroImage 144:74-82)\n")
        f.write(f"# Variant: {variant}, {n_vols} volumes, {n_pcs} PCs\n")
        f.write(
            f"# Variance of the simulated series explained: "
            f"{' '.join(f'{v * 100:.2f}%' for v in var_explained.tolist())}"
            f" (cumulative {cum:.2f}%)\n"
        )
        for row in pcs.cpu().numpy():
            f.write("  ".join(f"{v: .6f}" for v in row) + "\n")


def expand_mask_both(mask: Tensor) -> Tensor:
    """Expand mask for 'both' variant: repeat along z (dim 0) for concat."""
    return torch.cat([mask, mask], dim=0)
