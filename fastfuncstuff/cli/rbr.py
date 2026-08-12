"""``ffs_rbr`` — Recursive Boundary Registration: nonlinear boundary correction.

Fixes the *local* geometric distortion (dominantly phase-encode-axis for EPI)
that an affine cannot, by fitting a smooth multi-resolution control-point warp
that seats the anatomical WM boundary on the EPI's grey/white edge everywhere.
The nonlinear successor to ``ffs_bbr`` — run ``ffs_bbr`` first and feed its
refined ``epi2anat.aff12.1D`` here so the residual is small and within the BBR
capture range. See [[Recursive Boundary Registration]] (van Mourik 2019); method
in ``processing/rbr.py``.

Inputs mirror ``ffs_bbr``: the native EPI, the anat→EPI ``.aff12.1D`` (anat base,
EPI source), and a WM mask in anat space. Outputs mirror ``ffs_bbr`` too, with the
nonlinear warp folded into the anat-space EPI (and 0 outside the EPI FoV, so a
small-FoV EPI is never stretched into the empty region):

  * ``{prefix}_epi_in_anat.nii.gz`` — EPI on the anat grid, affine **and**
    nonlinear correction applied.
  * ``{prefix}_initial_epi_in_anat.nii.gz`` — affine only (before/after pair).
  * ``{prefix}_wm_in_epi.nii.gz`` / ``{prefix}_anat_in_epi.nii.gz`` — the WM
    boundary / structural cast into EPI space (affine only; aligns with the
    undistorted EPI / true anatomy).
  * ``{prefix}_wm_in_epi_warped.nii.gz`` / ``{prefix}_anat_in_epi_warped.nii.gz`` —
    the same casts distorted into the RAW-EPI frame (inverse warp), so a contour
    overlay on the native acquired EPI matches. These "move with" the fit.
  * ``{prefix}_epi_undistorted.nii.gz`` — the EPI in its own space with the local
    distortion removed (analysis-ready; overlays on ``anat_in_epi``).
  * ``{prefix}_warp.nii.gz`` — the composable nonlinear distortion warp (mm, AFNI
    DICOM, EPI grid), saved exactly like the MEDIC/qwarp warps. Catenate it
    source-side after the EPI→anat affine to fold moco/align/affine + distortion
    into a single resample; applied alone to the raw EPI it reproduces
    ``epi_undistorted``. This is the "last warp in the chain".
  * ``{prefix}_pe_disp.nii.gz`` — signed phase-encode displacement (EPI grid).

Boundary points are weighted by local EPI edge strength (reliability weighting),
so flat / filled / dropout regions of the WM mask surface don't drive the fit.
Assumes ~isotropic EPI (displacements/offsets in voxel space).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from fastfuncstuff.cli.bbr import _normalize_targets, _ribbon_wm
from fastfuncstuff.cli_utils import add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing.affine import apply_affine, load_matrix_chain
from fastfuncstuff.processing.bbr import (
    auto_polarity,
    boundary_reliability,
    correct_sign_fraction,
    extract_boundary_normals,
    extract_edge_normals,
    gradient_field,
    identity_params,
)
from fastfuncstuff.processing.io import load_image, save_image, save_warp_field
from fastfuncstuff.processing.rbr import (
    invert_displacement_field,
    optimize_rbr,
    resample_with_affine_field,
)
from fastfuncstuff.processing.tissue import build_tissue_design, tissue_projector
from fastfuncstuff.utils import REGISTRATION_TF32

_AXIS = {"x": 0, "y": 1, "z": 2}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_rbr",
        description=(
            "Recursive Boundary Registration — nonlinear boundary-based distortion\n"
            "correction. Fixes the LOCAL geometric distortion (dominantly phase-encode)\n"
            "that an affine can't, by fitting a smooth control-point warp that seats the\n"
            "anatomical WM boundary on the EPI grey/white edge everywhere.\n"
            "\n"
            "Run ffs_bbr FIRST and pass its refined epi2anat.aff12.1D here, so the anat\n"
            "starts in the closest linear space and the residual stays within the BBR\n"
            "capture range."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Direction / stack convention: identical to ffs_bbr. The .aff12.1D maps\n"
            "  anat(base) -> epi(source); pass the native EPI to -epi. Multiple affines\n"
            "  compose in AFNI -nwarp order (base-side -> source-side), e.g.\n"
            "    -1Dmatrix ref2anat.aff12.1D epi2ref.aff12.1D   (epi->ref->anat)\n"
            "  and need no reference image.\n"
            "\n"
            "Outputs (prefix_*): epi_in_anat / initial_epi_in_anat (EPI on anat grid,\n"
            "  after / before the warp); wm_in_epi (+ anat_in_epi with -anat) affine\n"
            "  casts, plus *_warped versions distorted into the raw-EPI frame for\n"
            "  overlay on the native EPI; epi_undistorted (EPI in its own space,\n"
            "  distortion removed, analysis-ready); warp (composable mm nonlinear warp\n"
            "  for ffs_nwarp, the last warp in the chain); pe_disp (signed PE disp).\n"
            "\n"
            "Typical use (after ffs_bbr):\n"
            "  ffs_rbr -epi epi_mean.nii -1Dmatrix epi_bbr_epi2anat.aff12.1D \\\n"
            "          -wm_mask wm.nii.gz -anat T1.nii -prefix epi_rbr\n"
        ),
    )
    req = p.add_argument_group("required inputs")
    req.add_argument(
        "-epi", required=True, help="Native (unaligned) EPI volume — the affine's source grid"
    )
    req.add_argument(
        "-1Dmatrix",
        dest="matrix",
        required=True,
        nargs="+",
        metavar="AFF",
        help="One or more anat->epi .aff12.1D affines, base-side->source-side "
        "(AFNI -nwarp order); ideally ffs_bbr's refined epi2anat output. "
        "No reference image needed for a stack.",
    )
    req.add_argument(
        "-wm_mask",
        default=None,
        help="WM (or WM/GM boundary) mask in anat space. Required for -target wm unless "
        "-lh_ribbon/-rh_ribbon are given. Not needed for edges/tissue only.",
    )
    req.add_argument("-prefix", required=True, help="Output prefix for all written files")
    req.add_argument(
        "-anat",
        default=None,
        help="Structural volume (anat grid). Written as prefix_anat_in_epi for overlay, and "
        "REQUIRED for -target edges (its gradient supplies the edge targets).",
    )

    tgt = p.add_argument_group("target controls")
    tgt.add_argument(
        "-target",
        nargs="+",
        default=["wm"],
        choices=["wm", "edges", "tissue", "both"],
        metavar="T",
        help="What the warp seats onto the EPI — one or more, summed (default wm). "
        "wm = WM/GM boundary via BBR contrast. edges = all anatomical edges by gradient "
        "DIRECTION (NGF; needs -anat). tissue = DENSE partial-volume match to soft "
        "GM/WM/CSF maps (-*_pve). both = wm+edges (legacy). Combine, e.g. `-target wm tissue`.",
    )
    tgt.add_argument(
        "-lh_ribbon",
        default=None,
        help="Left FreeSurfer ribbon (anat space). With -rh_ribbon, unioned into the WM mask "
        "(labels 2/41). Alternative to -wm_mask.",
    )
    tgt.add_argument("-rh_ribbon", default=None, help="Right FreeSurfer ribbon (see -lh_ribbon).")
    tgt.add_argument(
        "-wm_pve", default=None, help="WM tissue-probability map (anat); -target tissue."
    )
    tgt.add_argument(
        "-gm_pve", default=None, help="GM tissue-probability map (anat); -target tissue."
    )
    tgt.add_argument(
        "-csf_pve", default=None, help="CSF tissue-probability map (anat); -target tissue."
    )
    tgt.add_argument(
        "-tissue_weight",
        type=float,
        default=1.0,
        help="Weight of the tissue (partial-volume) term when combined (default 1.0).",
    )
    tgt.add_argument(
        "-tissue_nsample",
        type=int,
        default=20000,
        help="Random brain-voxel subsample for the dense tissue term (default 20000; 0 = all).",
    )
    tgt.add_argument(
        "-edge_percentile",
        type=float,
        default=75.0,
        help="For -target edges: keep anat gradients at/above this percentile "
        "as edge targets (default 75 = strongest quartile).",
    )
    tgt.add_argument(
        "-edge_blur",
        type=float,
        default=1.0,
        help="For -target edges: Gaussian blur (voxels) on the anat before its "
        "gradient (default 1.0), to match EPI smoothness and suppress noise edges.",
    )
    tgt.add_argument(
        "-ngf_weight",
        type=float,
        default=1.0,
        help="Weight of the edge (NGF) term relative to the WM BBR term (default 1.0).",
    )

    warp = p.add_argument_group("warp controls")
    warp.add_argument(
        "-pe_axis",
        default="y",
        choices=["x", "y", "z"],
        help="Phase-encode axis — the direction the warp is allowed to move (default y). "
        "EPI distortion is ~1D along PE.",
    )
    warp.add_argument(
        "-full3d",
        action="store_true",
        help="Let the warp move on ALL three axes (default: PE axis only, which is the "
        "physical prior and resists overfitting).",
    )
    warp.add_argument(
        "-spacings",
        type=float,
        nargs="+",
        default=None,
        help="Control-grid spacings in VOXELS, coarse->fine (e.g. `16 8 4`). "
        "Default: derived from volume size (~dim/4, /8, /16). Smaller = more local/flexible.",
    )
    warp.add_argument(
        "-reg_weight",
        type=float,
        default=1.0,
        help="Membrane smoothness weight (default 1.0). Higher = smoother/stiffer field "
        "(shrinks amplitude); lower = sharper local correction (risks overfitting).",
    )
    warp.add_argument(
        "-offsets",
        type=float,
        nargs="+",
        default=None,
        help="BBR sampling half-distance per level, in VOXELS (scalar or one per spacing; "
        "default 3.0 all levels). Wider on coarse levels widens the capture range.",
    )

    fit = p.add_argument_group("fit controls")
    fit.add_argument(
        "-polarity",
        default="auto",
        choices=["auto", "wm_bright", "wm_dark"],
        help="WM-vs-GM brightness in the EPI (default: auto). wm_bright=T1-like, "
        "wm_dark=T2*/EPI, auto=pick the better-fitting sign at the identity.",
    )
    fit.add_argument(
        "-no_reliability",
        action="store_true",
        help="Disable EPI-gradient reliability weighting. By default each boundary point is "
        "weighted by local EPI edge strength, so flat/filled/dropout regions (where the EPI "
        "has no grey/white edge) don't drive the fit and get filled in from confident neighbours.",
    )
    fit.add_argument(
        "-iters", type=int, default=200, help="Adam iteration cap per level (default 200)"
    )
    fit.add_argument("-lr", type=float, default=0.3, help="Adam learning rate (default 0.3)")
    fit.add_argument(
        "-device", default=None, help="Compute device: cuda / mps / cpu (default: auto-detect)"
    )
    add_verbose_arg(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    verb = args.verb

    device = setup_device(args.device, tf32=REGISTRATION_TF32)
    if verb:
        print(f"ffs_rbr: device={device}")

    targets = _normalize_targets(args.target)

    with spinner(f"Loading {Path(args.epi).name}"):
        epi, epi_hdr = load_image(args.epi, device=device)
    if epi.ndim == 4:
        epi = epi.mean(dim=0)

    # ── WM mask: explicit, or built from the FreeSurfer ribbon(s) ──
    wm = wm_hdr = None
    if args.wm_mask is not None:
        with spinner(f"Loading {Path(args.wm_mask).name}"):
            wm, wm_hdr = load_image(args.wm_mask, device=device)
    elif args.lh_ribbon or args.rh_ribbon:
        wm, wm_hdr = _ribbon_wm(args.lh_ribbon, args.rh_ribbon, device)

    # ── Validate requested targets have their inputs ──
    if "wm" in targets and wm is None:
        raise SystemExit("-target wm needs a WM boundary: give -wm_mask or -lh_ribbon/-rh_ribbon")
    if "edges" in targets and args.anat is None:
        raise SystemExit("-target edges requires -anat (its gradient supplies the edge targets)")
    tissue_paths = [(args.wm_pve, "wm"), (args.gm_pve, "gm"), (args.csf_pve, "csf")]
    tissue_paths = [(p, n) for p, n in tissue_paths if p is not None]
    if "tissue" in targets and len(tissue_paths) < 2:
        raise SystemExit("-target tissue needs ≥2 of -wm_pve / -gm_pve / -csf_pve")

    # ── Anat-space reference (affine base + output grid): WM mask, else a PVE/anat ──
    if wm_hdr is not None:
        assert wm is not None
        anat_ref_hdr, anat_ref_shape = wm_hdr, tuple(wm.shape)
    else:
        ref_img, anat_ref_hdr = load_image(
            args.wm_pve or args.gm_pve or args.anat or args.csf_pve, device=device
        )
        anat_ref_shape = tuple(ref_img.shape)

    # anat→epi voxel affine (composing the -1Dmatrix stack); invert to cast anat
    # masks/PVEs into EPI space.
    A = load_matrix_chain(
        args.matrix, base_affine=anat_ref_hdr["affine"], source_affine=epi_hdr["affine"]
    ).to(device=device, dtype=torch.float64)
    if verb and len(args.matrix) > 1:
        print(f"composed {len(args.matrix)} affines (base-side→source-side)")
    A_inv = torch.linalg.inv(A)

    def anat_to_epi(vol: torch.Tensor) -> torch.Tensor:
        return apply_affine(vol, A_inv.to(torch.float32), tuple(epi.shape), zero_outside=True)

    axes = [0, 1, 2] if args.full3d else [_AXIS[args.pe_axis]]

    # ── WM boundary term (BBR contrast) ──
    pts = torch.empty((0, 3), device=device)
    nrm = torch.empty((0, 3), device=device)
    reverse = False
    weight = None
    frac0 = float("nan")
    if "wm" in targets:
        assert wm is not None
        wm_epi = (anat_to_epi(wm) > 0.5).to(torch.float32)
        pts, nrm = extract_boundary_normals(wm_epi, device=device)
        if args.polarity == "auto":
            reverse = auto_polarity(epi, pts, nrm, offset=2.0, pivot=pts.mean(0))
        else:
            reverse = args.polarity == "wm_dark"
        frac0 = correct_sign_fraction(
            epi, pts, nrm, identity_params(device), offset=2.0, reverse=reverse
        )
        weight = None if args.no_reliability else boundary_reliability(epi, pts)
        if verb:
            low = 0.0 if weight is None else (weight < 0.2).float().mean().item()
            print(
                f"WM points: {len(pts)}; polarity {'wm_dark' if reverse else 'wm_bright'}; "
                f"{100 * low:.1f}% low-signal down-weighted; axes={axes}"
            )

    # ── Anat edge term (NGF, gradient-direction match on all edges) ──
    edge_pts = edge_nrm = epi_grad = edge_w = None
    if "edges" in targets:
        with spinner(f"Loading {Path(args.anat).name}"):
            anat_img, _ = load_image(args.anat, device=device)
        anat_epi = anat_to_epi(anat_img)
        edge_mask = anat_epi > (0.05 * anat_epi.max())  # skip background edges
        edge_pts, edge_nrm = extract_edge_normals(
            anat_epi,
            mask=edge_mask,
            blur=args.edge_blur,
            percentile=args.edge_percentile,
            device=device,
        )
        epi_grad = gradient_field(epi)
        edge_w = None if args.no_reliability else boundary_reliability(epi, edge_pts)
        if verb:
            print(f"Edge points: {len(edge_pts)}")

    # ── Tissue (dense partial-volume) design: cast the anat PVEs into EPI space ──
    tissue_coords = tissue_F = tissue_Fpinv = None
    if "tissue" in targets:
        tvols = []
        for pth, _name in tissue_paths:
            with spinner(f"Loading {Path(pth).name}"):
                v, _ = load_image(pth, device=device)
            tvols.append(anat_to_epi(v))
        n_sample = None if args.tissue_nsample <= 0 else args.tissue_nsample
        tissue_coords, tissue_F = build_tissue_design(tvols, n_sample=n_sample, device=device)
        tissue_Fpinv = tissue_projector(tissue_F)
        if verb:
            print(f"Tissue samples: {len(tissue_coords)} ({'+'.join(n for _, n in tissue_paths)})")

    res = optimize_rbr(
        epi,
        pts,
        nrm,
        axes=axes,
        spacings=args.spacings,
        offsets=args.offsets if args.offsets is not None else 3.0,
        reg_weight=args.reg_weight,
        reverse=reverse,
        weight=weight,
        edge_points=edge_pts,
        edge_normals=edge_nrm,
        grad_field=epi_grad,
        edge_weight=edge_w,
        ngf_weight=args.ngf_weight,
        tissue_coords=tissue_coords,
        tissue_F=tissue_F,
        tissue_Fpinv=tissue_Fpinv,
        tissue_weight=args.tissue_weight,
        iters=args.iters,
        lr=args.lr,
        verbose=bool(verb),
    )
    field = res["field"]
    pe_idx = _AXIS[args.pe_axis]
    max_disp = field.abs().max().item()
    qc = "" if frac0 != frac0 else f"correct-sign start {100 * frac0:.1f}%  |  "
    print(
        f"RBR cost {res['init_cost']:.4f} → {res['final_cost']:.4f}  |  "
        f"{qc}max |disp| {max_disp:.2f} vox"
    )

    prefix = args.prefix
    saved: list[str] = []
    eye = torch.eye(4, device=device, dtype=torch.float64)
    zero_field = torch.zeros_like(field)
    inv_field = invert_displacement_field(field)  # distort undistorted vols → raw frame

    def to_epi_affine(vol: torch.Tensor) -> torch.Tensor:
        return apply_affine(vol, A_inv.to(torch.float32), tuple(epi.shape), zero_outside=True)

    # ── Anat space (EPI resampled onto the anat grid; 0 outside the EPI FoV) ──
    with spinner("Writing epi_in_anat"):
        epi_in_anat = resample_with_affine_field(epi, A, field, anat_ref_shape)
        save_image(epi_in_anat, f"{prefix}_epi_in_anat.nii.gz", header_info=anat_ref_hdr)
        saved.append(f"{prefix}_epi_in_anat.nii.gz")
    with spinner("Writing initial_epi_in_anat"):
        initial = resample_with_affine_field(epi, A, zero_field, anat_ref_shape)
        save_image(initial, f"{prefix}_initial_epi_in_anat.nii.gz", header_info=anat_ref_hdr)
        saved.append(f"{prefix}_initial_epi_in_anat.nii.gz")

    # ── EPI space ──
    # Two coordinate frames live here, so we emit both an affine-only cast (aligns
    # with epi_undistorted / the true anatomy) and a warp-distorted cast (aligns
    # with the RAW acquired EPI, so a contour overlay on the native EPI matches).
    if wm is not None:
        with spinner("Writing wm_in_epi"):
            wm_in_epi = to_epi_affine(wm)
            save_image(wm_in_epi, f"{prefix}_wm_in_epi.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_wm_in_epi.nii.gz")
        with spinner("Writing wm_in_epi_warped"):
            wm_warped = resample_with_affine_field(wm_in_epi, eye, inv_field, tuple(epi.shape))
            save_image(wm_warped, f"{prefix}_wm_in_epi_warped.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_wm_in_epi_warped.nii.gz")
    if args.anat is not None:
        with spinner(f"Loading {Path(args.anat).name}"):
            anat, _ = load_image(args.anat, device=device)
        with spinner("Writing anat_in_epi"):
            anat_in_epi = to_epi_affine(anat)
            save_image(anat_in_epi, f"{prefix}_anat_in_epi.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_anat_in_epi.nii.gz")
        with spinner("Writing anat_in_epi_warped"):
            # Anat distorted into the raw-EPI frame (inverse warp): overlays on the
            # native EPI. This is the version that "moves with" the nonlinear fit.
            anat_warped = resample_with_affine_field(anat_in_epi, eye, inv_field, tuple(epi.shape))
            save_image(anat_warped, f"{prefix}_anat_in_epi_warped.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_anat_in_epi_warped.nii.gz")
    with spinner("Writing epi_undistorted"):
        # The EPI in its own space, distortion removed (edge pulled onto the
        # boundary): sample epi at p + d(p). Analysis-ready; overlays on anat_in_epi.
        epi_undist = resample_with_affine_field(epi, eye, field, tuple(epi.shape))
        save_image(epi_undist, f"{prefix}_epi_undistorted.nii.gz", header_info=epi_hdr)
        saved.append(f"{prefix}_epi_undistorted.nii.gz")

    # ── Composable nonlinear warp (mm, AFNI DICOM, EPI cardinal grid) ──
    # The pull field undistorted(p)=raw(p+d), saved exactly like the MEDIC/qwarp
    # warps so ffs_nwarp / 3dNwarpApply compose it. It is the SOURCE-SIDE (applied
    # first, closest to the raw EPI) nonlinear warp: catenate it after the affine,
    #   ffs_nwarp -source raw_epi -nwarp 'epi2anat.aff12.1D {prefix}_warp.nii.gz' \
    #             -master anat -prefix epi_in_anat
    # folds moco/align/affine-to-anat + this distortion into ONE resample. Applied
    # alone to the raw EPI it reproduces {prefix}_epi_undistorted.
    with spinner("Writing warp"):
        save_warp_field(
            field[0],
            field[1],
            field[2],
            f"{prefix}_warp.nii.gz",
            header_info=epi_hdr,
            units="mm",
        )
        saved.append(f"{prefix}_warp.nii.gz")

    # Tissue maps in functional space (affine + nonlinear undistortion applied) —
    # the aCompCor / laminar / partial-volume-correction deliverable. Cast to EPI
    # via the affine, then pull into the raw-EPI frame by the inverse field so they
    # align with the native EPI (same convention as the _warped anat/WM casts).
    for pth, name in tissue_paths:
        with spinner(f"Writing {name}_pve_in_epi"):
            v, _ = load_image(pth, device=device)
            v_epi = to_epi_affine(v)
            v_warp = resample_with_affine_field(v_epi, eye, inv_field, tuple(epi.shape))
            save_image(v_warp, f"{prefix}_{name}_pve_in_epi.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_{name}_pve_in_epi.nii.gz")

    # PE displacement map on the native EPI grid (QC / scrub like a flow map).
    with spinner("Writing pe_disp"):
        save_image(field[pe_idx], f"{prefix}_pe_disp.nii.gz", header_info=epi_hdr)
        saved.append(f"{prefix}_pe_disp.nii.gz")

    if verb:
        print("Saved: " + ", ".join(saved))


if __name__ == "__main__":
    sys.exit(main())
