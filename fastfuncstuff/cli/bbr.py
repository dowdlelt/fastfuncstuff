"""``ffs_bbr`` — Boundary-Based Registration refinement of an affine alignment.

Refines an existing EPI→anat affine (e.g. from ``ffs_allineate``) by seating the
anatomical WM/GM boundary onto the EPI's own grey/white edge — the classic
``bbregister`` idea, and the single-element case of Recursive Boundary
Registration ([[Recursive Boundary Registration]], van Mourik 2019). Cross-modal
where affine/intensity cost struggles, because it optimises the *geometric*
boundary rather than intensity similarity.

Given the EPI, one or more ``.aff12.1D`` affines (``anat``→``epi``, base-side→
source-side like an AFNI ``-nwarp`` stack — a whole ``epi→ref→anat`` chain works
with no reference image), and a WM mask in anat space, it composes the stack,
inverts it to cast the WM boundary into EPI space, runs BBR, then folds the
residual transform back in and writes (the nonlinear local distortion an affine
can't fix is the job of ``ffs_rbr``):

  * ``{prefix}_epi2anat.aff12.1D`` / ``{prefix}_anat2epi.aff12.1D`` — refined
    affines (AFNI DICOM ``.aff12.1D``), directly usable by ``ffs_allineate
    -1Dmatrix_apply`` / ``3dAllineate``.
  * ``{prefix}_initial_epi_in_anat.nii.gz`` / ``{prefix}_epi_in_anat.nii.gz`` —
    the EPI on the anat grid with the *original* vs *refined* affine (before/after
    the BBR refinement, for direct comparison).
  * ``{prefix}_wm_in_epi.nii.gz`` — the refined WM mask on the EPI grid (overlay
    on the EPI to check the boundary sits on the grey/white edge).
  * ``{prefix}_anat_in_epi.nii.gz`` — the structural on the EPI grid (with
    ``-anat``).

Method: ``processing/bbr.py``. This refines the *affine*; local EPI distortion
that no affine can fix is the job of the recursive (octree) stage — future work.

Assumes the WM mask (and optional ``-anat``) share the anat grid, and that the
matrix was saved with anat as base / EPI as source. Rotations, scales and the
sampling offset are treated in voxel space, so ~isotropic EPI is assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsHelpFormatter
from fastfuncstuff.cli_utils import add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing.affine import apply_affine, load_matrix_chain, save_matrix_1D
from fastfuncstuff.processing.bbr import (
    MODE_FREE_PARAMS,
    auto_polarity,
    bbr_cost,
    correct_sign_fraction,
    extract_boundary_normals,
    extract_edge_normals,
    gradient_field,
    identity_params,
    ngf_cost,
    optimize_bbr,
)
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.tissue import (
    build_tissue_design,
    tissue_projector,
    tissue_synthesis_cost,
)
from fastfuncstuff.utils import REGISTRATION_TF32


def _normalize_targets(tokens: list[str]) -> set[str]:
    """Expand the -target tokens into a set; 'both' is a legacy alias for wm+edges."""
    out: set[str] = set()
    for t in tokens:
        if t == "both":
            out |= {"wm", "edges"}
        else:
            out.add(t)
    return out


def _ribbon_wm(lh, rh, device):
    """Union L/R FreeSurfer ribbons into a binary WM mask (anat space).

    Standard ``ribbon.mgz`` labels 2/41 as WM; if a file is binary (no such
    labels) its whole extent is taken as the boundary region. Returns
    ``(wm_mask, header)`` on the given device.
    """
    vols = []
    hdr = None
    for pth in (lh, rh):
        if pth is None:
            continue
        v, hdr = load_image(pth, device=device)
        wm_lab = ((v == 2) | (v == 41)).to(torch.float32)
        vols.append(wm_lab if wm_lab.any() else (v > 0).to(torch.float32))
    if not vols:
        raise SystemExit("-lh_ribbon/-rh_ribbon: at least one ribbon file is required")
    wm = vols[0]
    for extra in vols[1:]:
        wm = ((wm + extra) > 0.5).to(torch.float32)
    return wm, hdr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_bbr",
        description=(
            "Boundary-Based Registration (BBR) refinement of an EPI→anat affine.\n"
            "\n"
            "Given an EPI, an existing EPI→anat affine (from ffs_allineate), and a WM\n"
            "mask in anat space, this seats the anatomical WM/GM boundary onto the EPI's\n"
            "own grey/white edge and folds the residual transform back into the affine —\n"
            "the classic bbregister idea, robust cross-modally because it optimises a\n"
            "geometric boundary, not intensity similarity. It refines the AFFINE only;\n"
            "for the local (nonlinear) distortion an affine can't fix, run ffs_rbr next."
        ),
        formatter_class=FfsHelpFormatter,
        epilog=(
            "Direction convention:\n"
            "  The .aff12.1D must map anat(base) -> epi(source), i.e. saved by\n"
            "  `ffs_allineate -base <anat> -source <epi> -1Dmatrix_save M`. Pass the\n"
            "  NATIVE (unaligned) EPI to -epi (the affine's source), not a resampled one.\n"
            "\n"
            "Multiple affines (-1Dmatrix a b c ...):\n"
            "  Composed in AFNI -nwarp order, base-side -> source-side (leftmost closest\n"
            "  to anat, rightmost closest to the EPI). An epi->ref->anat alignment is\n"
            "    -1Dmatrix ref2anat.aff12.1D epi2ref.aff12.1D\n"
            "  The reference image itself is NOT needed (composition is in DICOM mm).\n"
            "  The written epi2anat/anat2epi collapse the whole stack + the BBR\n"
            "  refinement into one matrix each.\n"
            "\n"
            "Targets (-target, combine freely):\n"
            "  wm     = WM/GM boundary via signed BBR contrast (classic; wide capture).\n"
            "  edges  = all anatomical edges by gradient direction (NGF; needs -anat).\n"
            "  tissue = dense partial-volume match to soft GM/WM/CSF maps (-*_pve);\n"
            "           sharp well / small capture, so best combined, e.g. `-target wm tissue`.\n"
            "  A WM boundary can come from -wm_mask OR -lh_ribbon/-rh_ribbon (labels 2/41).\n"
            "\n"
            "Outputs (prefix_*): epi2anat.aff12.1D, anat2epi.aff12.1D (refined affines);\n"
            "  initial_epi_in_anat / epi_in_anat (EPI on anat grid, before / after);\n"
            "  wm_in_epi (with a WM boundary), anat_in_epi (with -anat) for overlay on the\n"
            "  EPI; {wm,gm,csf}_pve_in_epi (given the PVEs) — tissue maps refined into\n"
            "  functional space (aCompCor / laminar / partial-volume-correction).\n"
            "\n"
            "Examples:\n"
            "  ffs_allineate -base T1.nii -source epi_mean.nii -prefix epi_al.nii \\\n"
            "                -1Dmatrix_save epi2anat.aff12.1D\n"
            "  ffs_bbr -epi epi_mean.nii -1Dmatrix epi2anat.aff12.1D \\\n"
            "          -wm_mask wm.nii.gz -anat T1.nii -prefix epi_bbr\n"
            "  # high-res: seat WM, then refine with the SPM tissue mixture\n"
            "  ffs_bbr -epi epi_mean.nii -1Dmatrix epi2anat.aff12.1D -wm_mask wm.nii.gz \\\n"
            "          -wm_pve c2.nii -gm_pve c1.nii -csf_pve c3.nii \\\n"
            "          -target wm tissue -prefix epi_bbr\n"
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
        "(AFNI -nwarp order). One matrix is the common case; give a stack to "
        "compose (e.g. `ref2anat.aff12.1D epi2ref.aff12.1D`). No reference image needed.",
    )
    req.add_argument(
        "-wm_mask",
        default=None,
        help="WM (or WM/GM boundary) mask in anat space. Required for -target wm unless "
        "-lh_ribbon/-rh_ribbon are given (which build it). Not needed for edges/tissue only.",
    )
    req.add_argument("-prefix", required=True, help="Output prefix for all written files")

    opt = p.add_argument_group("boundary / optional inputs")
    opt.add_argument(
        "-anat",
        default=None,
        help="Structural volume (anat grid). Written as prefix_anat_in_epi for overlay, and "
        "REQUIRED for -target edges (its gradient supplies the edge targets).",
    )
    opt.add_argument(
        "-target",
        nargs="+",
        default=["wm"],
        choices=["wm", "edges", "tissue", "both"],
        metavar="T",
        help="What to register to — one or more, summed (default wm). "
        "wm = WM/GM boundary via signed BBR contrast (classic). "
        "edges = ALL anatomical edges by gradient DIRECTION (NGF, polarity-agnostic; needs -anat). "
        "tissue = DENSE partial-volume match to soft GM/WM/CSF probability maps (needs "
        "-wm_pve/-gm_pve/-csf_pve; sharp well, best as a refinement — combine with wm). "
        "both = wm + edges (legacy alias). Combine freely, e.g. `-target wm tissue`.",
    )
    opt.add_argument(
        "-lh_ribbon",
        default=None,
        help="Left FreeSurfer ribbon (anat space). With -rh_ribbon, unioned into the WM mask "
        "(labels 2/41 = WM; a binary ribbon is taken as the boundary). Alternative to -wm_mask.",
    )
    opt.add_argument("-rh_ribbon", default=None, help="Right FreeSurfer ribbon (see -lh_ribbon).")
    opt.add_argument(
        "-wm_pve", default=None, help="WM tissue-probability map (anat space); -target tissue."
    )
    opt.add_argument(
        "-gm_pve", default=None, help="GM tissue-probability map (anat space); -target tissue."
    )
    opt.add_argument(
        "-csf_pve", default=None, help="CSF tissue-probability map (anat space); -target tissue."
    )
    opt.add_argument(
        "-tissue_weight",
        type=float,
        default=1.0,
        help="Weight of the tissue (partial-volume) cost when combined with wm/edges (default 1.0).",
    )
    opt.add_argument(
        "-tissue_nsample",
        type=int,
        default=20000,
        help="Random brain-voxel subsample for the dense tissue cost (default 20000; keeps "
        "high-res fits fast). 0 = use all voxels.",
    )
    opt.add_argument(
        "-edge_percentile",
        type=float,
        default=75.0,
        help="For -target edges/both: keep anat gradients at/above this percentile "
        "(default 75 = strongest 25%% of edges).",
    )
    opt.add_argument(
        "-edge_blur",
        type=float,
        default=1.0,
        help="For -target edges/both: Gaussian blur (voxels) on the anat before its gradient, "
        "to drop sub-EPI-resolution detail the EPI can't see (default 1.0).",
    )
    opt.add_argument(
        "-ngf_weight",
        type=float,
        default=1.0,
        help="For -target both: weight of the edge (NGF) cost relative to the WM cost "
        "(default 1.0).",
    )
    opt.add_argument(
        "-pial_mask",
        default=None,
        help="Filled interior-of-pial (brain) mask in anat space. Adds the GM/CSF (pial) "
        "boundary as a SECOND constraint (shares polarity with WM). Off by default. "
        "Must be filled, not the thin GM ribbon.",
    )
    opt.add_argument(
        "-pial_weight",
        type=float,
        default=0.5,
        help="Weight of pial-boundary points vs WM points, when -pial_mask is given "
        "(default 0.5 = pial half as trusted as WM).",
    )

    fit = p.add_argument_group("fit controls")
    fit.add_argument(
        "-mode",
        default="rigid",
        choices=sorted(MODE_FREE_PARAMS),
        help="Free degrees of freedom (default: rigid). "
        "rigid=6-DoF rot+trans (like bbregister); shift=trans only; "
        "similarity=rigid+scale (9); pe=translate+scale on the PE(y) axis; "
        "pe_shift=PE translate only. Use pe/pe_shift if the residual is pure "
        "phase-encode distortion.",
    )
    fit.add_argument(
        "-polarity",
        default="auto",
        choices=["auto", "wm_bright", "wm_dark"],
        help="WM-vs-GM brightness in the EPI (default: auto). "
        "wm_bright = WM brighter (T1-like); wm_dark = WM darker / GM brighter "
        "(T2*/EPI); auto = pick whichever fits better at the identity.",
    )
    fit.add_argument(
        "-offset",
        type=float,
        default=1.0,
        help="BBR sampling half-distance along the normal, in VOXELS (default 1.0). "
        "Larger = wider capture range but blurrier; ~0.3x cortical thickness is typical.",
    )
    fit.add_argument(
        "-offset_mm",
        type=float,
        default=None,
        help="Sampling half-distance in mm (overrides -offset; converted via mean EPI voxel size).",
    )
    fit.add_argument(
        "-coarse_range",
        type=float,
        default=4.0,
        help="Half-width (voxels) of the pre-Adam coarse translation search that establishes "
        "capture range (default 4.0; set 0 to skip). Should exceed the expected residual shift.",
    )
    fit.add_argument("-iters", type=int, default=300, help="Adam iteration cap (default 300)")
    fit.add_argument(
        "-lr",
        type=float,
        default=0.2,
        help="Adam starting learning rate (default 0.2), cosine-annealed to 0 over -iters "
        "with best-params kept. The dense -target tissue cost has a sharp, narrow well — if "
        "it barely moves (params stuck at ~lr-sized steps), lower this (e.g. 0.05) and/or "
        "raise -iters so it can settle.",
    )
    fit.add_argument(
        "-device", default=None, help="Compute device: cuda / mps / cpu (default: auto-detect)"
    )
    add_verbose_arg(p)
    return p.parse_args(argv)


def _voxel_sizes(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    verb = args.verb

    device = setup_device(args.device, tf32=REGISTRATION_TF32)
    if verb:
        print(f"ffs_bbr: device={device}")

    targets = _normalize_targets(args.target)

    # ── Load EPI ──
    with spinner(f"Loading {Path(args.epi).name}"):
        epi, epi_hdr = load_image(args.epi, device=device)
    if epi.ndim == 4:
        epi = epi.mean(dim=0)  # a lone reference; average a 4D input
    epi_aff = epi_hdr["affine"]

    # ── WM mask: an explicit mask, or built from the FreeSurfer ribbon(s) ──
    wm = wm_hdr = None
    if args.wm_mask is not None:
        with spinner(f"Loading {Path(args.wm_mask).name}"):
            wm, wm_hdr = load_image(args.wm_mask, device=device)
    elif args.lh_ribbon or args.rh_ribbon:
        wm, wm_hdr = _ribbon_wm(args.lh_ribbon, args.rh_ribbon, device)

    # ── Validate the requested targets have their inputs ──
    if "wm" in targets and wm is None:
        raise SystemExit("-target wm needs a WM boundary: give -wm_mask or -lh_ribbon/-rh_ribbon")
    if "edges" in targets and args.anat is None:
        raise SystemExit("-target edges requires -anat (its gradient supplies the edge targets)")
    tissue_paths = [(args.wm_pve, "wm"), (args.gm_pve, "gm"), (args.csf_pve, "csf")]
    tissue_paths = [(p, n) for p, n in tissue_paths if p is not None]
    if "tissue" in targets and len(tissue_paths) < 2:
        raise SystemExit("-target tissue needs ≥2 of -wm_pve / -gm_pve / -csf_pve")

    # ── Anat-space reference grid (base of the affine, output grid): WM mask if we
    #    have one, else a PVE / anat (edges/tissue-only runs). ──
    if wm_hdr is not None:
        assert wm is not None
        anat_ref_hdr, anat_ref_shape = wm_hdr, tuple(wm.shape)
    else:
        ref_path = args.wm_pve or args.gm_pve or args.anat or args.csf_pve
        ref_img, anat_ref_hdr = load_image(ref_path, device=device)
        anat_ref_shape = tuple(ref_img.shape)
    anat_aff = anat_ref_hdr["affine"]

    # ── Affine: anat→epi in voxel space (A), composing the -1Dmatrix stack.
    #    Invert to cast anat masks into EPI space. ──
    A = load_matrix_chain(args.matrix, base_affine=anat_aff, source_affine=epi_aff).to(
        device=device, dtype=torch.float64
    )
    if verb and len(args.matrix) > 1:
        print(f"composed {len(args.matrix)} affines (base-side→source-side)")
    A_inv = torch.linalg.inv(A)  # epi→anat voxel map

    def anat_to_epi(vol: torch.Tensor, mat_inv: torch.Tensor) -> torch.Tensor:
        # apply_affine(source, M, out_shape): M maps out(base) voxel → source voxel.
        return apply_affine(vol, mat_inv.to(torch.float32), tuple(epi.shape), zero_outside=True)

    # ── Offset (voxels) ──
    offset = args.offset
    if args.offset_mm is not None:
        offset = args.offset_mm / float(_voxel_sizes(epi_aff).mean())
        if verb:
            print(f"offset: {args.offset_mm} mm → {offset:.3f} vox")

    # ── WM boundary points (signed BBR contrast), + optional pial ──
    wm_pts = wm_nrm = wm_weight = reverse = None
    if "wm" in targets:
        wm_epi_bin = (anat_to_epi(wm, A_inv) > 0.5).to(torch.float32)
        wm_pts, wm_nrm = extract_boundary_normals(wm_epi_bin, device=device)
        if args.pial_mask is not None:
            with spinner(f"Loading {Path(args.pial_mask).name}"):
                pial, _ = load_image(args.pial_mask, device=device)
            pial_epi = (anat_to_epi(pial, A_inv) > 0.5).to(torch.float32)
            ppts, pnrm = extract_boundary_normals(pial_epi, device=device)
            wm_weight = torch.cat(
                [
                    torch.ones(len(wm_pts), device=device),
                    torch.full((len(ppts),), args.pial_weight, device=device),
                ]
            )
            wm_pts = torch.cat([wm_pts, ppts], dim=0)
            wm_nrm = torch.cat([wm_nrm, pnrm], dim=0)
        if args.polarity == "auto":
            reverse = auto_polarity(epi, wm_pts, wm_nrm, offset=offset, weight=wm_weight)
        else:
            reverse = args.polarity == "wm_dark"
        if verb:
            print(f"WM points: {len(wm_pts)}; polarity {'wm_dark' if reverse else 'wm_bright'}")

    # ── Anat edge points (NGF, gradient-direction match on all edges) ──
    edge_pts = edge_nrm = epi_grad = None
    if "edges" in targets:
        with spinner(f"Loading {Path(args.anat).name}"):
            anat_img, _ = load_image(args.anat, device=device)
        anat_epi = anat_to_epi(anat_img, A_inv)
        edge_mask = anat_epi > (0.05 * anat_epi.max())  # skip background edges
        edge_pts, edge_nrm = extract_edge_normals(
            anat_epi,
            mask=edge_mask,
            blur=args.edge_blur,
            percentile=args.edge_percentile,
            device=device,
        )
        epi_grad = gradient_field(epi)
        if verb:
            print(f"Edge points: {len(edge_pts)}")

    # ── Tissue (dense partial-volume) design: cast the anat PVEs into EPI space ──
    tissue_coords = tissue_F = tissue_Fpinv = None
    if "tissue" in targets:
        tvols = []
        for pth, _name in tissue_paths:
            with spinner(f"Loading {Path(pth).name}"):
                v, _ = load_image(pth, device=device)
            tvols.append(anat_to_epi(v, A_inv))
        n_sample = None if args.tissue_nsample <= 0 else args.tissue_nsample
        tissue_coords, tissue_F = build_tissue_design(tvols, n_sample=n_sample, device=device)
        tissue_Fpinv = tissue_projector(tissue_F)
        if verb:
            names = "+".join(n for _, n in tissue_paths)
            print(f"Tissue samples: {len(tissue_coords)} ({names})")

    # ── Assemble the cost (pivot spans all active sample sets) ──
    all_pts = torch.cat([p for p in (wm_pts, edge_pts, tissue_coords) if p is not None], dim=0)
    pivot = all_pts.mean(dim=0)

    def cost_fn(params: torch.Tensor) -> torch.Tensor:
        c = params.new_zeros(())
        if wm_pts is not None:
            c = c + bbr_cost(
                epi,
                wm_pts,
                wm_nrm,
                params,
                pivot=pivot,
                offset=offset,
                reverse=bool(reverse),
                weight=wm_weight,
            )
        if edge_pts is not None:
            c = c + args.ngf_weight * ngf_cost(epi_grad, edge_pts, edge_nrm, params, pivot=pivot)
        if tissue_coords is not None:
            c = c + args.tissue_weight * tissue_synthesis_cost(
                epi, tissue_coords, tissue_F, tissue_Fpinv, params, pivot=pivot
            )
        return c

    # ── Optimize ──
    res = optimize_bbr(
        epi,
        all_pts,
        torch.zeros_like(all_pts),  # only used for pivot/dtype; cost_fn drives the fit
        mode=args.mode,
        pivot=pivot,
        cost_fn=cost_fn,
        coarse_range=args.coarse_range,
        iters=args.iters,
        lr=args.lr,
        verbose=bool(verb),
    )
    T = res["matrix"].to(torch.float64)
    disp = res["params"].detach().cpu().numpy()
    qc = ""
    if wm_pts is not None:
        f0 = correct_sign_fraction(
            epi,
            wm_pts,
            wm_nrm,
            identity_params(device),
            pivot=pivot,
            offset=offset,
            reverse=bool(reverse),
        )
        f1 = correct_sign_fraction(
            epi, wm_pts, wm_nrm, res["params"], pivot=pivot, offset=offset, reverse=bool(reverse)
        )
        qc = f"correct-sign {100 * f0:.1f}% → {100 * f1:.1f}%  |  "
    print(
        f"cost {res['init_cost']:.4f} → {res['final_cost']:.4f}  |  {qc}"
        f"params rot={disp[0:3].round(3).tolist()} scale={disp[3:6].round(4).tolist()} "
        f"trans={disp[6:9].round(3).tolist()} (vox)"
    )

    # ── Compose refined affine: A' = T @ A (refine in EPI voxel space) ──
    # Keep the compose/inverse in float64 for accuracy; save_matrix_1D wants f32.
    A_new = T @ A
    A_new_inv = torch.linalg.inv(A_new)

    prefix = args.prefix
    save_matrix_1D(
        A_new.float(), f"{prefix}_epi2anat.aff12.1D", base_affine=anat_aff, source_affine=epi_aff
    )
    save_matrix_1D(
        A_new_inv.float(),
        f"{prefix}_anat2epi.aff12.1D",
        base_affine=epi_aff,
        source_affine=anat_aff,
    )

    saved = [f"{prefix}_epi2anat.aff12.1D", f"{prefix}_anat2epi.aff12.1D"]

    # EPI on the anat grid with the ORIGINAL affine (A) — the before/after pair.
    with spinner("Writing initial_epi_in_anat"):
        initial_epi_in_anat = apply_affine(
            epi, A.to(torch.float32), anat_ref_shape, zero_outside=True
        )
        save_image(
            initial_epi_in_anat, f"{prefix}_initial_epi_in_anat.nii.gz", header_info=anat_ref_hdr
        )
        saved.append(f"{prefix}_initial_epi_in_anat.nii.gz")

    # EPI resampled onto anat grid with the refined affine (A': anat→epi).
    with spinner("Writing epi_in_anat"):
        epi_in_anat = apply_affine(epi, A_new.to(torch.float32), anat_ref_shape, zero_outside=True)
        save_image(epi_in_anat, f"{prefix}_epi_in_anat.nii.gz", header_info=anat_ref_hdr)
        saved.append(f"{prefix}_epi_in_anat.nii.gz")

    # Refined WM mask on the EPI grid (QC overlay on the EPI).
    if wm is not None:
        with spinner("Writing wm_in_epi"):
            wm_in_epi = apply_affine(
                wm, A_new_inv.to(torch.float32), tuple(epi.shape), zero_outside=True
            )
            save_image(wm_in_epi, f"{prefix}_wm_in_epi.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_wm_in_epi.nii.gz")

    if args.anat is not None:
        with spinner(f"Loading {Path(args.anat).name}"):
            anat, _ = load_image(args.anat, device=device)
        with spinner("Writing anat_in_epi"):
            anat_in_epi = apply_affine(
                anat, A_new_inv.to(torch.float32), tuple(epi.shape), zero_outside=True
            )
            save_image(anat_in_epi, f"{prefix}_anat_in_epi.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_anat_in_epi.nii.gz")

    # Tissue probability maps in functional space, distortion-refined by A' — the
    # aCompCor / laminar / partial-volume-correction deliverable (affine-corrected;
    # ffs_rbr adds the nonlinear part).
    for pth, name in tissue_paths:
        with spinner(f"Writing {name}_pve_in_epi"):
            v, _ = load_image(pth, device=device)
            v_epi = apply_affine(
                v, A_new_inv.to(torch.float32), tuple(epi.shape), zero_outside=True
            )
            save_image(v_epi, f"{prefix}_{name}_pve_in_epi.nii.gz", header_info=epi_hdr)
            saved.append(f"{prefix}_{name}_pve_in_epi.nii.gz")

    if verb:
        print("Saved: " + ", ".join(saved))


if __name__ == "__main__":
    sys.exit(main())
