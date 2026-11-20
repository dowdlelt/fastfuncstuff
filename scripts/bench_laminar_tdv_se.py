#!/usr/bin/env python3
"""External SE referee for continuous laminar deveining on aligned ds003427.

Uses derivatives/autoproc/sub-XX_nordicA: aligned, frame-fixed GE and SE on the
same grid; depth.nii.gz was rebuilt with presurfer_al (see scripts/mk_maps.sh).
SE supplies coverage/brightness eligibility and external outcomes. No SE response
chooses physiological parameters, lambda, or responsive thresholds.
Reports matched-voxel temporal correlation and spatial block-beta agreement.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch

from fastfuncstuff.cli_utils import setup_device
from fastfuncstuff.design.matrices import build_task_design
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.glm.xval import project_out_nuisance_per_run
from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.laminar.pipeline import reference_p0
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.laminar_tdv import depth_patches, devein_volume_runs


def detrend(runs, tr):
    n = runs[0].shape[-1]
    psc = torch.cat([100 * (r / r.mean(-1, keepdim=True) - 1) for r in runs], -1)
    nuis = [construct_polynomial_matrix(n, 2, torch.device("cpu")) for _ in runs]
    out, _ = project_out_nuisance_per_run(
        psc,
        torch.empty(n * len(runs), 0),
        nuis,
        [i * n for i in range(len(runs))],
        device=torch.device("cpu"),
    )
    return out.reshape(len(psc), len(runs), n).permute(1, 0, 2).numpy()


def tcorr(x, y):
    x, y = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    return (x * y).sum(-1) / np.maximum(
        np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-12
    )


def evaluate(
    subject,
    base,
    outdir,
    device,
    targets,
    strengths,
    depths,
    save,
    s_d=0.4,
    transfer_mode="dynamic",
):
    begin = time.time()
    d = base / f"sub-{subject}_nordicA"
    dep, h = load_image(d / "depth.nii.gz")
    gm, _ = load_image(d / "gmfrac.nii.gz")
    brain, _ = load_image(d / "stage10.mask_brain.nii.zst")
    mask = (
        (gm.numpy() > 0.5)
        & (brain.numpy() > 0.5)
        & np.isfinite(dep.numpy())
        & (dep.numpy() >= 0)
        & (dep.numpy() <= 1)
    )
    full = {}
    headers = []
    for acq in ("Ge", "Se"):
        full[acq] = []
        for run in range(1, 4):
            a, ah = load_image(d / f"stage10.final.task-checkerboard{acq}.run-0{run}.nii.gz")
            if a.shape[1:] != dep.shape or not np.allclose(ah["affine"], h["affine"], atol=1e-4):
                raise ValueError("Dataset grids disagree: no implicit resampling allowed")
            full[acq].append(a)
            headers.append(ah)
    # Stable within-modality brightness checks plus shared interior slab coverage.
    coverage = np.ones(dep.shape, bool)
    for runs in full.values():
        means = torch.stack([a.mean(0) for a in runs]).mean(0).numpy()
        mask &= means > 0.2 * np.percentile(means[brain.numpy() > 0.5], 98)
        for a in runs:
            coverage &= torch.isfinite(a).all(0).numpy() & (a.amin(0).numpy() > 0)
    inner = ndi.minimum_filter(coverage.astype(np.uint8), size=(5, 3, 3), mode="constant") > 0
    mask &= inner
    ge = [a[:, mask].T.contiguous() for a in full["Ge"]]
    se = [a[:, mask].T.contiguous() for a in full["Se"]]
    shape = dep.shape
    del full
    depth = dep.numpy()[mask]
    spacing = np.sqrt(np.square(h["affine"][:3, :3]).sum(0))[::-1]
    labels = depth_patches(mask, dep.numpy(), spacing, 5)
    tr = float(headers[0]["header"].get_zooms()[3])
    T = ge[0].shape[1]
    side = (
        base.parent.parent
        / "ds003427-download"
        / f"sub-{subject}"
        / "func"
        / f"sub-{subject}_task-checkerboard_acq-ge_run-01_bold.json"
    )
    te = json.loads(side.read_text())["EchoTime"]
    spec = ModelSpec(
        N=depths, K=depths, TR=tr, TE=te, p0=replace(reference_p0(), s_d=s_d), device=device
    )
    gep, sep = detrend(ge, tr), detrend(se, tr)
    geavg, seavg = gep.mean(0), sep.mean(0)
    # Identical acquisition block schedule, reference SPMG1(15).
    from fastfuncstuff.design.hrf import get_spm_hrf_with_derivatives

    hrf = get_spm_hrf_with_derivatives(
        microtime_dt=0.1, hrf_duration=32, n_basis=1, device=torch.device("cpu")
    )
    x = build_task_design(
        hrf,
        T,
        [0],
        tr,
        0.1,
        event_onsets=[[np.arange(15.0, 240.0, 30.0)]],
        durations=[15.0],
        device=torch.device("cpu"),
    ).numpy()[:, 0]
    poly = construct_polynomial_matrix(T, 2, torch.device("cpu")).numpy()
    q, _ = np.linalg.qr(poly)
    x = x - q @ (q.T @ x)
    beta = lambda a: a @ x / (x @ x)
    # GE responsiveness is fixed before correction; SE only enters evaluation.
    gb = beta(geavg)
    rss = ((gep - gb[None, :, None] * x) ** 2).sum((0, 2))
    gt = gb / np.sqrt(np.maximum(rss / (3 * T - 10) / (3 * (x @ x)), 1e-20))
    crop = int(np.ceil(60 / tr))
    take = slice(crop, T - crop)
    sb = beta(seavg)
    reports = []
    outdir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        res = devein_volume_runs(
            ge,
            depth,
            labels,
            spec,
            target=target,
            transfer_mode=transfer_mode,
            strengths=tuple(strengths),
            verbose=True,
        )
        corrected = detrend(res.corrected, tr).mean(0)
        smooth = torch.stack(res.smooth_psc).mean(0).numpy()
        rawcorr = tcorr(geavg[:, take], seavg[:, take])
        corrcorr = tcorr(corrected[:, take], seavg[:, take])
        smcorr = tcorr(smooth[:, take], seavg[:, take])
        cb = beta(corrected)
        mb = beta(smooth)
        for selection, selected in [("all", res.valid), ("ge_positive", res.valid & (gt > 3))]:
            for lo, hi in [
                (0.0, 1.0),
                (0.0, 0.2),
                (0.2, 0.4),
                (0.4, 0.6),
                (0.6, 0.8),
                (0.8, 1.0001),
            ]:
                m = selected & (depth >= lo) & (depth < hi)
                if m.sum() < 5:
                    continue

                def spatial(a, selected=m):
                    return float(np.corrcoef(a[selected], sb[selected])[0, 1])

                report = dict(
                    subject=subject,
                    target=target,
                    selection=selection,
                    lo=lo,
                    hi=min(hi, 1.0),
                    n=int(m.sum()),
                    temporal_raw=float(np.median(rawcorr[m])),
                    temporal_corrected=float(np.median(corrcorr[m])),
                    temporal_smooth=float(np.median(smcorr[m])),
                    temporal_delta=float(np.median((corrcorr - rawcorr)[m])),
                    improved=float(np.mean(corrcorr[m] > rawcorr[m])),
                    spatial_raw=spatial(gb),
                    spatial_corrected=spatial(cb),
                    spatial_smooth=spatial(mb),
                    beta_retained=float(
                        np.median(cb[m] / np.where(abs(gb[m]) > 1e-5, gb[m], np.nan))
                    )
                    if selection == "ge_positive"
                    else None,
                    noise_gain=float(np.median(res.noise_gain[m])),
                )
                reports.append(report)
                print(json.dumps(report), flush=True)
        diagnostics = np.stack(
            [
                res.regularization,
                res.cv_r2,
                res.valid,
                rawcorr,
                corrcorr,
                smcorr,
                corrcorr - rawcorr,
            ]
        )
        mapvol = torch.zeros(len(diagnostics), *shape)
        mapvol[:, mask] = torch.from_numpy(diagnostics.astype(np.float32))
        save_image(
            mapvol,
            outdir / f"{subject}_{target}_scores.nii.gz",
            h,
            brick_labels=[
                "regularization",
                "ge_cv_r2",
                "valid",
                "r_raw_se",
                "r_corrected_se",
                "r_smooth_se",
                "delta_r",
            ],
        )
        if save:
            for name, arr in [
                ("ge", geavg),
                ("se", seavg),
                ("corrected", corrected),
                ("smooth", smooth),
            ]:
                out = torch.zeros(T, *shape)
                out[:, mask] = torch.from_numpy(arr.T.copy())
                save_image(out, outdir / f"{subject}_{target}_{name}_mean_psc.nii.gz", headers[0])
        np.savez_compressed(
            outdir / f"{subject}_{target}_voxel_scores.npz",
            depth=depth,
            valid=res.valid,
            ge_t=gt,
            raw_r=rawcorr,
            corrected_r=corrcorr,
            smooth_r=smcorr,
            raw_beta=gb,
            corrected_beta=cb,
            se_beta=sb,
            labels=labels,
            regularization=res.regularization,
        )
    (outdir / f"{subject}_report.json").write_text(
        json.dumps(
            dict(
                rows=reports,
                seconds=time.time() - begin,
                source=str(d),
                te=te,
                depths=depths,
                s_d=s_d,
                transfer_mode=transfer_mode,
                strengths=strengths,
                evaluation_crop_seconds=60,
            ),
            indent=2,
        )
    )
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-dataset",
        type=Path,
        required=True,
        help="Directory with prepared, same-grid GE, SE, and depth derivatives for ds003427",
    )
    p.add_argument("-output", type=Path, required=True)
    p.add_argument("-subjects", nargs="+", default=["03", "04", "05", "07"])
    p.add_argument("-targets", nargs="+", default=["local", "micro"])
    p.add_argument("-strengths", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument("-depths", type=int, default=5)
    p.add_argument("-s_d", type=float, default=0.4)
    p.add_argument("-transfer_mode", choices=("dynamic", "static"), default="dynamic")
    p.add_argument("-device", default="cpu,4")
    p.add_argument("-save_means", action="store_true")
    args = p.parse_args()
    dev = setup_device(args.device)
    rows = []
    for s in args.subjects:
        rows.extend(
            evaluate(
                s,
                args.dataset,
                args.output,
                dev,
                args.targets,
                args.strengths,
                args.depths,
                args.save_means,
                args.s_d,
                args.transfer_mode,
            )
        )
    (args.output / "report.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
