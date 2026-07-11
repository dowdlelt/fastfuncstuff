"""CLI for GPU multi-echo T2*/S0 estimation and optimal combination.

Command: ffs_t2smap (registered as entry point in pyproject.toml)

A GPU-first rebuild of tedana's ``t2smap``: estimates voxel-wise S0 and T2* from a
monoexponential decay, optimally combines echoes, and adds a leave-one-echo-out (LOEO)
QC plus an optional robust (Tukey-biweight) refit that down-weights echoes which disagree
with the decay model.

Usage:
    ffs_t2smap -input echo1.nii.gz echo2.nii.gz echo3.nii.gz \\
        -tes 0.015 0.039 0.063 -prefix sub01_ \\
        -fittype curvefit -device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import add_verbose_arg, parse_device_arg, spinner
from fastfuncstuff.processing import multiecho as me
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.utils import configure_torch_backends


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_t2smap",
        description="Multi-echo T2*/S0 fitting, optimal combination, and "
        "leave-one-echo-out QC (GPU, tedana t2smap compatible).",
    )
    parser.add_argument(
        "-input",
        "-d",
        nargs="+",
        required=True,
        help="Per-echo 4D files in ascending echo order (one file per echo time).",
    )
    parser.add_argument(
        "-tes",
        "-e",
        nargs="+",
        type=float,
        required=True,
        help="Echo times in seconds (BIDS). Millisecond values are accepted and "
        "converted automatically.",
    )
    parser.add_argument(
        "-prefix",
        required=True,
        help="Output prefix/stem. Outputs are <prefix>T2starmap.nii.gz, etc.",
    )
    parser.add_argument("-mask", default=None, help="Binary brain mask (strongly recommended).")
    parser.add_argument(
        "-automask",
        action="store_true",
        help="Derive a brain mask from the first echo instead of -mask.",
    )
    parser.add_argument(
        "-fittype",
        default="loglin",
        choices=["loglin", "curvefit", "robust"],
        help="loglin (fast log-linear), curvefit (batched nonlinear LM), or robust "
        "(curvefit with LOEO-driven echo down-weighting).",
    )
    parser.add_argument(
        "-fitmode",
        default="all",
        choices=["all", "ts"],
        help="all = one T2*/S0 per voxel; ts = per voxel and per timepoint (4D maps).",
    )
    parser.add_argument(
        "-fit_all_timepoints",
        "-fit-all-timepoints",
        action="store_true",
        help="With -fitmode all, fit jointly across every timepoint (tedana-exact) "
        "instead of the per-echo temporal mean (default, faster).",
    )
    parser.add_argument(
        "-combmode",
        default="t2s",
        choices=["t2s", "paid"],
        help="Echo combination: t2s (Posse 1999) or paid (Poser 2006).",
    )
    parser.add_argument(
        "-masktype",
        nargs="+",
        default=["dropout"],
        choices=["dropout", "decay", "none"],
        help="Adaptive-mask method(s) for counting good echoes per voxel.",
    )
    parser.add_argument(
        "-no_qc",
        "-no-qc",
        action="store_true",
        help="Skip the leave-one-echo-out QC maps.",
    )
    parser.add_argument(
        "-verbose_out",
        "-verbose-out",
        action="store_true",
        help="Also write curvefit variance/covariance maps.",
    )
    parser.add_argument("-device", default=None, help="PyTorch device (cuda, cpu, mps).")
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def _tes_to_ms(tes: list[float]) -> torch.Tensor:
    """Convert echo times to milliseconds, mirroring tedana's seconds-vs-ms heuristic."""
    t = torch.tensor(tes, dtype=torch.float32)
    if torch.all((t > 0) & (t < 1)):  # all in (0, 1) => seconds per BIDS
        t = t * 1000.0
    return t


def _unmask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scatter masked samples back to the spatial grid.

    Args:
        values: (V,) for a 3D map, (V, K) for a K-brick stack, or (V, T) timeseries.
        mask: (nz, ny, nx) boolean.

    Returns:
        (nz, ny, nx) for 3D or (K, nz, ny, nx) for a stack/timeseries.
    """
    spatial = mask.shape
    if values.ndim == 1:
        out = torch.zeros(spatial, dtype=values.dtype, device=values.device)
        out[mask] = values
        return out
    k = values.shape[1]
    out = torch.zeros((k, *spatial), dtype=values.dtype, device=values.device)
    out[:, mask] = values.T
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.input) != len(args.tes):
        raise SystemExit(
            f"Got {len(args.input)} echo file(s) but {len(args.tes)} echo time(s); "
            "provide one file per echo time."
        )

    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)
    tes = _tes_to_ms(args.tes).to(device)
    n_echos = len(tes)
    verb = args.verb
    t0 = time.time()

    # Load echoes -> (E, T, nz, ny, nx); a lone 3D volume reads as T=1.
    echoes = []
    header = None
    for path in args.input:
        vol, hdr = load_image(path, device=device)
        if vol.ndim == 3:
            vol = vol.unsqueeze(0)
        echoes.append(vol)
        header = header or hdr
    shapes = {tuple(v.shape) for v in echoes}
    if len(shapes) != 1:
        raise SystemExit(f"All echo files must share a shape; got {shapes}")
    data = torch.stack(echoes, dim=0)  # (E, T, nz, ny, nx)
    spatial = data.shape[2:]

    # Mask -> boolean (nz, ny, nx).
    if args.mask is not None:
        with spinner(f"Loading {Path(args.mask).name}"):
            mvol, _ = load_image(args.mask, device=device)
        if mvol.ndim == 4:
            mvol = mvol[0]
        if tuple(mvol.shape) != tuple(spatial):
            raise SystemExit("-mask grid does not match input grid")
        mask = mvol > 0.5
    elif args.automask:
        from fastfuncstuff.processing.mask import automask

        mask = automask(data[0, 0], device=device, verbose=verb >= 1) > 0
    else:
        # Default: voxels positive in every echo's temporal mean.
        mask = (data.mean(dim=1) > 0).all(dim=0)

    n_vox = int(mask.sum())
    if n_vox == 0:
        raise SystemExit("Mask is empty; nothing to fit.")

    # (E, T, nz, ny, nx) -> masked (V, E, T).
    data_cat = data[:, :, mask].permute(2, 0, 1).contiguous()  # (V, E, T)
    del echoes, data

    if verb >= 1:
        print(f"ffs_t2smap: {n_echos} echoes, {data_cat.shape[2]} TRs, {n_vox} voxels")
        print(f"  TEs (ms) = {[round(float(x), 3) for x in tes]}  device={device}")
        print(f"  fittype={args.fittype} fitmode={args.fitmode} combmode={args.combmode}")

    _, adaptive = me.make_adaptive_mask(data_cat, methods=tuple(args.masktype))

    # --- Fit ---
    if args.fittype == "robust":
        fit = me.fit_robust(
            data_cat,
            tes,
            adaptive,
            fitmode=args.fitmode,
            fit_all_timepoints=args.fit_all_timepoints,
            device=device,
            verbose=verb >= 1,
        )
    else:
        fit = me.fit_decay(
            data_cat,
            tes,
            adaptive,
            fittype=args.fittype,
            fitmode=args.fitmode,
            fit_all_timepoints=args.fit_all_timepoints,
            device=device,
            verbose=verb >= 1,
        )
    t2s, s0 = fit["t2s"], fit["s0"]

    # Floors/ceilings + full vs limited maps (per-voxel "all" maps only).
    if args.fitmode == "all":
        t2s_full, s0_full, t2s_lim, s0_lim = me.modify_t2s_s0_maps(t2s, s0, adaptive, tes)
    else:
        t2s_full, s0_full, t2s_lim, s0_lim = t2s, s0, t2s, s0

    # --- Save core maps ---
    def save(values, name, labels=None):
        save_image(
            _unmask(values, mask),
            f"{args.prefix}{name}.nii.gz",
            header_info=header,
            brick_labels=labels,
        )

    with spinner("Writing core maps"):
        save(t2s_full / 1000.0, "T2starmap")  # ms -> seconds for storage
        save(s0_full, "S0map")
        save(t2s_lim / 1000.0, "desc-limited_T2starmap")
        save(s0_lim, "desc-limited_S0map")
        save(adaptive.to(torch.float32), "adaptive_mask")

        rmse = me.rmse_of_fit(data_cat, tes, adaptive, t2s_full, s0_full)
        rmse = torch.nan_to_num(rmse, nan=0.0)
        save(rmse, "rmse")

        optcom = me.make_optcom(data_cat, tes, adaptive, t2s=t2s_full, combmode=args.combmode)
        save(optcom, "desc-optcom_bold")

        echo_labels = [f"echo-{i + 1}" for i in range(n_echos)]
        if args.fittype == "curvefit" or args.fittype == "robust":
            save(fit["failures"].to(torch.float32), "fit_failures")
            if args.verbose_out:
                save(fit["t2s_var"], "T2starvar")
                save(fit["s0_var"], "S0var")
                save(fit["t2s_s0_covar"], "T2star_S0_covar")
        if args.fittype == "robust":
            save(fit["echo_weight"], "echo_weight", labels=echo_labels)

    # --- Leave-one-echo-out QC ---
    if not args.no_qc:
        mean_echo = data_cat.mean(dim=2)
        avail = me.availability_weights(adaptive, n_echos)
        loeo_fittype = "loglin" if args.fittype == "loglin" else "curvefit"
        resid, resid_frac = me.leave_one_echo_out(
            mean_echo, tes, avail, fittype=loeo_fittype, device=device
        )
        save(torch.nan_to_num(resid, nan=0.0), "loeo_resid", labels=echo_labels)
        save(torch.nan_to_num(resid_frac, nan=0.0), "loeo_resid_frac", labels=echo_labels)

    if verb >= 1:
        print(f"  limited T2* median (s): {(t2s_lim[t2s_lim > 0] / 1000.0).median():.4f}")
        print(f"Saved maps with prefix: {args.prefix}")
        print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
