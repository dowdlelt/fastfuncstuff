#!/usr/bin/env python3
"""Debug PPCA dimensionality parity against FSL MELODIC outputs.

Usage example:
  conda run -n py312_movie_tasks python debug_ppca_melodic.py \
    --input /mnt/belegost/Data/nii_data/PROJECT_TASKFORCE/derivatives/global_proc/ses-01_task-mvpsA_run01_final.ica/filtered_func_data.nii.gz \
    --mask /mnt/belegost/Data/nii_data/PROJECT_TASKFORCE/derivatives/global_proc/ses-01_task-mvpsA_run01_final.ica/filtered_func_data.ica/mask.nii.gz \
    --melodic-dir /mnt/belegost/Data/nii_data/PROJECT_TASKFORCE/derivatives/global_proc/ses-01_task-mvpsA_run01_final.ica/filtered_func_data.ica
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fastfuncsim.ica_tools import (
    _adjust_eigenspectrum_melodic,
    _fsl_first_peak_k,
    _fsl_ppca_est_all,
    apply_polort_projection,
)
from fastfuncsim.pca import PCA


def _norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin <= 1e-15:
        return np.full_like(values, 0.5)
    return (values - vmin) / (vmax - vmin)


def _first_peak(values: np.ndarray, max_k: int) -> int:
    vals = _norm01(values)
    idx = 0
    ceiling = max(0, min(max_k - 1, len(vals) - 1))
    while idx < len(vals) - 1 and vals[idx] < vals[idx + 1] and idx < ceiling:
        idx += 1
    return idx + 1


def _compute_ppca(
    x_vox_t: torch.Tensor,
    n_components: int,
    n_eff: int,
    max_k: int,
) -> dict:
    x_t = x_vox_t.T
    pca = PCA(n_components=n_components, device=torch.device("cpu"))
    pca.fit(x_t)

    ev = pca.explained_variance_.detach().cpu().numpy().astype(np.float64)
    evr = pca.explained_variance_ratio_.detach().cpu().numpy().astype(np.float64)
    cum = np.cumsum(evr)

    adj_ev, max_ev = _adjust_eigenspectrum_melodic(ev, n_eff=n_eff, verbose=False)
    criteria = _fsl_ppca_est_all(adj_ev, N=n_eff)

    out = {
        "n_eigs_raw": int(len(ev)),
        "n_eigs_adj": int(len(adj_ev)),
        "max_ev": int(max_ev),
        "cumvar_61": float(cum[60]) if len(cum) >= 61 else None,
        "cumvar_70": float(cum[69]) if len(cum) >= 70 else None,
        "cumvar_80": float(cum[79]) if len(cum) >= 80 else None,
        "estimators": {
            name: int(_fsl_first_peak_k(values, max_k=max_k)) for name, values in criteria.items()
        },
        "lap_at": {},
    }

    lap_norm = _norm01(criteria["lap"])
    for k in (55, 58, 60, 61, 62, 65, 68, 70, 72, 75):
        if 1 <= k <= len(criteria["lap"]):
            out["lap_at"][str(k)] = {
                "norm": float(lap_norm[k - 1]),
                "raw": float(criteria["lap"][k - 1]),
            }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug MELODIC PPCA parity")
    parser.add_argument("--input", required=True, help="4D input NIfTI")
    parser.add_argument("--mask", required=True, help="3D mask NIfTI")
    parser.add_argument("--melodic-dir", required=True, help="FSL .ica directory")
    parser.add_argument("--polort", type=int, default=0)
    parser.add_argument("--n-eff", type=int, default=15499)
    parser.add_argument("--max-k", type=int, default=237)
    parser.add_argument(
        "--output-json",
        default="ppca_debug_report.json",
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    img = nib.load(args.input)
    mask_img = nib.load(args.mask)

    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0

    x = torch.from_numpy(data[mask, :])
    x = apply_polort_projection(x, polort=args.polort, device=torch.device("cpu"))
    std = x.std(dim=1, unbiased=False)
    nonconst = std > 1e-8
    x_norm = torch.zeros_like(x)
    x_norm[nonconst] = x[nonconst] / std[nonconst].unsqueeze(1)

    ppca_path = Path(args.melodic_dir) / "melodic_PPCA"
    melodic_ppca = np.loadtxt(ppca_path)

    report: dict[str, object] = {
        "input": str(Path(args.input)),
        "mask": str(Path(args.mask)),
        "melodic_dir": str(Path(args.melodic_dir)),
        "n_voxels": int(mask.sum()),
        "n_timepoints": int(data.shape[-1]),
        "voxel_sizes": [float(v) for v in img.header.get_zooms()[:3]],
        "n_eff": int(args.n_eff),
        "melodic_ppca_shape": [int(melodic_ppca.shape[0]), int(melodic_ppca.shape[1])],
        "melodic_ppca_col_peaks": {},
        "our_runs": {},
    }

    for j in range(melodic_ppca.shape[1]):
        col = melodic_ppca[:, j]
        report["melodic_ppca_col_peaks"][f"col_{j + 1}"] = {
            "first_peak": int(_first_peak(col, max_k=args.max_k)),
            "argmax": int(np.argmax(col) + 1),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    n_t = data.shape[-1]
    for n_components in (n_t - 1, n_t):
        key = f"pca_n_components_{n_components}"
        report["our_runs"][key] = _compute_ppca(
            x_vox_t=x_norm,
            n_components=n_components,
            n_eff=args.n_eff,
            max_k=args.max_k,
        )

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("PPCA debug complete")
    print(f"- melodic_PPCA rows: {melodic_ppca.shape[0]}")
    print(
        "- melodic col1 first_peak/argmax: "
        f"{report['melodic_ppca_col_peaks']['col_1']['first_peak']}/"
        f"{report['melodic_ppca_col_peaks']['col_1']['argmax']}"
    )
    print(f"- report: {out_path}")


if __name__ == "__main__":
    main()
