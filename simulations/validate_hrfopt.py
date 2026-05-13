#!/usr/bin/env python3
"""
HRFopt validation: can fit_glm_hrf_library_with_xval recover all 20 true HRFs?

Structure
---------
  N_VOXELS = N_HRFS × N_VOXELS_PER_HRF = 20 × 100 = 2000
  All 2000 voxels are fitted in ONE hrfopt call per permutation.
  Voxels i*100 … (i+1)*100 were generated with HRF i.
  Each voxel gets an independent noise draw.
  All voxels share the same onset schedule (realistic: one experiment, one timing).

Design types (by stimulus duration)
-------------------------------------
  duration < 6 s  →  fast event-related, TR=1.0 s, ISI 2–6 s from event offset
  duration ≥ 6 s  →  on-off block design,  TR=2.0 s, 20 s ON / 20 s OFF

Reports
-------
  - Per-HRF recovery rate (top-1 and top-3)
  - Mean rank of true HRF per group
  - 20×20 confusion matrix (summed across permutations)
  - CV-R² improvement over canonical per duration × noise × SNR
  - Design-matrix sanity checks (boxcar shape, regressor amplitude)
  - F-stat vs in-sample R² consistency
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastfuncstuff.design.builder import create_onset_matrix_microtime
from fastfuncstuff.design.hrf import get_hrf_library
from fastfuncstuff.design.hrf_selection import fit_glm_hrf_library_with_xval
from fastfuncstuff.design.matrices import convolve_hrf_microtime
from fastfuncstuff.simulation.noise import generate_ar1_noise, generate_arma_noise
from fastfuncstuff.utils import get_device

# ── global defaults ───────────────────────────────────────────────────────────
N_HRFS = 20
N_VOXELS_PER_HRF = 100          # 100 voxels per true HRF → 2000 total
N_VOXELS = N_HRFS * N_VOXELS_PER_HRF

N_RUNS = 6
MICROTIME_DT = 0.1

# Fast ER parameters (duration < 6 s)
TR_FAST = 1.0
N_TP_FAST = 300                  # 5 min @ 1 s TR
MIN_ISI_FAST = 2.0               # minimum ISI from event offset (seconds)
MAX_ISI_FAST = 6.0

# Block design parameters (duration ≥ 6 s)
TR_BLOCK = 2.0
N_TP_BLOCK = 150                 # 5 min @ 2 s TR
BLOCK_ON_S = 20.0
BLOCK_OFF_S = 20.0

DURATIONS_FAST  = [1.0, 3.0, 5.0]
DURATIONS_BLOCK = [10.0, 15.0]
DURATIONS_ALL   = DURATIONS_FAST + DURATIONS_BLOCK

NOISE_TYPES = ["white", "ar1", "arma", "physiological"]
SNRS        = [0.5, 1.0, 2.0, 4.0]
N_PERMS     = 5                  # independent onset schedules to average over


# ── onset generators ──────────────────────────────────────────────────────────

def _fast_er_onsets(n_runs: int, n_tp: int, tr: float, duration: float,
                    seed: int) -> list[np.ndarray]:
    """Jittered fast event-related onsets for one condition, per run."""
    rng = np.random.default_rng(seed)
    run_dur_s = n_tp * tr
    onsets_per_run = []
    for _ in range(n_runs):
        t = tr * 2                # start two TRs in
        onsets = []
        while True:
            onsets.append(t)
            isi = MIN_ISI_FAST + rng.uniform(0.0, MAX_ISI_FAST - MIN_ISI_FAST)
            t += duration + isi
            if t + duration + MIN_ISI_FAST > run_dur_s - tr * 2:
                break
        onsets_per_run.append(np.array(onsets, dtype=np.float32))
    return onsets_per_run


def _block_onsets(n_runs: int, n_tp: int, tr: float, duration: float,
                  seed: int) -> list[np.ndarray]:  # noqa: ARG001
    """Fixed on-off block onsets for one condition, per run."""
    run_dur_s = n_tp * tr
    block_period = BLOCK_ON_S + BLOCK_OFF_S
    onsets_per_run = []
    for _ in range(n_runs):
        onsets = []
        t = BLOCK_OFF_S / 2          # start mid-off-block so we have baseline
        while t + duration < run_dur_s - BLOCK_OFF_S / 2:
            onsets.append(t)
            t += block_period
        onsets_per_run.append(np.array(onsets, dtype=np.float32))
    return onsets_per_run


def make_all_onsets(duration: float, n_runs: int, n_tp: int, tr: float,
                    seed: int) -> list[list[np.ndarray]]:
    """Return all_onsets[condition][run] for a single condition."""
    if duration < 6.0:
        per_run = _fast_er_onsets(n_runs, n_tp, tr, duration, seed)
    else:
        per_run = _block_onsets(n_runs, n_tp, tr, duration, seed)
    return [per_run]          # one condition


# ── noise generators ──────────────────────────────────────────────────────────

def _make_noise(noise_type: str, n_voxels: int, n_tp: int, target_std: float,
                device: torch.device, seed: int) -> torch.Tensor:
    """(n_voxels, n_tp) noise with given type and std."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if noise_type == "white":
        noise = torch.randn(n_voxels, n_tp, device=device)

    elif noise_type == "ar1":
        rho = float(rng.uniform(0.2, 0.45))
        rows = [generate_ar1_noise(rho, n_tp, device=device) for _ in range(n_voxels)]
        noise = torch.stack(rows)

    elif noise_type == "arma":
        rows = [
            generate_arma_noise([rng.uniform(0.2, 0.4)], [rng.uniform(0.05, 0.2)],
                                n_tp, device=device)
            for _ in range(n_voxels)
        ]
        noise = torch.stack(rows)

    elif noise_type == "physiological":
        # AR(1) + low-frequency drift
        rho = float(rng.uniform(0.4, 0.6))
        rows = [generate_ar1_noise(rho, n_tp, device=device) for _ in range(n_voxels)]
        noise = torch.stack(rows)
        t = torch.linspace(-1, 1, n_tp, device=device)
        drift = (0.3 * t + 0.15 * (3 * t**2 - 1) / 2)
        noise = noise + drift.unsqueeze(0) * torch.randn(n_voxels, 1, device=device).abs()

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # scale to target std per voxel
    cur_std = noise.std(dim=1, keepdim=True).clamp(min=1e-6)
    return noise / cur_std * target_std


# ── data simulation ───────────────────────────────────────────────────────────

def simulate_2000_voxels(
    hrf_library: torch.Tensor,          # (N_HRFS, hrf_len)
    all_onsets: list[list[np.ndarray]], # [cond][run] → np.ndarray
    duration: float,
    noise_type: str,
    snr: float,
    run_starts: list[int],
    n_tp: int,
    tr: float,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """
    Build (N_VOXELS, n_tp) data where voxels [i*100:(i+1)*100] use HRF i.

    Each voxel gets an independent noise draw. Betas are drawn per-voxel from
    a positive normal distribution so no two voxels are identical.
    """
    rng = np.random.default_rng(seed)

    onset_matrix = create_onset_matrix_microtime(
        all_onsets=all_onsets,
        run_starts=run_starts,
        tr=tr,
        n_timepoints=n_tp,
        microtime_dt=MICROTIME_DT,
        stim_durations=[duration],
        device=device,
    )  # (n_tp * bins_per_tr, 1)

    # Pre-compute all 20 design columns (one per HRF)
    designs = []
    for h in range(N_HRFS):
        d = convolve_hrf_microtime(
            onset_matrix, hrf_library[h], n_tp, tr, MICROTIME_DT,
            run_starts=run_starts, device=device,
        )  # (n_tp, 1)
        designs.append(d[:, 0])               # (n_tp,)
    # stack: (N_HRFS, n_tp)
    designs_mat = torch.stack(designs, dim=0)

    # Betas per voxel: (N_VOXELS,) — positive, variable across voxels
    betas_np = np.abs(rng.normal(1.0, 0.3, size=N_VOXELS)).clip(0.3).astype(np.float32)
    betas = torch.from_numpy(betas_np).to(device)

    # Signal per voxel: voxel i uses HRF (i // N_VOXELS_PER_HRF)
    hrf_assignments = torch.arange(N_VOXELS, device=device) // N_VOXELS_PER_HRF
    # signal[v] = betas[v] * designs_mat[hrf_assignments[v]]
    signal = betas.unsqueeze(1) * designs_mat[hrf_assignments]  # (N_VOXELS, n_tp)

    # Noise: per-voxel, scaled to target SNR
    signal_std = signal.std(dim=1).clamp(min=1e-6)
    target_noise_std = (signal_std / snr).mean().item()

    noise = _make_noise(noise_type, N_VOXELS, n_tp, target_noise_std, device, seed + 1)

    # Baseline + signal + noise
    data = 100.0 + signal + noise
    return data.cpu()


# ── run hrfopt ────────────────────────────────────────────────────────────────

def run_hrfopt(
    data: torch.Tensor,
    hrf_library: torch.Tensor,
    all_onsets: list[list[np.ndarray]],
    duration: float,
    run_starts: list[int],
    n_tp: int,
    tr: float,
    device: torch.device,
):
    onset_matrix = create_onset_matrix_microtime(
        all_onsets=all_onsets,
        run_starts=run_starts,
        tr=tr,
        n_timepoints=n_tp,
        microtime_dt=MICROTIME_DT,
        stim_durations=[duration],
        device=device,
    )
    return fit_glm_hrf_library_with_xval(
        data=data,
        onsets=onset_matrix,
        hrf_library=hrf_library,
        tr=tr,
        run_starts=run_starts,
        stim_durations=[duration],
        cv_strategy=1,       # LORO
        metric="cod",
        microtime_dt=MICROTIME_DT,
        polort=None,
        device=device,
        verbose=False,
    )


# ── recovery metrics ──────────────────────────────────────────────────────────

def compute_recovery(r2_all: torch.Tensor, n_hrfs: int | None = None,
                     n_per_hrf: int | None = None):
    """
    r2_all: (N_VOXELS, N_HRFS) CV R² for every HRF at every voxel.
    Returns per-HRF metrics and a confusion matrix.
    """
    if n_hrfs is None:
        n_hrfs = N_HRFS
    if n_per_hrf is None:
        n_per_hrf = N_VOXELS_PER_HRF
    # rank of true HRF per voxel (0 = best)
    n_vox = n_hrfs * n_per_hrf
    hrf_true = torch.arange(n_vox) // n_per_hrf             # (n_vox,)
    r2_true_per_vox = r2_all[torch.arange(n_vox), hrf_true] # (n_vox,)
    rank_per_vox = (r2_all > r2_true_per_vox.unsqueeze(1)).sum(dim=1).float()
    best_hrf = r2_all.argmax(dim=1)                          # (n_vox,) predicted HRF

    # Per-HRF stats
    per_hrf_top1   = torch.zeros(n_hrfs)
    per_hrf_top3   = torch.zeros(n_hrfs)
    per_hrf_rank   = torch.zeros(n_hrfs)
    confusion      = torch.zeros(n_hrfs, n_hrfs, dtype=torch.long)

    for h in range(n_hrfs):
        idx = slice(h * n_per_hrf, (h + 1) * n_per_hrf)
        ranks_h = rank_per_vox[idx]
        best_h  = best_hrf[idx]
        per_hrf_top1[h]  = (ranks_h == 0).float().mean()
        per_hrf_top3[h]  = (ranks_h <= 2).float().mean()
        per_hrf_rank[h]  = ranks_h.mean()
        for v in range(n_per_hrf):
            confusion[h, int(best_h[v].item())] += 1

    return {
        "per_hrf_top1":  per_hrf_top1.numpy(),
        "per_hrf_top3":  per_hrf_top3.numpy(),
        "per_hrf_rank":  per_hrf_rank.numpy(),
        "confusion":     confusion.numpy(),
        "global_top1":   per_hrf_top1.mean().item(),
        "global_top3":   per_hrf_top3.mean().item(),
        "global_rank":   per_hrf_rank.mean().item(),
    }


# ── design-matrix sanity ──────────────────────────────────────────────────────

def validate_designs(hrf_library: torch.Tensor, device: torch.device) -> bool:
    print("\n" + "=" * 72)
    print("PART 1 — Design Matrix Sanity")
    print("=" * 72)
    print(f"  {'dur':>6}  {'type':>8}  {'boxcar':>6}  {'nonzero':>7}  "
          f"{'differ':>6}  {'peak_min':>8}  {'peak_max':>8}  status")

    all_ok = True
    for dur in DURATIONS_ALL:
        if dur < 6.0:
            tr, n_tp, dtype = TR_FAST, N_TP_FAST, "fast_er"
        else:
            tr, n_tp, dtype = TR_BLOCK, N_TP_BLOCK, "block"

        run_starts = list(range(0, n_tp * N_RUNS, n_tp))
        all_onsets = make_all_onsets(dur, N_RUNS, n_tp, tr, seed=0)

        onset_matrix = create_onset_matrix_microtime(
            all_onsets=all_onsets,
            run_starts=run_starts,
            tr=tr,
            n_timepoints=n_tp * N_RUNS,
            microtime_dt=MICROTIME_DT,
            stim_durations=[dur],
            device=device,
        )

        # Check boxcar: all non-zero values should be 1.0
        vals = onset_matrix[:, 0]
        boxcar_ok = bool(vals[vals > 0].sub(1.0).abs().max().item() < 0.01)

        # Check designs across HRFs
        full_tp = n_tp * N_RUNS
        designs = torch.stack([
            convolve_hrf_microtime(onset_matrix, hrf_library[h], full_tp, tr,
                                   MICROTIME_DT, run_starts=run_starts, device=device)[:, 0]
            for h in range(N_HRFS)
        ])  # (N_HRFS, full_tp)

        peaks = designs.abs().max(dim=1).values
        nonzero_ok = bool((peaks > 0.3).all().item())
        differ_ok  = bool(designs.std(dim=0).max().item() > 0.01)
        peak_min, peak_max = peaks.min().item(), peaks.max().item()
        ok = boxcar_ok and nonzero_ok and differ_ok
        if not ok:
            all_ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  {dur:6.1f}  {dtype:>8}  {str(boxcar_ok):>6}  "
              f"{str(nonzero_ok):>7}  {str(differ_ok):>6}  "
              f"{peak_min:8.3f}  {peak_max:8.3f}  {status}")
    return all_ok


# ── main recovery experiment ──────────────────────────────────────────────────

def run_recovery(
    hrf_library: torch.Tensor,
    device: torch.device,
    durations: list[float],
    noise_types: list[str],
    snrs: list[float],
    n_perms: int,
    verbose: bool = False,
) -> list[dict]:
    print("\n" + "=" * 72)
    print("PART 2 — HRF Recovery (2000 voxels, 20 HRFs × 100 voxels)")
    print("=" * 72)
    total = len(durations) * len(noise_types) * len(snrs) * n_perms
    print(f"  Configurations: {len(durations)}dur × {len(noise_types)}noise × "
          f"{len(snrs)}snr × {n_perms}perm = {total} total hrfopt calls\n")

    records = []
    run_n = 0
    t0 = time.time()

    for dur in durations:
        if dur < 6.0:
            tr, n_tp = TR_FAST, N_TP_FAST
            dtype_label = "fast_er"
        else:
            tr, n_tp = TR_BLOCK, N_TP_BLOCK
            dtype_label = "block"

        run_starts = list(range(0, n_tp * N_RUNS, n_tp))
        total_tp = n_tp * N_RUNS

        for noise_type in noise_types:
            for snr in snrs:
                # Accumulate confusion and per-HRF stats over perms
                agg_top1   = np.zeros(N_HRFS)
                agg_top3   = np.zeros(N_HRFS)
                agg_rank   = np.zeros(N_HRFS)
                agg_conf   = np.zeros((N_HRFS, N_HRFS), dtype=int)
                agg_r2imp  = []

                for perm in range(n_perms):
                    run_n += 1
                    seed = perm * 1000 + int(dur * 10) + hash(noise_type) % 100

                    # Fresh onset schedule for each permutation
                    all_onsets = make_all_onsets(dur, N_RUNS, n_tp, tr, seed=seed)

                    # Simulate 2000 voxels
                    data = simulate_2000_voxels(
                        hrf_library, all_onsets, dur, noise_type, snr,
                        run_starts, total_tp, tr, device, seed=seed + 500,
                    )

                    # Run hrfopt
                    try:
                        res = run_hrfopt(data, hrf_library, all_onsets, dur,
                                         run_starts, total_tp, tr, device)
                    except Exception as exc:
                        print(f"  ERROR: dur={dur} noise={noise_type} snr={snr} "
                              f"perm={perm}: {exc}")
                        continue

                    r2_all  = res.xval_r2_all_hrfs.cpu()   # (N_VOXELS, N_HRFS)
                    r2_best = res.xval_r2_best.mean().item()
                    r2_can  = res.xval_r2_canonical.mean().item() \
                              if res.xval_r2_canonical is not None else float("nan")

                    m = compute_recovery(r2_all)
                    agg_top1  += m["per_hrf_top1"]
                    agg_top3  += m["per_hrf_top3"]
                    agg_rank  += m["per_hrf_rank"]
                    agg_conf  += m["confusion"]
                    agg_r2imp.append(r2_best - r2_can)

                    if verbose:
                        print(f"    perm={perm}  top1={m['global_top1']:.1%}  "
                              f"top3={m['global_top3']:.1%}  "
                              f"rank={m['global_rank']:.2f}  "
                              f"ΔR²={r2_best - r2_can:+.4f}")

                agg_top1 /= n_perms
                agg_top3 /= n_perms
                agg_rank /= n_perms

                elapsed = time.time() - t0
                pct = 100 * run_n / total
                print(
                    f"  dur={dur:5.1f}s ({dtype_label:7s})  "
                    f"{noise_type:14s}  snr={snr:.1f}  "
                    f"top1={agg_top1.mean():.1%}  "
                    f"top3={agg_top3.mean():.1%}  "
                    f"rank={agg_rank.mean():.2f}/{N_HRFS-1}  "
                    f"ΔR²={np.mean(agg_r2imp):+.4f}  "
                    f"[{pct:.0f}% {elapsed:.0f}s]"
                )

                if verbose:
                    # Print per-HRF breakdown
                    print(f"    Per-HRF top1: "
                          + " ".join(f"{v:.0%}" for v in agg_top1))
                    # Print worst HRFs
                    worst = np.argsort(agg_top1)[:5]
                    print(f"    Worst HRFs (top-1): "
                          + ", ".join(f"HRF{w}={agg_top1[w]:.0%}" for w in worst))

                records.append({
                    "duration":      dur,
                    "design_type":   dtype_label,
                    "noise_type":    noise_type,
                    "snr":           snr,
                    "global_top1":   float(agg_top1.mean()),
                    "global_top3":   float(agg_top3.mean()),
                    "global_rank":   float(agg_rank.mean()),
                    "per_hrf_top1":  agg_top1,
                    "per_hrf_top3":  agg_top3,
                    "per_hrf_rank":  agg_rank,
                    "confusion":     agg_conf,
                    "mean_r2_improvement": float(np.mean(agg_r2imp)),
                    "n_perms":       n_perms,
                })

    return records


# ── F-stat / R² consistency ───────────────────────────────────────────────────

def check_consistency(hrf_library: torch.Tensor, device: torch.device) -> bool:
    print("\n" + "=" * 72)
    print("PART 3 — F-stat / R² Consistency (truth = HRF 10, dur=3s, AR1, SNR=2)")
    print("=" * 72)

    dur, tr, n_tp = 3.0, TR_FAST, N_TP_FAST
    run_starts = list(range(0, n_tp * N_RUNS, n_tp))
    all_onsets = make_all_onsets(dur, N_RUNS, n_tp, tr, seed=42)

    data = simulate_2000_voxels(
        hrf_library, all_onsets, dur, "ar1", 2.0,
        run_starts, n_tp * N_RUNS, tr, device, seed=9999,
    )

    res = run_hrfopt(data, hrf_library, all_onsets, dur,
                     run_starts, n_tp * N_RUNS, tr, device)

    print(f"  CV R² canonical : {res.xval_r2_canonical.mean().item():.4f}")
    print(f"  CV R² hrfopt    : {res.xval_r2_best.mean().item():.4f}")

    ok = True
    if res.final_results and res.canonical_results:
        hr = res.final_results.r2.mean().item()
        cr = res.canonical_results.r2.mean().item()
        print(f"  In-sample R² canonical : {cr:.4f}")
        print(f"  In-sample R² hrfopt    : {hr:.4f}")
        if hr < cr - 0.01:
            print("  WARN: hrfopt in-sample R² lower than canonical")
            ok = False
        else:
            print("  PASS: in-sample R² consistent")

    hf = res.final_results.fstats if res.final_results else None
    cf = res.canonical_results.fstats if res.canonical_results else None
    if hf is not None and cf is not None:
        print(f"  F-stat canonical : {cf.mean().item():.1f}")
        print(f"  F-stat hrfopt    : {hf.mean().item():.1f}")
        if hf.mean().item() < cf.mean().item() - 1.0:
            print("  WARN: hrfopt F-stat lower than canonical")
            ok = False
        else:
            print("  PASS: F-stat consistent")
    else:
        missing = "hrfopt" if hf is None else "canonical"
        print(f"  WARN: {missing} results missing F-stats")
        ok = False

    # Rank of true HRF for each group
    r2_all = res.xval_r2_all_hrfs.cpu()
    m = compute_recovery(r2_all)
    print(f"\n  Global top-1: {m['global_top1']:.1%}  "
          f"top-3: {m['global_top3']:.1%}  "
          f"mean rank: {m['global_rank']:.2f}/{N_HRFS-1}")
    return ok


# ── confusion matrix pretty-print ─────────────────────────────────────────────

def print_confusion(confusion: np.ndarray) -> None:
    """Print a compact 20×20 confusion matrix."""
    n = confusion.shape[0]
    correct = np.diag(confusion)
    total   = confusion.sum(axis=1)
    pct     = np.where(total > 0, 100 * correct / total, 0)
    print("\n  Confusion matrix (rows=true, cols=predicted). "
          "Diagonal = correct; off-diagonal = errors.")
    header = "     " + "".join(f"{i:3d}" for i in range(n))
    print(f"  {header}")
    for i in range(n):
        row = "".join(
            f"{confusion[i, j]:3d}" if i != j else f"\033[1m{confusion[i, j]:3d}\033[0m"
            for j in range(n)
        )
        print(f"  {i:3d}: {row}  ({pct[i]:5.1f}%)")


# ── summary tables ────────────────────────────────────────────────────────────

def print_summary(records: list[dict], durations: list[float],
                  noise_types: list[str], snrs: list[float]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    # By duration (avg across noise/snr)
    print(f"\n  Recovery by duration (avg over noise types and SNRs):")
    print(f"  {'dur':>6}  {'type':>8}  {'top-1':>7}  {'top-3':>7}  "
          f"{'mean rank':>9}  {'ΔR²':>8}")
    for dur in durations:
        recs = [r for r in records if r["duration"] == dur]
        if not recs:
            continue
        t1 = np.mean([r["global_top1"] for r in recs])
        t3 = np.mean([r["global_top3"] for r in recs])
        rk = np.mean([r["global_rank"] for r in recs])
        di = np.mean([r["mean_r2_improvement"] for r in recs])
        dt = recs[0]["design_type"]
        print(f"  {dur:6.1f}  {dt:>8}  {t1:7.1%}  {t3:7.1%}  {rk:9.2f}  {di:+8.4f}")

    # By noise type (avg across dur/snr)
    print(f"\n  Recovery by noise type (avg over durations and SNRs):")
    print(f"  {'noise':>14}  {'top-1':>7}  {'top-3':>7}  {'mean rank':>9}  {'ΔR²':>8}")
    for nt in noise_types:
        recs = [r for r in records if r["noise_type"] == nt]
        if not recs:
            continue
        t1 = np.mean([r["global_top1"] for r in recs])
        t3 = np.mean([r["global_top3"] for r in recs])
        rk = np.mean([r["global_rank"] for r in recs])
        di = np.mean([r["mean_r2_improvement"] for r in recs])
        print(f"  {nt:>14}  {t1:7.1%}  {t3:7.1%}  {rk:9.2f}  {di:+8.4f}")

    # By SNR (avg across dur/noise)
    print(f"\n  Recovery by SNR (avg over durations and noise types):")
    print(f"  {'SNR':>6}  {'top-1':>7}  {'top-3':>7}  {'mean rank':>9}  {'ΔR²':>8}")
    for snr in snrs:
        recs = [r for r in records if abs(r["snr"] - snr) < 1e-6]
        if not recs:
            continue
        t1 = np.mean([r["global_top1"] for r in recs])
        t3 = np.mean([r["global_top3"] for r in recs])
        rk = np.mean([r["global_rank"] for r in recs])
        di = np.mean([r["mean_r2_improvement"] for r in recs])
        print(f"  {snr:6.1f}  {t1:7.1%}  {t3:7.1%}  {rk:9.2f}  {di:+8.4f}")

    # Confusion matrix (summed over all configs at highest SNR)
    snr_max = max(snrs)
    hi_snr_recs = [r for r in records if abs(r["snr"] - snr_max) < 1e-6]
    if hi_snr_recs:
        total_conf = sum(r["confusion"] for r in hi_snr_recs)
        print(f"\n  Confusion matrix at SNR={snr_max} (summed over "
              f"{len(hi_snr_recs)} configs):")
        print_confusion(total_conf)

    # Flag problems
    problems = []
    for r in records:
        if r["global_top1"] < 0.5 and r["snr"] >= 2.0:
            problems.append(
                f"    top-1 <50% at SNR={r['snr']}: "
                f"dur={r['duration']}s  noise={r['noise_type']}"
            )
        if r["mean_r2_improvement"] < -0.005 and r["snr"] >= 2.0:
            problems.append(
                f"    R² degradation at SNR={r['snr']}: "
                f"dur={r['duration']}s  noise={r['noise_type']}  "
                f"ΔR²={r['mean_r2_improvement']:+.4f}"
            )
    if problems:
        print("\n  PROBLEMS DETECTED:")
        for p in problems:
            print(p)
    else:
        print("\n  No major problems detected.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _patch_globals(n_hrfs: int, n_voxels: int) -> None:
    """Patch module-level constants for quick mode."""
    global N_HRFS, N_VOXELS
    N_HRFS = n_hrfs
    N_VOXELS = n_voxels


# ── CLI ───────────────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate ffs_hrfopt: HRF recovery across all library members"
    )
    p.add_argument("-device", default=None, help="Force device (cpu/cuda)")
    p.add_argument("-quick", action="store_true",
                   help="Quick smoke-test: 3 HRFs→300 voxels, 2 dur, 2 noise, 2 SNR, 1 perm")
    p.add_argument("-durations", nargs="+", type=float, default=None)
    p.add_argument("-noise", nargs="+", default=None,
                   choices=["white", "ar1", "arma", "physiological"])
    p.add_argument("-snrs", nargs="+", type=float, default=None)
    p.add_argument("-n_perms", type=int, default=N_PERMS)
    p.add_argument("-verbose", "-v", action="store_true",
                   help="Print per-permutation results and per-HRF breakdown")
    return p


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    print(f"Loading HRF library ({N_HRFS} HRFs, stim_duration=0 impulse responses)...")
    hrf_library = get_hrf_library(
        mode="library",
        stim_duration=0.0,
        microtime_dt=MICROTIME_DT,
        n_hrfs=N_HRFS,
        device=device,
    )
    print(f"  HRF library: {hrf_library.shape}  "
          f"[peak_min={hrf_library.abs().max(dim=1).values.min():.3f}  "
          f"peak_max={hrf_library.abs().max(dim=1).values.max():.3f}]")

    durations   = args.durations   or DURATIONS_ALL
    noise_types = args.noise       or NOISE_TYPES
    snrs        = args.snrs        or SNRS
    n_perms     = args.n_perms

    if args.quick:
        N_HRFS_QUICK = 5
        hrf_library = hrf_library[:N_HRFS_QUICK]
        _patch_globals(N_HRFS_QUICK, N_HRFS_QUICK * N_VOXELS_PER_HRF)
        durations   = [1.0, 5.0, 15.0]
        noise_types = ["white", "ar1"]
        snrs        = [1.0, 2.0]
        n_perms     = 1
        print(f"Quick mode: {N_HRFS_QUICK} HRFs × {N_VOXELS_PER_HRF} = "
              f"{N_HRFS_QUICK * N_VOXELS_PER_HRF} voxels, "
              f"{len(durations)} dur, {len(noise_types)} noise, {len(snrs)} snr, {n_perms} perm")
    else:
        print(f"Full mode: {N_HRFS} HRFs × {N_VOXELS_PER_HRF} = {N_VOXELS} voxels")

    # Part 1
    design_ok = validate_designs(hrf_library, device)

    # Part 2
    records = run_recovery(
        hrf_library, device, durations, noise_types, snrs, n_perms, args.verbose
    )

    # Part 3
    fstat_ok = check_consistency(hrf_library, device)

    # Summary
    print_summary(records, durations, noise_types, snrs)
    print(f"\n  Design sanity: {'PASS' if design_ok else 'FAIL'}")
    print(f"  F-stat/R² consistency: {'PASS' if fstat_ok else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
