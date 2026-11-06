# NIfTI compression benchmark: Apple M4 Max

Why the toolbox writes `.nii.zst` for intermediates (and what `.nii.gz` costs).
Produced by `scripts/bench_nifti_compression.py` and `scripts/bench_nifti_write_path.py`.
Linux (6 threads) was measured separately; this directory is the Mac side only.

Machine: M4 Max (10 P + 4 E cores), 36 GB, macOS 15.7, RAM-disk scratch, 10 threads.
isal 1.9.1, pigz 2.8 (zlib-ng build), zstd/pzstd 1.5.7, rapidgzip 0.16.0.

## What it shows (10 threads, compress seconds)

| File | ISA-L 2 (.gz) | `pzstd -1` | `pzstd -3` | pigz -1 |
|---|---|---|---|---|
| t1 (49 MB) | 0.037 | 0.015 | 0.034 | 0.080 |
| warp (136 MB) | 0.053 | 0.025 | 0.046 | 0.110 |
| bold (860 MB) | 0.558 | 0.135 | 0.501 | 1.352 |
| prf (996 MB) | 0.654 | 0.164 | 0.198 | 2.000 |
| 5D warp (3.5 GB) | 1.514 | 0.608 | 0.654 | 4.103 |

- `pzstd -1` is the fastest compressor on every file, at a similar file size to gzip.
- Decompression: `pzstd -1` 0.249 s vs rapidgzip 1.92 s on the 3.5 GB warp (7.7x);
  0.092 s vs 0.346 s on `bold` (3.8x). Gzip decode is single-stream, which is the whole story.
- ISA-L does **not** gain on ARM; it loses ground relative to Linux.
- The Linux 64 KB pipe fix has no effect here (macOS has no equivalent); the write gain in the
  toolbox routes comes from skipping nibabel's slice copies.
- `-1` is the best default for both reads and writes; it pays from about 4 MB.

## Files

| File | Content |
|---|---|
| `fig1_compress.png`, `fig2_decompress.png` | Codec speed per file |
| `fig3_tradeoff.png` | Level sweep (`bold` only) |
| `fig4_end_to_end.png` | Toolbox read/write routes |
| `fig5_crossover.png` | File size where parallel zstd starts paying |
| `fig6_nvme.png` | On-disk (fsync writes, warm reads) |
| `results.csv`, `results_disk.csv`, `crossover.csv`, `*meta.json` | Raw data |

## Read before quoting

- Not identical to the Linux run: 10 vs 6 threads, pzstd 1.5.7 vs 1.5.5, zlib-ng pigz, and
  different files (only the 3.5 GB warp is shared).
- The 3.5 GB warp codec run pushed swap from 1.1 to 7.4 GB, so its absolute times may be inflated.
- Load average peaked at 6.8 during the runs.
- Disk reads are warm (macOS cannot evict one file from the page cache).
- fig4 shows the 3.5 GB warp at 1.07 s write / 0.97 s read via `ffs-zst`; the write-path harness
  measured 0.70 s / 0.69 s. Different harnesses, not reconciled.
- Not explained: ISA-L level 3 faster than level 2, and level 0 giving ~103% of original size.
- Not run: stock pigz, the 7.9 GB warp.
