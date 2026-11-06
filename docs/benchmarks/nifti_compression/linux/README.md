# NIfTI compression benchmark: Linux (Xeon W-2133)

Why the toolbox writes `.nii.zst` for intermediates (and what `.nii.gz` costs).
Produced by `scripts/bench_nifti_compression.py` and `scripts/bench_nifti_write_path.py`.
The Mac side is in `../mac_m4max/`.

Machine: Xeon W-2133 (6 cores / 12 threads), 251 GB, Linux 7.0, tmpfs scratch for the codec
runs, NVMe for the disk run, 6 threads. isal 1.9.1, pigz 2.8, zstd 1.5.7, pzstd 1.5.5,
rapidgzip 0.16.0. Codec, end-to-end and disk runs: 2026-11-06 on commit `a50006ea`, idle
machine (load average median 2.7). Crossover and write-path A/B: 2026-11-05.

## What it shows

Compress seconds, 6 threads:

| File | ISA-L 2 (.gz) | `pzstd -1` | `pzstd -3` | pigz -1 |
|---|---|---|---|---|
| t1 (98 MB) | 0.13 | 0.08 | 0.32 | 0.48 |
| bold (258 MB) | 0.21 | 0.27 | 0.47 | 0.75 |
| 5D warp (524 MB) | 0.37 | 0.35 | 0.70 | 1.51 |
| raw mag (894 MB) | 0.93 | 0.54 | 1.88 | 3.67 |
| 5D warp (3.5 GB) | 1.77 | 1.59 | 3.11 | 10.71 |
| 5D flow series (7.9 GB) | 3.20 | 2.99 | 4.85 | 10.13 |

What `save_nifti` / `load_nifti` cost a Python caller (array to file, file to float32 array,
tmpfs; fig4), against nibabel with single-thread ISA-L:

| File | write `.zst` | write `.gz` | nibabel write | read `.zst` | read `.gz` | nibabel read |
|---|---|---|---|---|---|---|
| preproc bold (658 MB) | 0.42 | 0.45 | 1.46 | 0.38 | 0.95 | 2.51 |
| raw mag (894 MB) | 0.61 | 0.94 | 4.03 | 0.74 | 2.38 | 3.87 |
| 5D warp (3.5 GB) | 2.15 | 2.06 | 9.06 | 1.76 | 5.06 | 9.77 |
| 5D flow series (7.9 GB) | 4.74 | 3.78 | 14.06 | 3.95 | 11.64 | 11.95 |

- **Reads are where zst wins.** `load_nifti` on `.nii.zst` is the fastest compressed read on
  10 of 11 files and ties nibabel on the 33 MB mask: 2.9x faster than `.nii.gz` on the 3.5 GB
  warp and on the 7.9 GB series, and 2-6.6x faster than nibabel on everything over 100 MB. At the codec level `pzstd -1` decodes 0.64 s vs rapidgzip
  2.71 s on the 3.5 GB warp (4.2x). Gzip decode is single-stream, which is the whole story.
- **Writes are a tie with threaded ISA-L.** `pzstd -1` is the fastest compressor on 7 of 11
  files and ISA-L on the other 4. Through the toolbox, `.nii.gz` writes a little faster on
  the biggest warps (3.78 vs 4.74 s at 7.9 GB). Above 100 MB both are 1.5-6.6x faster than nibabel;
  on the 33 MB mask nibabel's in-process write wins (0.03 vs 0.05 s).
- **Compressed can beat uncompressed.** Writing the 3.5 GB warp as `.nii.zst` (2.15 s)
  beats writing a plain `.nii` (3.66 s), because there are half as many bytes to move.
- **On the NVMe** (fig6; writes include fsync, reads start cold) the picture holds: the 3.5 GB
  warp writes in 2.74 s as `.zst` vs 2.57 s `.gz` and 4.47 s plain, and reads in 1.76 s vs
  5.10 s `.gz` and 9.08 s for nibabel. The 894 MB raw mag reads as fast from `.zst` (0.75 s)
  as from a plain `.nii` (0.76 s).
- `-1` is the right default. It pays from about 4 MB on writes and 8-16 MB on reads, and costs
  at most 5 ms below that, so no size switch is needed (fig5).
- What each write-path fix bought (`write_path.csv`, NVMe): the 7.9 GB warp's `.nii.zst` write
  went from 13.5 s to 5.2 s and the 3.5 GB warp's from 6.0 s to 2.7 s, from the 1 MiB pipe plus
  writing without nibabel's slice copies. Cold reads are 16-23% faster with the 1 MiB pipe.

## Files

| File | Content |
|---|---|
| `fig1_compress.png`, `fig2_decompress.png` | Codec speed per file |
| `fig3_tradeoff.png` | Level sweep |
| `fig4_end_to_end.png` | Toolbox read/write routes against nibabel |
| `fig5_crossover.png` | File size where parallel zstd starts paying |
| `fig6_nvme.png` | On-disk (fsync writes, cold reads) |
| `results.csv`, `results_disk.csv`, `crossover.csv`, `*meta.json` | Raw data for the figures |
| `write_path.csv` | `bench_nifti_write_path.py` A/B of each fix (`summary` subcommand reads it) |

## Read before quoting

- Not identical to the Mac run: 6 vs 10 threads, pzstd 1.5.5 vs 1.5.7, stock vs zlib-ng pigz,
  and different files (only the 3.5 GB warp is shared).
- The end-to-end read routes all read the same file per format (`pzstd -3` for `.zst`,
  `pigz -1` for `.gz`), so the route is the only difference. `-1` and `-3` reads were within 3%
  in the write-path A/B.
- `crossover.csv` and `write_path.csv` are from the day before, under more load; the crossover
  is codec-only and the A/B alternated routes, so neither depends on the later code.
- ISA-L in the nibabel routes is single-threaded, as nibabel would call it; the toolbox's
  `.nii.gz` route uses threaded ISA-L.
- Not run: new codecs (OpenZL, Blosc2, libdeflate).
