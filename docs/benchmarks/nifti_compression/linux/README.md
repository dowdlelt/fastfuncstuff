# NIfTI compression benchmark: Linux (Xeon W-2133)

Why the toolbox writes `.nii.zst` for intermediates (and what `.nii.gz` costs).
Produced by `scripts/bench_nifti_compression.py` and `scripts/bench_nifti_write_path.py`
on 2026-11-05. The Mac side is in `../mac_m4max/`.

Machine: Xeon W-2133 (6 cores / 12 threads), 251 GB, Linux 7.0, tmpfs scratch for the codec
runs, NVMe for the disk runs, 6 threads. isal 1.9.1, pigz 2.8, zstd 1.5.7, pzstd 1.5.5,
rapidgzip 0.16.0.

## What it shows (6 threads, compress seconds)

| File | ISA-L 2 (.gz) | `pzstd -1` | `pzstd -3` | pigz -1 |
|---|---|---|---|---|
| t1 (98 MB) | 0.13 | 0.08 | 0.32 | 0.48 |
| bold (258 MB) | 0.20 | 0.28 | 0.47 | 0.77 |
| 5D warp (524 MB) | 0.41 | 0.36 | 0.74 | 1.54 |
| raw mag (894 MB) | 0.92 | 0.59 | 1.99 | 3.69 |
| 5D warp (3.5 GB) | 1.87 | 1.61 | 2.98 | 10.75 |
| 5D flow series (7.9 GB) | 3.32 | 3.07 | 5.06 | 10.53 |

- `pzstd -1` is the fastest compressor on 7 of 11 files; threaded ISA-L wins the other 4, at
  similar file sizes. Against `pzstd -3`, ISA-L wins 10 of 11. ISA-L writes are a median 3.8x
  faster than pigz -1.
- Decompression is where zstd wins outright: `pzstd -1` 1.05 s vs rapidgzip 5.84 s on the
  3.5 GB warp (5.6x); 0.11 s vs 0.39 s on `bold` (3.5x); median 3.1x across the 11 files.
  Gzip decode is single-stream, which is the whole story.
- `-1` is the best default for reads and writes. It pays from about 4 MB on writes and
  8-16 MB on reads, and costs at most 5 ms below that, so no size switch is needed.
- The toolbox's own routes after the write-path fixes (`write_path.csv`, NVMe, write timed with
  fsync, reads cold): 7.9 GB warp `.nii.zst` write 13.5 s -> 5.2 s, 3.5 GB 6.0 s -> 2.7 s,
  894 MB 1.00 s -> 0.84 s, 658 MB 0.96 s -> 0.59 s. Cold reads are 16-23% faster with the
  1 MiB pipe, and `-1` vs `-3` reads are within 3%.

## Files

| File | Content |
|---|---|
| `fig1_compress.png`, `fig2_decompress.png` | Codec speed per file |
| `fig3_tradeoff.png` | Level sweep |
| `fig4_end_to_end.png` | Toolbox read/write routes (old code, see below) |
| `fig5_crossover.png` | File size where parallel zstd starts paying |
| `fig6_nvme.png` | On-disk (fsync writes, cold reads) |
| `results.csv`, `results_disk.csv`, `crossover.csv`, `*meta.json` | Raw data for the figures |
| `write_path.csv` | `bench_nifti_write_path.py` A/B of each fix (`summary` subcommand reads it) |

## Read before quoting

- Not identical to the Mac run: 6 vs 10 threads, pzstd 1.5.5 vs 1.5.7, stock vs zlib-ng pigz,
  and different files (only the 3.5 GB warp is shared).
- **fig4 and the "real" panels of fig6 time the toolbox routes as they were mid-session, not as
  shipped.** fig4 predates every write-path change (level 1 default, streaming into the
  compressor, threaded ISA-L for `.gz`); fig6 predates the 1 MiB pipe and the no-copy write.
  The codec panels are unaffected. For the shipped routes use `write_path.csv`.
- Load average during the codec run: median 8.4, max 12.3, on a 6-thread benchmark. The
  write-path A/B ran quieter (median 2.1, max 3.9).
- Float32 `.nii` "reads" through nibabel are memory-mapped, so near-zero read times for plain
  `.nii` are not loads.
- The figures were regenerated on 2026-11-06 from the saved CSVs with the current plotting
  code. The disk run had no `disk_meta.json` (the script did not write one yet); the one here
  was written by hand, with `cold_reads: true` because Linux evicts with `posix_fadvise`.
- Not run: new codecs (OpenZL, Blosc2, libdeflate).
