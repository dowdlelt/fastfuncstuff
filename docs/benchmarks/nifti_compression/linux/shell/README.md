# Shell-only codec benchmark: Linux (Xeon W-2133)

The same 11 files as `../`, timed by `scripts/bench_codecs.sh`: every tool through
its own command line, from a bash loop. No Python is in the timed path, so nothing about
how the toolbox calls a codec, or how the Python harness does, can tilt the result.
Plotted with `scripts/plot_codecs.py codecs.csv -split 4,4,3`.

Run 2026-11-06, 4 threads, 3 shuffled repetitions, RAM disk to RAM disk; every decoder
checked against the plain bytes before timing. ISA-L is its own `igzip` CLI, built from
2.31 with threads (see "Read before quoting").

## What it shows (4 threads, seconds)

| File | `igzip -1` | `igzip -2` | `pzstd -1` | gzip read (rapidgzip) | `pzstd` read | read ratio |
|---|---|---|---|---|---|---|
| brain mask (33 MB) | 0.023 | 0.029 | 0.028 | 0.072 | 0.015 | 4.7x |
| bold (258 MB) | 0.330 | 0.329 | 0.423 | 0.261 | 0.158 | 1.7x |
| preproc bold (658 MB) | 0.565 | 0.562 | 0.463 | 0.822 | 0.222 | 3.7x |
| raw mag (894 MB) | 1.590 | 1.557 | 0.886 | 1.327 | 0.349 | 3.8x |
| 4D warp series (2.6 GB) | 1.738 | 1.837 | 1.773 | 1.548 | 0.584 | 2.7x |
| 5D warp (3.5 GB) | 2.760 | 3.047 | 2.226 | 3.232 | 0.753 | 4.3x |
| 5D flow series (7.9 GB) | 3.684 | 3.744 | 3.267 | 6.204 | 1.135 | 5.5x |

- **It agrees with the Python benchmark.** Threaded ISA-L out-compresses `pzstd -1` on 4 of
  11 files, the *same four* as in `../` (mask, bold, masked PSC, 4D warp series), and the
  rest go to `pzstd -1` by margins of 10-45%. Writes are a near-tie.
- **Reads are not.** `pzstd` reads a `pzstd` file a median 3.7x faster than rapidgzip, the
  fastest gzip reader, reads any gzip (1.7x-5.5x). ISA-L's own decoder, on one core because
  that is all gzip allows, is 2-3x slower again.
- **ISA-L level 1 is as good as 2.** Faster or tied on 9 of 11 files at the same size
  (5D warp 2.76 vs 3.05 s). Level 1 is the fastest ISA-L level on 7 files, 2 on 2, and 3
  on 2 (bold, raw mag), so no level wins everywhere. `save_nifti` uses level 2.
- **Who wrote a zst file matters.** Only `pzstd` output decodes in parallel; a `zstd -T4`
  file reads at single-thread speed even through `pzstd` (5D warp 2.41 s vs 0.75 s). The
  toolbox writes with `pzstd`. Who wrote a gzip file barely matters to rapidgzip.

## Files

| File | Content |
|---|---|
| `codecs_1.png`, `codecs_2.png`, `codecs_3.png` | Smallest four, middle four, largest three files |
| `codecs.csv` | Every timing: file, op, method, rep, seconds, bytes, plain bytes, load |
| `codecs.csv.meta` | CPU, threads, tool versions |

## Read before quoting

- **The box was busy.** A 3dQwarp ran throughout; load average median 6.4 on 6 cores.
  Every tool got the same 4 threads in shuffled order, so rankings are fair, but absolute
  times are inflated, more so for the tools using all 4 threads.
- **ISA-L needs a threaded build.** Distro `igzip` builds (Ubuntu's `isal` 2.31 included)
  accept `-T` and ignore it, because ISA-L's build never defines `HAVE_THREADS`. This run
  used ISA-L 2.31 built with `./configure CFLAGS="-O2 -DHAVE_THREADS" LIBS=-lpthread`
  (needs nasm); `-T 6` then ran 2.4x faster than `-T 1`. The script refuses an `igzip`
  whose CPU time does not exceed its wall time.
- Decoding is to `/dev/null`: codec speed, not a load into an array. `../` has the
  toolbox's real routes.
- `pigz -d` inflates on one core whatever `-p` says; it is drawn faded with the other
  single-core decoders.
- Reproduce: `IGZIP=/path/to/igzip scripts/bench_codecs.sh -o codecs.csv -t 4 -r 3 <files>`.
