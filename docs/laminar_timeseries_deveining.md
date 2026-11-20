# Laminar timeseries deveining (`ffs_laminar_tdv`) — EXPERIMENTAL

The full method write-up — idea, each step, what can and cannot be identified,
the traps found building it, and every result so far — lives in the wiki:
`../fmri_wiki/concepts/Laminar timeseries deveining.md` (tool page:
`../fmri_wiki/software/ffs_laminar_tdv.md`). This page is the in-repo summary.

**Status.** Recovers known answers in simulation. On MindsEye 7T data
(sub-ME462566) likely-vein voxels lose ~3× more than with the column-only version
while typical voxels barely change, but nothing has yet shown that the removed
signal is vascular and the kept signal local. The column-only version moved GE
slightly *away* from SE on ds003427. Simultaneous VASO/BOLD is the planned
referee. The CLI prints an EXPERIMENTAL banner; metadata records it.

**Method in brief.**

1. Columns: geodesic midthickness patches through each ribbon voxel's owning
   vertex (shared with `ffs_hrftdv`), equivolume depth; a 2 mm shell past the
   ribbon inherits each nearest ribbon voxel's column and depth.
2. Column profiles: per-run PSC, Legendre drift removed, precision-weighted
   average into `-depths` bins (veins otherwise inflate their column).
3. Linearised Havlicek–Uludag vasculature → depth × depth transfer per frequency.
4. Ridge inverse per column and frequency → CBF histories → local and
   transported BOLD at every depth.
5. Per voxel: regress its data on local (free), transport (gain → 1) and
   d(transport)/dt (shift → 0), ridge strength by leave-one-run-out; remove
   gain × transport + shift × derivative. Event-free, so rest works.
6. Optional: `-events` + `-cv_p0` choose the fixed physiology by pooled
   leave-one-run-out design prediction.

```sh
ffs_laminar_tdv -input run-0{1..6}.nii.zst -fs_subj $SUBJECTS_DIR/sub-01 \
  -ridge 0.8 -prefix ltdv/sub-01
```

`ffs_laminar_tdv -help` documents every flag. The historical GE/SE validation
used OpenNeuro ds003427 with aligned, frame-fixed GE and SE runs and cortical
depth maps on the same grid. The [scoring script](../scripts/bench_laminar_tdv_se.py)
takes the prepared derivatives directory as `-dataset`. The
[plotting script](../scripts/plot_laminar_tdv_se.py) reads its report.
