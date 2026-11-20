# Motion-correction ground truth

Real anatomy, synthetic motion: the scripts behind [`docs/moco.md`](../../docs/moco.md).

1. **`gen_truth.py FINE OUT [options]`** moves a high-resolution 3D GRE volume
   (a DFT on all three axes, so FFT shears move it exactly) at its native grid on
   the GPU and samples it the way an EPI would: k-space truncation in-plane, and
   along z either a box slice profile centred on each sample (2D) or truncation (3D).
   - `-factor FX FY FZ -ncoarse NX NY NZ`: the coarse grid (e.g. 6 6 6 / 94 94 56
     takes 0.35 mm to 2.1 mm; 6 6 7 / 94 94 48 gives 2.1x2.1x2.45 mm voxels).
   - `-trace AFF12 REF TAG` (repeatable): replay a real moco trace (`-1Dmatrix_save`
     output, with the dataset it came from); rotations are re-orthonormalized.
   - `-battery`: translation sweeps 0.025-1 voxel per axis, rotation sweeps
     0.005-0.5 deg, and random 6-DOF jitter / small / moderate motion.
   - `-noise 0 0.02 0.05`: Gaussian noise as a fraction of the brain mean.
2. **`score_motion.py OUT EST.aff12.1D [...] [-gain] [-sweeps]`**: mean | max error of
   the displacement field over the brain mask, per motion group (pivot-free, so
   parameter conventions cannot leak in). `-sweeps` prints the sub-voxel bias curve,
   `-gain` estimated/true size for small motions.
3. **`score_resample.py OUT SERIES.nii [modes]`**: undoes the TRUE motion with each
   1D shear kernel through the same plan and reports the brain residual, the
   temporal std it injects, and the error in a shell outside the brain (ringing).

```bash
python gen_truth.py anat_T2star_0p35mm.nii.gz truth/syn -battery -noise 0 0.02 0.05
ffs_moco -input truth/syn_2d_n020.nii -base 0 -1Dmatrix_save est_fourier.aff12.1D
ffs_moco -input truth/syn_2d_n020.nii -base 0 -interp heptic -1Dmatrix_save est_heptic.aff12.1D
3dvolreg -Fourier -base 0 -prefix /tmp/vr.nii -1Dmatrix_save est_vr.aff12.1D truth/syn_2d_n020.nii
python score_motion.py truth/syn est_fourier.aff12.1D est_heptic.aff12.1D est_vr.aff12.1D -sweeps -gain
python score_resample.py truth/syn truth/syn_2d_n000.nii heptic,wsinc5,fourier
```

4. **`plot_figures.py MANIFEST.json OUTDIR`** draws the seven figures in
   `docs/moco/` from the testbeds and the estimate files a manifest names (see
   `plot_manifest_example.json`, paths relative to the working directory).
