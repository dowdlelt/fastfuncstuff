# Synthetic AFNI fixtures

The onset files contain invented times for two 360-volume runs. AFNI 25.3.03
generated the reference matrix with the following command from this directory.

```sh
3dDeconvolve -nodata 720 1 -concat '1D: 0 360' -local_times -polort 3 \
  -num_stimts 2 -stim_times 1 movie.txt 'SPMG1(5)' -stim_label 1 movie \
  -stim_times 2 prompt.txt 'SPMG1(5)' -stim_label 2 prompt \
  -x1D X.xmat.1D -x1D_stop
```

The two image formats contain the same 4 × 5 × 6 × 8 array of int16 ones.
To regenerate them in an empty working directory with nibabel and AFNI installed:

```sh
python -c 'import nibabel as nib, numpy as np; nib.save(nib.Nifti1Image(np.ones((4, 5, 6, 8), dtype=np.int16), np.eye(4)), "ones.nii.gz")'
3dcopy ones.nii.gz ones+orig
3dNotes -HH 'Synthetic all-ones AFNI test fixture' ones+orig.HEAD
gzip -n ones+orig.BRIK
```
