# First, 3dDeconvolve
time 3dDeconvolve -overwrite \
    -input \
        scaled_afni_mni_task-localizer_run-1.nii.gz \
        scaled_afni_mni_task-localizer_run-2.nii.gz \
        scaled_afni_mni_task-localizer_run-3.nii.gz \
        scaled_afni_mni_task-localizer_run-4.nii.gz \
        scaled_afni_mni_task-localizer_run-5.nii.gz \
    -polort A \
    -num_stimts 5 \
    -float \
    -stim_times_IM 1 timing_files/onsets.localizer.times.faces.txt 'SPMG1(3)' \
    -stim_label 1 faces \
    -stim_times_IM 2 timing_files/onsets.localizer.times.bodies.txt 'SPMG1(3)' \
    -stim_label 2 bodies \
    -stim_times_IM 3 timing_files/onsets.localizer.times.objects.txt 'SPMG1(3)' \
    -stim_label 3 objects \
    -stim_times_IM 4 timing_files/onsets.localizer.times.scenes.txt 'SPMG1(3)' \
    -stim_label 4 scenes \
    -stim_times_IM 5 timing_files/onsets.localizer.times.scrambled.txt 'SPMG1(3)' \
    -stim_label 5 scrambled \
    -jobs 10 -noFDR  \
    -tout -x1D  IM_X.xmat.1D -bucket IM_afni_stats_localizer.nii.gz

time tcsh IM_afni_stats_localizer.REML_cmd -overwrite -nofdr


ime ffs_reml \
          -input \
              scaled_afni_mni_task-localizer_run-1.nii.gz \
              scaled_afni_mni_task-localizer_run-2.nii.gz \
              scaled_afni_mni_task-localizer_run-3.nii.gz \
              scaled_afni_mni_task-localizer_run-4.nii.gz \
              scaled_afni_mni_task-localizer_run-5.nii.gz \
          -matrix IM_X.xmat.1D \
          -use_double \
          -Obuck IM_ffs_stats_localizer.nii.gz \
          -tout


 time ffs_reml \
          -input \
              scaled_afni_mni_task-localizer_run-1.nii.gz \
              scaled_afni_mni_task-localizer_run-2.nii.gz \
              scaled_afni_mni_task-localizer_run-3.nii.gz \
              scaled_afni_mni_task-localizer_run-4.nii.gz \
              scaled_afni_mni_task-localizer_run-5.nii.gz \
          -matrix IM_X.xmat.1D \
          -Rbuck ffs_stats_IM_localizer_REML.nii.gz \
          -Rvar ffs_stats_IM_localizer_REML_var.nii.gz \
          -tout


