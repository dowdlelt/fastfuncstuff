# Create an MNI automask for these so we can focus on the releavant parts. 
# we do not need to time this. we just need this output file in the processing folder. 
3dAutomask -prefix MNI_automask.nii.gz ffs_mni_resampled_task-localizer_run-1.nii.gz

# We are running a TENT model - this is like an FIR model. 
# It produces iresp outputs -these are time curves, that are predicted HRFs - for each condition, so one iresp per condition.

time 3dDeconvolve -overwrite \
    -input \
        ffs_mni_resampled_task-localizer_run-1.nii.gz \
        ffs_mni_resampled_task-localizer_run-2.nii.gz \
        ffs_mni_resampled_task-localizer_run-3.nii.gz \
        ffs_mni_resampled_task-localizer_run-4.nii.gz \
        ffs_mni_resampled_task-localizer_run-5.nii.gz \
    -polort A \
    -num_stimts 5 \
    -float \
    -stim_times 1 timing_files/onsets.localizer.times.faces.txt 'TENT(0,15,11)' \
    -stim_label 1 faces \
    -stim_times 2 timing_files/onsets.localizer.times.bodies.txt 'TENT(0,15,11)' \
    -stim_label 2 bodies \
    -stim_times 3 timing_files/onsets.localizer.times.objects.txt 'TENT(0,15,11)' \
    -stim_label 3 objects \
    -stim_times 4 timing_files/onsets.localizer.times.scenes.txt 'TENT(0,15,11)' \
    -stim_label 4 scenes \
    -stim_times 5 timing_files/onsets.localizer.times.scrambled.txt 'TENT(0,15,11)' \
    -stim_label 5 scrambled \
    -jobs 10 -noFDR  \
    -iresp 1 iresp_faces -iresp 2 iresp_bodies -iresp 3 iresp_objects -iresp 4 iresp_scenes -iresp 5 iresp_scrambled \
    -mask MNI_automask.nii.gz \
    -x1D  TENT_X.xmat.1D -bucket TENT_afni_stats_localizer.nii.gz

# We do not need REML here - this is deconvolution, predicting the shape of the response without constraitn. We don't even care about sats. no contrast either.
### FFS version is here
ffs_deconvolve \
 -input ffs_mni_resampled_task-localizer_run-*.nii.gz \
 -onsets ./timing_files/onsets.localizer.times.bodies.txt ./timing_files/onsets.localizer.times.faces.txt ./timing_files/onsets.localizer.times.objects.txt ./timing_files/onsets.localizer.times.scenes.txt ./timing_files/onsets.localizer.times.scrambled.txt  \
 -prefix ffs_tent \
 -model TENT \
 -window 0 15 \
 -verbose

# the outputs of this again, should be in the ffs_glm_IM folder. 
# They look like this ffs_tent_iresp_bodies.nii.gz, based on unique part of contion onset file names. 

## PASS CRITERIA. 
# The iresps should have the same length - but thats simple. table stakes. 
# Each paired condition, based on names (AFNI faces ffs Faces) should have a timeseries correlation (voxel 1 corr with voxel 1) - median across image should be > 0.95 for every condition. ignore zeros (masked out duh)
# In addition, the spatial pattern of the middle timepoint of each iresp pair (AFNI, FFS) should have a spatial correlation of > 0.95 across the brain (again, ignore zeros).

# Here ends stage glm_TENT


# HERE begins stage glm_IM

# NOW 3dDeconvolve, IM model. This output should be organized like the other afni_glm output - excapt this is afni_glm_IM
# This creates a beta/t-stat pairing of each condition. 
# This is one beta per event (so lots of betas in this case. )
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
    -mask MNI_automask.nii.gz \
    -tout -x1D  IM_X.xmat.1D -bucket IM_afni_stats_localizer.nii.gz


# This replicaates the OLS IM model above, using the exact same matrix. Output is same order as 3dDeconvolve
time ffs_reml \
          -input \
              scaled_afni_mni_task-localizer_run-1.nii.gz \
              scaled_afni_mni_task-localizer_run-2.nii.gz \
              scaled_afni_mni_task-localizer_run-3.nii.gz \
              scaled_afni_mni_task-localizer_run-4.nii.gz \
              scaled_afni_mni_task-localizer_run-5.nii.gz \
          -matrix IM_X.xmat.1D \
          -use_double \
          -Obuck IM_ffs_stats_localizer.nii.gz \
          -mask MNI_automask.nii.gz \
          -tout


## Pass condition. 
# First, within the mask the betas (which is every 2nd volume, starting with 1 so, 1,3,etc) should have incredbily high correlations over time (so correlate each and every voxel, then take the median across voxels) - this should be > 0.95 for each condition.
# Second, the spatial pattern of each condition (so beta 1 across voxels should correlate with beta 1 across voxels, etc) should also be > 0.95. Again, ignore zeros.
# This should also be true for the t-stats, which are every 2nd volume starting with 2 (so 2,4, etc).
# and finally the 0th volume is the F-stat that should be highly correlated. 



## Here ends glm_IM



## Here begins glm_IM_REML. This is the same model as above, but with REML.

# This runs the same but with REML, which should give more accurate t-stats and perhaps more "accurate" betas. 
time tcsh IM_afni_stats_localizer.REML_cmd -overwrite -nofdr


# And this replicates the REML IM model, again, same matrix. We also get REML_var parameters here (a and b are most important). 
# 3dREMLfit above (in the tcsh command) also spits out AFNIs REMLvar - a and b are in the same place. 
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
          -mask MNI_automask.nii.gz \
          -tout

# Ok Pass condtiion

## Pass condition. 
# First, within the mask the betas (which is every 2nd volume, starting with 1 so, 1,3,etc) should have incredbily high correlations over time (so correlate each and every voxel, then take the median across voxels) - this should be > 0.95 for each condition.
# Second, the spatial pattern of each condition (so beta 1 across voxels should correlate with beta 1 across voxels, etc) should also be > 0.95. Again, ignore zeros.
# This should also be true for the t-stats, which are every 2nd volume starting with 2 (so 2,4, etc).
# and finally the 0th volume is the F-stat that should be highly correlated. 
# Note this is likely to fail, due to slight differences in the REML estimation, but we still need to check it. Lets set these to 0.9 instead of 0.95 and see what happens. 
