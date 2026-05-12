#!/bin/bash

## PREP WORK. NONE OF THIS IS TIMED!
# cd to the test data directory, keeping in mind that it is a variable
cd $FFS_BENCHMARK_DATA


# Download the data from AWS - we just need one run of the magntiude and phase data. 
 aws s3 sync --no-sign-request --exclude "*" --include "*sub-03_task-checkerboard_acq-ge_run-01_*.nii.gz" s3://openneuro.org/ds003427 ds003427-download/

# ok, we've got it
# (base) [21:13:41] logan@Chronos /home/logan/.fastfuncstuff/test_data  
# > tree ds003427-download/
# ds003427-download/
# └── sub-03
#     └── func
#         ├── sub-03_task-checkerboard_acq-ge_run-01_bold.nii.gz
#         └── sub-03_task-checkerboard_acq-ge_run-01_phase.nii.gz
cd ds003427-download

mkdir processing
cd processing

# Now, we get the motion params, from the bold data (aka, magnitude. )
3dvolreg -overwrite -heptic -prefix ignore_moco.nii.gz -1Dmatrix_save moco_params.aff12.1D ../sub-03/func/sub-03_task-checkerboard_acq-ge_run-01_bold.nii.gz 

# unwrap the phase data, using magnitude, and romeo
~/Dropbox/Resources/code/mritools_ubuntu-24.04_4.7.1/bin/romeo \
 -p ../sub-03/func/sub-03_task-checkerboard_acq-ge_run-01_phase.nii.gz \
 -m ../sub-03/func/sub-03_task-checkerboard_acq-ge_run-01_bold.nii.gz \
 -t epi \
 -o unwrapped_phase.nii.gz \
 -v

# Now, we have to apply these, using ffs_nwarp - we can use direct, because we have unwrapped the phase (more or less), good enough for now, TODO revisit later. 
ffs_nwarp \
    -source ../sub-03/func/sub-03_task-checkerboard_acq-ge_run-01_bold.nii.gz  \
    -phase unwrapped_phase.nii.gz \
    -nwarp moco_params.aff12.1D \
    -phase_warp direct \
    -prefix method_direct_aligned.nii.gz \
    -phase_units rad

3dAutomask -overwrite -prefix automask.nii.gz ../sub-03/func/sub-03_task-checkerboard_acq-ge_run-01_bold.nii.gz 


# THe benchmark should then run the phaseprep code we created and time that as the ref, and then run the ffs_phasereg code with the same input and settings and time that. 
# For the love of god don't use these filenames. but this is the comparable command. 
ffs_phasereg -magnitude method_direct_aligned.nii.gz -phase method_direct_aligned_phase.nii.gz -prefix test_totallynew_nosgf_nwarp_pol1 -polort 1 -task_removal none -verbose

# THen the benchmark should compare params, and voxel timeseries, within an automask. 