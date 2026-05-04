#!/bin/bash

# NOTE (2026-04-28): This script is a legacy prototype.
# The canonical, maintained benchmark framework is `ffs_benchmark`.
# See the wiki (external, ~/Dropbox/Resources/code/fmri_wiki) for source-of-truth docs:
# - software/ffs_benchmark.md
# - principles/Benchmark validation.md

# TODO this downloded data, etc should be gitignored. We don't want to upload all of this!
# and if it is the git history, we need to remove that, cause it'll be huge. 

# Get current directory of this script and go there. 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# TODO  check timing of afni versus ffs for each command that we have. 
# TODO combine the time estimates for ffs_alineate and ffs_qwarp, since sswarper does "both" f those. 
# TODO - one point is to never skip FFS outputs, in the REAL real test code, because validating against afni is the point - this is the ultimate test suite, in that sense. 
# TODO - and it could, in theory, add a lot of coverage, because it will test a lot of scripts, in an end to end fashion. 
cd $DIR

# TODO - we will have to time the AFNI functions, the first time they run, and cache that (because after that we skip for speed)
# TODO -  the test is for both comparing and timing. We want to confirm our putputs are good, and we want to see how our speeds compare. 
# So, for first run, create a  cache file, that list functions and times, and dates, for this architecture (and what OMP_NUM_THREADS is set to). 
# If I run this on a new machine, like a mac, some things might not run, but we should get time for those that do. 
# TODO - after its all working, we want a pretty plot, showing bars for the general thing (motion correction, warping, OLS, GLSQ, etc) and how long AFNI takes versus FFS 
# And again, architectures there - ffs (5070Ti), ffs (Mac M4 Pro), etc etc. 
# As I test on more devices, we add details - but I don't want to have to rerun eversingle thing of course. 

# TODO - there are some missing skips in here as well (3dAllineate I think, as a start). 

# TODO - the aws check is somewhat slow, maybe we should just check if the data is there, and then run sync command only if needed. 
# Need the anat and func folders, with nii files in each one. Doesn't need to be perfect, check 
# mkdir \
#     -p ds005165-download

# # check
# # Download one subject and one session from openneuro. 
# aws s3 sync \
#     --no-sign-request s3://openneuro.org/ds005165 ds005165-download/ --exclude "*" --include "sub-01/ses-01/*"
# # we use sub-01 in include and not */sub or /sub  because we don't want derivatives/sub-01

cd ds005165-download

# AFNI VERSION


mkdir -p processing
cd processing

MOCO="Motion correction"
###  AFNI
# We have to check motion correction
for task in localizer rest; do
    for run in 1 2 3 4 5 ; do
        if [ -f "afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold.nii" ]; then
            echo "Motion correction output for task ${task}, run ${run} already exists. Skipping motion correction step."
        else
            echo "Running motion correction for run ${run}."
            time 3dvolreg -overwrite \
                -heptic \
                -prefix afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -base 0 \
                -1Dfile afni_motion_correction_task-${task}_run-${run}.1D \
                -1Dmatrix_save afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold_mat.aff12.1D \
                ../sub-01/ses-01/func/sub-01_ses-01_task-${task}_run-${run}_bold.nii
            echo "Motion correction for task ${task}, run ${run} done."
            echo "-----------------------------------------------------------------"

            # Quick mean
            3dTstat -overwrite \
                -prefix afni_mean_sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold.nii
        fi
    done
done


for task in localizer rest; do
    for run in 1 2 3 4 5 ; do
        if [ -f "ffs_moco_sub-01_ses-01_task-${task}_run-${run}_bold.nii" ]; then
            echo "Motion correction output for task ${task}, run ${run} already exists. Skipping motion correction step."
        else
            echo "Running motion correction for run ${run}."
            time ## FFS version of same
            time ffs_moco \
                -input ../sub-01/ses-01/func/sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -interp heptic \
                -final heptic \
                -weight_automask -base 0 -1Dfile ffs_motion_correction_task-${task}_run-${run}.1D \
                -1Dmatrix_save ffs_moco_sub-01_ses-01_task-${task}_run-${run}_bold_mat.aff12.1D \
                -prefix ffs_moco_sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -save_mean

            echo "Motion correction for task ${task}, run ${run} done."
            echo "-----------------------------------------------------------------"
            mv mean_ffs_moco_sub-01_ses-01_task-${task}_run-${run}_bold.nii ffs_mean_sub-01_ses-01_task-${task}_run-${run}_bold.nii
        fi
    done
done

# TODO compre the motion parmeters between ffs and afni - are they similar?
# TODO compare the mean image between ffs and afni - should be highly correlated

for task in localizer rest; do
    for run in 2 3 4 5 ; do
        if [ -f "afni_mean_sub-01_ses-01_task-${task}_run-${run}_bold_al.nii" ]; then
                echo "Aligned mean output for task ${task}, run ${run} already exists. Skipping alignment step."
        else
            # Align means to means, we'll need this for final warping. 
            3dAllineate -overwrite -cost lpa -onepass \
                -source_automask -autoweight \
                -warp  shr \
                -base afni_mean_sub-01_ses-01_task-localizer_run-1_bold.nii \
                -source afni_mean_sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -prefix afni_mean_sub-01_ses-01_task-${task}_run-${run}_bold_al.nii \
                -1Dmatrix_save afni_mean_${task}_run-${run}_to_localizer_run-1_mat.aff12.1D  
        fi
    done
done

for task in rest; do
    for run in 1 2 3 4 5 ; do
        if [ -f "afni_mean_sub-01_ses-01_task-${task}_run-${run}_bold_al.nii" ]; then
            echo "Aligned mean output for task ${task}, run ${run} already exists. Skipping alignment step."
        else
            echo "Running alignment for task ${task}, run ${run}."
            # Align means to means, we'll need this for final warping. 
            3dAllineate -overwrite -cost lpa -onepass \
                -source_automask -autoweight \
                -warp  shr \
                -base afni_mean_sub-01_ses-01_task-localizer_run-1_bold.nii \
                -source afni_mean_sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -prefix afni_mean_sub-01_ses-01_task-${task}_run-${run}_bold_al.nii \
                -1Dmatrix_save afni_mean_${task}_run-${run}_to_localizer_run-1_mat.aff12.1D  
        fi
    done
done

3dTcat -overwrite -prefix cat_al_means.nii.gz \
    afni_mean_sub-01_ses-01_task-localizer_run-1_bold.nii \
    afni_mean_sub-01_ses-01_task-localizer_run-2_bold_al.nii \
    afni_mean_sub-01_ses-01_task-localizer_run-3_bold_al.nii \
    afni_mean_sub-01_ses-01_task-localizer_run-4_bold_al.nii \
    afni_mean_sub-01_ses-01_task-localizer_run-5_bold_al.nii \
    afni_mean_sub-01_ses-01_task-rest_run-1_bold_al.nii \
    afni_mean_sub-01_ses-01_task-rest_run-2_bold_al.nii \
    afni_mean_sub-01_ses-01_task-rest_run-3_bold_al.nii \
    afni_mean_sub-01_ses-01_task-rest_run-4_bold_al.nii \
    afni_mean_sub-01_ses-01_task-rest_run-5_bold_al.nii

SLICETIME="Current"
# This is fine for one run. - we want to validate against ffs only, not for analysis. 
time 3dTshift -overwrite  \
    -prefix afni_tshift_sub-01_ses-01_task-localizer_run-1_bold.nii \
    -tzero 0 \
    -tpattern @$DIR/ds005165_slicetiming.txt \
    -wsinc9 \
    ../sub-01/ses-01/func/sub-01_ses-01_task-localizer_run-1_bold.nii

echo "Slice timing correction done."
echo "-----------------------------------------------------------------"

### FFS tshift
time ffs_slicetime \
    -input ../sub-01/ses-01/func/sub-01_ses-01_task-localizer_run-1_bold.nii \
    -tzero 0 \
    -wsinc9 \
    -tpattern $DIR/ds005165_slicetiming.txt \
    -prefix ffs_tshift_sub-01_ses-01_task-localizer_run-1_bold.nii

# TODO - correlated all voxels (a new utility for ffs_util?) - should have R =1, for  nearly all. 
# This would be between the timeseries of a fni and ffs above. 

# This script should have an overwrite option, we should skip if it exist, unless  overwrite was requested. 
if [ -f "sswarper_output/anatQQ.sub-01.nii" ]; then
    echo "SSwarper output already exists. Skipping sswarper step."
else
    # This should be else if or somethig.
    echo "Running sswarper for skull stripping and warping to MNI space."
    time sswarper2 \
        -input ../sub-01/ses-01/anat/sub-01_ses-01_T1w.nii \
        -base    MNI152_2009_template_SSW.nii.gz  \
        -subid sub-01 \
        -odir sswarper_output 
fi

# Here, we want to use the skull stripped (anatSS)  from afni and see what our MNI normalization woudld look like. 
mkdir -p ffs_warper
cd ffs_warper
# Set abin to the location of afni, so that ffs can find it.
ABINDIR=`which afni`
# now, strip the last part of the path to get the directory
ABINDIR=${ABINDIR%/*}

ALIGN="Alignment"
# we have to allineate first 
# TODO confirm that ffs_allineate resamples to base. 
if [ -f "al_ffs_anatSS.sub-01.nii" ]; then
    echo "Allineate output already exists. Skipping allineate step."
else
    time ffs_allineate \
        -source ../sswarper_output/anatSS.sub-01.nii \
        -base ${ABINDIR}/MNI152_2009_template.nii.gz  \
        -prefix al_ffs_anatSS.sub-01.nii \
        -source_automask -autoweight \
        -cost lpa -lpa_sigma 5
fi

if [ -f "anatFFS.sub-01.nii" ]; then
    echo "FFS Qwarp output already exists. Skipping FFS Qwarp step."
else
    time ffs_qwarp \
        -source al_ffs_anatSS.sub-01.nii \
        -base ${ABINDIR}/MNI152_2009_template.nii.gz  \
        -minpatch 11 -lpa \
        -prefix anatFFS.sub-01.nii 
fi

cd ..

# TODO how similar is our volume to the AFNI version (anatFFS versus anatQQ ). correlation?
# TODO confirm voxel resolution (may need to add a -dxyz option to ffs_qwarp for final voxel resolution of output (ffs_nwarp supports this already))


echo "Skull stripping and warping done. "
echo "-----------------------------------------------------------------"

# make a MUCH smaller master. 
3dAutobox -overwrite -npad 3 -prefix  sswarper_output/autobox_anatQQ.sub-01.nii \
    sswarper_output/anatQQ.sub-01.nii   

cp sswarper_output/autobox_anatQQ.sub-01.nii ./
# if this exist, skip skip anat_al_keep_e2a_only_mat.aff12.1D
if [ -f "anat_al_keep_e2a_only_mat.aff12.1D" ]; then
    echo "anat_al_keep_e2a_only_mat.aff12.1D already exists. Skipping matrix extraction step."
else
    time align_epi_anat.py -overwrite \
    -rigid_body \
    -anat_has_skull no \
    -anat2epi \
    -anat ./sswarper_output/anatSS.sub-01.nii \
    -epi afni_mean_sub-01_ses-01_task-localizer_run-1_bold.nii \
    -epi_base 0 \
    -suffix _al

cat_matvec anatSS.sub-01_al_mat.aff12.1D -I -ONELINE > anat_al_keep_e2a_only_mat.aff12.1D
fi

NWARPAPPLY="Warping"
if [ -f "afni_mni_task-localizer_run-1.nii.gz" ]; then
    echo "Warped output for localizer run 1 already exists. Skipping warping step."
else
    time 3dNwarpApply -overwrite \
        -master ./autobox_anatQQ.sub-01.nii \
        -dxyz 3.0  \
        -wsinc5 \
        -nwarp "./sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D afni_moco_sub-01_ses-01_task-localizer_run-1_bold_mat.aff12.1D" \
        -source ../sub-01/ses-01/func/sub-01_ses-01_task-localizer_run-1_bold.nii \
        -prefix afni_mni_task-localizer_run-1.nii.gz
    echo "Run warped to final space "
    echo "-----------------------------------------------------------------"
fi

## FFS nwarp apply here - we want to apply the same stuff, but with FFS - same input, same warps, but our alg. 
if [ -f "ffs_mni_task-localizer_run-1.nii.gz" ]; then
    echo "Warped output for localizer run 1 already exists. Skipping warping step."
else
    time ffs_nwarp \
        -master ./autobox_anatQQ.sub-01.nii \
        -dxyz 3.0  \
        -save_mean \
        -interp wsinc5 \
        -nwarp "./sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D afni_moco_sub-01_ses-01_task-localizer_run-1_bold_mat.aff12.1D" \
        -source ../sub-01/ses-01/func/sub-01_ses-01_task-localizer_run-1_bold.nii \
        -prefix ffs_mni_task-localizer_run-1.nii.gz
    echo "Run warped to final space "
    echo "-----------------------------------------------------------------"
fi

for task in localizer; do
    for run in 2 3 4 5 ; do
        if [ -f "afni_mni_task-${task}_run-${run}.nii.gz" ]; then
            echo "Warped output for task ${task}, run ${run} already exists. Skipping warping step."
        else
            echo "Running warping for task ${task}, run ${run}."
            time 3dNwarpApply -overwrite \
                -master ./autobox_anatQQ.sub-01.nii \
                -dxyz 3.0  \
                -wsinc5 \
                -nwarp "./sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D afni_mean_${task}_run-${run}_to_localizer_run-1_mat.aff12.1D   afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold_mat.aff12.1D" \
                -source ../sub-01/ses-01/func/sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -prefix afni_mni_task-${task}_run-${run}.nii.gz
        fi

        if [ -f "ffs_mni_task-${task}_run-${run}.nii.gz" ]; then
            echo "Warped output for task ${task}, run ${run} already exists. Skipping warping step."
        else
            echo "Running warping for task ${task}, run ${run}."
            time ffs_nwarp \
                -master ./autobox_anatQQ.sub-01.nii \
                -dxyz 3.0  \
                -interp wsinc5 \
                -nwarp "./sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D afni_mean_${task}_run-${run}_to_localizer_run-1_mat.aff12.1D   afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold_mat.aff12.1D" \
                -source ../sub-01/ses-01/func/sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -prefix ffs_mni_task-${task}_run-${run}.nii.gz
        fi

    done
done

for task in rest; do
    for run in 1 2 3 4 5 ; do
        if [ -f "afni_mni_task-${task}_run-${run}.nii.gz" ]; then
            echo "Warped output for task ${task}, run ${run} already exists. Skipping warping step."
        else
            echo "Running warping for task ${task}, run ${run}."
        time 3dNwarpApply -overwrite \
                -master ./autobox_anatQQ.sub-01.nii \
                -dxyz 3.0  \
                -wsinc5 \
                -nwarp "./sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D afni_mean_${task}_run-${run}_to_localizer_run-1_mat.aff12.1D afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold_mat.aff12.1D" \
                -source ../sub-01/ses-01/func/sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -prefix afni_mni_task-${task}_run-${run}.nii.gz
        fi

        if [ -f "ffs_mni_task-${task}_run-${run}.nii.gz" ]; then
            echo "Warped output for task ${task}, run ${run} already exists. Skipping warping step."
        else
            echo "Running warping for task ${task}, run ${run}."
            time ffs_nwarp \
                -master ./autobox_anatQQ.sub-01.nii \
                -dxyz 3.0  \
                -interp wsinc5 \
                -nwarp "./sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D afni_mean_${task}_run-${run}_to_localizer_run-1_mat.aff12.1D   afni_moco_sub-01_ses-01_task-${task}_run-${run}_bold_mat.aff12.1D" \
                -source ../sub-01/ses-01/func/sub-01_ses-01_task-${task}_run-${run}_bold.nii \
                -prefix ffs_mni_task-${task}_run-${run}.nii.gz
        fi

    done
done
# TODO - correlate timeseries? compare means? make sure our data is aligned, and should be near identical
# TODO clock this - everything should be faster!


# Next run the model. 
mkdir -p timing_files
timing_tool.py -write_multi_timing timing_files/onsets.localizer. -multi_timing_ncol_tsv ../sub-01/ses-01/func/sub-01_ses-01_task-localizer_run-*_events.tsv 

# > ls timing_files/
# onsets.localizer.times.bodies.txt     onsets.localizer.times.fix.txt        onsets.localizer.times.scenes.txt
# onsets.localizer.times.faces.txt      onsets.localizer.times.objects.txt    onsets.localizer.times.scrambled.txt

# scale the data
for run in 1 2 3 4 5 ; do
    if [ -f "scaled_afni_mni_task-localizer_run-${run}.nii.gz" ]; then
        echo "Scaled output for localizer run ${run} already exists. Skipping scaling step."
    else
        echo "Running scaling for localizer run ${run}."
        3dTstat -overwrite -prefix mean_this_localizer.nii.gz afni_mni_task-localizer_run-${run}.nii.gz
        3dcalc -overwrite -prefix scaled_afni_mni_task-localizer_run-${run}.nii.gz \
            -a afni_mni_task-localizer_run-${run}.nii.gz \
            -b mean_this_localizer.nii.gz \
            -expr 'a/b*100'
    fi
done

GLMS="Happening"

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
    -stim_times 1 timing_files/onsets.localizer.times.faces.txt 'SPMG1(3)' \
    -stim_label 1 faces \
    -stim_times 2 timing_files/onsets.localizer.times.bodies.txt 'SPMG1(3)' \
    -stim_label 2 bodies \
    -stim_times 3 timing_files/onsets.localizer.times.objects.txt 'SPMG1(3)' \
    -stim_label 3 objects \
    -stim_times 4 timing_files/onsets.localizer.times.scenes.txt 'SPMG1(3)' \
    -stim_label 4 scenes \
    -stim_times 5 timing_files/onsets.localizer.times.scrambled.txt 'SPMG1(3)' \
    -stim_label 5 scrambled \
    -jobs 10 -noFDR  \
    -gltsym 'SYM: +1*faces -1*objects'                                           \
    -glt_label 1 faces_vs_objects \
    -gltsym 'SYM: +1*faces -1*scenes'                                           \
    -glt_label 2 faces_vs_scenes \
    -gltsym 'SYM: +1*faces -1*scrambled'                                           \
    -glt_label 3 faces_vs_scrambled \
    -tout -x1D  X.xmat.1D -bucket afni_stats_localizer.nii.gz

# Next 3dREMLfit
time tcsh afni_stats_localizer.REML_cmd -overwrite -nofdr 

## Repeat with FFS (note we use AFNI input again). 
# TODO - ffs_reml arg parse doesn't produce help (says need input, but help check should trigger before that, and should show help. )
# First we check using the AFNI Matrix directly
time ffs_reml \
    -input \
        scaled_afni_mni_task-localizer_run-1.nii.gz \
        scaled_afni_mni_task-localizer_run-2.nii.gz \
        scaled_afni_mni_task-localizer_run-3.nii.gz \
        scaled_afni_mni_task-localizer_run-4.nii.gz \
        scaled_afni_mni_task-localizer_run-5.nii.gz \
    -matrix X.xmat.1D \
    -use_double \
    -Obuck ffs_stats_localizer.nii.gz \
    -tout

# TODO compare betas, etc betweent he OLS results of afni and ffs. 

# TODO - check for torch compile, jit numba etc improvements in this mixed CPU/GPU code. 
# TODO - for example , this batches over grid search - 12 runs of this, eac one taking 14 seconds. If the first was double, and then the others were half, that would be a big win. 
# Next, actually do the REML step with ffs
time ffs_reml \
    -input \
        scaled_afni_mni_task-localizer_run-1.nii.gz \
        scaled_afni_mni_task-localizer_run-2.nii.gz \
        scaled_afni_mni_task-localizer_run-3.nii.gz \
        scaled_afni_mni_task-localizer_run-4.nii.gz \
        scaled_afni_mni_task-localizer_run-5.nii.gz \
    -matrix X.xmat.1D \
    -use_double \
    -Rbuck ffs_stats_localizer_REML.nii.gz \
    -Rvar ffs_stats_localizer_REML_var.nii.gz \
    -tout

# TODO compare betas, etc betweent he REML results of afni and ffs.
# This includes the a/b  paramaeters in the reml_var files from AFNI and FFS. 

ICA="ICA"
#  Then we  examine ICA - each run separately, and then all runs together.
if [ -f "all_rest_melodic.ica/melodic_pcaD" ]; then
    echo "Melodic output for rest runs already exists. Skipping melodic step."
else
    time melodic -i afni_mni_task-rest_run-1.nii.gz,afni_mni_task-rest_run-2.nii.gz,afni_mni_task-rest_run-3.nii.gz,afni_mni_task-rest_run-4.nii.gz,afni_mni_task-rest_run-5.nii.gz \
    -o all_rest_melodic.ica --bgthreshold=3 --tr=1.7500 --report --guireport=all_rest_melodic.ica/report.html \
    -d 0 --mmthresh=0.5 --Oall --Ostats -v
fi

if [ -f "all_localizer_melodic.ica/melodic_pcaD" ]; then
    echo "Melodic output for localizer runs already exists. Skipping melodic step."
else
    time melodic -i afni_mni_task-localizer_run-1.nii.gz,afni_mni_task-localizer_run-2.nii.gz,afni_mni_task-localizer_run-3.nii.gz,afni_mni_task-localizer_run-4.nii.gz,afni_mni_task-localizer_run-5.nii.gz \
        -o all_localizer_melodic.ica --bgthreshold=3 --tr=1.7500 --report --guireport=all_localizer_melodic.ica/report.html \
        -d 0 --mmthresh=0.5 --Oall --Ostats -v 

fi

#############
#### FFS ICA VERSIONS
#####
if [ -f "all_rest_ffs.ica/melodic_pcaD" ]; then
    echo "Melodic output for rest runs already exists. Skipping melodic step."
else
    # TODO - progress bar for iterations, leave on screen. 
    # TODO - leave GMM progress bar on screen as well - also, does this converge fully? or does it miss some? 
    time ffs_ica -input afni_mni_task-rest_run-1.nii.gz \
        afni_mni_task-rest_run-2.nii.gz \
        afni_mni_task-rest_run-3.nii.gz \
        afni_mni_task-rest_run-4.nii.gz \
        afni_mni_task-rest_run-5.nii.gz \
        -mask all_rest_melodic.ica/mask.nii.gz \
        -temp_concat \
    -prefix all_rest_ffs -verbose
fi

if [ -f "all_localizer_ffs.ica/melodic_pcaD" ]; then
    echo "Melodic output for localizer runs already exists. Skipping melodic step."
else
    time ffs_ica -input afni_mni_task-localizer_run-1.nii.gz \
        afni_mni_task-localizer_run-2.nii.gz \
        afni_mni_task-localizer_run-3.nii.gz \
        afni_mni_task-localizer_run-4.nii.gz \
        afni_mni_task-localizer_run-5.nii.gz \
        -mask all_localizer_melodic.ica/mask.nii.gz \
        -temp_concat \
        -prefix all_localizer_ffs -verbose

fi

#  Now run GLMsingle, at  each stage, and reproduce the same steps, final result with ffs. 


