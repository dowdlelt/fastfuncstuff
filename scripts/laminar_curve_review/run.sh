#!/usr/bin/env bash
# Question: do zero-pinned / finer knots change the real ROI x depth curves (the negative lag-0 knot)?
set -uo pipefail
R=/mnt/belegost/Projects/HiHiFaces/fine_outputs/ffs_autoproc_sub-3010_ses-fine_nonlmoco.results
SUBJ=/mnt/belegost/Data/fs_subjects/sub-3010_acq-3t_HiHiFaces
OUT=$1; PY=/home/logan/miniconda3/envs/py314_ffs/bin/python
EV=(); MOT=(); IN=(); i=0
for t in expres fixate gender; do for r in 1 2 3; do
  i=$((i + 1))
  EV+=($R/stimuli/sub-3010_ses-fine_task-${t}_run-${r}_events.tsv)
  MOT+=(-ortvec_run $R/stage02.moco.ses-fine.task-$t.run-$r.motion.1D motion $i)
  IN+=($R/stage10.final.ses-fine.task-$t.run-$r.nii.zst)
done; done
fit(){ name=$1; shift
  ( time $PY -m fastfuncstuff.cli.deconvolve -input "${IN[@]}" -events "${EV[@]}" -event_cols onset duration stim_contrast -event_prefix task "${MOT[@]}" \
      -window 0 14.8 -mask $R/stage10.mask_epi.nii.zst -atlas $R/roi_lh_v1.lh.label $R/roi_rh_ffa.rh.label -fs_subj $SUBJ -depths 7 9 11 -equivolume \
      -device cuda -prefix $OUT/$name "$@" ) > $OUT/log_$name.txt 2>&1
  echo "$name exit $? $(grep real $OUT/log_$name.txt)"; }
fit tz17g -model TENTzero -tent-n-basis 17 -tent-smooth global
fit ols17 -model TENT -tent-n-basis 17
fit cz9 -model CSPLINzero
fit ols9 -model TENT
