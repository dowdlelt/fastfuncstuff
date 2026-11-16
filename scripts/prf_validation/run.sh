#!/usr/bin/env bash
# ffs_pyrf vs analyzePRF on one subject of ds003787 (NYU retinotopy).
# Produces everything docs/prf_validation.md reports. Steps are skipped when
# their output exists, so a rerun only redoes what was deleted.
#
#   scripts/prf_validation/run.sh [WORK]
#
# Needs: aws CLI (anonymous S3), MATLAB with the Parallel Computing Toolbox,
# analyzePRF at ~/Dropbox/Resources/code/matlab_toolboxes/analyzePRF.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-/home/logan/miniconda3/envs/py314_ffs/bin/python}
BIN=$(dirname "$PY")
DATA=${FFS_BENCHMARK_DATA_DIR:-$REPO/test_data}/ds003787-download
WORK=${1:-$DATA/derivatives/ffs_prf_validation}
SCREEN=${SCREEN:-0.1}        # screening R2 a vertex needs to be handed to analyzePRF
NWORKERS=${NWORKERS:-6}      # analyzePRF parpool size (one per physical core)
RUNS="01 02 03 04 05 06"
SUB=sub-wlsubj001; SES=ses-nyu3t01
S3=s3://openneuro.org/ds003787
mkdir -p "$WORK"

# 1. Download: 6 of the 12 identical bar runs, fsnative surface series (~1.4 GB).
fetch() { [ -s "$DATA/$1" ] || aws s3 cp --no-sign-request --quiet "$S3/$1" "$DATA/$1"; }
for r in $RUNS; do for h in L R; do
  fetch derivatives/fmriprep/$SUB/$SES/func/${SUB}_${SES}_task-prf_acq-PA_run-${r}_space-fsnative_hemi-$h.func.mgz
done; fetch derivatives/stimulus_apertures/$SUB/$SES/${SUB}_${SES}_task-prf_run-${r}_aperture.mat; done
fetch $SUB/$SES/func/${SUB}_${SES}_task-prf_acq-PA_run-01_bold.json
for h in lh rh; do
  fetch derivatives/ROIs/$SUB/$h.ROIs_V1-4.mgz
  for m in angle eccen sigma vexpl x y; do fetch derivatives/prfanalyze-vista/$SUB/$SES/$h.$m.mgz; done
  for s in white pial sphere sulc; do fetch derivatives/freesurfer/$SUB/surf/$h.$s; done
done

# 2. Same numbers for both tools: GIfTI series + the .mat aperture as .npy.
[ -s "$WORK/meta.json" ] || "$PY" "$HERE/prepare.py" "$DATA" "$WORK"
INPUTS=$(for r in $RUNS; do printf '%s ' "$WORK/runs/run-$r.lh.func.gii"; done)
APER="$WORK/aperture.npy"
STIMS=$(for r in $RUNS; do printf '%s ' "$APER"; done)

pyrf() {  # pyrf PREFIX DEVICE EXTRA... ; wall time to PREFIX.time
  local prefix=$1 device=$2; shift 2
  [ -s "$prefix.time" ] && return
  local t0=$(date +%s.%N) started=$(date -Iseconds) commit=$(git -C "$REPO" rev-parse HEAD)
  local dirty=$(git -C "$REPO" status --porcelain --untracked-files=no -- fastfuncstuff | wc -l)
  "$BIN/ffs_pyrf" -input $INPUTS -stimulus $STIMS -prefix "$prefix" -device "$device" "$@" \
    2>&1 | tee "$prefix.log"
  echo "$(date +%s.%N) - $t0" | bc > "$prefix.time"
  # Which code produced this fit: a rerun skips it while the .time file exists.
  {
    echo "started: $started"
    echo "commit: $commit"
    echo "dirty: $dirty tracked files changed under fastfuncstuff/"
    echo "device: $device"
    echo "host: $(hostname)"
    echo "args: $*"
  } > "$prefix.provenance"
}

# 3. Whole cortex, both hemispheres, screened: the configuration a user runs,
#    timed end to end. Its screen map is the vertex set analyzePRF gets.
pyrf "$WORK/ffs_cortex_gpu" cuda -hrf canonical -screen "$SCREEN" -save_screen -xval none

# 4. analyzePRF on exactly the vertices the screen passed.
[ -s "$WORK/apr_input.mat" ] || "$PY" "$HERE/export_for_matlab.py" "$WORK" "$WORK/ffs_cortex_gpu" --screen "$SCREEN"
[ -s "$WORK/apr_output.mat" ] || matlab -batch \
  "addpath('$HERE'); run_analyzeprf('$WORK/apr_input.mat', '$WORK/apr_output.mat', $NWORKERS)" \
  2>&1 | tee "$WORK/apr.log"

# 5. Parity: analyzePRF's vertices, its model choices (canonical HRF, CSS staged
#    exponent, polynomial degree 2) and its units (no percent scaling), so the
#    two fits see identical numbers.
[ -s "$WORK/runs/aprmask.lh.shape.gii" ] || "$PY" "$HERE/write_mask.py" "$WORK" "$WORK/apr_input.mat"
pyrf "$WORK/ffs_parity" cuda -hrf canonical -mask "$WORK/runs/aprmask.lh.shape.gii" -dont_scale -xval none

# 6. What the options buy, each with split-half cross-validation on the same screen:
#    canonical HRF, the HRF library, data-driven denoising, and both.
pyrf "$WORK/ffs_canon"    cuda -hrf canonical -screen "$SCREEN" -xval halves
pyrf "$WORK/ffs_lib"      cuda -hrf library   -screen "$SCREEN" -xval halves
pyrf "$WORK/ffs_canon_dn" cuda -hrf canonical -screen "$SCREEN" -xval halves -denoise -save_denoise
pyrf "$WORK/ffs_lib_dn"   cuda -hrf library   -screen "$SCREEN" -xval halves -denoise -save_denoise

# 7. Figures and numbers.
"$PY" "$HERE/compare.py" "$DATA" "$WORK" "$REPO/docs/prf_validation"
"$PY" "$HERE/variants.py" "$DATA" "$WORK" "$REPO/docs/prf_validation"
