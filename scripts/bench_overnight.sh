#!/usr/bin/env bash
# Work through the benchmark worklist unattended until a stop time.
#
#   scripts/bench_overnight.sh [UNTIL=07:00] [DEVICE=cuda] [extra work -run flags]
#   scripts/bench_overnight.sh 07:00 cpu -workers 0 -threads 6   # core-rich CPU host
#
# -workers N -threads T runs N workers side by side, each pinned to T physical cores and
# every tool told to use T threads (-workers 0 = as many as fit the cores / CPU quota).
#
# Runs as a transient systemd user unit so it survives the terminal (and any Claude
# session) closing, with a memory cap: an uncapped run once let oomd kill the whole
# terminal (memory: cap memory on ad-hoc runs). Plan units first with
#   python -m fastfuncstuff.benchmark.work -plan CHECK -datasets ... -n N
# Morning: python -m fastfuncstuff.benchmark.work -status / -results CHECK
#          journalctl --user -u <unit>   or the log under $ROOT/logs/
set -euo pipefail

UNTIL="${1:-07:00}"
DEVICE="${2:-cuda}"
shift $(( $# < 2 ? $# : 2 ))
EXTRA="${*:-}"
# whole-run cap; this host has 251 GB. Lower it (and FFS_MATLAB_MAX_GB) on smaller hosts.
PY="${FFS_PYTHON:-/home/logan/miniconda3/envs/py314_ffs/bin/python}"
: "${FFS_BENCHMARK_DATA_DIR:?set FFS_BENCHMARK_DATA_DIR (the bench root sits beside it)}"
ROOT="${FFS_BENCH_ROOT:-$(dirname "$FFS_BENCHMARK_DATA_DIR")/bench}"
mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d-%H%M)"
LOG="$ROOT/logs/night-$STAMP.log"
UNIT="ffs-bench-$STAMP"

systemd-run --user --unit="$UNIT" --collect \
    -p MemoryMax="${FFS_BENCH_MEMORY_MAX:-200G}" -p MemorySwapMax=0 \
    --setenv=PATH --setenv=FFS_TPM --setenv=FSLDIR --setenv=FSLOUTPUTTYPE --setenv=FFS_BENCHMARK_DATA_DIR --setenv=FFS_MATLAB_MAX_GB \
    --setenv=FFS_BENCH_ROOT="$ROOT" --setenv=PYTHONUNBUFFERED=1 \
    --working-directory="$ROOT" \
    bash -c "exec '$PY' -m fastfuncstuff.benchmark.work -run -until '$UNTIL' -device '$DEVICE' $EXTRA >> '$LOG' 2>&1"

echo "started $UNIT until $UNTIL on $DEVICE $EXTRA"
echo "log:    $LOG"
echo "stop:   systemctl --user stop $UNIT"
