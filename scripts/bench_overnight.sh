#!/usr/bin/env bash
# Work through the benchmark worklist unattended until a stop time.
#
#   scripts/bench_overnight.sh [UNTIL=07:00] [DEVICE=cuda]
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
PY="${FFS_PYTHON:-/home/logan/miniconda3/envs/py314_ffs/bin/python}"
: "${FFS_BENCHMARK_DATA_DIR:?set FFS_BENCHMARK_DATA_DIR (the bench root sits beside it)}"
ROOT="${FFS_BENCH_ROOT:-$(dirname "$FFS_BENCHMARK_DATA_DIR")/bench}"
mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d-%H%M)"
LOG="$ROOT/logs/night-$STAMP.log"
UNIT="ffs-bench-$STAMP"

systemd-run --user --unit="$UNIT" --collect \
    -p MemoryMax=96G -p MemorySwapMax=0 \
    --setenv=PATH --setenv=FSLDIR --setenv=FSLOUTPUTTYPE --setenv=FFS_BENCHMARK_DATA_DIR \
    --setenv=FFS_BENCH_ROOT="$ROOT" --setenv=PYTHONUNBUFFERED=1 \
    --working-directory="$ROOT" \
    bash -c "exec '$PY' -m fastfuncstuff.benchmark.work -run -until '$UNTIL' -device '$DEVICE' >> '$LOG' 2>&1"

echo "started $UNIT until $UNTIL on $DEVICE"
echo "log:    $LOG"
echo "stop:   systemctl --user stop $UNIT"
