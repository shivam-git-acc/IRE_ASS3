#!/usr/bin/env bash
# Regenerate every number, figure and table in the report from the data files.
#
#   bash run_all.sh            full run: SIFT-1M + GloVe-100, all five parts
#   bash run_all.sh --quick    small subset, under 15 minutes on a laptop
#
# Everything is written under ./out (results, figures, logs) and ./scratch (downloaded
# datasets, .npy arrays). Neither is committed. A Hugging Face token is NOT required: if
# HF_TOKEN is absent the mirror is skipped and all output stays local.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

QUICK=""
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK="--quick" ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

export A3_OUT="${A3_OUT:-$HERE/out}"
export A3_SCRATCH="${A3_SCRATCH:-$HERE/scratch}"
mkdir -p "$A3_OUT/results" "$A3_OUT/figures" "$A3_OUT/logs" "$A3_SCRATCH"

PY="${PYTHON:-python3}"
LOG="$A3_OUT/logs/run_all_$(date +%Y%m%d_%H%M%S).log"

banner() { printf '\n=== %s ===\n' "$1" | tee -a "$LOG"; }
stage()  { banner "$1"; shift; "$@" 2>&1 | tee -a "$LOG"; }

banner "environment"
{
  "$PY" -c "import sys; print('python', sys.version.split()[0])"
  "$PY" -c "import faiss; print('faiss', faiss.__version__)"
  "$PY" -c "import numpy; print('numpy', numpy.__version__)"
  echo "mode: ${QUICK:---full}"
  echo "out:  $A3_OUT"
} 2>&1 | tee -a "$LOG"

banner "unit tests for the recall metric"
"$PY" -m pytest -q tests/ 2>&1 | tee -a "$LOG"

stage "stage 0 — data + exact ground truth" "$PY" prepare_data.py $QUICK
stage "part 1 — exact baseline and timing"  "$PY" part1_exact.py $QUICK
stage "part 2 — SIFT"                       "$PY" part2_benchmark.py --dataset sift $QUICK

if [ -z "$QUICK" ]; then
  # GloVe is the expensive half (~3 h); quick mode covers SIFT only, which is enough to
  # exercise every code path end to end.
  stage "part 2 — GloVe" "$PY" part2_benchmark.py --dataset glove
fi

stage "part 3 — scaling"       "$PY" part3_scaling.py $QUICK
stage "part 4 — hard geometry" "$PY" part4_geometry.py $QUICK
stage "stage 6 — figures and tables" "$PY" make_report_assets.py

banner "done"
{
  echo "results : $A3_OUT/results/results.csv"
  echo "figures : $A3_OUT/figures/"
  echo "log     : $LOG"
  ls -1 "$A3_OUT/figures" 2>/dev/null | sed 's/^/  /'
} 2>&1 | tee -a "$LOG"
