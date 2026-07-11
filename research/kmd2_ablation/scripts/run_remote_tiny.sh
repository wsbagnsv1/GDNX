#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="$ROOT_DIR/research/kmd2_ablation/configs/smoke.json"
OUT=""
DEVICE="cpu"
JOB_INDEX=0
NUM_JOBS=1
SUMMARIZE=0

usage() {
  echo "usage: $0 --out PATH [--config PATH] [--device cpu|cuda:N] [--job-index N] [--num-jobs N] [--summarize]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --job-index) JOB_INDEX="$2"; shift 2 ;;
    --num-jobs) NUM_JOBS="$2"; shift 2 ;;
    --summarize) SUMMARIZE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$OUT" ]]; then
  usage
  exit 2
fi

cd "$ROOT_DIR"
python -m research.kmd2_ablation.run_ablation preflight \
  --backend tiny --config "$CONFIG" --out "$OUT" \
  --device "$DEVICE" --dtype float32 \
  --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" --resume --dry-run
python -m research.kmd2_ablation.run_ablation run \
  --backend tiny --config "$CONFIG" --out "$OUT" \
  --device "$DEVICE" --dtype float32 \
  --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" --resume
if [[ "$NUM_JOBS" -eq 1 || "$SUMMARIZE" -eq 1 ]]; then
  python -m research.kmd2_ablation.run_ablation summarize \
    --backend tiny --config "$CONFIG" --out "$OUT" \
    --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" --resume
else
  echo "sharded worker complete; run one post-array coordinator with --summarize" >&2
fi
