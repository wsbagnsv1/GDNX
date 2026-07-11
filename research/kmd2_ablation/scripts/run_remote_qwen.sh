#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="$ROOT_DIR/research/kmd2_ablation/configs/qwen_exact_cache.json"
MODEL=""
TOKENIZER=""
NATIVE_CHECKPOINT=""
DATA=""
TEACHER_MODEL=""
ASSETS_MANIFEST=""
OUT=""
STUDENT_DEVICE="cuda:0"
TEACHER_DEVICE="cuda:0"
DTYPE="bfloat16"
JOB_INDEX=0
NUM_JOBS=1
SMOKE=0
SUMMARIZE=0

usage() {
  echo "usage: $0 [--smoke] --model PATH --native-checkpoint PATH --data PATH --out PATH [--teacher-model PATH] [--tokenizer PATH] [--assets-manifest PATH] [--config PATH] [--student-device cuda:N] [--teacher-device cuda:N] [--dtype bfloat16|float32] [--job-index N] [--num-jobs N] [--summarize]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --native-checkpoint) NATIVE_CHECKPOINT="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --assets-manifest) ASSETS_MANIFEST="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --student-device) STUDENT_DEVICE="$2"; shift 2 ;;
    --teacher-device) TEACHER_DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --job-index) JOB_INDEX="$2"; shift 2 ;;
    --num-jobs) NUM_JOBS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --summarize) SUMMARIZE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$NATIVE_CHECKPOINT" || -z "$DATA" || -z "$OUT" ]]; then
  usage
  exit 2
fi
if [[ "$SMOKE" -eq 0 && -z "$TEACHER_MODEL" ]]; then
  echo "--teacher-model is required outside --smoke" >&2
  exit 2
fi

if [[ "$SMOKE" -eq 1 ]]; then
  SMOKE_CONFIG="$OUT/.generated/qwen-smoke.json"
  python - "$CONFIG" "$SMOKE_CONFIG" <<'PY'
import json
import os
import sys
import uuid
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
raw = json.loads(source.read_text(encoding="utf-8"))
raw["task"]["params"]["objective"] = "synthetic_only"
raw["task"]["params"]["example_ids"] = raw["task"]["params"]["example_ids"][:1]
raw["task"]["params"]["training_window_example_counts"] = [1]
raw["task"]["params"]["training_window_token_counts"] = [64]
raw["task"]["params"]["episodes_per_cell"] = 1
raw["task"]["params"]["free_generation_subset_per_cell"] = 1
raw["budget"] = {"tokens": 64, "updates": 1}
raw["lengths"] = {"curriculum": [64], "extrapolation": [64]}
raw["cache"]["width"] = 4
raw["cache"]["block_size"] = 16
raw["runtime"]["output_path"] = "outputs/qwen-smoke"
payload = json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n"
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    if destination.read_text(encoding="utf-8") != payload:
        raise SystemExit("existing generated smoke config conflicts")
else:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
PY
  CONFIG="$SMOKE_CONFIG"
fi

NATIVE_R_OUT="$(python - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["task"]["params"].get("native_r_out")
if type(value) is not int or value < 1:
    raise SystemExit("task.params.native_r_out must be a positive integer")
print(value)
PY
)"
export GDN3_FAST_SCAN=1
export GDN3_KMD2_ROUT="$NATIVE_R_OUT"

COMMON=(
  --backend qwen --mode heal --config "$CONFIG" --out "$OUT"
  --model "$MODEL" --native-checkpoint "$NATIVE_CHECKPOINT" --data "$DATA"
  --student-device "$STUDENT_DEVICE" --dtype "$DTYPE"
  --job-index "$JOB_INDEX" --num-jobs "$NUM_JOBS" --resume
)
if [[ -n "$TOKENIZER" ]]; then COMMON+=(--tokenizer "$TOKENIZER"); fi
if [[ -n "$ASSETS_MANIFEST" ]]; then COMMON+=(--assets-manifest "$ASSETS_MANIFEST"); fi
if [[ -n "$TEACHER_MODEL" ]]; then
  COMMON+=(--teacher-model "$TEACHER_MODEL" --teacher-device "$TEACHER_DEVICE")
fi

cd "$ROOT_DIR"
python -m research.kmd2_ablation.run_ablation preflight "${COMMON[@]}" --dry-run
python -m research.kmd2_ablation.run_ablation run "${COMMON[@]}"
if [[ "$NUM_JOBS" -eq 1 || "$SUMMARIZE" -eq 1 ]]; then
  python -m research.kmd2_ablation.run_ablation summarize "${COMMON[@]}"
else
  echo "sharded worker complete; run one post-array coordinator with --summarize" >&2
fi
