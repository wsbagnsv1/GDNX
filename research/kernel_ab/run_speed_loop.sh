#!/bin/bash
# A/B kernel-speed auto-research loop — one competitor.
# Restart-based stateless-agent + disk-ledger (same pattern as run_auto_research.sh):
# each iteration is ONE bounded `pi -p` turn that makes one concrete speed improvement
# to research/kernel_ab/<WS>/cand_scan.py, benches it (appends to <WS>/leaderboard.jsonl),
# updates <WS>/notes.md, and exits. State lives on disk so context compaction never
# matters and a crash only costs one turn.
#
# Usage:
#   PROVIDER=glm MODEL=canada-quant/GLM-5.2-W4A16-MTP GPU=0 SESSION=kernel-glm WS=glm \
#     nohup bash research/kernel_ab/run_speed_loop.sh > research/kernel_ab/glm/loop.out 2>&1 &
# Stop: touch research/kernel_ab/<WS>/STOP   (checked between turns)  or kill the process.
set -u
cd /home/dev/gdn3_fable
export PATH="/home/dev/.local/share/pi-node/node-v22.23.1-linux-x64/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU   # in-process this remaps to cuda:0 -> pins this loop to physical GPU $GPU

WS_DIR="research/kernel_ab/$WS"
LOG="$WS_DIR/loop.log"

PROMPT="Do ONE bounded unit of KMD-2 scan-kernel speed optimization this turn, per \
research/kernel_ab/BRIEF.md. Your workspace is research/kernel_ab/$WS/ (edit ONLY files \
in there). Your GPU is $GPU (pass --device cuda:0; CUDA_VISIBLE_DEVICES already pins it). \
First skim research/kernel_ab/$WS/leaderboard.jsonl and research/kernel_ab/$WS/notes.md so \
you don't repeat work. Then: form one hypothesis, edit research/kernel_ab/$WS/cand_scan.py, \
bench it with research/kernel_ab/bench_scan.py (--cand research/kernel_ab/$WS/cand_scan.py \
--leaderboard research/kernel_ab/$WS/leaderboard.jsonl --device cuda:0 --note '<desc>'), and \
append your result + next idea to research/kernel_ab/$WS/notes.md. The candidate must stay \
correct vs research/kernel_ab/ref_scan.py (fwd relMSE<2e-3, grad relMSE<1e-2) or it is \
DISQUALIFIED. Do exactly one improvement, then STOP."

i=0
while true; do
  [ -f "$WS_DIR/STOP" ] && { echo "STOP present — exiting @ $(date -Is)" | tee -a "$LOG"; break; }
  i=$((i+1))
  echo "===== [$WS] turn $i @ $(date -Is) =====" >> "$LOG"
  timeout 3000 pi -p --session-id "gdn3-$SESSION" --approve \
    --provider "$PROVIDER" --model "$MODEL" \
    --tools bash,read,write,edit \
    "$PROMPT" >> "$LOG" 2>&1
  echo "----- [$WS] turn $i done rc=$? @ $(date -Is) -----" >> "$LOG"
  sleep 8
done
