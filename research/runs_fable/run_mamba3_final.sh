#!/bin/bash
# Finalization suite: author's Mamba-3-inspired ideas on the working stack.
# Controls on disk: fr_np64_r1 (conv) = 0.977; fr3_np64_r1_P32R16_ste = 0.926;
# fr2_np64_r4_ortho = 0.922.
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/testbed_mqar.py "$@" --steps 6000 --eval_every 1000 --keys_vocab 128 \
     --n_pairs 64 --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "^step|final_tokacc"
}
run m3_rot            --r 1 --rot
run m3_rot_trap       --r 1 --rot --trap
run m3_rout4          --r 1 --r_out 4
run m3_r4_rout4_ortho --r 4 --r_out 4 --slot_ortho 0.1
run m3_rot_ste        --r 1 --rot --compact_P 32 --compact_R 16 --compact_ste
echo "SUITE DONE"
