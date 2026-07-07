#!/bin/bash
# Seed replication for the two decision-relevant signals before finalizing:
# (a) rotation-under-compaction (+2.7 pts?), (b) output-MIMO widening (+0.7 pts?).
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/testbed_mqar.py "$@" --steps 6000 --eval_every 3000 --keys_vocab 128 \
     --n_pairs 64 --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "final_tokacc"
}
for s in 1 2; do
  run sd${s}_ste       --r 1 --compact_P 32 --compact_R 16 --compact_ste --seed $s
  run sd${s}_rot_ste   --r 1 --rot --compact_P 32 --compact_R 16 --compact_ste --seed $s
  run sd${s}_r1        --r 1 --seed $s
  run sd${s}_rout4     --r 1 --r_out 4 --seed $s
done
echo "SUITE DONE"
