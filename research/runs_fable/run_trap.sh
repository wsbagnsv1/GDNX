#!/bin/bash
# Conv vs Mamba-3 trapezoidal: does the principled width-2 in-recurrence conv
# (trap + channel biases) replace the external short conv?  (arXiv:2603.15569)
cd /home/dev/gdn3_fable
PY=/home/dev/gdn3_qwen35_package/.venv/bin/python
run() {
  name=$1; shift
  echo "=== $name ($*) ==="
  $PY research/testbed_mqar.py "$@" --steps 6000 --eval_every 1000 --keys_vocab 128 \
     --device cuda:1 --out research/runs_fable/$name.json 2>&1 \
     | grep -E "^step|final_tokacc"
}
run tr_np16_noconv_trap --n_pairs 16 --r 1 --no_conv --trap
run tr_np64_noconv_trap --n_pairs 64 --r 1 --no_conv --trap
run tr_np64_conv_trap   --n_pairs 64 --r 1 --trap
echo "SUITE DONE"
