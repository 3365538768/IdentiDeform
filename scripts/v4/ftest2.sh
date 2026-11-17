#!/usr/bin/env bash
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
rm -rf data/v4_forcetest
k=0
for i in 6 0; do
  nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((k+6)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i --root data/v4_forcetest --g-all -2" > /dev/null 2>&1 < /dev/null &
  k=$((k+1))
done
