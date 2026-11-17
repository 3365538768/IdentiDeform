#!/usr/bin/env bash
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
rm -rf data/v4_ftest/F30
k=0
for F in 30; do for i in 0 1 2; do
  nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((k%8)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i --root data/v4_ftest/F$F --g-all -2 --press-force $F" > /dev/null 2>&1 < /dev/null &
  k=$((k+1)); done; done
