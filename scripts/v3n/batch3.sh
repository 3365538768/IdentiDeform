#!/usr/bin/env bash
# V3 review batch 3 (a800-4): press / press-hold-release with three press modes
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
g=0
for mode in sticky gafter gsmall; do
  for i in 0 1 2 18 19 20 15 16 17 33 34 35; do
    nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i --root data/v3n_press_$mode --press-mode $mode" > docs/v3n/logs/press_${mode}_$i.log 2>&1 < /dev/null &
    g=$((g+1)); sleep 1
  done
done
