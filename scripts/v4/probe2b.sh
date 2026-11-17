#!/usr/bin/env bash
# Probe 2b (11-15 17:10): held weak actions with a firm grip (25 % segment, 30 % closure), E/rho x1 vs x4
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
IDX="6 7 8 24 9 10 11 27 12 13 14 30"
g=0
run() { nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py $*" > /dev/null 2>&1 < /dev/null & g=$((g+1)); sleep 1; }
for i in $IDX; do
  run --split val --indices $i --root data/v3n_probe/K1 --g-all -2 --weak hold
  run --split val --indices $i --root data/v3n_probe/K4 --g-all -2 --weak hold --e-scale 4
done
