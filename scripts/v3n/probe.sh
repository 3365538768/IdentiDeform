#!/usr/bin/env bash
# Identifiability probe (11-15): how much does the video change when E/rho is x4, V3 setup vs V3 + g=2 on every action
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
IDX="0 1 2 18 3 4 5 21 6 7 8 24 9 10 11 27 12 13 14 30 15 16 17 33"
g=0; GPUS=(0 1 2 3 5 6 7)
run() { nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=${GPUS[$((g%7))]} && .venv/bin/python scripts/v3n/gen_v3.py $*" > /dev/null 2>&1 < /dev/null & g=$((g+1)); sleep 1; }
for i in $IDX; do
  run --split val --indices $i --root data/v3n_probe/A4 --e-scale 4
  run --split val --indices $i --root data/v3n_probe/G1 --g-all -2
  run --split val --indices $i --root data/v3n_probe/G4 --g-all -2 --e-scale 4
done
