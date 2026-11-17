#!/usr/bin/env bash
# Probe 3 (11-15 16:25): press / press-hold-release with fast release + sticky bed, E/rho x1 vs x4
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
IDX="0 1 2 18 15 16 17 33"
g=0
run() { nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py $*" > /dev/null 2>&1 < /dev/null & g=$((g+1)); sleep 1; }
for i in $IDX; do
  run --split val --indices $i --root data/v3n_probe/R1 --release fast
  run --split val --indices $i --root data/v3n_probe/R4 --release fast --e-scale 4
done
