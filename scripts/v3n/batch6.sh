#!/usr/bin/env bash
# V3 review round 3b (a800-4): press / press-hold-release / bend / stretch with one-sided press plate and grip-aware axes
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
IDX="0 1 2 18 19 20 36 37 38 54 55 56 15 16 17 33 34 35 51 52 53 69 70 71 12 13 14 30 31 32 48 49 50 66 67 68 9 10 11 27 28 29 45 46 47 63 64 65"
for i in $IDX; do rm -rf data/v3n_r3/val/$(printf %06d $i); done
g=0
for i in $IDX; do
  nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i --root data/v3n_r3" > docs/v3n/logs/r3b_$i.log 2>&1 < /dev/null &
  g=$((g+1)); sleep 1
done
