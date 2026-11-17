#!/usr/bin/env bash
# V3 review batch 2: grip-based shear/stretch (6-11) + one more action cycle (18-35)
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
for i in 6 7 8 9 10 11; do rm -rf data/v3_main/val/$(printf %06d $i); done
g=0
for i in 6 7 8 9 10 11 $(seq 18 35); do
  nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i" > docs/v3n/logs/smoke_$i.log 2>&1 < /dev/null &
  g=$((g+1)); sleep 1
done
