#!/usr/bin/env bash
# V3 review round 3 (a800-4 part): final setup (support bed, grippers, g=2 press/drop, v3 ranges, screened objects)
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
rm -rf data/v3n_r3
g=0
for i in $(seq 0 39); do
  nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i --root data/v3n_r3" > docs/v3n/logs/r3_$i.log 2>&1 < /dev/null &
  g=$((g+1)); sleep 1
done
