#!/usr/bin/env bash
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
rm -rf data/v3n_sanity
for i in 0 18 9; do
  nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((i%8)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i --root data/v3n_sanity" > docs/v3n/logs/sanity_$i.log 2>&1 < /dev/null &
done
