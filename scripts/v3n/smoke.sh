#!/usr/bin/env bash
# V3 contact-setup smoke test: 15 val samples (5 actions x 3 laws) on a800-4
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
rm -rf data/v3_main/val data/v3n_dbg scripts/v3n/_dbg_gen_*.py; mkdir -p docs/v3n/logs
g=0
for i in 0 1 2 6 7 8 9 10 11 12 13 14 15 16 17; do
  nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices $i" > docs/v3n/logs/smoke_$i.log 2>&1 < /dev/null &
  g=$((g+1)); sleep 1
done
