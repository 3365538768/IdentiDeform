#!/usr/bin/env bash
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
for i in 7 6; do rm -rf data/v3n_dbg/val/$(printf %06d $i); done
nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=7 && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices 7 --root data/v3n_dbg" > docs/v3n/logs/dbg7.log 2>&1 < /dev/null &
nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=6 && .venv/bin/python scripts/v3n/gen_v3.py --split val --indices 6 --root data/v3n_dbg" > docs/v3n/logs/dbg6.log 2>&1 < /dev/null &
