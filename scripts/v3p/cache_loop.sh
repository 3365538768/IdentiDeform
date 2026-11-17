#!/bin/bash
# V3p: incremental V-JEPA caching while generation runs (a800-4, nohup). Every cycle: manifest of every completed
# sample (val, test, train < TRAIN_MAX) -> cache_v3.py skips files that exist. Stops when docs/v3p/CACHE_LOOP_STOP exists.
#   nohup bash scripts/v3p/cache_loop.sh 4000 0 > docs/v3p/logs/cache_loop.log 2>&1 &
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test && source scripts/r1_env.sh
TRAIN_MAX=${1:-4000}; GPU=${2:-0}
while [ ! -e docs/v3p/CACHE_LOOP_STOP ]; do
  date -u +%H:%M:%S
  .venv/bin/python scripts/v3p/build_manifest.py --name v3p_cache_inc --train-max $TRAIN_MAX && \
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python scripts/v3p/cache_v3.py --manifest v3p_cache_inc --status-output docs/v3p/cache_inc.json 2>&1 | grep -v Warning | tail -2
  cat docs/v3p/cache_inc.json | tr -d '\n '; echo
  sleep 240
done
