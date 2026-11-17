#!/usr/bin/env bash
# usage: scripts/v3v/run_bg.sh <name> <real_eval args...>   (GPU 7 only; log docs/v3v/logs/<name>.log)
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R; export CUDA_VISIBLE_DEVICES=7; name=$1; shift
echo "$(date -Is) $(hostname) START $*" >> docs/v3v/logs/$name.log
bash scripts/v3f/replay_env.sh scripts/v3v/real_eval.py "$@" >> docs/v3v/logs/$name.log 2>&1; rc=$?
echo "$(date -Is) END rc=$rc" >> docs/v3v/logs/$name.log
