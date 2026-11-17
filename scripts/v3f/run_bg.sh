#!/usr/bin/env bash
# usage: scripts/v3f/run_bg.sh <gpu> <name> <cmd...>   (runs from project root with r1_env, log docs/v3f/logs/<name>.log)
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R; source scripts/r1_env.sh; export CUDA_VISIBLE_DEVICES=$1; name=$2; shift 2
echo "$(date -Is) START $*" >> docs/v3f/logs/$name.log
"$@" >> docs/v3f/logs/$name.log 2>&1; rc=$?
echo "$(date -Is) END rc=$rc" >> docs/v3f/logs/$name.log
