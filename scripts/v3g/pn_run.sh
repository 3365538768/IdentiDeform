#!/bin/bash
# Usage: bash scripts/v3g/pn_run.sh <cfgname e.g. torus_v11> <gpu>   (run from rsi_real_test root)
set -u
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R; source scripts/v3g/pn_env.sh
name=$1; gpu=$2
log=$R/docs/v3g/logs/pn_${name}.log
n=1; while [ -e $R/docs/v3g/logs/pn_${name}.a$n.log ]; do n=$((n+1)); done; [ -e $log ] && mv $log $R/docs/v3g/logs/pn_${name}.a$n.log
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
cd $R/external/pacnerf
echo "V3G_START $(date +%s) $(date -u +%FT%TZ) host=$(hostname) gpu=$gpu $(nvidia-smi --query-gpu=name --format=csv,noheader -i $gpu)" > $log
$PNV/bin/python train.py --config=$R/scripts/v3g/pn_cfgs/${name}.py 2>&1 | python3 -I $R/scripts/v3g/ts.py >> $log
rc=${PIPESTATUS[0]}
echo "V3G_END $(date +%s) rc=$rc" >> $log
