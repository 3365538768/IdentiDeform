#!/bin/bash
# v3g: run official Spring-Gaus train.py (own configs, all 10 cams) on scenes sequentially on one GPU.
# usage: sg_run.sh <gpu> <timeout_sec_per_scene> scene...
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R; source ../env.sh >/dev/null 2>&1; source scripts/v3g/sg_env.sh
G=$1; TO=$2; shift 2
P=$SG_VENV/bin/python
for s in "$@"; do
  L=$R/docs/v3g/logs/sg_train_$s.log; M=$R/docs/v3g/logs/sg_mem_$s.txt
  echo "START $s gpu=$G $(date -u +%FT%TZ) $(date +%s)" > $R/docs/v3g/logs/sg_time_$s.txt
  cd $R/external/v3g_springgaus
  timeout $TO $P train.py -g $G --cfg config/mpm_synthetic/$s.yaml --eval_cam 5 --exp_id v3g_$s > $L 2>&1 &
  TP=$!
  sleep 5; PY=$(pgrep -P $TP | head -1); echo "pid $PY" > $M
  while kill -0 $TP 2>/dev/null; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | /usr/bin/grep "^$PY," >> $M; sleep 30; done
  wait $TP; rc=$?
  cd $R
  echo "END $s rc=$rc $(date -u +%FT%TZ) $(date +%s)" >> $R/docs/v3g/logs/sg_time_$s.txt
done
echo "ALL_DONE $(date -u +%FT%TZ)" >> $R/docs/v3g/logs/sg_time_all.txt
