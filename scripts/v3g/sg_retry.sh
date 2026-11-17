#!/bin/bash
# v3g: after the main runner finishes, retry scenes whose run failed (e.g. CUDA OOM from co-located jobs).
# Static/velocity checkpoints in checkpoints/<scene> are reused by train.py, so only the dynamic stage reruns.
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test; cd $R
until /usr/bin/grep -q ALL_DONE docs/v3g/logs/sg_time_all.txt 2>/dev/null; do sleep 60; done
for s in "$@"; do
  for try in 2 3; do
    if [ -e docs/v3g/logs/sg_time_$s.txt ] && /usr/bin/grep -q "rc=0" docs/v3g/logs/sg_time_$s.txt; then break; fi
    [ -e docs/v3g/logs/sg_time_$s.txt ] && mv docs/v3g/logs/sg_time_$s.txt docs/v3g/logs/sg_time_${s}_try$((try-1))_fail.txt && mv docs/v3g/logs/sg_train_$s.log docs/v3g/logs/sg_train_${s}_try$((try-1))_fail.log
    rm -f docs/v3g/logs/sg_time_all.txt
    scripts/v3g/sg_run.sh 7 25200 $s
  done
done
echo "RETRY_DONE $(date -u +%FT%TZ)" >> docs/v3g/logs/sg_retry_done.txt
