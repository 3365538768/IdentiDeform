#!/bin/bash
# v3g: memory-gated retries on GPU 7 (co-located jobs caused CUDA OOM). Waits until >= 20 GB free before each launch.
# Static/velocity checkpoints are reused, so retries rerun only the dynamic stage (+eval).
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test; cd $R
DEADLINE=$(date -d "2026-11-15T02:30:00Z" +%s)
for s in "$@"; do
  for try in 3 4 5 6 7 8; do
    if [ -e docs/v3g/logs/sg_time_$s.txt ] && /usr/bin/grep -q "rc=0" docs/v3g/logs/sg_time_$s.txt; then break; fi
    if [ -e docs/v3g/logs/sg_time_$s.txt ]; then
      n=1; while [ -e docs/v3g/logs/sg_time_${s}_fail$n.txt ]; do n=$((n+1)); done
      mv docs/v3g/logs/sg_time_$s.txt docs/v3g/logs/sg_time_${s}_fail$n.txt; mv docs/v3g/logs/sg_train_$s.log docs/v3g/logs/sg_train_${s}_fail$n.log
    fi
    while true; do
      used=$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      [ $((81920 - used)) -ge 20000 ] && break
      echo "$(date -u +%FT%TZ) wait $s free=$((81920-used))"; sleep 120
    done
    left=$((DEADLINE - $(date +%s))); [ $left -lt 3600 ] && { echo "deadline: skip $s"; break; }
    rm -f docs/v3g/logs/sg_time_all.txt
    scripts/v3g/sg_run.sh 7 $(( left < 25200 ? left : 25200 )) $s
  done
done
echo "RETRY2_DONE $(date -u +%FT%TZ)" >> docs/v3g/logs/sg_retry2_done.txt
