#!/bin/bash
# Writes this host's nvidia-smi snapshot to .cache/gpu_<host>.json every 15 s for monitor_r.
H=${1:-a800-4}; OUT=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/.cache/gpu_$H.json
while true; do
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | python3 -c "
import sys,json,time
k=[\"index\",\"util\",\"mem_used\",\"mem_total\",\"temp\",\"power\"]
g=[dict(zip(k,[float(x) for x in l.split(\",\")])) for l in sys.stdin if l.strip()]
json.dump({\"host\":\"$H\",\"updated\":time.time(),\"gpus\":g},open(\"$OUT.tmp\",\"w\"))" && mv -f $OUT.tmp $OUT
  sleep 15
done
