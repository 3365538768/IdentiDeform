#!/bin/bash
# Simple v3g scheduler: launches pending PAC-NeRF runs on GPUs 3/0 when free memory allows (max 4 own runs per GPU, GPUs 3 and 7).
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test; cd $R
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  gl="3 7"
  for g in $gl; do
    [ ${#pending[@]} -eq 0 ] && break
    free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
    mine=$(for p in $(pgrep -f "pacnerf/bin/python train.py") $(pgrep -f "^bash scripts/v3g/pn_job2.sh"); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | /usr/bin/grep -x "CUDA_VISIBLE_DEVICES=$g"; done | wc -l)
    if [ "$free" -gt 14000 ] && [ "$mine" -lt 5 ]; then
      j=${pending[0]}; pending=("${pending[@]:1}")
      echo "$(date -u +%FT%TZ) launch $j gpu=$g free=$free mine=$mine"
      CUDA_VISIBLE_DEVICES=$g nohup bash scripts/v3g/pn_job2.sh $j $g > /dev/null 2>&1 &
      sleep 240
    fi
  done
  sleep 60
done
echo "$(date -u +%FT%TZ) all launched"
