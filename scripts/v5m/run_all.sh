#!/bin/bash
# V5m: all arms x seeds on one GPU (seed 29 of every arm first). nohup bash scripts/v5m/run_all.sh 7 > docs/v5m/run_all.log 2>&1 &
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test && source scripts/r1_env.sh; export CUDA_VISIBLE_DEVICES=${1:-7}
for s in 29 17 43; do
  for arm in vision vision_action vision_touch_action touch_action vta_mixv5f; do
    a=$arm; extra=""; [ $arm == vta_mixv5f ] && { a=vision_touch_action; extra=--mix-v5f; }
    out=runs/v5m_${arm}_seed$s
    [ -e $out/status.json ] && continue
    echo "$(date -u +%H:%M:%S) start $out"; .venv/bin/python scripts/v5m/train_mm.py --arm $a $extra --seed $s --out $out > $out.log 2>&1 || echo "FAILED $out"
  done
  echo "$(date -u +%H:%M:%S) seed $s done"
done
