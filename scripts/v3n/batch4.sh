#!/usr/bin/env bash
# V3 review batch 4 (a800-4): gripper reruns after the particle clamp; full cycle with v3 ranges; extra drops with v3 ranges
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
for i in 7 10 26; do rm -rf data/v3n_grip/val/$(printf %06d $i); done
rm -rf data/v3n_dbg
g=0
run() { nohup bash -c "source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES=$((g%8)) && .venv/bin/python scripts/v3n/gen_v3.py $*" > docs/v3n/logs/b4_$g.log 2>&1 < /dev/null & g=$((g+1)); sleep 1; }
for i in 7 10 26; do run --split val --indices $i --root data/v3n_grip; done
for i in $(seq 0 35) 39 40 41 57 58 59; do run --split val --indices $i --root data/v3n_v3r --ranges v3; done
