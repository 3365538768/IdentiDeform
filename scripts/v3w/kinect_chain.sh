#!/bin/bash
# Secondary view (pre-registered 'if time permits'): wait for the Kinect build, then predict + evaluate.
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
while ! /usr/bin/grep -q '^done' docs/v3w/logs/build_fall_kinect.log; do sleep 20; done
source scripts/r1_env.sh; export CUDA_VISIBLE_DEVICES=7
.venv/bin/python -I scripts/v3w/phys101_fall.py predict data/v3w/fall_kinect docs/v3w/pred_kinect.json
.venv/bin/python -I scripts/v3w/phys101_eval.py data/v3w/fall_kinect docs/v3w/pred_kinect.json data/external_datasets/v3w_phys101/raw/objects/mass data/external_datasets/v3w_phys101/raw/objects/vol docs/v3w/RESULTS_kinect.json
echo CHAIN_DONE
