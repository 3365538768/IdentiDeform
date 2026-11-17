#!/usr/bin/env bash
# V3f PIXIE: run scripts/v3f/pixie_features.py in the isolated .engine-envs/pixie venv (torch etc. borrowed read-only
# from .venv via PYTHONPATH, like .engine-envs/gsim). Usage: pixie_run.sh GPU OUTDIR [args...]
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R && source scripts/r1_env.sh
export CUDA_VISIBLE_DEVICES=$1; shift
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat:$R/.venv/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$R/external/diff-gaussian-rasterization-depthalpha:$R/.engine-envs/pixie/lib/python3.12/site-packages:$R/.venv/lib/python3.12/site-packages"
cd $R/data/v3f_pixie
exec $R/.engine-envs/pixie/bin/python $R/scripts/v3f/pixie_features.py "$@"
