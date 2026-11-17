#!/usr/bin/env bash
# As scripts/v3e/run_squeeze.sh, but runs the V5g grasp-and-lift wrapper.
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R && source scripts/r1_env.sh
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat:$R/.venv/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$R/external/upstream_shim:$R/upstream/DeformFieldBench:$R/upstream/DeformFieldBench/simulation:$R/upstream/DeformFieldBench/simulation/gaussian-splatting:$R/external/simple-knn:$R/external/diff-gaussian-rasterization:$R/.engine-envs/gsim/lib/python3.12/site-packages:$R/.venv/lib/python3.12/site-packages"
cd $R/upstream/DeformFieldBench/simulation
exec $R/.engine-envs/gsim/bin/python $R/scripts/v5g/grasp_main.py "$@"
