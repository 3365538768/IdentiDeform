#!/usr/bin/env bash
# Run upstream modified_simulation.py in the gsim env (torch/cv2 borrowed from .venv, GS rasterizer built in external/).
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R && source scripts/r1_env.sh
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat:$R/.venv/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$R/external/upstream_shim:$R/upstream/DeformFieldBench:$R/upstream/DeformFieldBench/simulation:$R/upstream/DeformFieldBench/simulation/gaussian-splatting:$R/external/simple-knn:$R/external/diff-gaussian-rasterization:$R/.engine-envs/gsim/lib/python3.12/site-packages:$R/.venv/lib/python3.12/site-packages"
cd $R/upstream/DeformFieldBench/simulation
exec $R/.engine-envs/gsim/bin/python modified_simulation.py "$@"
