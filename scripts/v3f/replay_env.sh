#!/usr/bin/env bash
# gsim python (warp) + torch libs from .venv, as scripts/v2/run_upstream_v2.sh; script adds sys.path itself (-I)
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat:$R/.venv/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec $R/.engine-envs/gsim/bin/python -I "$@"
