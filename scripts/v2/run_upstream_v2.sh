#!/usr/bin/env bash
# Run DeformFieldBench's simulation/modified_simulation.py (MPM + 3DGS rendering) through scripts/v2/upstream_main.py.
# Expects the DeformFieldBench checkout at upstream/DeformFieldBench and its Python environment in $GSIM_PYTHON
# (default: python). Set CUDA_VISIBLE_DEVICES as needed.
set -e
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$R/upstream/DeformFieldBench:$R/upstream/DeformFieldBench/simulation:$R/upstream/DeformFieldBench/simulation/gaussian-splatting${PYTHONPATH:+:$PYTHONPATH}"
cd "$R/upstream/DeformFieldBench/simulation"
exec "${GSIM_PYTHON:-python}" "$R/scripts/v2/upstream_main.py" "$@"
