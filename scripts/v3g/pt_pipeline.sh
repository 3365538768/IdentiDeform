#!/bin/bash
# v3g PhysTwin pipeline on one case: CMA-ES (official 20 iters) -> first-order (official 200 iters) -> inference -> eval
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
cd $R; source scripts/v3g/pt_env.sh
CASE=${CASE:-single_push_sloth}; GPU=${GPU:-3}
W=$R/runs/v3g_phystwin; mkdir -p $W/data
ln -sfn $R/data/external_datasets/v3g_phystwin $W/data/different_types
ln -sfn $R/external/v3g_phystwin/configs $W/configs
cd $W
PY="$PT_VENV/bin/python -u"; T=$W/timing.jsonl
TF=$($PT_VENV/bin/python -I -c "import json;print(json.load(open('data/different_types/$CASE/split.json'))['train'][1])")
export CUDA_VISIBLE_DEVICES=$GPU
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv -i $GPU
run() { echo "=== $1 start $(date -u +%FT%TZ)"; $PY $R/scripts/v3g/pt_run.py "$@" --timing $T; rc=$?; echo "=== $1 rc=$rc $(date -u +%FT%TZ)"; [ $rc -eq 0 ] || { echo PIPELINE_ABORT; exit 1; }; }
[ -f experiments_optimization/$CASE/optimal_params.pkl ] || run cma --case $CASE --base_path ./data/different_types --case_name $CASE --train_frame $TF
run train --case $CASE --base_path ./data/different_types --case_name $CASE --train_frame $TF
run infer --case $CASE --base_path ./data/different_types --case_name $CASE
run eval_chamfer --case $CASE
run eval_track --case $CASE
echo PIPELINE_DONE
