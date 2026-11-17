#!/usr/bin/env bash
# V3s: PhysTwin's own chamfer/track evaluation (scripts/v3l/pt_eval.py) of the V3-prior MPM search, budget 0 and budget 12.
set -e
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test; cd $R; CASE=$1; D=data/v3s/pt
b0=$(python3 -c "import json;print(json.load(open('$D/search_prior.json'))['budget0']['vertices'])")
b12=$(python3 -c "import json;print(json.load(open('$D/search_prior.json'))['best']['vertices'])")
.engine-envs/v3g_phystwin/bin/python -I scripts/v3l/pt_eval.py $CASE $D/$b0 --out $D/eval_mpm_prior_b0.json
.engine-envs/v3g_phystwin/bin/python -I scripts/v3l/pt_eval.py $CASE $D/$b12 --out $D/eval_mpm_prior_b12.json
