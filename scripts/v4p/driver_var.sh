#!/bin/bash
# Quick prior check driver for a data variant (a800-4, nohup): every cycle build the manifest (train<1000 + val) with the
# given root args, cache new samples (GPU $CGPU); when train=1000 and val=283 (or stalled >=99%): index, configs, 9 jobs
# (seed 29 first) + optional report job -> $JOBS, then run the pool on GPUs $TGPUS.
#   nohup bash scripts/v4p/driver_var.sh <manifest> <run_prefix> <plan_id> <jobs.json> <cache_gpu> <train_gpus> "<build_manifest root args>" > log 2>&1 &
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test && source scripts/r1_env.sh; PY=.venv/bin/python
MAN=$1; PRE=$2; PID=$3; JOBS=$4; CGPU=$5; TGPUS=$6; ROOTARGS=$7
last=""; still=0
while true; do
  out=$($PY scripts/v3p/build_manifest.py $ROOTARGS --name $MAN --splits train,val --train-max 1000 2>&1 | tail -1)
  t=$(echo "$out" | sed -n "s/.*'train': \([0-9]*\).*/\1/p"); v=$(echo "$out" | sed -n "s/.*'val': \([0-9]*\).*/\1/p")
  echo "$(date -u +%H:%M:%S) train=$t val=$v"
  [ -n "$t" ] && [ "$t" -gt 0 ] && CUDA_VISIBLE_DEVICES=$CGPU $PY scripts/v3p/cache_v3.py --manifest $MAN --status-output docs/v5p/cache_$MAN.json 2>&1 | grep -v -i warn | tail -1
  [ "$t" == "1000" ] && [ "$v" == "283" ] && break
  if [ "$t $v" == "$last" ] && [ "${t:-0}" -ge 990 ] && [ "${v:-0}" -ge 280 ]; then still=$((still+1)); else still=0; fi
  [ $still -ge 6 ] && { echo "stalled, proceeding"; break; }
  last="$t $v"; sleep 120
done
CUDA_VISIBLE_DEVICES=$CGPU $PY scripts/v3p/cache_v3.py --manifest $MAN --status-output docs/v5p/cache_$MAN.json 2>&1 | tail -1
$PY scripts/v3p/cache_v3.py --manifest $MAN --finalize --index-dir data/prepared/${MAN}_vjl || exit 1
$PY scripts/v3p/make_configs.py --manifest $MAN --prefix $PRE
$PY - <<PYEOF
import json
order=[('B',29),('C',29),('D',29),('B',17),('B',43),('C',17),('C',43),('D',17),('D',43)]
jobs=[{'id':f'$PRE'+f'_{g}_seed{s}','cmd':f'.venv/bin/python scripts/v3p/train_job.py configs/v3p/$PRE'+f'_{g}_seed{s}.json --preload','busy':f'$PRE'+f'_{g}_seed{s}[.](json|resume)'} for g,s in order]
json.dump(jobs,open('$JOBS','w'),indent=1); print('queued')
PYEOF
echo "$(date -u +%H:%M:%S) training"
$PY scripts/v3p/pool.py $JOBS --gpus $TGPUS --per-gpu 1 --max-total 4 --done-dir runs/v5p_done
echo "$(date -u +%H:%M:%S) pool finished"
