#!/bin/bash
# V4 prior check driver (a800-4, nohup). V4 = V3 for press/drop/tap30hold (read from data/v3_main, identical samples),
# regenerated shear/stretch/bend from data/v4_main. Every cycle: manifest (train<1000 + val) -> incremental V-JEPA cache
# (GPU $GPU). When train=1000 and val=283 (or stalled at >=99%): final index, configs v4t1k_*, jobs -> docs/v4p/train_jobs_a4.json,
# then run the pool (seed 29 of B/C/D first).
#   nohup bash scripts/v4p/driver_v4.sh 7 > docs/v4p/driver_v4.log 2>&1 &
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test && source scripts/r1_env.sh; PY=.venv/bin/python; GPU=${1:-7}
MAN=v4main_t1000; ROOTARGS="--root data/v4_main --prefix v4main --alt press,drop,tap30hold=data/v3_main:v3main"
last=""; still=0
while true; do
  out=$($PY scripts/v3p/build_manifest.py $ROOTARGS --name $MAN --splits train,val --train-max 1000 2>&1 | tail -1)
  t=$(echo "$out" | sed -n "s/.*'train': \([0-9]*\).*/\1/p"); v=$(echo "$out" | sed -n "s/.*'val': \([0-9]*\).*/\1/p")
  echo "$(date -u +%H:%M:%S) train=$t val=$v"
  [ -n "$t" ] && [ "$t" -gt 0 ] && CUDA_VISIBLE_DEVICES=$GPU $PY scripts/v3p/cache_v3.py --manifest $MAN --status-output docs/v4p/cache_inc.json 2>&1 | grep -v -i warn | tail -1
  $PY scripts/v3p/plan_update.py --id v4p --detail "V3 t1000 对照组 9 个运行完成；V4（shear/stretch/bend 重新生成，其余=V3）train $t/1000, val $v/283，特征增量提取中" >/dev/null
  [ "$t" == "1000" ] && [ "$v" == "283" ] && break
  if [ "$t $v" == "$last" ] && [ "${t:-0}" -ge 990 ] && [ "${v:-0}" -ge 280 ]; then still=$((still+1)); else still=0; fi
  [ $still -ge 6 ] && { echo "stalled, proceeding"; break; }
  last="$t $v"; sleep 120
done
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/v3p/cache_v3.py --manifest $MAN --status-output docs/v4p/cache_inc.json 2>&1 | tail -1
$PY scripts/v3p/cache_v3.py --manifest $MAN --finalize --index-dir data/prepared/${MAN}_vjl || exit 1
$PY scripts/v3p/make_configs.py --manifest $MAN --prefix v4t1k
$PY - <<PYEOF
import json
order=[('B',29),('C',29),('D',29),('B',17),('B',43),('C',17),('C',43),('D',17),('D',43)]
jobs=[{'id':f'v4t1k_{g}_seed{s}','cmd':f'.venv/bin/python scripts/v3p/train_job.py configs/v3p/v4t1k_{g}_seed{s}.json --preload','busy':f'v4t1k_{g}_seed{s}[.](json|resume)'} for g,s in order]
jobs.append({'id':'v4p_report','gpu':False,'after':[j['id'] for j in jobs],'cmd':'.venv/bin/python scripts/v4p/prior_check.py > docs/v4p/prior_check.log 2>&1'})
json.dump(jobs,open('docs/v4p/train_jobs_a4.json','w'),indent=1); print('queued')
PYEOF
$PY scripts/v3p/plan_update.py --id v4p --detail "V4 数据齐（train 1000 / val 283），V4 B/C/D×3 训练中（seed29 优先）" >/dev/null
$PY scripts/v3p/pool.py docs/v4p/train_jobs_a4.json --gpus 0,1,2,3,4,5,6 --per-gpu 1 --max-total 7 --done-dir runs/v4p_done
echo "$(date -u +%H:%M:%S) pool finished"
