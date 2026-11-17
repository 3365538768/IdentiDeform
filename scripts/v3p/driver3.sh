#!/bin/bash
# V3p main driver (a800-4, nohup): wait for train 0..TRAIN_MAX-1 -> manifest/cache/index -> configs -> 9 B/C/D runs
# (pool on a800-4 + r_queue on a800-2) -> test_once evals -> baselines -> hand-off.
#   nohup bash scripts/v3p/driver3.sh 4000 v3main_t4000 v3main4k > docs/v3p/logs/driver_t4000.log 2>&1 &
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test && source scripts/r1_env.sh
TMAX=$1; MAN=$2; PRE=$3; PY=.venv/bin/python
count() { $PY - <<PYEOF
import json,glob
n=0
for f in glob.glob('data/v3_main/train/*/result.json'):
    i=int(f.split('/')[-2])
    if i < $TMAX:
        try: n += json.load(open(f)).get('exit')==0
        except Exception: pass
print(n)
PYEOF
}
last=-1; still=0
while true; do
  n=$(count); echo "$(date -u +%H:%M:%S) train<$TMAX done=$n"
  [ "$n" -ge "$TMAX" ] && break
  if [ "$n" -eq "$last" ] && [ "$n" -ge $((TMAX*99/100)) ]; then still=$((still+1)); else still=0; fi
  [ $still -ge 6 ] && { echo "stalled at $n (>=99%), proceeding"; break; }
  last=$n; sleep 300
done
$PY scripts/v3p/plan_update.py --detail "train<$TMAX 已生成 ($n)，建 manifest/特征索引并启动 B/C/D×3 训练（$PRE）" >/dev/null
$PY scripts/v3p/build_manifest.py --name $MAN --train-max $TMAX --min-train 1000 || exit 1
CUDA_VISIBLE_DEVICES=0 $PY scripts/v3p/cache_v3.py --manifest $MAN --status-output docs/v3p/cache_$MAN.json 2>&1 | tail -2
$PY scripts/v3p/cache_v3.py --manifest $MAN --finalize --index-dir data/prepared/${MAN}_vjl || exit 1
$PY scripts/v3p/make_configs.py --manifest $MAN --prefix $PRE
for s in val test; do $PY scripts/v3p/baselines.py --manifest $MAN --prefix $PRE --split $s; done
# a800-2 share (r_queue): 3 runs; a800-4 pool: the other 6 (C_seed29 first).
$PY - <<PYEOF
import json
pre='$PRE'
a2=[('B',17),('D',17),('B',43)]
a4=[('C',29),('C',17),('C',43),('B',29),('D',29),('D',43)]
ev=lambda g,s:(f".venv/bin/python -m fieldbridge.cli evaluate --config configs/v3p/{pre}_{g}_seed{s}.json "
               f"--checkpoint runs/{pre}_{g}_seed{s}/best_material.pt --split test --output runs/{pre}_{g}_seed{s}/test_once > runs/{pre}_{g}_seed{s}/test_once.log")
q={'phase':'v3p','max_per_gpu':10,'gpus':[0,1,2,3,4,5,6,7],'note':'V3p prior training share on a800-2 (rest on a800-4 pool docs/v3p/jobs_a4_$PRE.json)','jobs':[]}
try: q=json.load(open('docs/v3p/queue.json'))
except Exception: pass
ids={j['id'] for j in q['jobs']}
for g,s in a2:
    t=f'{pre}_{g}_seed{s}'
    for j in [{'id':t,'kind':'cmd','cmd':f'.venv/bin/python scripts/v3p/train_job.py configs/v3p/{t}.json --preload','group':f'{g}-{pre}','seed':s,'mem_mib':20000,'retries':3},
              {'id':t+'_test','kind':'cmd','group':'v3p-eval','mem_mib':8000,'retries':1,'after':[t],'cmd':ev(g,s)}]:
        if j['id'] not in ids: q['jobs'].append(j)
json.dump(q,open('docs/v3p/queue.json','w'),indent=1)
jobs=[]
try: jobs=json.load(open('docs/v3p/jobs_a4_$PRE.json'))
except Exception: pass
ids={j['id'] for j in jobs}
for g,s in a4:
    t=f'{pre}_{g}_seed{s}'
    for j in [{'id':t,'cmd':f'.venv/bin/python scripts/v3p/train_job.py configs/v3p/{t}.json --preload','busy':t+'[.](json|resume)'},
              {'id':t+'_test','after':[t],'cmd':ev(g,s)}]:
        if j['id'] not in ids: jobs.append(j)
# (no hand-off job: the 4000-sample model is only a data-scaling point; HANDOFF.json stays the 2000-sample prior)
json.dump(jobs,open('docs/v3p/jobs_a4_$PRE.json','w'),indent=1)
print('queued')
PYEOF
echo "$(date -u +%H:%M:%S) launching pool"
$PY scripts/v3p/pool.py docs/v3p/jobs_a4_$PRE.json --gpus 1,2,3,4,5,6,7 --per-gpu 1 --max-total 4
echo "$(date -u +%H:%M:%S) pool finished"
