#!/bin/bash
# Full V5b prior (a800-4 GPU 7, nohup): wait for data/v5b_main train 0-1999 + val 283 + test 300 (caching as data
# arrives) -> manifest v5bmain_t2000 (stats from V5b train 0-1999) -> index, configs v5b2k_*, baselines ->
# B/C/D x3 (seed 29 first, RAM preload) -> GATE (val) -> test_once x9 -> docs/v5p/HANDOFF.json.
#   nohup bash scripts/v5p/driver_full.sh 7 > docs/v5p/driver_full.log 2>&1 &
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test && source scripts/r1_env.sh; PY=.venv/bin/python; GPU=${1:-7}
MAN=v5bmain_t2000; PRE=v5b2k; ROOTARGS="--root data/v5b_main --prefix v5bmain --ranges v5"; JOBS=docs/v5p/train_jobs_full.json
last=""; still=0
while true; do
  out=$($PY scripts/v3p/build_manifest.py $ROOTARGS --name $MAN --splits train,val,test --train-max 2000 2>&1 | tail -1)
  t=$(echo "$out" | sed -n "s/.*'train': \([0-9]*\).*/\1/p"); v=$(echo "$out" | sed -n "s/.*'val': \([0-9]*\).*/\1/p"); s=$(echo "$out" | sed -n "s/.*'test': \([0-9]*\).*/\1/p")
  echo "$(date -u +%H:%M:%S) train=$t val=$v test=$s"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/v3p/cache_v3.py --manifest $MAN --status-output docs/v5p/cache_$MAN.json 2>&1 | grep -v -i warn | tail -1
  $PY scripts/v3p/plan_update.py --id v5p --detail "快检完成（docs/v5/PRIOR_CHECK）；V5b 全量：train $t/2000, test $s/300 生成+特征提取中，齐后训练 B/C/D×3 → GATE → HANDOFF" >/dev/null
  [ "$t" == "2000" ] && [ "$v" == "283" ] && [ "$s" == "300" ] && break
  if [ "$t $v $s" == "$last" ] && [ "${t:-0}" -ge 1980 ] && [ "${v:-0}" -ge 280 ] && [ "${s:-0}" -ge 297 ]; then still=$((still+1)); else still=0; fi
  [ $still -ge 6 ] && { echo "stalled, proceeding"; break; }
  last="$t $v $s"; sleep 120
done
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/v3p/cache_v3.py --manifest $MAN --status-output docs/v5p/cache_$MAN.json 2>&1 | tail -1
$PY scripts/v3p/cache_v3.py --manifest $MAN --finalize --index-dir data/prepared/${MAN}_vjl || exit 1
$PY scripts/v3p/make_configs.py --manifest $MAN --prefix $PRE
for sp in val test; do $PY scripts/v3p/baselines.py --manifest $MAN --prefix $PRE --split $sp; done
$PY - <<PYEOF
import json
pre='$PRE'
order=[('B',29),('C',29),('D',29),('B',17),('B',43),('C',17),('C',43),('D',17),('D',43)]
ev=lambda t:(f"test -e runs/{t}/test_once/predictions.jsonl || .venv/bin/python -m fieldbridge.cli evaluate --config configs/v3p/{t}.json "
             f"--checkpoint runs/{t}/best_material.pt --split test --output runs/{t}/test_once > runs/{t}/test_once.log")
runs=[f'{pre}_{g}_seed{s}' for g,s in order]
jobs=[{'id':t,'cmd':f'.venv/bin/python scripts/v3p/train_job.py configs/v3p/{t}.json --preload','busy':t+'[.](json|resume)'} for t in runs]
jobs.append({'id':pre+'_gate','gpu':False,'after':runs,'cmd':'.venv/bin/python scripts/v5p/gate.py > docs/v5p/gate.log 2>&1'})
jobs+=[{'id':t+'_test','after':[t, pre+'_gate'],'cmd':ev(t)} for t in runs]
note=json.dumps({'status':'final: V5b full prior (train 0-1999), main_run C_seed29 + 9-run ensemble',
                 'note_for_c1_refine':'V5b data: data/v5b_main (all six actions; held shear/stretch/bend; V5 ranges). Re-point c1_refine stats/manifest/sid/data_root to the fields above and its generator to the V5b settings of scripts/v3n/gen_v3.py used for data/v5b_main. prediction_dir: val = best val epoch dir (r1_report.best_dir), test = test_once.'})
jobs.append({'id':pre+'_handoff','gpu':False,'after':[t+'_test' for t in runs],
             'cmd':".venv/bin/python scripts/v3p/export_handoff.py --prefix "+pre+" --manifest $MAN --docs docs/v5p --data-root data/v5b_main --sid-prefix v5bmain --extra '"+note+"' > docs/v5p/handoff.log"})
json.dump(jobs,open('$JOBS','w'),indent=1); print('queued', len(jobs))
PYEOF
$PY scripts/v3p/plan_update.py --id v5p --detail "V5b 全量数据齐，B/C/D×3 训练中（GPU7）→ GATE → test_once → HANDOFF" >/dev/null
$PY scripts/v3p/pool.py $JOBS --gpus $GPU --per-gpu 5 --max-total 5 --done-dir runs/v5p_done
echo "$(date -u +%H:%M:%S) pool finished"
