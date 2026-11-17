"""Pick ~60 V2-main validation samples stratified by (GT law x observed action); writes docs/v3b/C1_SAMPLES.json."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(l) for l in (ROOT / 'data/manifests/v2main.jsonl').read_text().splitlines() if l.strip()]
val = sorted(r['sample_id'] for r in rows if r['split'] == 'val')
cells = {}
for sid in val:
    g = json.loads((ROOT / f'data/v2_main/val/{sid.rsplit("/", 1)[1]}/gt.json').read_text())
    cells.setdefault((g['law'], g['action']), []).append(int(sid.rsplit('/', 1)[1]))
rng = random.Random(20261113)
chosen, extra = [], []
for key in sorted(cells):
    pool = sorted(cells[key]); rng.shuffle(pool)
    chosen += [(key, i) for i in pool[:3]]; extra += [(key, i) for i in pool[3:4]]
rng.shuffle(extra)
chosen += extra[:60 - len(chosen)]
chosen.sort(key=lambda c: c[1])
# GT re-simulation of the observed action (noise floor) for one sample per action.
seen, gt_obs = set(), []
for (law, act), i in chosen:
    if act not in seen:
        seen.add(act); gt_obs.append(i)
out = {'seed': 20261113, 'cells': {f'{l}/{a}': len(v) for (l, a), v in sorted(cells.items())},
       'samples': [{'index': i, 'law': l, 'action': a, 'gt_observed': i in gt_obs} for (l, a), i in chosen]}
(ROOT / 'docs/v3b/C1_SAMPLES.json').write_text(json.dumps(out, indent=1))
print(len(chosen), out['cells'])
