"""V3d: pick 30 V2-main TEST samples stratified by (GT law x observed action) with the selection logic of
scripts/v3b/c1_select.py (same seed 20261113; only the split, the per-cell quota and the total change:
1 per cell (18 cells) + 12 second picks from a shuffled extra pool). Uses only gt.json law/action labels,
no model output or simulation result. Writes docs/v3d/C1_TEST_SAMPLES.json."""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PER_CELL, TOTAL, SEED = 1, 30, 20261113
rows = [json.loads(l) for l in (ROOT / 'data/manifests/v2main_withtest.jsonl').read_text().splitlines() if l.strip()]
test = sorted(r['sample_id'] for r in rows if r['split'] == 'test')
assert len(test) == 300, len(test)
cells = {}
for sid in test:
    g = json.loads((ROOT / f'data/v2_main/test/{sid.rsplit("/", 1)[1]}/gt.json').read_text())
    cells.setdefault((g['law'], g['action']), []).append(int(sid.rsplit('/', 1)[1]))
rng = random.Random(SEED)
chosen, extra = [], []
for key in sorted(cells):
    pool = sorted(cells[key]); rng.shuffle(pool)
    chosen += [(key, i) for i in pool[:PER_CELL]]; extra += [(key, i) for i in pool[PER_CELL:PER_CELL + 1]]
rng.shuffle(extra)
chosen += extra[:TOTAL - len(chosen)]
chosen.sort(key=lambda c: c[1])
seen, gt_obs = set(), []
for (law, act), i in chosen:
    if act not in seen:
        seen.add(act); gt_obs.append(i)
out = {'split': 'test', 'seed': SEED, 'per_cell': PER_CELL, 'total': TOTAL,
       'cells': {f'{l}/{a}': len(v) for (l, a), v in sorted(cells.items())},
       'samples': [{'index': i, 'law': l, 'action': a, 'gt_observed': i in gt_obs} for (l, a), i in chosen]}
(ROOT / 'docs/v3d/C1_TEST_SAMPLES.json').write_text(json.dumps(out, indent=1))
print(len(chosen), out['cells'])
