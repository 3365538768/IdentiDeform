"""V3r: gripper-task sample selection on V3 val, the same rule as V2 (V3e).

V2 rule: the V3e gripper set = all C1 refinement val samples (docs/v3b/C1_SAMPLES.json, scripts/v3b/c1_select.py:
60 val samples stratified by GT law x observed action, 3 per cell + 6 from a shuffled pool of 4th picks, seed 20261113),
with a pre-specified stratified wave 1 (10 plasticine + 10 metal + 6 jelly, seed 20261114, scripts/v3e/make_jobs.py).
Samples whose GT squeeze simulations all fail are excluded at evaluation time (V2: 5 ice_telescope samples).
V3: the C1 val set is the V3q one (docs/v3q/PROTOCOL.json, the same c1_select logic on V3 val indices 0-282 by generator
labels, failed generations replaced from the same cell's pool), so the refined twins (V3q prior_top2 / lawmean_best3)
exist for exactly these samples. The selection is recomputed here and must equal the V3q protocol.

  python3 scripts/v3r/select.py      -> docs/v3r/SAMPLES.json
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAWS = ['jelly', 'metal', 'plasticine']
ACTIONS = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']
QUOTA = {'plasticine': 10, 'metal': 10, 'jelly': 6}


def c1_select(indices, per_cell=3, total=60, seed=20261113):
    cells = {}
    for i in indices:
        cells.setdefault((LAWS[i % 3], ACTIONS[(i // 3) % 6]), []).append(i)
    rng = random.Random(seed); chosen, extra = [], []
    for key in sorted(cells):
        pool = sorted(cells[key]); rng.shuffle(pool)
        chosen += [(key, i) for i in pool[:per_cell]]; extra += [(key, i) for i in pool[per_cell:per_cell + 1]]
    rng.shuffle(extra)
    chosen += extra[:total - len(chosen)]
    chosen.sort(key=lambda c: c[1])
    return [{'index': i, 'law': l, 'action': a} for (l, a), i in chosen]


def main():
    mine = c1_select(range(283))
    proto = ROOT / 'docs/v3q/PROTOCOL.json'
    src, samples = 'recomputed (V3q protocol not yet written)', mine
    if proto.exists():
        v = json.loads(proto.read_text())['samples']['val']['samples']
        samples = [{'index': s['index'], 'law': s['law'], 'action': s['action']} for s in v]
        src = 'docs/v3q/PROTOCOL.json val (identical to the recomputation)' if [s['index'] for s in mine] == [s['index'] for s in samples] \
            else 'docs/v3q/PROTOCOL.json val (DIFFERS from the recomputation: V3q replacement of failed generations)'
    rng = random.Random(20261114); wave1 = []
    for law, k in QUOTA.items():
        pool = sorted(s['index'] for s in samples if s['law'] == law)
        wave1 += sorted(rng.sample(pool, k))
    indices = wave1 + sorted(s['index'] for s in samples if s['index'] not in wave1)
    out = {'rule': 'V2 rule: V3e gripper set = the C1 val set (scripts/v3b/c1_select.py logic, seed 20261113) on V3 val 0-282; '
                   'wave 1 = scripts/v3e/make_jobs.py quota 10 plasticine / 10 metal / 6 jelly, seed 20261114',
           'source': src, 'samples': samples, 'seed': 20261114, 'quota_wave1': QUOTA, 'wave1': wave1, 'indices': indices}
    (ROOT / 'docs/v3r').mkdir(parents=True, exist_ok=True)
    (ROOT / 'docs/v3r/SAMPLES.json').write_text(json.dumps(out, indent=1))
    print(len(samples), src); print('indices', indices)


if __name__ == '__main__':
    main()
