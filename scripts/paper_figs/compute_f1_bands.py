"""Stage 1 for F1: object-bootstrap 95% bands at every simulation budget k.

Val (60 C1 samples): re-uses scripts/v3h/report.py loaders (state.json under data/v3b_refine/v2main_C_seed29),
checks every pooled curve value against docs/v3h/PRIOR_QUALITY.json prior_top1_val60, and bootstraps objects
(2000 draws, seed 17, same resampler as report.py).
Test (30 samples): per-sample ident_err stored in docs/v3d/C1_TEST_RESULTS.json samples; checked against the
stored curves; same object bootstrap.
Output: docs/paper_figs/data/f1_bands.json
  .venv/bin/python scripts/paper_figs/compute_f1_bands.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3h'))
import report as R  # noqa: E402

ARMS = ['prior_top1', 'prior_top2', 'lawmean_gt', 'lawmean_best3']
out = {'note': __doc__.strip(), 'draws': R.DRAWS, 'seed': R.SEED}

# ---- val 60
pq = json.loads((ROOT / 'docs/v3h/PRIOR_QUALITY.json').read_text())['prior_top1_val60']['result']
idx60 = [p['index'] for p in json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']]
S = {a: {i: r for i in idx60 if (r := R.load_reused(a, i)) is not None} for a in ['prior'] + ARMS}
idx = [i for i in idx60 if all(i in S[a] for a in ARMS)]
assert len(idx) == pq['n_samples'] == 60, len(idx)
groups = R.groups_of(S, idx); rng = np.random.default_rng(R.SEED)
boots = [R.boot_sel(groups, rng) for _ in range(R.DRAWS)]
val = {'n_samples': len(idx), 'n_objects': len(groups), 'arms': {}}
for a in ARMS:
    rows = {}
    for k in range(25):
        m = R.pooled(S, a, idx, k)
        if m is None:
            continue
        ref = pq['arms'][a]['curve'].get(str(k))
        assert ref is not None and abs(ref - m) < 1e-9, (a, k, ref, m)
        bs = [R.pooled(S, a, s, k) for s in boots]
        rows[str(k)] = {'mean': m, 'ci95': np.quantile(bs, [.025, .975]).tolist()}
    for k, v in pq['arms'][a].get('mae', {}).items():  # stored CIs must be reproduced
        assert np.allclose(rows[k]['ci95'], v['ci95']), (a, k, rows[k]['ci95'], v['ci95'])
    val['arms'][a] = rows
out['val60'] = val

# ---- test 30
T = json.loads((ROOT / 'docs/v3d/C1_TEST_RESULTS.json').read_text())
stored = {**T['curves'], 'prior_top1': T['prior_top1_addendum']['curves']['prior_top1']}
samp = {a: {s['index']: s for s in T['samples'][a]} for a in ARMS}
tidx = T['analysed_indices']
obj = {i: samp['prior_top2'][i]['object'] for i in tidx}
g = {}
for i in tidx:
    g.setdefault(obj[i], []).append(i)
tgroups = list(g.values()); rng = np.random.default_rng(R.SEED)
tboots = [R.boot_sel(tgroups, rng) for _ in range(R.DRAWS)]

def tpooled(a, sel, k):
    e = []
    for i in sel:
        v = samp[a][i]['ident_err'].get(k)
        if v is None:
            return None
        e += [x for x in v if x is not None]
    return float(np.mean(e))

test = {'n_samples': len(tidx), 'n_objects': len(tgroups), 'arms': {}}
for a in ARMS:
    rows = {}
    for k in sorted(stored[a], key=int):
        m = tpooled(a, tidx, k)
        if m is None:
            continue
        assert abs(m - stored[a][k]['identifiable_mae_pooled']) < 1e-9, (a, k, m, stored[a][k]['identifiable_mae_pooled'])
        bs = [tpooled(a, s, k) for s in tboots]
        rows[k] = {'mean': m, 'ci95': np.quantile(bs, [.025, .975]).tolist()}
    test['arms'][a] = rows
out['test30'] = test
d = ROOT / 'docs/paper_figs/data'; d.mkdir(parents=True, exist_ok=True)
(d / 'f1_bands.json').write_text(json.dumps(out, indent=1) + '\n')
for s in ('val60', 'test30'):
    for a, r in out[s]['arms'].items():
        print(s, a, {k: round(v['mean'], 3) for k, v in r.items() if k in ('0', '3', '4', '12', '24')})
