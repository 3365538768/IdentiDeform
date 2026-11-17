"""F1(b) bands on the enlarged test set: docs/v3d original 30 + docs/v3x new 30 (paired-complete samples only).

Same procedure as scripts/paper_figs/compute_f1_bands.py: pooled identifiable z-MAE over all identifiable abs errors of the
selected samples; object bootstrap (groups = objects, rng = default_rng(SEED), DRAWS draws of boot_sel, percentile 95 %),
DRAWS/SEED/boot_sel taken from scripts/v3h/report.py. Per-sample errors at budget k come from scripts/v3x/report.py err_at
(the loader behind docs/v3x/RESULTS.json).
Sets (as in RESULTS.json table2): prior_top1 / lawmean_best3 on the paired-complete top1+best3 set; lawmean_gt on its
paired-complete set with prior_top1 (bootstrapped separately, own rng seeded SEED). prior_top2 is not run on the new samples
and is omitted. Checks: every pooled mean equals docs/v3x/RESULTS.json combined_60.table2, and on the original 30 the
per-sample loader reproduces docs/paper_figs/data/f1_bands.json test30 means.
Output: docs/paper_figs/data/f1_bands_test_enlarged.json
  .venv/bin/python scripts/paper_figs/compute_f1_bands_test_enlarged.py
"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]


def mod(name, p):
    spec = importlib.util.spec_from_file_location(name, ROOT / p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


R = mod('v3h_report', 'scripts/v3h/report.py')
X = mod('v3x_report', 'scripts/v3x/report.py')
KS = [0, 3, 4, 8, 12, 16, 24]
orig = [s['index'] for s in json.loads((ROOT / 'docs/v3d/C1_TEST_SAMPLES.json').read_text())['samples']]
new = json.loads((ROOT / 'docs/v3x/SAMPLES.json').read_text())['new_order']
ARMS = ['prior_top1', 'lawmean_best3', 'lawmean_gt']
S = {}
for i in orig:
    S[i] = {a: X.load('data/v3d_refine_test', i, a) for a in ARMS}
for i in new:
    S[i] = {a: X.load('data/v3x_refine_test', i, a) for a in ARMS}
obj = {i: next(s for s in S[i].values() if s)['object'] for i in S}
done = lambda i, a: bool(S[i][a] and S[i][a].get('complete'))
E = lambda i, a, k: X.err_at(S[i][a], k)


def pooled(a, sel, k):
    e = []
    for i in sel:
        v = E(i, a, k)
        if v is None:
            return None
        e += v
    return float(np.mean(e))


def bands(sel, arms):
    g = {}
    for i in sel:
        g.setdefault(obj[i], []).append(i)
    groups = list(g.values()); rng = np.random.default_rng(R.SEED)
    boots = [R.boot_sel(groups, rng) for _ in range(R.DRAWS)]
    out = {'n_samples': len(sel), 'n_objects': len(groups), 'n_original': sum(i in orig for i in sel),
           'n_new': sum(i in new for i in sel), 'arms': {}}
    for a in arms:
        rows = {}
        for k in KS:
            if a == 'lawmean_best3' and k < 3:
                continue
            m = pooled(a, sel, k)
            bs = [pooled(a, s, k) for s in boots]
            rows[str(k)] = {'mean': m, 'ci95': np.quantile(bs, [.025, .975]).tolist()}
        out['arms'][a] = rows
    return out


all_idx = orig + new
P = [i for i in all_idx if done(i, 'prior_top1') and done(i, 'lawmean_best3')]
G = [i for i in all_idx if done(i, 'prior_top1') and done(i, 'lawmean_gt')]
res = {'note': __doc__.strip(), 'draws': R.DRAWS, 'seed': R.SEED,
       'top1_best3_set': bands(P, ['prior_top1', 'lawmean_best3']),
       'gt_set': bands(G, ['lawmean_gt', 'prior_top1'])}
V = json.loads((ROOT / 'docs/v3x/RESULTS.json').read_text())
T2 = V['combined_60']['table2']
for key, arms in (('top1_best3_set', ['prior_top1', 'lawmean_best3']), ('gt_set', ['lawmean_gt'])):
    for a in arms:
        assert res[key]['n_samples'] == T2[a]['n'], (a, res[key]['n_samples'], T2[a]['n'])
        for k in (0, 3, 4, 12, 24):
            if f'k{k}' in T2[a]:
                assert abs(res[key]['arms'][a][str(k)]['mean'] - T2[a][f'k{k}']) < 1e-9, (a, k)
B30 = json.loads((ROOT / 'docs/paper_figs/data/f1_bands.json').read_text())['test30']['arms']
for a in ARMS:
    for k, r in B30[a].items():
        if int(k) in KS:
            assert abs(pooled(a, orig, int(k)) - r['mean']) < 1e-9, ('orig30', a, k, pooled(a, orig, int(k)), r['mean'])
sp = V['combined_60']['lawmean_gt_sims_to_reach_prior_k0']
assert abs(sp['prior_k0_mae'] - res['gt_set']['arms']['prior_top1']['0']['mean']) < 1e-9
res['sims_saved'] = {'source': 'docs/v3x/RESULTS.json combined_60.lawmean_gt_sims_to_reach_prior_k0', **sp}
res['source_generated_utc'] = V['generated_utc']
d = ROOT / 'docs/paper_figs/data'
(d / 'f1_bands_test_enlarged.json').write_text(json.dumps(res, indent=1) + '\n')
for key in ('top1_best3_set', 'gt_set'):
    r = res[key]; print(key, r['n_samples'], r['n_objects'], r['n_original'], r['n_new'])
    for a, rows in r['arms'].items():
        print(' ', a, {k: (round(v['mean'], 3), [round(x, 3) for x in v['ci95']]) for k, v in rows.items()})
print('sims', sp['sims'], sp['ci95'], sp['n'], sp['objects'])
