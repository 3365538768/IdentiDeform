"""V3w eval: endpoints exactly as docs/v3w/PROTOCOL.json.  eval <out_dir> <pred.json> <mass> <vol> <results.json>"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
out, predf, massf, volf, resf = sys.argv[1:6]
index = json.loads((Path(out) / 'index.json').read_text()); pred = json.loads(Path(predf).read_text())
mass = dict((l.split()[0], float(l.split()[1])) for l in open(massf) if l.strip())
vol = dict((l.split()[0], float(l.split()[1])) for l in open(volf) if l.strip())
CAT = {'rubber': 'rubber', 'h_rubber': 'rubber', 'foam': 'foam', 'p_block': 'plastic', 'p_doll': 'plastic', 'p_ring': 'plastic',
       'p_toy': 'plastic', 'w_block': 'wood', 'w_pole': 'wood', 'h_wood': 'wood', 'm_coin': 'metal', 'cardboard': 'cardboard'}
LIT = {'rubber': 3.5, 'foam': 5.4, 'plastic': 6.3, 'wood': 7.2, 'metal': 7.3}
RIGID = {'plastic', 'wood', 'metal'}
cat = lambda o: CAT[o.rsplit('_', 1)[0]]
PRIMARY = ['table', 'metal', 'glass']; B = 10000; rng = np.random.default_rng(0)


def trial_rows(surfs):
    rows = []
    for tid, r in index.items():
        if r['surface'] not in surfs or 'excluded' in r or tid not in pred: continue
        p = pred[tid]
        rows.append({'obj': r['object'], 'cat': cat(r['object']), 'motion': p['motion']['log10_E_over_rho_x1000'],
                     'static': p['static']['log10_E_over_rho_x1000'], 'squash': -r['squash'],
                     'law': p['motion']['law_prob_mean'], 'rho': p['motion']['rho_decoded']})
    return rows


def objects(rows):
    O = {}
    for r in rows: O.setdefault(r['obj'], []).append(r)
    return {o: {'cat': v[0]['cat'], 'n_trials': len(v), **{k: float(np.mean([x[k] for x in v])) for k in ('motion', 'static', 'squash', 'rho')},
                'law': {l: float(np.mean([x['law'][l] for x in v])) for l in v[0]['law']}} for o, v in O.items()}


def auc(soft, rigid):
    s = np.asarray(soft)[:, None]; r = np.asarray(rigid)[None, :]
    return float(np.mean((s < r) + 0.5 * (s == r)))


def endpoints(O, keys=('motion', 'static', 'squash')):
    soft = [o for o in O if O[o]['cat'] == 'rubber']; rig = [o for o in O if O[o]['cat'] in RIGID]
    sp = [o for o in O if O[o]['cat'] in LIT]
    res = {'n_soft': len(soft), 'n_rigid': len(rig), 'n_spearman': len(sp)}
    if not soft or not rig: return res
    bs = [(rng.choice(soft, len(soft)), rng.choice(rig, len(rig)), rng.choice(sp, len(sp))) for _ in range(B)]
    boot = {}
    for k in keys:
        a = auc([O[o][k] for o in soft], [O[o][k] for o in rig])
        rh = float(spearmanr([O[o][k] for o in sp], [LIT[O[o]['cat']] for o in sp]).statistic)
        A = np.array([auc([O[o][k] for o in s], [O[o][k] for o in r]) for s, r, _ in bs])
        R = np.array([spearmanr([O[o][k] for o in q], [LIT[O[o]['cat']] for o in q]).statistic for _, _, q in bs])
        boot[k] = (A, R)
        res[k] = {'P1_auc': a, 'P1_ci': np.nanpercentile(A, [2.5, 97.5]).tolist(), 'P2_spearman': rh, 'P2_ci': np.nanpercentile(R, [2.5, 97.5]).tolist()}
    for b in keys[1:]:
        dA = boot['motion'][0] - boot[b][0]; dR = boot['motion'][1] - boot[b][1]
        res[f'motion_minus_{b}'] = {'P1_diff': res['motion']['P1_auc'] - res[b]['P1_auc'], 'P1_ci': np.nanpercentile(dA, [2.5, 97.5]).tolist(),
                                    'P2_diff': res['motion']['P2_spearman'] - res[b]['P2_spearman'], 'P2_ci': np.nanpercentile(dR, [2.5, 97.5]).tolist()}
    return res


excl = {}
for tid, r in index.items():
    excl.setdefault(cat(r['object']), {'total': 0}); excl[cat(r['object'])]['total'] += 1
    if 'excluded' in r: excl[cat(r['object'])][r['excluded']] = excl[cat(r['object'])].get(r['excluded'], 0) + 1
Op = objects(trial_rows(PRIMARY)); Oa = objects(trial_rows(['table', 'metal', 'glass', 'foam', 'rug']))
R = {'protocol': 'docs/v3w/PROTOCOL.json', 'exclusions_by_category_all_surfaces': excl,
     'primary_surfaces': PRIMARY, 'primary': endpoints(Op), 'S1_all_surfaces': endpoints(Oa),
     'S1_per_surface': {s: endpoints(objects(trial_rows([s]))) for s in ['table', 'metal', 'glass', 'foam', 'rug']}}
tr = trial_rows(PRIMARY)
R['S2_trial_level_auc'] = {k: auc([t[k] for t in tr if t['cat'] == 'rubber'], [t[k] for t in tr if t['cat'] in RIGID]) for k in ('motion', 'static', 'squash')}
R['S2_n_trials'] = {'soft': sum(t['cat'] == 'rubber' for t in tr), 'rigid': sum(t['cat'] in RIGID for t in tr)}
C = {}
for o, v in Op.items(): C.setdefault(v['cat'], []).append(v)
R['per_category_primary'] = {c: {'n_objects': len(v), 'n_trials': sum(x['n_trials'] for x in v),
                                 'pred_log10_E_over_rho_x1000_mean': float(np.mean([x['motion'] for x in v])),
                                 'static_mean': float(np.mean([x['static'] for x in v])), 'squash_mean': float(np.mean([-x['squash'] for x in v])),
                                 'law_prob_mean': {l: float(np.mean([x['law'][l] for x in v])) for l in v[0]['law']}} for c, v in sorted(C.items())}
dens = {o: mass[o] / vol[o] * 1000 for o in Op}
R['S4_rho_vs_measured_density'] = {'spearman': float(spearmanr([Op[o]['rho'] for o in Op], [dens[o] for o in Op]).statistic), 'n': len(Op)}
R['objects_primary'] = {o: {**v, 'measured_density_kg_m3': dens[o]} for o, v in sorted(Op.items())}
Path(resf).write_text(json.dumps(R, indent=1))
p = R['primary']
print(json.dumps({k: p[k] for k in p if not k.startswith('n_')}, indent=1)); print(p['n_soft'], p['n_rigid'], p['n_spearman'])
print(json.dumps(R['per_category_primary'], indent=1)); print(R['S2_trial_level_auc'], R['S4_rho_vs_measured_density'], json.dumps(excl))
