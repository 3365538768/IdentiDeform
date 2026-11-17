"""Aggregate V3b C1 (simulation-in-the-loop refinement) into docs/v3b/C1_REFINE_RESULTS.json (validation only).

Arms (same objective/optimizer, same samples): prior (model MAP law + prediction, fixed step 0.5 decade),
prior_sigma (same start, initial step = model predictive std), lawmean_gt (privileged GT law, train mean of
that law), lawmean_best3 (non-privileged: the 3 law means are simulated first, search continues from the
best). Accuracy is reported on the identifiable targets E/rho and sigma_y/rho (train-standardized z of
ln1p specific values; sigma_y only where defined), pooled over (sample, target) elements.
Speedup: smallest number of uninformed simulations whose curve reaches the prior arm's error at k sims
(linear interpolation in log budget), with an object-clustered bootstrap over samples.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho']
ARMS = ['prior', 'prior_sigma', 'lawmean_gt', 'lawmean_best3', 'prior_top2']


def load_states(root, run, plan):
    out = {a: {} for a in ARMS}
    for p in plan:
        for arm in ARMS:
            f = ROOT / root / run / f"{p['index']:06d}" / ('' if arm == 'prior' else arm) / 'state.json'
            if f.exists():
                s = json.loads(f.read_text())
                if s.get('budgets'):
                    s['_complete'] = bool(s.get('complete'))
                    out[arm][p['index']] = s
    return out


def ident_err(s, k):
    b = s['budgets'].get(str(k))
    if b is None:
        return None
    e = b['abs_err_z']
    return [v for j, v in enumerate(e) if j in (0, 3) and v is not None]


def curve(states, idx, budgets, fn):
    out = {}
    for k in budgets:
        vals = [fn(states[i], k) for i in idx if i in states and str(k) in states[i]['budgets']]
        vals = [v for v in vals if v is not None]
        if len(vals) == len(idx):
            out[k] = vals
    return out


def pooled_mae(states, idx, k):
    e = []
    for i in idx:
        v = ident_err(states[i], k)
        if v is None:
            return None
        e += v
    return float(np.mean(e)) if e else None


def crossing(prior_mae, unc):
    """Smallest uninformed budget whose (interpolated, log-budget) curve reaches prior_mae; inf if never."""
    ks = sorted(unc)
    for a, b in zip(ks, ks[1:]):
        if unc[a] <= prior_mae:
            return float(a)
        if unc[b] <= prior_mae:
            t = (unc[a] - prior_mae) / (unc[a] - unc[b])
            return float(math.exp(math.log(a) + t * (math.log(b) - math.log(a))))
    return float(ks[0]) if ks and unc[ks[0]] <= prior_mae else math.inf


def speedup(sp, su, idx, prior_ks, unc_ks, draws=1000, seed=17):
    objs = {}
    for i in idx:
        objs.setdefault(sp[i]['object'], []).append(i)
    groups = list(objs.values())
    def point(sel):
        uc = {k: pooled_mae(su, sel, k) for k in unc_ks}
        uc = {k: v for k, v in uc.items() if v is not None}
        return {k: crossing(pooled_mae(sp, sel, k), uc) for k in prior_ks}, uc
    est, uc = point(idx)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in prior_ks}
    for _ in range(draws):
        sel = [i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]
        b, _ = point(sel)
        for k in prior_ks:
            boots[k].append(b[k])
    res = {}
    for k in prior_ks:
        arr = np.array(boots[k])
        q = np.quantile(arr, [.025, .5, .975])
        fmt = lambda v: None if not np.isfinite(v) else float(v)
        res[str(k)] = {'prior_mae': pooled_mae(sp, idx, k), 'uninformed_sims_to_match': fmt(est[k]),
                       'censored_above_max_budget': bool(not np.isfinite(est[k])),
                       'speedup_factor': (fmt(est[k] / k) if k else None),
                       'bootstrap_sims_to_match_ci95': [fmt(q[0]), fmt(q[2])],
                       'bootstrap_fraction_censored': float(np.mean(~np.isfinite(arr))),
                       'speedup_factor_ci95': ([fmt(q[0] / k), fmt(q[2] / k)] if k else None)}
    return {'n_samples': len(idx), 'n_objects': len(groups), 'uninformed_curve': {str(k): v for k, v in uc.items()},
            'max_uninformed_budget': max(unc_ks), 'by_prior_budget': res}


def boot_mean(values, groups, draws=2000, seed=17):
    by = {}
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(v)
    x = np.array([np.mean(v) for v in by.values()])
    r = np.random.default_rng(seed).choice(x, size=(draws, len(x)), replace=True).mean(1)
    return {'mean': float(np.mean(values)), 'n': len(values), 'objects': len(x), 'ci95': np.quantile(r, [.025, .975]).tolist()}


def arm_table(states, idx, budgets):
    rows = {}
    for k in budgets:
        have = [i for i in idx if str(k) in states[i]['budgets']]
        if len(have) < len(idx):
            continue
        bs = [states[i]['budgets'][str(k)] for i in idx]
        e = [v for i in idx for v in ident_err(states[i], k)]
        eE = [b['abs_err_z'][0] for b in bs]
        eS = [b['abs_err_z'][3] for b in bs if b['abs_err_z'][3] is not None]
        fits = [b['observed_fit'] for b in bs if b.get('observed_fit') and not b['observed_fit'].get('failed')]
        rep = [states[i]['replay'][str(k)] for i in idx if isinstance(states[i]['replay'].get(str(k)), dict)
               and 'iou' in states[i]['replay'][str(k)]]
        rows[str(k)] = {'identifiable_mae_pooled': float(np.mean(e)), 'E_over_rho_mae': float(np.mean(eE)),
                        'sigma_y_over_rho_mae': float(np.mean(eS)) if eS else None, 'n_sigma': len(eS),
                        'all4_mae_pooled': float(np.mean([v for b in bs for v in b['abs_err_z'] if v is not None])),
                        'law_accuracy': float(np.mean([b['law'] == states[i]['gt_law'] for b, i in zip(bs, idx)])) if all('law' in b for b in bs) else None,
                        'observed_iou': float(np.mean([f['iou'] for f in fits])) if fits else None,
                        'observed_rgb_mae': float(np.mean([f['rgb_mae_union'] for f in fits])) if fits else None,
                        'n_observed_fit': len(fits),
                        'replay_iou': float(np.mean([r['iou'] for r in rep])) if rep else None,
                        'replay_rgb_mae': float(np.mean([r['rgb_mae_union'] for r in rep])) if rep else None,
                        'n_replay': len(rep),
                        'wall_seconds_mean': float(np.mean([b['wall_seconds'] for b in bs])),
                        'sims_mean': float(np.mean([b['sims'] for b in bs]))}
    return rows


def main():
    a = argparse.ArgumentParser(); a.add_argument('--run', default='v2main_C_seed29')
    a.add_argument('--root', default='data/v3b_refine'); a.add_argument('--output', default='docs/v3b/C1_REFINE_RESULTS.json')
    args = a.parse_args()
    plan = json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']
    subset = json.loads((ROOT / 'docs/v3b/queue.json').read_text()).get('c1_subset48', [])
    S = load_states(args.root, args.run, plan)
    out = {'protocol': __doc__.strip() + f' Start model: best-val-epoch predictions of {args.run}. Objective (1-IoU)+RGB_MAE_union/255 '
           'vs the observed 3-view 128-frame video; compass search over log10(E/rho) [and log10(sigma_y/rho) unless the law is '
           'jelly], opportunistic, halving the step after an unsuccessful poll; budget k = best of the first k simulations of '
           'one search (nested). nu, rho fixed (prior: predicted; law-mean arms: law train mean); hardening/softening 0.25. '
           'Replay = unseen action ACTIONS[(i+3)%6] vs a GT replay. Validation split only; test untouched.',
           'planned_samples': len(plan), 'subset48': subset,
           'states': {arm: {'any': len(S[arm]), 'complete': sum(s['_complete'] for s in S[arm].values())} for arm in ARMS}}
    budgets = {'prior': [0, 3, 4, 8, 12, 16, 24], 'prior_sigma': [0, 3, 4, 8, 12], 'lawmean_gt': [0, 3, 4, 8, 12, 16, 24, 32, 48],
               'lawmean_best3': [3, 4, 8, 12, 16, 24, 32, 48], 'prior_top2': [0, 2, 3, 4, 8, 12, 16, 24]}
    # Common sample set: complete in prior and both uninformed arms (24-budget jobs).
    def complete_at(arm, k):
        return {i for i, s in S[arm].items() if str(k) in s['budgets'] and s['budgets'][str(k)]['sims'] >= min(k, 1) and
                (s['_complete'] or len(s['evals']) >= k)}
    common = sorted(complete_at('prior', 24) & complete_at('lawmean_gt', 24) & complete_at('lawmean_best3', 24))
    out['common_samples'] = len(common)
    out['curves_common'] = {arm: arm_table(S[arm], common, [k for k in budgets[arm] if k <= 24]) for arm in ARMS if arm not in ('prior_sigma', 'prior_top2')}
    cs = sorted(set(common) & complete_at('prior_sigma', 12))
    out['prior_sigma_vs_prior_budget12'] = {'n': len(cs),
        'prior_sigma': arm_table(S['prior_sigma'], cs, [0, 4, 8, 12]), 'prior': arm_table(S['prior'], cs, [0, 4, 8, 12])}
    if cs:
        d = [np.mean(ident_err(S['prior_sigma'][i], 12)) - np.mean(ident_err(S['prior'][i], 12)) for i in cs]
        out['prior_sigma_vs_prior_budget12']['sample_delta_identifiable_mae'] = boot_mean(d, [S['prior'][i]['object'] for i in cs])
    if common:
        objs = [S['prior'][i]['object'] for i in common]
        for arm in ('prior', 'lawmean_gt', 'lawmean_best3'):
            for k in (12, 24):
                d = [np.mean(ident_err(S[arm][i], k)) - np.mean(ident_err(S[arm][i], 0 if arm != 'lawmean_best3' else 3)) for i in common]
                out.setdefault('refinement_gain', {})[f'{arm}_budget{k}_minus_start'] = boot_mean(d, objs)
        for k in (0, 4, 12, 24):
            for u in ('lawmean_gt', 'lawmean_best3'):
                d = [np.mean(ident_err(S['prior'][i], k)) - np.mean(ident_err(S[u][i], k if k else 3 if u == 'lawmean_best3' else 0)) for i in common]
                out.setdefault('prior_minus_uninformed_same_budget', {})[f'{u}_budget{k}'] = boot_mean(d, objs)
        out['speedup_all'] = {u: speedup(S['prior'], S[u], common, [0, 4, 12, 24], [k for k in budgets[u] if 3 <= k <= 24])
                              for u in ('lawmean_gt', 'lawmean_best3')}
        sub = sorted(set(subset) & set(common) & complete_at('lawmean_gt', 48) & complete_at('lawmean_best3', 48))
        out['subset48_samples'] = len(sub)
        if sub:
            out['curves_subset48'] = {arm: arm_table(S[arm], sub, budgets[arm]) for arm in ('prior', 'lawmean_gt', 'lawmean_best3')}
            out['speedup_subset48'] = {u: speedup(S['prior'], S[u], sub, [0, 4, 12, 24], [k for k in budgets[u] if 3 <= k <= 48])
                                       for u in ('lawmean_gt', 'lawmean_best3')}
    # prior_top2: 9-run B/C/D ensemble prediction simulated under its top-2 laws (2 probe sims), then the same
    # search; budget 0 = 'prior_ens' (ensemble prediction, top-1 law, no sims). Fair counterpart of lawmean_best3.
    c2 = sorted(set(common) & complete_at('prior_top2', 24))
    t2 = {'n': len(c2), 'partial': len(c2) < len(plan),
          'note': 'budget k counts ALL simulations incl. law probes (prior_top2: 2, lawmean_best3: 3); '
                  'search_only_accounting relabels each curve by k minus its probes.'}
    if c2:
        o2 = [S['prior'][i]['object'] for i in c2]
        t2['curves'] = {arm: arm_table(S[arm], c2, [k for k in budgets[arm] if k <= 24]) for arm in ('prior_top2', 'prior', 'lawmean_gt', 'lawmean_best3')}
        t2['search_only_accounting'] = {arm: {str(int(k) - probes): v['identifiable_mae_pooled'] for k, v in t2['curves'][arm].items() if int(k) >= probes}
                                        for arm, probes in (('prior_top2', 2), ('lawmean_best3', 3), ('prior', 0), ('lawmean_gt', 0))}
        t2['prior_ens_budget0_vs_prior_budget0'] = boot_mean([np.mean(ident_err(S['prior_top2'][i], 0)) - np.mean(ident_err(S['prior'][i], 0)) for i in c2], o2)
        for k in (0, 4, 12, 24):
            for other in ('lawmean_best3', 'lawmean_gt', 'prior'):
                ko = (3 if k == 0 else k) if other == 'lawmean_best3' else k
                d = [np.mean(ident_err(S['prior_top2'][i], k)) - np.mean(ident_err(S[other][i], ko)) for i in c2]
                t2.setdefault('top2_minus_other_same_budget', {})[f'{other}_budget{k}'] = boot_mean(d, o2)
        t2['speedup'] = {u: speedup(S['prior_top2'], S[u], c2, [0, 4, 12, 24], [k for k in budgets[u] if 3 <= k <= 24])
                         for u in ('lawmean_gt', 'lawmean_best3')}
    out['prior_top2'] = t2
    gto = [s['gt_observed'] for s in S['prior'].values() if 'iou' in s.get('gt_observed', {})]
    if gto:
        out['noise_floor_gt_resim_observed'] = {'n': len(gto), 'iou': float(np.mean([g['iou'] for g in gto])),
                                                'rgb_mae_union': float(np.mean([g['rgb_mae_union'] for g in gto]))}
    allev = [e for arm in ARMS for s in S[arm].values() for e in s['evals']]
    out['simulations'] = {'n': len(allev), 'failed': sum(bool(e.get('failed')) for e in allev),
                          'median_seconds': float(np.median([e['seconds'] for e in allev])) if allev else None}
    out['samples'] = {arm: [{'index': i, 'object': s['object'], 'gt_law': s['gt_law'], 'law': s.get('law'),
                             'action': s['observed_action'], 'complete': s['_complete'],
                             'ident_err': {k: ident_err(s, k) for k in s['budgets']},
                             'x': {k: b['x'] for k, b in s['budgets'].items()},
                             'replay_iou': {k: v.get('iou') for k, v in s['replay'].items() if isinstance(v, dict)},
                             'evals': [[e.get('law'), e['x'], round(e['loss'], 4)] for e in s['evals']]}
                            for i, s in sorted(S[arm].items())] for arm in ARMS}
    (ROOT / args.output).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'samples'}, indent=1)[:20000])


if __name__ == '__main__':
    main()
