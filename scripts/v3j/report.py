"""V3j report -> docs/v3j/RESULTS.json (figure: scripts/v3j/make_figure.py) (validation only; see docs/v3j/PROTOCOL.json).

Reused compass arms (prior_top1, lawmean_gt, lawmean_best3 in data/v3b_refine) are re-derived at every k with
scripts/v3h/report.py reused_errs (checked against their stored budgets). All tables at budget k use the samples
that have budget k in every arm of that table (listed). CIs: paired object-clustered bootstrap (2000 draws, seed 17).

  .venv/bin/python scripts/v3j/report.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3b'))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('v3h_report', ROOT / 'scripts/v3h/report.py')
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)  # scripts/v3h/report.py (reused_errs, ident)

PROTO = json.loads((ROOT / 'docs/v3j/PROTOCOL.json').read_text())
IDX = PROTO['subset']['indices']
KS = [0, 4, 12, 24]
OLD = {'prior_top1': 'prior_top1', 'lawmean_gt': 'lawmean_gt', 'lawmean_best3': 'lawmean_best3'}
NEW = ['cma_prior', 'cma_gtmean', 'cma_best3', 'u4_prior', 'u4_gtmean']
EXP1 = ['prior_top1', 'lawmean_gt', 'lawmean_best3', 'cma_prior', 'cma_gtmean', 'cma_best3']
EXP2 = ['prior_top1', 'u4_prior', 'lawmean_gt', 'u4_gtmean']
LABEL = {'prior_top1': 'V-JEPA prior, compass 2-D', 'lawmean_gt': 'GT-law mean (priv.), compass 2-D',
         'lawmean_best3': 'best-of-3 law means, compass 2-D', 'cma_prior': 'V-JEPA prior, CMA-ES (s0=0.5)',
         'cma_gtmean': 'GT-law mean (priv.), CMA-ES (s0=1)', 'cma_best3': 'best-of-3 law means, CMA-ES (s0=1)',
         'u4_prior': 'V-JEPA prior, compass 4-D (unfrozen h, nu)', 'u4_gtmean': 'GT-law mean (priv.), compass 4-D'}
DRAWS, SEED = 2000, 17


def load():
    S = {}
    for arm, sub in OLD.items():
        S[arm] = {}
        for i in IDX:
            s = json.loads((H.C.ROOT / 'data/v3b_refine/v2main_C_seed29' / f'{i:06d}' / sub / 'state.json').read_text())
            cur = H.reused_errs(s, arm, 24)
            n = len(s['evals'])
            fit = {}
            for k in KS:
                if k and n:
                    e = min(s['evals'][:min(k, n)], key=lambda e: e['loss']); fit[k] = e['iou']
            S[arm][i] = {'object': s['object'], 'gt_law': s['gt_law'], 'action': s['observed_action'],
                         'curve': {k: v['err'] for k, v in cur.items()}, 'law': {k: v['law'] for k, v in cur.items()},
                         'sims': n, 'failed': sum(bool(e.get('failed')) for e in s['evals']), 'reused': 0,
                         'nu': {k: s['nu'] for k in range(25)}, 'h': {k: 0.25 for k in range(25)}, 'fit_iou': fit,
                         'replay': {int(k): v['iou'] for k, v in s.get('replay', {}).items() if k.isdigit() and isinstance(v, dict) and 'iou' in v}}
    for arm in NEW:
        S[arm] = {}
        for i in IDX:
            f = ROOT / 'data/v3j_refine' / arm / f'{i:06d}' / 'state.json'
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            if not s.get('complete') or 'budgets' not in s:
                if 'x0f' not in s or 'budgets' not in s:
                    continue
                # Interrupted extension (e.g. a budget-24 job cut at the deadline): best-of-first-k is final for every
                # k <= number of stored simulations, so the nested curve is re-derived up to that k (same rule as c1j_refine).
                s['budgets'] = derive(s, arm)
            b = s['budgets']
            S[arm][i] = {'object': s['object'], 'gt_law': s['gt_law'], 'action': s['observed_action'],
                         'curve': {int(k): H.ident(v['abs_err_z']) for k, v in b.items()}, 'law': {int(k): v['law'] for k, v in b.items()},
                         'sims': len(s['evals']), 'failed': sum(bool(e.get('failed')) for e in s['evals']),
                         'reused': sum('reused_from' in e for e in s['evals']), 'budget_run': s.get('budget_run'),
                         'nu': {int(k): v['nu'] for k, v in b.items()}, 'h': {int(k): v['hardening'] for k, v in b.items()},
                         'fit_iou': {int(k): (v['observed_fit'] or {}).get('iou') for k, v in b.items() if v['observed_fit']},
                         'replay': {int(k): v['iou'] for k, v in s.get('replay', {}).items() if k.isdigit() and 'iou' in v},
                         'gt_nuisance': s['gt_nuisance']}
    return S


def derive(s, arm):
    m = np.array(s['target_mask'], bool); out = {}
    for k in range(len(s['evals']) + 1):
        if arm == 'cma_best3' and k < 3:
            continue
        if k == 0:
            lk, xf = s['law'], s['x0f']
        else:
            e = min(s['evals'][:k], key=lambda e: e['loss']); lk, xf = e['law'], e['xf']
        z = list(s['base_z']) if lk == s['law'] else list(H.MEANS[lk])
        z[0] = H.C.encode(0, 10 ** xf[0])
        if lk != 'jelly':
            z[3] = H.C.encode(3, 10 ** xf[1])
        elif arm not in ('cma_prior', 'u4_prior'):
            z[3] = 0.0
        err = np.abs(np.array(z) - np.array(s['target_z']))
        fe = next((e for e in s['evals'][:k] if e['law'] == lk and e['xf'] == xf), None) if k else None
        out[str(k)] = {'law': lk, 'xf': xf, 'nu': xf[3], 'hardening': xf[2] if lk != 'jelly' else None,
                       'abs_err_z': [float(v) if mm else None for v, mm in zip(err, m)],
                       'observed_fit': None if fe is None else {'iou': fe['iou']}}
    return out


def common(S, arms, k):
    return [i for i in IDX if all(i in S[a] and k in S[a][i]['curve'] for a in arms)]


def pooled(S, arm, sel, k):
    e = [v for i in sel for v in S[arm][i]['curve'][k]]
    return float(np.mean(e)) if e else None


def groups(S, sel):
    g = {}
    for i in sel:
        g.setdefault(S['prior_top1'][i]['object'], []).append(i)
    return list(g.values())


def boot(S, sel, fn):
    rng = np.random.default_rng(SEED); G = groups(S, sel); out = []
    for _ in range(DRAWS):
        pick = [i for j in rng.integers(0, len(G), len(G)) for i in G[j]]
        v = fn(pick)
        if v is not None:
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None


def table(S, arms, ref='prior_top1'):
    out = {}
    for k in KS:
        a_k = [a for a in arms if not (k == 0 and a in ('lawmean_best3', 'cma_best3'))]
        sel = common(S, a_k, k)
        row = {'n_samples': len(sel), 'samples': sel, 'arms': {}}
        for a in a_k:
            d = {'mae': pooled(S, a, sel, k), 'mae_ci': boot(S, sel, lambda p: pooled(S, a, p, k))}
            if a != ref and sel:
                d['diff_vs_prior_top1'] = d['mae'] - pooled(S, ref, sel, k)
                d['diff_ci'] = boot(S, sel, lambda p: pooled(S, a, p, k) - pooled(S, ref, p, k))
            row['arms'][a] = d
        out[str(k)] = row
    return out


def sims_to_target(S, arms):
    """Target T = lawmean_gt pooled MAE at 24 on the samples with k=24 in every arm; k* = first crossing (linear interp.)."""
    sel = [i for i in common(S, arms, 24) if all(all(k in S[a][i]['curve'] for k in range(3 if a in ('lawmean_best3', 'cma_best3') else 0, 25)) for a in arms)]

    def kstar(a, p, T):
        ks = list(range(3 if a in ('lawmean_best3', 'cma_best3') else 0, 25)); c = [pooled(S, a, p, k) for k in ks]
        for j, (k, v) in enumerate(zip(ks, c)):
            if v <= T + 1e-12:
                if j == 0:
                    return float(k)
                return float(ks[j - 1] + (c[j - 1] - T) / (c[j - 1] - v))
        return None
    out = {'n_samples': len(sel), 'samples': sel}
    if not sel:
        return out
    T = pooled(S, 'lawmean_gt', sel, 24); out['target'] = T; out['arms'] = {}
    for a in arms:
        rng = np.random.default_rng(SEED); G = groups(S, sel); draws = []
        for _ in range(DRAWS):
            p = [i for j in rng.integers(0, len(G), len(G)) for i in G[j]]
            draws.append(kstar(a, p, pooled(S, 'lawmean_gt', p, 24)))
        fin = [d if d is not None else 25.0 for d in draws]
        out['arms'][a] = {'k_star': kstar(a, sel, T), 'ci_censored_at_25': [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))],
                          'frac_draws_reached': float(np.mean([d is not None for d in draws]))}
    return out


def curves(S, arms):
    sel = common(S, arms, 24)
    return {'n_samples': len(sel), 'samples': sel,
            'arms': {a: {str(k): pooled(S, a, sel, k) for k in range(25) if all(k in S[a][i]['curve'] for i in sel)} for a in arms}}


def nuisance(S, arms):
    out = {}
    for k in KS:
        sel = common(S, arms, k); row = {'n_samples': len(sel), 'arms': {}}
        for a in arms:
            nu_e, h_e, fit, rep = [], [], [], []
            for i in sel:
                gn = S['u4_prior'][i]['gt_nuisance'] if i in S['u4_prior'] else S['u4_gtmean'][i]['gt_nuisance']
                r = S[a][i]
                nu_e.append(abs(float(np.clip(r['nu'][k], 0.05, 0.48)) - gn['nu']))
                if gn['hardening'] is not None and r['law'][k] != 'jelly' and r['h'].get(k) is not None:
                    h_e.append(abs(r['h'][k] - gn['hardening']))
                if k in r['fit_iou'] and r['fit_iou'][k] is not None:
                    fit.append(r['fit_iou'][k])
            rsel = [i for i in sel if all(k in S[b][i]['replay'] for b in arms)]
            rep = [S[a][i]['replay'][k] for i in rsel]
            row['arms'][a] = {'nu_abs_err': float(np.mean(nu_e)), 'hardening_abs_err': float(np.mean(h_e)) if h_e else None, 'n_h': len(h_e),
                              'observed_fit_iou': float(np.mean(fit)) if fit else None,
                              'replay_iou': float(np.mean(rep)) if rep else None, 'n_replay': len(rsel)}
        out[str(k)] = row
    return out


def main():
    S = load()
    res = {'protocol': 'docs/v3j/PROTOCOL.json', 'subset': IDX,
           'coverage': {a: {'samples': len(S[a]), 'budget24': sum(1 for i in S[a] if 24 in S[a][i]['curve']),
                            'sims_total': sum(r['sims'] for r in S[a].values()), 'sims_reused': sum(r['reused'] for r in S[a].values()),
                            'failed_sims': sum(r['failed'] for r in S[a].values()),
                            'samples_with_failure': sorted(i for i, r in S[a].items() if r['failed'])} for a in S},
           'exp1_table': table(S, EXP1), 'exp2_table': table(S, EXP2), 'exp2_nuisance': nuisance(S, EXP2),
           'exp1_curves': curves(S, EXP1), 'exp2_curves': curves(S, EXP2),
           'sims_to_target_exp1': sims_to_target(S, EXP1), 'sims_to_target_exp2': sims_to_target(S, EXP2)}
    # Arms lacking budget 24 everywhere (e.g. cma_prior if dropped) are still reported at 0/4/12 via separate tables.
    for name, arms in (('exp1', EXP1), ('exp2', EXP2)):
        full = [a for a in arms if sum(1 for i in S[a] if 24 in S[a][i]['curve']) >= 1]
        if full != arms:
            res[f'{name}_table_24_available_arms'] = table(S, full)
            res[f'{name}_curves_24'] = curves(S, full); res[f'sims_to_target_{name}_24'] = sims_to_target(S, full)
    (ROOT / 'docs/v3j/RESULTS.json').write_text(json.dumps(res, indent=1) + '\n')
    for name in ('exp1_table', 'exp2_table'):
        print(name)
        for k, row in res[name].items():
            print(' k', k, 'n', row['n_samples'], {a: (round(d['mae'], 3) if d['mae'] is not None else None,
                                                       [round(x, 3) for x in d.get('diff_ci') or []]) for a, d in row['arms'].items()})
    print(json.dumps(res['coverage'])); print(json.dumps({k: res[k] for k in res if k.startswith('sims_to')})[:2000])
    print(json.dumps(res['exp2_nuisance'])[:3000])


if __name__ == '__main__':
    main()
