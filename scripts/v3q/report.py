"""V3q report -> docs/v3q/RESULTS.json and docs/paper_figs/data/f1_bands_v3.json (see docs/v3q/PROTOCOL.json).

Errors are recomputed from the stored estimates (log10 specific x at every nested budget k) against gt.json:
  z error (primary, V3 prior stats) = |ln1p(10^x_hat) - ln1p(10^x_gt)| / std_j ; also V2 z units and log10 decades.
Identifiable targets: E/rho always; sigma_y/rho where the GT law has it and it is not censored (or prior target_mask).
If the chosen law is jelly, sigma_y estimate = prior prediction (prior arms) / all-law train mean (law-mean arms), as V2.
CIs: object-clustered bootstrap (2000 draws, seed 17), resampler of scripts/v3h/report.py.

  .venv/bin/python scripts/v3q/report.py
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3h')); sys.path.insert(0, str(ROOT / 'scripts/v3b'))
DRAWS, SEED = 2000, 17
REFINE = os.environ.get('V3Q_REFINE', 'data/v3q_refine')
OUT_RES = os.environ.get('V3Q_RESULTS', 'docs/v3q/RESULTS.json')
OUT_BANDS = os.environ.get('V3Q_BANDS', 'docs/paper_figs/data/f1_bands_v3.json')
ARMS = ['prior_top1', 'prior_top2', 'lawmean_gt', 'lawmean_best3']
PROBES = {'prior_top1': 0, 'prior_top2': 2, 'lawmean_gt': 0, 'lawmean_best3': 3}
KS = list(range(25))
PROTO = json.loads((ROOT / 'docs/v3q/PROTOCOL.json').read_text())
LM = json.loads((ROOT / 'docs/v3q/LAW_MEANS.json').read_text())
V2STATS = json.loads((ROOT / 'data/manifests/v2main.specific.stats.json').read_text())
V2REF = {'val': {'prior_top1': {0: .344, 4: .327, 12: .283, 24: .236}, 'lawmean_gt': {0: .674, 4: .577, 12: .339, 24: .257},
                 'lawmean_best3': {4: .771, 12: .536, 24: .482}},
         'test': {'prior_top1': {0: .496, 4: .573, 12: .430, 24: .298}}}


import functools


@functools.lru_cache(maxsize=None)
def prior_meta(split):
    f = ROOT / f'docs/v3q/PRIOR_ENSEMBLE_{split}.json'
    return json.loads(f.read_text()) if f.exists() else None


def stats_for():
    for split in ('val', 'test'):
        p = prior_meta(split)
        if p and p.get('stats'):
            return p['stats'], p.get('stats_source', 'prior hand-off')
    return LM['stats_provisional'], 'docs/v3q/LAW_MEANS.json stats_provisional (sampler over planned V3 train indices)'


STATS, STATS_SRC = stats_for()


def errs(s, k, split):
    """[(param j, z_v3, z_v2, decades)] identifiable errors at budget k (None if k not available)."""
    b = s['budgets'].get(str(k))
    if b is None:
        return None
    gtE, gtS = s['gt_x_log10']
    sig_defined = gtS is not None and not s.get('censored_sigma')
    pm = prior_meta(split)
    if pm is not None and str(s['index']) in pm['samples'] and pm['samples'][str(s['index'])].get('target_mask') is not None:
        sig_defined = bool(pm['samples'][str(s['index'])]['target_mask'][3])
    xE, xS = b['x']
    if b['law'] == 'jelly':
        xS = s['prior_x_log10_unclipped'][1] if s['arm'].startswith('prior') else math.log10(math.expm1(LM['all_law_logS']))
    out = []
    for j, xh, xg in ((0, xE, gtE), (3, xS, gtS)):
        if j == 3 and not sig_defined:
            continue
        d = abs(math.log1p(10 ** xh) - math.log1p(10 ** xg))
        out.append((j, d / STATS['std'][j], d / V2STATS['std'][j], abs(xh - xg)))
    return out


def load(split):
    plan = PROTO['samples'][split]['samples']
    repl = json.loads((ROOT / 'docs/v3q/SAMPLES_RESOLVED.json').read_text())[split] if (ROOT / 'docs/v3q/SAMPLES_RESOLVED.json').exists() else [p['index'] for p in plan]
    S = {a: {} for a in ARMS}
    for i in repl:
        for a in ARMS:
            f = ROOT / f'{REFINE}/{split}/{i:06d}/{a}/state.json'
            if f.exists():
                s = json.loads(f.read_text())
                if s.get('complete'):
                    S[a][i] = s
    return repl, S


def pooled(S, a, sel, k, split, unit=1):
    e = []
    for i in sel:
        v = errs(S[a][i], k, split)
        if v is None:
            return None
        e += [x[unit] for x in v]
    return float(np.mean(e)) if e else None


def groups(S0, idx):
    g = {}
    for i in idx:
        g.setdefault(S0[i]['object'], []).append(i)
    return list(g.values())


def boot_sel(gr, rng):
    return [i for g in rng.integers(0, len(gr), len(gr)) for i in gr[g]]


def cross_log(prior_mae, unc):
    ks = sorted(k for k in unc if k > 0)
    for a, b in zip(ks, ks[1:]):
        if unc[a] <= prior_mae:
            return float(a)
        if unc[b] <= prior_mae:
            t = (unc[a] - prior_mae) / (unc[a] - unc[b])
            return float(math.exp(math.log(a) + t * (math.log(b) - math.log(a))))
    return float(ks[0]) if ks and unc[ks[0]] <= prior_mae else math.inf


def cross_lin(target, curve):
    ks = sorted(curve)
    if curve[ks[0]] <= target:
        return float(ks[0])
    for a, b in zip(ks, ks[1:]):
        if curve[b] <= target:
            return float(a + (curve[a] - target) / (curve[a] - curve[b]) * (b - a))
    return math.inf


def q(arr):
    arr = np.array(arr, float); f = lambda v: None if not np.isfinite(v) else float(v)
    return {'ci95': [f(np.quantile(arr, .025)), f(np.quantile(arr, .975))], 'fraction_censored': float(np.mean(~np.isfinite(arr)))}


def analyse(split):
    idx_all, S = load(split)
    present = [a for a in ARMS if S[a]]
    res = {'planned': len(idx_all), 'complete_per_arm': {a: len(S[a]) for a in ARMS}}
    if not present:
        return res, None
    # max budget available per arm (best3 may be cut to 12)
    kmax = {a: min(s.get('complete_budget', 24) for s in S[a].values()) for a in present}
    common = [i for i in idx_all if all(i in S[a] for a in present)]
    res.update(arms_present=present, kmax=kmax, analysed_indices=common, n_analysed=len(common), partial=len(common) < len(idx_all))
    if not common:
        return res, None
    S0 = S[present[0]]; gr = groups(S0, common); rng = np.random.default_rng(SEED)
    boots = [boot_sel(gr, rng) for _ in range(DRAWS)]
    res['n_objects'] = len(gr)
    curves, bands = {}, {}
    for a in present:
        ks = [k for k in KS if k <= kmax[a] and not (a == 'lawmean_best3' and k == 0)]
        curves[a], bands[a] = {}, {}
        for k in ks:
            m = pooled(S, a, common, k, split)
            bs = [pooled(S, a, sel, k, split) for sel in boots]
            bands[a][str(k)] = {'mean': m, 'ci95': np.quantile(bs, [.025, .975]).tolist()}
            curves[a][str(k)] = {'z_mae_v3': m, 'ci95': bands[a][str(k)]['ci95'], 'z_mae_v2units': pooled(S, a, common, k, split, 2),
                                 'log10_decades_mae': pooled(S, a, common, k, split, 3),
                                 'E_z_mae': float(np.mean([e[1] for i in common for e in errs(S[a][i], k, split) if e[0] == 0])),
                                 'sigma_z_mae': (lambda v: float(np.mean(v)) if v else None)([e[1] for i in common for e in errs(S[a][i], k, split) if e[0] == 3]),
                                 'law_accuracy': float(np.mean([S[a][i]['budgets'][str(k)]['law'] == S[a][i]['gt_law'] for i in common])),
                                 'observed_iou': (lambda v: float(np.mean(v)) if v else None)([S[a][i]['budgets'][str(k)].get('iou') for i in common if S[a][i]['budgets'][str(k)].get('iou') is not None])}
    res['curves'] = curves
    res['search_only_accounting'] = {a: {str(int(k) - PROBES[a]): v['z_mae_v3'] for k, v in curves[a].items() if int(k) >= PROBES[a]} for a in present}
    # sims saved vs lawmean_gt
    if 'lawmean_gt' in present:
        unc_ks = [k for k in KS if k <= kmax['lawmean_gt']]
        res['sims_saved_vs_lawmean_gt'] = {}
        for a in [x for x in ('prior_top1', 'prior_top2') if x in present]:
            out = {}
            for k in (0, 4, 12):
                if k > kmax[a]:
                    continue
                def point(sel):
                    unc = {kk: pooled(S, 'lawmean_gt', sel, kk, split) for kk in unc_ks}
                    pm = pooled(S, a, sel, k, split)
                    return cross_log(pm, unc), cross_lin(pm, unc)
                est = point(common); bl = [point(sel) for sel in boots]
                f = lambda v: None if not np.isfinite(v) else float(v)
                out[str(k)] = {'prior_mae': pooled(S, a, common, k, split), 'sims_to_match_log_interp': f(est[0]),
                               'sims_saved_log': f(est[0] - k) if np.isfinite(est[0]) else None,
                               'log_interp_bootstrap': q([b[0] for b in bl]), 'sims_to_match_linear': f(est[1]),
                               'linear_bootstrap': q([b[1] for b in bl]), 'censored_above_24': bool(not np.isfinite(est[0]))}
            res['sims_saved_vs_lawmean_gt'][a] = out
    # paired differences prior_top1 - other at equal budgets
    if 'prior_top1' in present:
        def boot_mean(vals, objs):
            by = {}
            for v, o in zip(vals, objs):
                by.setdefault(o, []).append(v)
            x = np.array([np.mean(v) for v in by.values()])
            r = np.random.default_rng(SEED).choice(x, size=(DRAWS, len(x)), replace=True).mean(1)
            return {'mean': float(np.mean(vals)), 'ci95': np.quantile(r, [.025, .975]).tolist()}
        objs = [S0[i]['object'] for i in common]
        sm = lambda a, i, k: float(np.mean([e[1] for e in errs(S[a][i], k, split)]))
        for other in [x for x in present if x != 'prior_top1']:
            for k in (0, 4, 12, 24):
                ko = (3 if k == 0 else k) if other == 'lawmean_best3' else k
                if ko > kmax[other] or k > kmax['prior_top1']:
                    continue
                res.setdefault('top1_minus_other_same_budget', {})[f'{other}_k{k}'] = boot_mean([sm('prior_top1', i, k) - sm(other, i, ko) for i in common], objs)
    # post-refinement vs initial error (per sample, no-probe arms)
    fit = {}
    for k in (4, 12, 24):
        pts = {}
        for a in [x for x in ('prior_top1', 'lawmean_gt') if x in present and k <= kmax[x]]:
            pts[a] = [(np.mean([e[1] for e in errs(S[a][i], 0, split)]), np.mean([e[1] for e in errs(S[a][i], k, split)])) for i in common]
        if not pts:
            continue
        def lf(p):
            x, y = np.array(p).T; sl, ic = np.polyfit(x, y, 1)
            return {'slope': float(sl), 'intercept': float(ic), 'r': float(np.corrcoef(x, y)[0, 1]), 'n': len(p)}
        fit[str(k)] = {'pooled_per_sample': lf([v for p in pts.values() for v in p]), **{f'{a}_per_sample': lf(p) for a, p in pts.items()},
                       'arm_points': {a: [float(np.mean([e[1] for i in common for e in errs(S[a][i], 0, split)])), curves[a][str(k)]['z_mae_v3']] for a in pts}}
    res['post_vs_initial_error'] = fit
    # failures
    res['simulation_failures'] = {a: {'failed': sum(bool(e.get('failed')) for i in common for e in S[a][i]['evals']),
                                      'total': sum(len(S[a][i]['evals']) for i in common),
                                      'failed_start_or_probe': sum(bool(e.get('failed')) for i in common for e in S[a][i]['evals'][:max(PROBES[a], 1)]),
                                      'samples_with_failure': sum(any(e.get('failed') for e in S[a][i]['evals']) for i in common),
                                      'search_exhausted_before_budget': sum(bool(S[a][i].get('search_exhausted')) for i in common)} for a in present}
    allev = [e for a in present for i in common for e in S[a][i]['evals']]
    uniq = {e['cache'] for e in allev if 'cache' in e}
    res['simulations'] = {'evaluations': len(allev), 'unique_simulations': len(uniq),
                          'median_seconds': float(np.median([e['seconds'] for e in allev])) if allev else None}
    # law choice of the search arms
    res['law_choice'] = {a: {'start_law_accuracy': float(np.mean([S[a][i].get('law_start', S[a][i]['law']) == S[a][i]['gt_law'] for i in common])) if a != 'lawmean_best3' else None,
                             'search_law_accuracy': float(np.mean([S[a][i]['law'] == S[a][i]['gt_law'] for i in common]))} for a in present}
    gto = []
    for p in PROTO['samples'][split]['samples']:
        f = ROOT / f'{REFINE}/{split}/{p["index"]:06d}/gt_observed/state.json'
        if f.exists():
            gto.append(json.loads(f.read_text()))
    res['noise_floor_gt_resim_observed'] = {'n': len(gto), 'iou': float(np.mean([g['iou'] for g in gto if 'iou' in g])) if gto else None,
                                            'rgb_mae_union': float(np.mean([g['rgb_mae_union'] for g in gto if 'iou' in g])) if gto else None,
                                            'loss': float(np.mean([g['loss'] for g in gto if 'iou' in g])) if gto else None,
                                            'per_action': {g['observed_action']: {k: g.get(k) for k in ('iou', 'rgb_mae_union', 'loss', 'failed')} for g in gto}}
    res['samples'] = {a: [{'index': i, 'object': S[a][i]['object'], 'gt_law': S[a][i]['gt_law'], 'action': S[a][i]['observed_action'], 'law': S[a][i]['law'],
                           'err_z_v3': {str(k): [e[1] for e in errs(S[a][i], k, split)] for k in (0, 4, 12, 24) if str(k) in S[a][i]['budgets']},
                           'evals': [[e['law'], e['x'], round(e['loss'], 4)] for e in S[a][i]['evals']]} for i in common] for a in present}
    return res, {'n_samples': len(common), 'n_objects': len(gr), 'arms': bands}


def v2_per_sample_fit():
    """Like-for-like per-sample post-vs-initial fit on the V2 val60 states (scripts/v3h/report.py loaders)."""
    try:
        import report as R
    except Exception as e:  # noqa
        return {'error': str(e)}
    idx60 = [p['index'] for p in json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']]
    out = {}
    for k in (4, 12, 24):
        pts = {}
        for a in ('prior_top1', 'lawmean_gt'):
            rows = [R.load_reused(a, i) for i in idx60]
            pts[a] = [(np.mean(r['curve'][0]['err']), np.mean(r['curve'][k]['err'])) for r in rows if r is not None]
        def lf(p):
            x, y = np.array(p).T; sl, ic = np.polyfit(x, y, 1)
            return {'slope': float(sl), 'intercept': float(ic), 'r': float(np.corrcoef(x, y)[0, 1]), 'n': len(p)}
        out[str(k)] = {'pooled_per_sample': lf([v for p in pts.values() for v in p]), **{f'{a}_per_sample': lf(p) for a, p in pts.items()}}
    return out


def main():
    out = {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'protocol': 'docs/v3q/PROTOCOL.json',
           'z_stats': STATS_SRC, 'note': 'k counts all simulations incl. probes; z_mae_v3 = primary (V3 z units), z_mae_v2units / '
           'log10_decades_mae for the V2 comparison (V2 numbers are V2 z units).'}
    bands = {'note': 'V3q F1-style bands: identifiable z-MAE (V3 z units) with object-bootstrap 95% CIs (2000 draws, seed 17) at every k; '
             'same structure as docs/paper_figs/data/f1_bands.json (keys val60/test30 -> n_samples, n_objects, arms -> <arm> -> <k> -> mean, ci95). '
             'Written by scripts/v3q/report.py.', 'draws': DRAWS, 'seed': SEED}
    for split, key in (('val', 'val60'), ('test', 'test30')):
        r, b = analyse(split)
        out[split] = r
        if b:
            bands[key] = b
            bands.setdefault('sims_saved_prior_top1_k0_vs_lawmean_gt', {})[key] = (r.get('sims_saved_vs_lawmean_gt', {}).get('prior_top1', {}).get('0', {}).get('sims_to_match_log_interp'))
    out['v2_per_sample_post_vs_initial_fit_val60'] = v2_per_sample_fit()
    comp = {}
    for split in ('val', 'test'):
        for a, ref in V2REF[split].items():
            c = out[split].get('curves', {}).get(a)
            comp[f'{split}_{a}'] = {str(k): {'v2_z': v, 'v3_z': (c or {}).get(str(k), {}).get('z_mae_v3'), 'v3_in_v2_z_units': (c or {}).get(str(k), {}).get('z_mae_v2units'),
                                             'v3_log10_decades': (c or {}).get(str(k), {}).get('log10_decades_mae')} for k, v in ref.items()}
    out['v2_comparison'] = comp
    (ROOT / OUT_RES).write_text(json.dumps(out, indent=1) + '\n')
    (ROOT / OUT_BANDS).write_text(json.dumps(bands, indent=1) + '\n')
    for split in ('val', 'test'):
        for a, c in out[split].get('curves', {}).items():
            print(split, a, out[split]['n_analysed'], {k: round(v['z_mae_v3'], 3) for k, v in c.items() if k in ('0', '3', '4', '12', '24')})


if __name__ == '__main__':
    main()
