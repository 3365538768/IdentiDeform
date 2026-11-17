"""V3h report: prior quality vs simulation-in-the-loop refinement (validation only) -> docs/v3h/PRIOR_QUALITY.json
(figure: scripts/v3h/make_figure.py). See docs/v3h/PROTOCOL.json (pre-registered arms, subset, noise draws).

Reused C1 arms (data/v3b_refine/v2main_C_seed29): curves at every k are recomputed from the stored evaluation
sequence with the exact c1_refine.py budget rule (checked against the stored budgets), so all arms are compared
at the same k on the same samples. Also breaks down the 60-sample C1 results by observed action and by GT law.

  .venv/bin/python scripts/v3h/report.py [--allow-partial]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3b'))
import c1_refine as C  # noqa: E402

PROTO = json.loads((ROOT / 'docs/v3h/PROTOCOL.json').read_text())
REUSED = {'prior': '', 'prior_top2': 'prior_top2', 'lawmean_gt': 'lawmean_gt', 'lawmean_best3': 'lawmean_best3', 'prior_top1': 'prior_top1'}
NEW = ['dino_c', 'gtnoise_0.25', 'gtnoise_0.5', 'gtnoise_1', 'ens_map']
PROBES = {'prior_top2': 2, 'lawmean_best3': 3}
LABEL = {'prior_top1': 'V-JEPA 9-ens top-1 law (no probes)', 'prior': 'V-JEPA single (C s29)', 'prior_top2': 'V-JEPA 9-ens + top-2 probes', 'ens_map': 'V-JEPA 9-ens MAP',
         'dino_c': 'DINOv3 C 3-ens', 'lawmean_gt': 'GT-law train mean (priv.)', 'lawmean_best3': 'best-of-3 law means',
         'gtnoise_0.25': 'GT + N(0,0.25 dec) (priv.)', 'gtnoise_0.5': 'GT + N(0,0.5 dec) (priv.)', 'gtnoise_1': 'GT + N(0,1 dec) (priv.)'}
MEANS = C.law_means()
DRAWS, SEED = 2000, 17


def ident(abs_err):
    return [v for j, v in enumerate(abs_err) if j in (0, 3) and v is not None]


def reused_errs(s, arm, kmax):
    """Identifiable abs errors at k=0..kmax, re-derived exactly as c1_refine.py's budget block."""
    law, law0, x0, base_z = s['law'], s.get('law_start', s['law']), s['x0'], s['base_z']
    out = {}
    tz, m = np.array(s['target_z']), np.array(s['target_mask'], bool)
    for k in range(kmax + 1):
        n = min(k, len(s['evals']))
        if arm == 'lawmean_best3' and n == 0:
            continue
        if n == 0:
            lk, x = law0, x0
        else:
            e = min(s['evals'][:n], key=lambda e: e['loss']); lk, x = e.get('law', law), e['x']
        z = list(base_z) if (lk == law or arm in ('prior_top2', 'prior_top1')) else list(MEANS[lk])
        z[0] = C.encode(0, 10 ** x[0])
        if lk != 'jelly':
            z[3] = C.encode(3, 10 ** x[1])
        elif arm not in ('prior', 'prior_sigma', 'prior_top2', 'prior_top1'):
            z[3] = 0.0
        err = np.abs(np.array(z) - tz)
        out[k] = {'err': ident([float(v) if mm else None for v, mm in zip(err, m)]), 'law': lk}
        b = s['budgets'].get(str(k))
        if b is not None:
            assert np.allclose(out[k]['err'], ident(b['abs_err_z']), atol=1e-6), (s['sample_id'], arm, k)
    return out


def load_reused(arm, i):
    f = ROOT / 'data/v3b_refine/v2main_C_seed29' / f'{i:06d}' / REUSED[arm] / 'state.json'
    if not f.exists():
        assert arm == 'prior_top1', f
        return None
    s = json.loads(f.read_text())
    if not s.get('complete'):
        assert arm == 'prior_top1', f
        return None
    return {'object': s['object'], 'action': s['observed_action'], 'gt_law': s['gt_law'], 'curve': reused_errs(s, arm, 24),
            'n_evals': len(s['evals']), 'failed': sum(bool(e.get('failed')) for e in s['evals']),
            'replay': {int(k): v['iou'] for k, v in s.get('replay', {}).items() if k.isdigit() and isinstance(v, dict) and 'iou' in v},
            'observed_iou': {k: (b.get('observed_fit') or {}).get('iou') for k, b in s['budgets'].items()}}


def load(idx_list, allow_partial):
    S = {}
    for arm, sub in REUSED.items():
        S[arm] = {}
        for i in idx_list:
            r = load_reused(arm, i)
            if r is not None:
                S[arm][i] = r
    for arm in NEW:
        S[arm] = {}
        for i in idx_list:
            f = ROOT / 'data/v3h_refine' / arm / f'{i:06d}' / 'state.json'
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            if not s.get('complete'):
                continue
            S[arm][i] = {'object': s['object'], 'action': s['observed_action'], 'gt_law': s['gt_law'],
                         'curve': {int(k): {'err': ident(b['abs_err_z']), 'law': b['law']} for k, b in s['budgets'].items()},
                         'n_evals': len(s['evals']), 'failed': sum(bool(e.get('failed')) for e in s['evals']),
                         'seconds': [e['seconds'] for e in s['evals']]}
    return S


def pooled(S, arm, sel, k):
    e = []
    for i in sel:
        c = S[arm][i]['curve'].get(k)
        if c is None:
            return None
        e += c['err']
    return float(np.mean(e)) if e else None


def groups_of(S, idx):
    g = {}
    for i in idx:
        g.setdefault(S['prior'][i]['object'], []).append(i)
    return list(g.values())


def boot_sel(groups, rng):
    return [i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]


def crossing(curve, target):
    """Smallest (linearly interpolated) k at which the pooled curve reaches target; inf if never within the curve."""
    ks = sorted(curve)
    if curve[ks[0]] <= target:
        return float(ks[0])
    for a, b in zip(ks, ks[1:]):
        if curve[b] <= target:
            return float(a + (curve[a] - target) / (curve[a] - curve[b]) * (b - a))
    return math.inf


def ci(vals):
    v = np.array(vals, float)
    fin = v[np.isfinite(v)]
    q = lambda p: (None if not np.isfinite(x := float(np.quantile(v, p))) else x)
    return {'ci95': [q(.025), q(.975)], 'median': q(.5), 'fraction_censored': float(np.mean(~np.isfinite(v))),
            'n_finite': int(len(fin))}


def analyse(S, idx, arms, kmax_new=12):
    groups = groups_of(S, idx); rng = np.random.default_rng(SEED)
    boots = [boot_sel(groups, rng) for _ in range(DRAWS)]
    init_k = lambda a: PROBES.get(a, 0) if a == 'lawmean_best3' else 0
    kmax = lambda a: kmax_new if a in NEW else 24
    out = {'n_samples': len(idx), 'n_objects': len(groups), 'arms': {}}
    T = {'T24_lawmean_gt': lambda sel: pooled(S, 'lawmean_gt', sel, 24), 'T12_lawmean_gt': lambda sel: pooled(S, 'lawmean_gt', sel, 12)}
    out['targets'] = {t: f(idx) for t, f in T.items()}
    for a in arms:
        curve = {k: pooled(S, a, idx, k) for k in range(kmax(a) + 1)}
        curve = {k: v for k, v in curve.items() if v is not None}
        row = {'label': LABEL[a], 'probes_in_budget': PROBES.get(a, 0), 'init_budget': init_k(a), 'curve': curve,
               'law_accuracy_final': float(np.mean([S[a][i]['curve'][max(S[a][i]['curve'])]['law'] == S[a][i]['gt_law'] for i in idx])),
               'law_accuracy_start': float(np.mean([S[a][i]['curve'][min(S[a][i]['curve'])]['law'] == S[a][i]['gt_law'] for i in idx])),
               'failed_sims': sum(S[a][i]['failed'] for i in idx if a in NEW) if a in NEW else None,
               'sims_run': sum(S[a][i]['n_evals'] for i in idx) if a in NEW else None}
        for k in sorted({init_k(a), 4, 12, 24} & set(curve)):
            bs = [pooled(S, a, sel, k) for sel in boots]
            row.setdefault('mae', {})[str(k)] = {'mean': curve[k], 'ci95': np.quantile(bs, [.025, .975]).tolist()}
            if a != 'lawmean_gt':
                ko = k if k in (4, 12, 24) else 0
                d = lambda sel: pooled(S, a, sel, k) - pooled(S, 'lawmean_gt', sel, ko)
                row.setdefault('minus_lawmean_gt_same_budget', {})[str(k)] = {'mean': d(idx), 'ci95': np.quantile([d(s) for s in boots], [.025, .975]).tolist(),
                                                                              'lawmean_gt_budget': ko}
        k0 = init_k(a)
        sm = lambda i, k: float(np.mean(S[a][i]['curve'][k]['err']))
        row['fraction_samples_error_increased'] = {str(k): float(np.mean([sm(i, k) > sm(i, k0) + 1e-9 for i in idx])) for k in (4, 12)}
        for k in (4, 12):
            g = lambda sel: pooled(S, a, sel, k) - pooled(S, a, sel, k0)
            row.setdefault('gain_vs_init', {})[str(k)] = {'mean': g(idx), 'ci95': np.quantile([g(s) for s in boots], [.025, .975]).tolist()}
        for t, f in T.items():
            def cross(sel):
                c = {k: pooled(S, a, sel, k) for k in range(kmax(a) + 1)}
                return crossing({k: v for k, v in c.items() if v is not None}, f(sel))
            est = cross(idx)
            row.setdefault('sims_to_target', {})[t] = {'estimate': None if not np.isfinite(est) else est,
                                                       'censored_above': None if np.isfinite(est) else max(curve),
                                                       'bootstrap': ci([cross(s) for s in boots])}
        out['arms'][a] = row
    return out


def breakdown60(S60, idx):
    out = {}
    arms = [a for a in ('prior', 'prior_top1', 'prior_top2', 'lawmean_gt', 'lawmean_best3') if all(i in S60[a] for i in idx)]
    for key in ('action', 'gt_law', 'drop_vs_driven'):
        out[key] = {}
        val = lambda i: (S60['prior'][i]['action'] if key == 'action' else S60['prior'][i]['gt_law'] if key == 'gt_law'
                         else ('drop' if S60['prior'][i]['action'] == 'drop' else 'driven(bend/press/shear/stretch/tap30hold)'))
        for v in sorted({val(i) for i in idx}):
            sel = [i for i in idx if val(i) == v]; groups = groups_of(S60, sel); rng = np.random.default_rng(SEED)
            boots = [boot_sel(groups, rng) for _ in range(DRAWS)]
            r = {'n': len(sel), 'n_objects': len(groups)}
            for a in arms:
                r[a] = {str(k): pooled(S60, a, sel, k) for k in ((3, 4, 12, 24) if a == 'lawmean_best3' else (0, 4, 12, 24))}
            for a in [x for x in ('prior', 'prior_top1', 'prior_top2') if x in arms]:
                for k in (0, 4, 12, 24):
                    for u in ('lawmean_gt', 'lawmean_best3'):
                        ku = (3 if k == 0 else k) if u == 'lawmean_best3' else k
                        d = lambda s: pooled(S60, a, s, k) - pooled(S60, u, s, ku)
                        r.setdefault('diff', {})[f'{a}_minus_{u}_k{k}'] = {'mean': d(sel), 'ci95': np.quantile([d(s) for s in boots], [.025, .975]).tolist()}
            r['prior_law_accuracy'] = float(np.mean([S60['prior'][i]['curve'][0]['law'] == S60['prior'][i]['gt_law'] for i in sel]))
            r['ensemble_top1_law_accuracy'] = float(np.mean([S60['prior_top2'][i]['curve'][0]['law'] == S60['prior_top2'][i]['gt_law'] for i in sel]))
            r['prior_top2_law_accuracy_k24'] = float(np.mean([S60['prior_top2'][i]['curve'][24]['law'] == S60['prior_top2'][i]['gt_law'] for i in sel]))
            out[key][v] = r
    out['arms'] = arms
    return out


def top1_val60(S60, idx):
    """prior_top1 (ensemble, top-1 law, no probes; scripts/v3d/c1_refine.py --split val) vs the other C1 arms on val."""
    arms = ['prior_top1', 'prior_top2', 'prior', 'lawmean_gt', 'lawmean_best3']
    res = analyse(S60, idx, arms)
    groups = groups_of(S60, idx); rng = np.random.default_rng(SEED)
    boots = [boot_sel(groups, rng) for _ in range(DRAWS)]
    diffs = {}
    for k in (0, 4, 12, 24):
        for u in arms[1:]:
            ku = (3 if k == 0 else k) if u == 'lawmean_best3' else k
            d = lambda s: pooled(S60, 'prior_top1', s, k) - pooled(S60, u, s, ku)
            diffs[f'top1_minus_{u}_k{k}'] = {'mean': d(idx), 'ci95': np.quantile([d(s) for s in boots], [.025, .975]).tolist(), 'other_budget': ku}
    iou = {}
    for a in arms:
        for k in (0, 4, 12, 24):
            rep = [S60[a][i]['replay'].get(k) for i in idx if S60[a][i]['replay'].get(k) is not None]
            obs = [S60[a][i]['observed_iou'].get(str(k)) for i in idx if S60[a][i]['observed_iou'].get(str(k)) is not None]
            iou.setdefault(a, {})[str(k)] = {'replay_iou': float(np.mean(rep)) if rep else None, 'n_replay': len(rep),
                                             'observed_iou': float(np.mean(obs)) if obs else None, 'n_observed': len(obs)}
    return {'n': len(idx), 'complete': len(idx) == 60, 'result': res, 'paired': diffs, 'iou': iou,
            'failed_sims_top1': sum(S60['prior_top1'][i]['failed'] for i in idx), 'sims_top1': sum(S60['prior_top1'][i]['n_evals'] for i in idx)}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--allow-partial', action='store_true'); args = a.parse_args()
    idx = PROTO['subset']['indices']
    S = load(idx, args.allow_partial)
    done = {arm: sorted(S[arm]) for arm in NEW}
    primary = ['dino_c', 'gtnoise_0.25', 'gtnoise_0.5', 'gtnoise_1']
    common = [i for i in idx if all(i in S[arm] for arm in primary)]
    if len(common) < len(idx) and not args.allow_partial:
        raise SystemExit(f'incomplete: {len(common)}/{len(idx)} (use --allow-partial)')
    arms = [a for a in REUSED if all(i in S[a] for i in common)] + primary
    if not common and args.allow_partial:  # smoke test of the reused-arm path only
        common = list(idx); arms = [a for a in REUSED if all(i in S[a] for i in common)]
    res = analyse(S, common, arms)
    out = {'protocol': 'docs/v3h/PROTOCOL.json (pre-registered)', 'note': __doc__.strip(), 'complete_samples_per_new_arm': {k: len(v) for k, v in done.items()},
           'common_samples': common, 'main': res}
    ens_common = [i for i in common if i in S['ens_map']]
    if ens_common:
        out['ens_map_secondary'] = {'n': len(ens_common), 'complete': len(ens_common) == len(idx),
                                    'result': analyse(S, ens_common, ['ens_map', 'prior', 'prior_top2', 'lawmean_gt', 'dino_c', 'gtnoise_0.5'])}
    # Init-quality scaling across all arms with no probes (budget-0 init -> budget-k error).
    np_arms = [x for x in arms if x not in PROBES]
    xs = np.array([res['arms'][x]['curve'][0] for x in np_arms])
    for k in (4, 12) if len(np_arms) > 2 else ():
        ys = np.array([res['arms'][x]['curve'][k] for x in np_arms])
        sl, ic = np.polyfit(xs, ys, 1)
        out.setdefault('scaling_fit_no_probe_arms', {})[str(k)] = {'arms': np_arms, 'slope': float(sl), 'intercept': float(ic),
                                                                    'r': float(np.corrcoef(xs, ys)[0, 1])}
    plan = json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']
    idx60 = [p['index'] for p in plan]
    S60 = {arm: {i: r for i in idx60 if (r := load_reused(arm, i)) is not None} for arm in REUSED}
    out['breakdown_60'] = breakdown60(S60, idx60)
    t1 = [i for i in idx60 if i in S60['prior_top1']]
    if t1:
        out['prior_top1_val60'] = top1_val60(S60, t1)
    secs = [v for arm in NEW for s in S[arm].values() for v in s['seconds']]
    out['new_simulations'] = {'n': len(secs), 'failed': sum(s['failed'] for arm in NEW for s in S[arm].values()),
                              'median_seconds': float(np.median(secs)) if secs else None}
    (ROOT / 'docs/v3h/PRIOR_QUALITY.json').write_text(json.dumps(out, indent=1) + '\n')
    for x in arms:
        r = res['arms'][x]
        print(f"{x:14s} " + ' '.join(f"k{k}={r['curve'].get(k, float('nan')):.3f}" for k in (0, 3, 4, 12, 24)),
              {t: (v['estimate'], v['bootstrap']['ci95']) for t, v in r['sims_to_target'].items()})


if __name__ == '__main__':
    main()
