"""V5g: twin-planned grasp-and-lift evaluation -> docs/v5g/RESULTS.json (+ per-sample table).

Outcome of a candidate (position, grip level) in a simulation (thresholds from docs/v5g/PREREG.md, env-overridable):
  FAIL-drop   : left > T_DROP (more than that fraction of the object did not follow the jaws during the hold: slip / tear / droop),
  FAIL-damage : |permanent jaw crush| > T_DMG (aligned span loss of the gripped segment after release),
  SUCCESS     : otherwise; failed simulation = FAIL (drop).
Planner (any twin): among candidates predicted SUCCESS pick the lowest grip level (gentlest grasp); ties -> lower predicted 'left'
(less sag), then centre before end; none -> ABSTAIN. Execution = GT simulation of the chosen candidate.
Success = executed GT outcome SUCCESS, or ABSTAIN when no candidate is GT-feasible. Oracle = the same planner on GT.
Force regret (relative) = (GT grip force at the chosen grasp - GT grip force at the oracle grasp) / oracle force, on samples where the
chosen grasp succeeded and an oracle grasp exists; level regret = |level index chosen - level index oracle| (abstain/fail counted).
Fixed: one (position, level) for all objects, chosen leave-one-out on the other samples' GT outcomes (max success, ties -> gentler).
CIs: object (cluster) bootstrap, 2000 draws, percentile 95 %.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T_DROP = float(os.environ.get('V5G_T_DROP', '0.5'))
T_DMG = float(os.environ.get('V5G_T_DMG', '0.05'))
ARMS = ['gt', 'prior_vt', 'prior_vision', 'lawmean_pred', 'fixed_loo']
LABEL = {'gt': 'GT oracle', 'prior_vt': 'vision + touch prior twin', 'prior_vision': 'vision prior twin (9-run)',
         'lawmean_pred': 'class-mean twin', 'fixed_loo': 'fixed grasp (leave-one-out)'}


CANDS = os.environ.get('V5G_CANDS', 'end:0.03,end:0.10,end:0.25,end:0.40,centre:0.03,centre:0.10,centre:0.25,centre:0.40').split(',')
# 08:45 deviation: the analysis uses these 8 candidates for every sample (levels 0.06 / 0.16 were dropped from unstarted jobs)


def load_state(sd, m):
    """(sample dir, method) -> state restricted to CANDS, or None if any of them is not simulated yet (cache state, not the
    per-method state.json, which is only written when all 12 originally planned candidates exist)."""
    tw = sd / m / 'twin.json'
    if not tw.exists():
        return None
    t = json.loads(tw.read_text())
    try:
        st = json.loads((sd / 'cache' / t['cache'] / 'state.json').read_text())
    except (OSError, ValueError):
        return None
    runs = st.get('runs', {})
    if not all(c in runs and (runs[c].get('failed') or 'metrics' in runs[c] or runs[c].get('ref_failed')) for c in CANDS):
        return None
    return {'runs': {c: runs[c] for c in ['none:0'] + CANDS if c in runs}, 'meta': t['meta'], 'params': t['params'], 'complete': True}


def outcome(r):
    m = r.get('metrics')
    if r.get('failed') or m is None:
        return 'drop'
    if m['left'] > T_DROP:
        return 'drop'
    if abs(m['crush']) > T_DMG:
        return 'damage'
    return 'success'


def plan(runs, levels):
    feas = [(levels.index(r['level']), (r.get('metrics') or {}).get('left', 1), 0 if r['pos'] == 'centre' else 1, c)
            for c, r in runs.items() if c != 'none:0' and outcome(r) == 'success']
    return min(feas)[3] if feas else None


def boot(rows, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for r in rows:
        by[r['object']].append(r)
    objs = sorted(by)
    v = [fn([r for o in rng.choice(objs, len(objs)) for r in by[o]]) for _ in range(n)]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]


def main():
    a = argparse.ArgumentParser(); a.add_argument('--runs', default='data/v5g/runs/val'); a.add_argument('--out', default='docs/v5g/RESULTS_core.json')
    args = a.parse_args()
    samples = []
    for sd in sorted((ROOT / args.runs).iterdir()):
        if not sd.name.isdigit():
            continue
        st = {}
        for m in ARMS[:-1]:
            s = load_state(sd, m)
            if s is not None:
                st[m] = s
        if 'gt' not in st:
            continue
        if os.environ.get('V5G_REQUIRE_ALL') == '1' and not all(m in st for m in ARMS[:-1]):
            continue  # paired-complete samples only (scripts/v5g/analyze.py)
        g = json.loads((ROOT / f'data/v5b_main/val/{sd.name}/gt.json').read_text())
        samples.append({'index': int(sd.name), 'law': g['law'], 'object': g['object'], 'states': st})
    cands = sorted({c for s in samples for c in s['states']['gt']['runs'] if c != 'none:0'})
    levels = sorted({s['states']['gt']['runs'][c]['level'] for s in samples for c in s['states']['gt']['runs'] if c != 'none:0'})
    rows = []
    for s in samples:
        G = s['states']['gt']['runs']; orc = plan(G, levels)
        gf = lambda c: (G[c].get('metrics') or {}).get('grip')
        row = {'index': s['index'], 'law': s['law'], 'object': s['object'], 'oracle': orc,
               'gt_outcomes': {c: outcome(G[c]) for c in cands}, 'methods': {}}
        for m, st in s['states'].items():
            ch = plan(st['runs'], levels)
            res = outcome(G[ch]) if ch else ('abstain_ok' if orc is None else 'abstain_miss')
            row['methods'][m] = {'chosen': ch, 'gt_outcome': res, 'twin_law': st['meta'].get('law'),
                                 'predicted': {c: outcome(st['runs'][c]) for c in cands if c in st['runs']}}
        rows.append(row)
    # fixed grasp, leave-one-out
    for i, r in enumerate(rows):
        others = [o for j, o in enumerate(rows) if j != i]
        opts = [None] + cands  # None = always abstain
        sc = lambda o, c: (o['oracle'] is None) if c is None else (o['gt_outcomes'][c] == 'success')
        best = max(opts, key=lambda c: (sum(sc(o, c) for o in others), 1 if c is None else -levels.index(float(c.split(':')[1]))))
        res = ('abstain_ok' if r['oracle'] is None else 'abstain_miss') if best is None else r['gt_outcomes'][best]
        r['methods']['fixed_loo'] = {'chosen': best, 'gt_outcome': res}
    for r in rows:
        G = next(s for s in samples if s['index'] == r['index'])['states']['gt']['runs']
        for m, v in r['methods'].items():
            ch, o = v['chosen'], v['gt_outcome']
            v['success'] = o in ('success', 'abstain_ok'); v['drop'] = o == 'drop'; v['damage'] = o == 'damage'; v['missed'] = o == 'abstain_miss'
            li = lambda c: levels.index(G[c]['level']) + 1 if c else 0
            v['level_regret'] = abs(li(ch) - li(r['oracle']))
            fo = (G[r['oracle']].get('metrics') or {}).get('grip') if r['oracle'] else None
            fc = (G[ch].get('metrics') or {}).get('grip') if ch else None
            v['force_regret_rel'] = (fc - fo) / fo if (o == 'success' and fo and fc is not None) else None
            v['grip_force_gt'] = fc
    summ = {}
    for m in ARMS:
        rs = [r for r in rows if m in r['methods']]
        if not rs:
            continue
        mean = lambda k: (lambda p: float(np.mean([x['methods'][m][k] for x in p])) if p else None)
        fr = lambda p: (float(np.mean(v)) if (v := [x['methods'][m]['force_regret_rel'] for x in p if x['methods'][m]['force_regret_rel'] is not None]) else None)
        summ[m] = {'label': LABEL[m], 'n': len(rs), **{k: mean(k)(rs) for k in ('success', 'drop', 'damage', 'missed', 'level_regret')},
                   'success_ci95': boot(rs, mean('success')), 'drop_ci95': boot(rs, mean('drop')), 'damage_ci95': boot(rs, mean('damage')),
                   'level_regret_ci95': boot(rs, mean('level_regret')), 'force_regret_rel': fr(rs), 'force_regret_rel_ci95': boot(rs, fr),
                   'n_force_regret': sum(x['methods'][m]['force_regret_rel'] is not None for x in rs),
                   'law_accuracy': (float(np.mean([x['methods'][m]['twin_law'] == x['law'] for x in rs])) if m not in ('gt', 'fixed_loo') else None)}
    paired = {}
    for name, (x, y) in {'vt-minus-vision': ('prior_vt', 'prior_vision'), 'vision-minus-classmean': ('prior_vision', 'lawmean_pred'),
                         'vt-minus-classmean': ('prior_vt', 'lawmean_pred'), 'vt-minus-fixed': ('prior_vt', 'fixed_loo'),
                         'vision-minus-fixed': ('prior_vision', 'fixed_loo'), 'classmean-minus-fixed': ('lawmean_pred', 'fixed_loo')}.items():
        rs = [r for r in rows if x in r['methods'] and y in r['methods']]
        if not rs:
            continue
        d = lambda k: (lambda p: float(np.mean([float(r['methods'][x][k]) - float(r['methods'][y][k]) for r in p])))
        paired[name] = {'n': len(rs), 'n_objects': len({r['object'] for r in rs}),
                        **{f'{k}_diff': d(k)(rs) for k in ('success', 'drop', 'damage', 'level_regret')},
                        **{f'{k}_diff_ci95': boot(rs, d(k)) for k in ('success', 'drop', 'damage', 'level_regret')}}
    gt_hist = {c: {o: sum(r['gt_outcomes'][c] == o for r in rows) for o in ('success', 'drop', 'damage')} for c in cands}
    out = {'thresholds': {'T_DROP_left': T_DROP, 'T_DMG_crush': T_DMG}, 'candidates': cands, 'levels': levels, 'n_samples': len(rows),
           'summary': summ, 'paired': paired, 'gt_outcome_by_candidate': gt_hist,
           'oracle_histogram': {str(k): sum(r['oracle'] == k for r in rows) for k in [None] + cands},
           'fixed_global_success': ({c: float(np.mean([r['gt_outcomes'][c] == 'success' for r in rows])) for c in cands} | {'abstain': float(np.mean([r['oracle'] is None for r in rows]))}) if rows else {},
           'samples': rows}
    p = ROOT / args.out; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1))
    for m, v in summ.items():
        print(f"{m:14s} n={v['n']} succ={v['success']:.3f} {v['success_ci95']} drop={v['drop']:.2f} dmg={v['damage']:.2f} miss={v['missed']:.2f} "
              f"lvl_regret={v['level_regret']:.2f} force_regret={v['force_regret_rel']}")
    print('oracle', out['oracle_histogram']); print('fixed global', out['fixed_global_success'])
    for k, v in paired.items():
        print(k, v['n'], round(v['success_diff'], 3), v['success_diff_ci95'])


if __name__ == '__main__':
    main()
