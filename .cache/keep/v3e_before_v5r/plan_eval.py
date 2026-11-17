"""V3e gripper task: planning, execution outcome, success/regret, object-bootstrap CIs -> docs/v3e/GRIPPER_TASK.json.

Planning (per method, in its twin): feasible = {d in DEPTHS : d >= D_MIN, twin sim ok, twin dent(d) <= TAU};
choose d* = max(feasible), or ABSTAIN (refuse to grasp) if empty. Execution = the GT-parameter squeeze at d*
(the GT runs at every candidate depth are simulated once and shared; they are also the oracle twin).
Success: d* chosen and GT dent(d*) <= TAU, or ABSTAIN when no candidate is GT-feasible (correct refusal).
Regret: |d* - d_oracle| with ABSTAIN coded as d = 0.
Fixed heuristic: one depth (or abstain) for every object, chosen leave-one-out on the OTHER evaluated samples
(maximum success; ties -> larger d), i.e. tuned on disjoint samples with GT outcomes.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import squeeze_sim as S  # noqa: E402

ROOT = S.ROOT
METHODS = ['gt', 'refined_top2', 'ensemble_prior', 'refined_best3', 'fixed_loo', 'lawmean_pred']
LABEL = {'gt': 'GT parameters (oracle twin)', 'refined_top2': 'prior_top2 + 12-sim refinement (ours)',
         'ensemble_prior': 'ensemble prior, no refinement (0 sims)', 'refined_best3': 'lawmean_best3 + 12-sim refinement (uninformed)',
         'fixed_loo': 'fixed depth (leave-one-out tuned)', 'lawmean_pred': 'law-mean parameters, predicted law'}


ABS = False


def dents(state, metric):
    out = {}
    for d in S.DEPTHS:
        r = state['runs'].get(f'{d:.3f}')
        if r is None:
            continue
        out[d] = None if (r.get('failed') or 'metrics' not in r) else (abs(r['metrics'][metric]) if ABS else r['metrics'][metric])
    return out


def plan(dd, tau):
    feas = [d for d, v in dd.items() if d >= S.D_MIN - 1e-9 and v is not None and v <= tau]
    return max(feas) if feas else None


def boot(rows, objs, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); uo = sorted(set(objs)); by = defaultdict(list)
    for r, o in zip(rows, objs):
        by[o].append(r)
    vals = []
    for _ in range(n):
        pick = [r for o in rng.choice(uo, len(uo)) for r in by[o]]
        vals.append(fn(pick))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    a = argparse.ArgumentParser(); a.add_argument('--runs', default='data/v3e/runs'); a.add_argument('--metric', default='crush'); a.add_argument('--abs', action='store_true'); a.add_argument('--dmin', type=float, default=None)
    a.add_argument('--tau', type=float, default=S.TAU); a.add_argument('--only', default='')  # 'wave1' or comma list
    a.add_argument('--out', default='docs/v3e/GRIPPER_TASK.json'); a.add_argument('--samples', default='docs/v3e/SAMPLES.json'); args = a.parse_args()
    global ABS
    ABS = args.abs
    if args.dmin is not None:
        S.D_MIN = args.dmin
    only = None
    if args.only == 'wave1':
        only = set(json.loads((ROOT / args.samples).read_text())['wave1'])
    elif args.only:
        only = {int(v) for v in args.only.split(',')}
    samples = []
    for sd in sorted((ROOT / args.runs).iterdir()):
        if not sd.name.isdigit() or (only is not None and int(sd.name) not in only):
            continue
        st = {}
        for m in METHODS:
            p = sd / m / 'state.json'
            if p.exists() and json.loads(p.read_text()).get('complete'):
                st[m] = json.loads(p.read_text())
        if 'gt' not in st:
            continue
        gt = json.loads((ROOT / f'{S.DATA}/{int(sd.name):06d}/gt.json').read_text())
        samples.append({'index': int(sd.name), 'law': gt['law'], 'object': gt['object'], 'states': st})
    rows, excluded = [], []
    for s in samples:
        gd = dents(s['states']['gt'], args.metric)
        if not gd or all(v is None for v in gd.values()):
            # GT execution itself failed (MPM blow-up with GT parameters, or GT reference run failed): not evaluable
            excluded.append({'index': s['index'], 'law': s['law'], 'object': s['object'], 'reason': 'GT simulations failed'})
            continue
        oracle = plan(gd, args.tau)
        gt_ok = lambda d: d is not None and gd.get(d) is not None and gd[d] <= args.tau and d >= S.D_MIN - 1e-9
        row = {'index': s['index'], 'law': s['law'], 'object': s['object'], 'gt_dent': {f'{d:.2f}': v for d, v in gd.items()},
               'oracle_d': oracle, 'methods': {}}
        for m, st in s['states'].items():
            td = dents(st, args.metric); ch = plan(td, args.tau)
            succ = (ch is None and oracle is None) or gt_ok(ch)
            sims_plan = sum(1 for r in st['runs'].values())
            meta = st.get('meta', {})
            row['methods'][m] = {'twin_law': meta.get('law'), 'twin_params': st['params'], 'twin_dent': {f'{d:.2f}': v for d, v in td.items()},
                                 'chosen_d': ch, 'success': bool(succ), 'regret': abs((ch or 0) - (oracle or 0)),
                                 'damaged': bool(ch is not None and not gt_ok(ch)), 'missed_grasp': bool(ch is None and oracle is not None),
                                 'gt_dent_at_choice': gd.get(ch) if ch is not None else None,
                                 'utility': (ch if gt_ok(ch) else 0.0) if ch is not None else 0.0,
                                 'sims_refine': meta.get('sims_refine', 0), 'sims_plan': sims_plan,
                                 'wall_seconds_refine': meta.get('refine_wall_seconds', 0.0),
                                 'wall_seconds_plan': float(sum(r.get('seconds', 0) for r in st['runs'].values()))}
        rows.append(row)
    # fixed depth, leave-one-out
    options = [None] + list(S.DEPTHS)
    def fixed_success(r, d):
        if d is None:
            return r['oracle_d'] is None
        v = r['gt_dent'].get(f'{d:.2f}')
        return v is not None and v <= args.tau and d >= S.D_MIN - 1e-9
    for i, r in enumerate(rows):
        others = [o for j, o in enumerate(rows) if j != i]
        best = max(options, key=lambda d: (sum(fixed_success(o, d) for o in others), -1 if d is None else d))
        r['methods']['fixed_loo'] = {'chosen_d': best, 'success': bool(fixed_success(r, best)),
                                     'utility': (best if fixed_success(r, best) else 0.0) if best is not None else 0.0, 'regret': abs((best or 0) - (r['oracle_d'] or 0)),
                                     'damaged': bool(best is not None and not fixed_success(r, best)),
                                     'missed_grasp': bool(best is None and r['oracle_d'] is not None), 'sims_refine': 0, 'sims_plan': 0,
                                     'wall_seconds_refine': 0.0, 'wall_seconds_plan': 0.0}
    fixed_global = {('abstain' if d is None else f'{d:.2f}'): float(np.mean([fixed_success(r, d) for r in rows])) for d in options}
    # summary
    objs = [r['object'] for r in rows]; summary = {}
    complete = [r for r in rows if all(m in r['methods'] for m in METHODS)]
    cobjs = [r['object'] for r in complete]
    for m in METHODS:
        get = lambda rs, k: [x['methods'][m][k] for x in rs]
        mrows = [r for r in rows if m in r['methods']]
        ent = {'label': LABEL[m], 'n': len(mrows)}
        if not mrows:
            summary[m] = ent; continue
        groups = [('all', mrows)] + [(law, [r for r in mrows if r['law'] == law]) for law in S.LAWS] + \
                 [('plastic_laws', [r for r in mrows if r['law'] != 'jelly'])]
        for grp, rs in groups:
            if not rs:
                continue
            ob = [r['object'] for r in rs]
            ent[grp] = {'n': len(rs), 'success': float(np.mean(get(rs, 'success'))),
                        'success_ci95_object_bootstrap': boot(rs, ob, lambda p: np.mean([x['methods'][m]['success'] for x in p])),
                        'regret_mean': float(np.mean(get(rs, 'regret'))),
                        'regret_ci95_object_bootstrap': boot(rs, ob, lambda p: np.mean([x['methods'][m]['regret'] for x in p])),
                        'damage_rate': float(np.mean(get(rs, 'damaged'))), 'missed_grasp_rate': float(np.mean(get(rs, 'missed_grasp'))),
                        'mean_chosen_d': float(np.mean([v or 0 for v in get(rs, 'chosen_d')])),
                        'utility_mean': float(np.mean(get(rs, 'utility'))),
                        'utility_ci95_object_bootstrap': boot(rs, ob, lambda p: np.mean([x['methods'][m]['utility'] for x in p])),
                        'oracle_utility_mean': float(np.mean([x['oracle_d'] or 0.0 for x in rs]))}
        ent['sims_per_object'] = float(np.mean(get(mrows, 'sims_refine'))) + float(np.mean(get(mrows, 'sims_plan')))
        ent['wall_seconds_per_object_refine'] = float(np.mean(get(mrows, 'wall_seconds_refine')))
        ent['wall_seconds_per_object_plan'] = float(np.mean(get(mrows, 'wall_seconds_plan')))
        summary[m] = ent
    paired = {}
    for m in METHODS:
        for b in ('ensemble_prior', 'lawmean_pred', 'fixed_loo', 'refined_best3'):
            both = [r for r in rows if m in r['methods'] and b in r['methods']]
            if m == b or not both:
                continue
            f = lambda p: np.mean([x['methods'][m]['success'] - x['methods'][b]['success'] for x in p])
            g = lambda p: np.mean([x['methods'][m]['regret'] - x['methods'][b]['regret'] for x in p])
            ob = [r['object'] for r in both]
            u = lambda p: np.mean([x['methods'][m]['utility'] - x['methods'][b]['utility'] for x in p])
            paired[f'{m}-minus-{b}'] = {'n': len(both), 'success_diff': float(f(both)), 'success_ci95_object_bootstrap': boot(both, ob, f),
                                        'utility_diff': float(u(both)), 'utility_ci95_object_bootstrap': boot(both, ob, u),
                                        'regret_diff': float(g(both)), 'regret_ci95_object_bootstrap': boot(both, ob, g)}
    out = {'protocol': {
        'task': 'damage-aware parallel-jaw squeeze: two rigid sticky plates close on the object along its narrower horizontal axis '
                'by d*W/2 each over 1.0 s, hold until 1.5 s, open (plates removed), free relaxation until 4.0 s; gravity, floor collider; '
                'same MPM settings as V2 main/C1 (low damping, substep 1e-4, n_grid from template).',
        'decision': 'largest candidate depth d whose predicted permanent crush <= tau and d >= d_min; else abstain (refuse grasp)',
        'candidate_depths': S.DEPTHS, 'd_min': S.D_MIN, 'tau': args.tau, 'damage_metric': args.metric,
        'damage_metric': args.metric + (' (absolute value)' if args.abs else ''),
        'damage_metric_definition': {
            'crush': 'PRIMARY. permanent crush = (span_ref - span_sq_aligned) / W: span = 1-99 % quantile extent along the squeeze axis of all MPM '
                     'particles averaged over 3.5-4.0 s (after release); the squeezed configuration is first rigidly (Procrustes) aligned to the '
                     'reference (d=0 run, same parameters), so rigid rotation after release is not counted; W = initial width along the squeeze '
                     'axis (narrower horizontal axis)',
            'wloss': 'same without rigid alignment (the originally planned metric; contaminated by rigid rotation of thin objects after release, '
                     'see scripts/v3e/diag_rotation.py) - sensitivity only',
            'dent_p95': 'P95 over particles of |aligned final position - reference final position| / W - sensitivity only'}[args.metric],
        'gravity': 'off for the task (g=0): under the V2-main gravity setting most elastoplastic objects slump to a puddle in <0.5 s '
                   '(pilot figure docs/v3e/figs/pilot_particles_gravity.png), which makes a grasp ill-posed',
        'execution': 'GT-parameter simulation at the chosen depth (shared GT runs at every candidate depth; Warp-MPM run-to-run nondeterminism noted separately)',
        'success': 'chosen d is GT-feasible, or abstain when no candidate depth is GT-feasible',
        'regret': '|d_chosen - d_oracle|, abstain = 0',
        'utility': 'secondary (added after the primary-metric switch, before final results): achieved safe grasp depth = d_chosen if GT-feasible '
                   'else 0 (damage or abstain); oracle utility = d_oracle',
        'twins': 'object geometry = observed object for every method; twins differ only in material law/parameters; hardening/softening '
                 'are not predicted (0.25 for twins, GT values for GT)',
        'fixed_heuristic': 'single depth or abstain for all objects, chosen leave-one-out on the other evaluated samples using their GT outcomes',
        'bootstrap': 'object (cluster) bootstrap over object identities, 2000 resamples, percentile 95% CI'},
        'summary': summary, 'paired_success_differences': paired, 'fixed_depth_success_global': fixed_global,
        'n_samples': len(rows), 'n_complete': len(complete), 'excluded_samples': excluded, 'samples': rows}
    p = ROOT / args.out; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1))
    print(f'n={len(rows)} complete={len(complete)} metric={args.metric} tau={args.tau}')
    for m in METHODS:
        e = summary[m]
        if 'all' not in e:
            continue
        al = e['all']
        per = ' '.join(f"{g}={e[g]['success']:.2f}" for g in S.LAWS if g in e)
        print(f"{m:15s} n={e['n']} succ={al['success']:.3f} CI={al['success_ci95_object_bootstrap'][0]:.2f}-{al['success_ci95_object_bootstrap'][1]:.2f} "
              f"regret={al['regret_mean']:.3f} util={al['utility_mean']:.3f}/{al['oracle_utility_mean']:.3f} dmg={al['damage_rate']:.2f} miss={al['missed_grasp_rate']:.2f} {per} sims={e['sims_per_object']:.1f}")
    print('fixed global', fixed_global)
    print('excluded', excluded)
    print('oracle depths', [r['oracle_d'] for r in rows])


if __name__ == '__main__':
    main()
