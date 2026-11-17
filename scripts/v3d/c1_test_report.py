"""Aggregate the pre-registered V3d C1 TEST run (docs/v3d/C1_TEST_PROTOCOL.json) into docs/v3d/C1_TEST_RESULTS.json.
Metric/aggregation functions are imported unchanged from scripts/v3b/c1_report.py (val analysis).
Analysis set = test samples whose three arms (prior_top2, lawmean_gt, lawmean_best3) are complete at budget 24."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3b'))
import c1_report  # noqa: E402
from c1_report import load_states, ident_err, arm_table, boot_mean, speedup  # noqa: E402
c1_report.ARMS = list(c1_report.ARMS) + ['prior_top1']  # v3d addendum arm (state dir <idx>/prior_top1)

ARMS3 = ('prior_top2', 'lawmean_gt', 'lawmean_best3')
BUDGETS = {'prior_top1': [0, 3, 4, 8, 12, 16, 24], 'prior_top2': [0, 2, 3, 4, 8, 12, 16, 24], 'lawmean_gt': [0, 3, 4, 8, 12, 16, 24], 'lawmean_best3': [3, 4, 8, 12, 16, 24]}
PROBES = {'prior_top1': 0, 'prior_top2': 2, 'lawmean_best3': 3, 'lawmean_gt': 0}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--root', default='data/v3d_refine_test')
    a.add_argument('--output', default='docs/v3d/C1_TEST_RESULTS.json'); args = a.parse_args()
    plan = json.loads((ROOT / 'docs/v3d/C1_TEST_SAMPLES.json').read_text())['samples']
    S = load_states(args.root, 'v2main_C_seed29', plan)
    def complete_at(arm, k):
        return {i for i, s in S[arm].items() if str(k) in s['budgets'] and (s['_complete'] or len(s['evals']) >= k)}
    common = sorted(set.intersection(*[complete_at(arm, 24) for arm in ARMS3]))
    out = {'split': 'test', 'protocol': 'docs/v3d/C1_TEST_PROTOCOL.json', 'planned_samples': len(plan), 'n_analysed': len(common),
           'partial': len(common) < len(plan), 'analysed_indices': common,
           'states': {arm: {'any': len(S[arm]), 'complete': sum(s['_complete'] for s in S[arm].values())} for arm in ARMS3 + ('prior_top1',)},
           'note': 'budget k counts ALL simulations incl. law probes (prior_top2: 2, lawmean_best3: 3); '
                   'search_only_accounting relabels each curve by k minus its probes. prior_top2 budget 0 = prior_ens '
                   '(9-run ensemble mean prediction, top-1 law, no simulation).'}
    if common:
        objs = [S['prior_top2'][i]['object'] for i in common]
        out['n_objects'] = len(set(objs))
        out['curves'] = {arm: arm_table(S[arm], common, BUDGETS[arm]) for arm in ARMS3}
        out['search_only_accounting'] = {arm: {str(int(k) - PROBES[arm]): v['identifiable_mae_pooled'] for k, v in out['curves'][arm].items()
                                               if int(k) >= PROBES[arm]} for arm in ARMS3}
        for k in (0, 4, 12, 24):
            for other in ('lawmean_best3', 'lawmean_gt'):
                ko = (3 if k == 0 else k) if other == 'lawmean_best3' else k
                d = [np.mean(ident_err(S['prior_top2'][i], k)) - np.mean(ident_err(S[other][i], ko)) for i in common]
                out.setdefault('top2_minus_other_same_budget', {})[f'{other}_budget{k}'] = boot_mean(d, objs)
        for arm, k0 in (('prior_top2', 0), ('lawmean_gt', 0), ('lawmean_best3', 3)):
            for k in (12, 24):
                d = [np.mean(ident_err(S[arm][i], k)) - np.mean(ident_err(S[arm][i], k0)) for i in common]
                out.setdefault('refinement_gain', {})[f'{arm}_budget{k}_minus_budget{k0}'] = boot_mean(d, objs)
        out['speedup'] = {u: speedup(S['prior_top2'], S[u], common, [0, 4, 12, 24], [k for k in BUDGETS[u] if 3 <= k <= 24])
                          for u in ('lawmean_gt', 'lawmean_best3')}
        ok = [i for i in common if S['prior_top2'][i]['budgets']['0']['law'] == S['prior_top2'][i]['gt_law']]
        out['prior_top2_by_top1_law_correct'] = {
            'n_correct': len(ok), 'n_wrong': len(common) - len(ok),
            'correct_mae': {k: float(np.mean([v for i in ok for v in ident_err(S['prior_top2'][i], k)])) if ok else None for k in (0, 4, 12, 24)},
            'wrong_mae': {k: float(np.mean([v for i in common if i not in ok for v in ident_err(S['prior_top2'][i], k)]))
                          if len(ok) < len(common) else None for k in (0, 4, 12, 24)}}
        out['replay_gt_failed_samples'] = sorted(i for i in common if S['prior_top2'][i]['replay'].get('gt_failed'))
    # Addendum arm prior_top1 (ensemble prediction, top-1 law, no probes, same search) on the analysed samples where it is complete.
    c1 = sorted(set(common) & complete_at('prior_top1', 24))
    t1 = {'n': len(c1), 'partial': len(c1) < len(plan), 'note': 'addendum arm, registered before any test simulation'}
    if c1:
        o1 = [S['prior_top1'][i]['object'] for i in c1]
        t1['curves'] = {arm: arm_table(S[arm], c1, BUDGETS[arm]) for arm in ('prior_top1',) + ARMS3}
        for k in (0, 4, 12, 24):
            for other in ('lawmean_best3', 'lawmean_gt', 'prior_top2'):
                ko = (3 if k == 0 else k) if other == 'lawmean_best3' else k
                d = [np.mean(ident_err(S['prior_top1'][i], k)) - np.mean(ident_err(S[other][i], ko)) for i in c1]
                t1.setdefault('top1_minus_other_same_budget', {})[f'{other}_budget{k}'] = boot_mean(d, o1)
        for k in (12, 24):
            d = [np.mean(ident_err(S['prior_top1'][i], k)) - np.mean(ident_err(S['prior_top1'][i], 0)) for i in c1]
            t1.setdefault('refinement_gain', {})[f'prior_top1_budget{k}_minus_budget0'] = boot_mean(d, o1)
        t1['speedup'] = {u: speedup(S['prior_top1'], S[u], c1, [0, 4, 12, 24], [k for k in BUDGETS[u] if 3 <= k <= 24])
                         for u in ('lawmean_gt', 'lawmean_best3')}
        t1['simulation_failures'] = {'failed': sum(bool(e.get('failed')) for i in c1 for e in S['prior_top1'][i]['evals']),
                                     'total': sum(len(S['prior_top1'][i]['evals']) for i in c1)}
    out['prior_top1_addendum'] = t1
    gto = [s['gt_observed'] for s in S['prior_top2'].values() if 'iou' in s.get('gt_observed', {})]
    out['noise_floor_gt_resim_observed'] = {'n': len(gto), 'iou': float(np.mean([g['iou'] for g in gto])) if gto else None,
                                            'rgb_mae_union': float(np.mean([g['rgb_mae_union'] for g in gto])) if gto else None}
    out['simulation_failures'] = {arm: {'failed': sum(bool(e.get('failed')) for i in common for e in S[arm][i]['evals']),
                                        'total': sum(len(S[arm][i]['evals']) for i in common),
                                        'failed_start_or_probe': sum(bool(e.get('failed')) for i in common for e in S[arm][i]['evals'][:max(PROBES[arm], 1)])}
                                  for arm in ARMS3}
    allev = [e for arm in ARMS3 for i in common for e in S[arm][i]['evals']]
    out['simulations'] = {'n': len(allev), 'failed': sum(bool(e.get('failed')) for e in allev),
                          'median_seconds': float(np.median([e['seconds'] for e in allev])) if allev else None}
    # Val reference (docs/v3b/C1_REFINE_RESULTS.json, n=60), copied for side-by-side reading.
    val = json.loads((ROOT / 'docs/v3b/C1_REFINE_RESULTS.json').read_text())['prior_top2']
    out['val_reference'] = {'n': val['n'], 'curves_identifiable_mae': {arm: {k: v['identifiable_mae_pooled'] for k, v in val['curves'][arm].items()}
                                                                     for arm in ARMS3},
                            'top2_minus_other_same_budget': {k: {'mean': v['mean'], 'ci95': v['ci95']} for k, v in val['top2_minus_other_same_budget'].items()
                                                             if not k.startswith('prior_')},
                            'speedup_sims_to_match': {u: {k: v['uninformed_sims_to_match'] for k, v in val['speedup'][u]['by_prior_budget'].items()}
                                                      for u in ('lawmean_gt', 'lawmean_best3')}}
    if common:
        t = out['top2_minus_other_same_budget']
        h1 = all(t[f'lawmean_best3_budget{k}']['ci95'][1] < 0 for k in (0, 4, 12, 24))
        m0 = out['speedup']['lawmean_gt']['by_prior_budget']['0']['uninformed_sims_to_match']
        out['preregistered_replication'] = {
            'H1_top2_beats_nonprivileged_best3_every_budget': h1,
            'H2_privileged_sims_to_match_top2_budget0': m0 if m0 is not None else '>24 (censored)',
            'H2_replicates_(>=6_sims_incl_censored)': (m0 is None) or (m0 >= 6),
            'H3_top2_minus_lawmean_gt_negative_ci_at_budget0_and_4': all(t[f'lawmean_gt_budget{k}']['ci95'][1] < 0 for k in (0, 4))}
    out['samples'] = {arm: [{'index': i, 'object': s['object'], 'gt_law': s['gt_law'], 'law': s.get('law'), 'action': s['observed_action'],
                             'complete': s['_complete'], 'ident_err': {k: ident_err(s, k) for k in s['budgets']},
                             'x': {k: b['x'] for k, b in s['budgets'].items()},
                             'replay_iou': {k: v.get('iou') for k, v in s['replay'].items() if isinstance(v, dict)},
                             'evals': [[e.get('law'), e['x'], round(e['loss'], 4)] for e in s['evals']]}
                            for i, s in sorted(S[arm].items())] for arm in ARMS3 + ('prior_top1',)}
    (ROOT / args.output).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('samples',)}, indent=1)[:12000])


if __name__ == '__main__':
    main()
