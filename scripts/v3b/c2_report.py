"""V3b C2 report: soft law-mixture head vs v2main C, plus seed / B-C-D ensembles (validation only).

Each run is scored at its best validation epoch (r1_report.best_dir). The paired object bootstrap is the
same as scripts/bootstrap_object_ci.py (object-macro standardized MAE, 2000 draws, seed 17), done in memory
so ensembles and the MAP variant can be compared too. Gate (preregistered in the task):
  mean relative MAE improvement >= 2 % AND >= 2/3 seeds with the object CI entirely below 0.

  .venv/bin/python scripts/v3b/c2_report.py --output docs/v3b/C2_MIXTURE_RESULTS.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402

SEEDS = (17, 29, 43)
LAW_NAMES = {0: 'jelly', 1: 'metal', 2: 'plasticine'}


def load(run):
    path = ROOT / 'runs' / run
    if not (path / 'status.json').exists():
        return None
    d, best = best_dir(path)
    rows = {r['sample_id']: r for r in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines()) if r}
    return {'run': run, 'dir': str(d.relative_to(ROOT)), 'best_epoch': best['epoch'], 'val_mae_logged': best['val_mae'],
            'status': json.loads((path / 'status.json').read_text()).get('status'), 'rows': rows}


def errors(rows, key='prediction_z'):
    out = {}
    for sid, r in rows.items():
        m = np.array(r['target_mask'], bool)
        out[sid] = np.abs(np.array(r[key]) - np.array(r['target_z']))[m]
    return out


def mae(err):
    return float(np.concatenate(list(err.values())).mean())


def summary(rows, err):
    laws = sorted({r['law_id'] for r in rows.values()})
    acts = sorted({r['action'] for r in rows.values()})
    per_param = np.zeros(4); count = np.zeros(4)
    for sid, r in rows.items():
        m = np.array(r['target_mask'], bool)
        e = np.abs(np.array(r.get('_pz', r['prediction_z'])) - np.array(r['target_z']))
        per_param[m] += e[m]; count[m] += 1
    return {'standardized_mae': mae(err),
            'law_accuracy': float(np.mean([r['predicted_law_id'] == r['law_id'] for r in rows.values()])),
            'by_law': {LAW_NAMES[k]: mae({s: e for s, e in err.items() if rows[s]['law_id'] == k}) for k in laws},
            'by_action': {a: mae({s: e for s, e in err.items() if rows[s]['action'] == a}) for a in acts},
            'by_parameter': dict(zip(['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho'], (per_param / np.maximum(count, 1)).tolist()))}


def paired(rows, err_a, err_b, draws=2000, seed=17):
    groups = {}
    for sid in err_a:
        if len(err_a[sid]):
            groups.setdefault((rows[sid]['source'], rows[sid]['object_id']), []).append(err_b[sid].mean() - err_a[sid].mean())
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
    a, b = mae(err_a), mae(err_b)
    return {'control_mae': a, 'candidate_mae': b, 'relative_improvement': (a - b) / a, 'objects': len(values),
            'mean_object_delta_candidate_minus_control': float(values.mean()),
            'paired_bootstrap_95_ci': np.quantile(resampled, [.025, .975]).tolist(),
            'ci_below_zero': bool(np.quantile(resampled, .975) < 0)}


def with_pred(rows, z, law=None):
    out = {}
    for sid, r in rows.items():
        out[sid] = dict(r, prediction_z=list(map(float, z[sid])), _pz=list(map(float, z[sid])))
        if law is not None:
            out[sid]['predicted_law_id'] = int(law[sid])
    return out


def ensemble(runs):
    sids = sorted(runs[0]['rows'])
    z = {s: np.mean([r['rows'][s]['prediction_z'] for r in runs], 0) for s in sids}
    law = {s: np.bincount([r['rows'][s]['predicted_law_id'] for r in runs], minlength=3).argmax() for s in sids}
    return with_pred(runs[0]['rows'], z, law)


def gate(pairs):
    imp = [p['relative_improvement'] for p in pairs]
    below = sum(p['ci_below_zero'] for p in pairs)
    return {'mean_relative_improvement': float(np.mean(imp)), 'seeds_ci_below_zero': int(below), 'seeds': len(pairs),
            'passed': bool(len(pairs) == 3 and np.mean(imp) >= .02 and below >= 2)}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--output'); a.add_argument('--candidate', default='v3b_C2mix')
    args = a.parse_args()
    runs = {(g, s): load(f'{g}_seed{s}') for g in ('v2main_B', 'v2main_C', 'v2main_D', args.candidate) for s in SEEDS}
    runs = {k: v for k, v in runs.items() if v is not None}
    report = {'protocol': 'Validation split only (best validation epoch per run, so val numbers are mildly optimistic for '
                          'every arm alike); V2 test split untouched. Standardized MAE = mean |z error| over observed '
                          'specific targets (E/rho, nu, rho, sigma_y/rho; ln1p, train-standardized), as evaluation.py.',
              'gate_rule': 'mean relative MAE improvement >= 2% and >= 2/3 seeds whose paired object-bootstrap 95% CI '
                           '(candidate - control) lies entirely below 0',
              'runs': {}, 'pairs': {}, 'ensembles': {}}
    for (g, s), r in sorted(runs.items()):
        err = errors(r['rows'])
        entry = {'dir': r['dir'], 'best_epoch': r['best_epoch'], 'status': r['status'], **summary(r['rows'], err)}
        if g == args.candidate:
            mp, oracle = {}, {}
            for sid, row in r['rows'].items():
                means = np.array(row['mixture_means']); k = int(np.argmax(row['mixture_logits']))
                mp[sid] = means[k]; oracle[sid] = means[row['law_id']] if row['law_id'] >= 0 else means[k]
            r['map_rows'] = with_pred(r['rows'], mp)
            entry['map_variant'] = summary(r['map_rows'], errors(r['map_rows']))
            orows = with_pred(r['rows'], oracle)
            entry['oracle_law_component_diagnostic'] = summary(orows, errors(orows))['standardized_mae']
        report['runs'][f'{g}_seed{s}'] = entry
    def pair_set(control, cand_key, cand_rows_fn):
        rows = []
        for s in SEEDS:
            if (control, s) in runs and (args.candidate, s) in runs:
                c = runs[(control, s)]['rows']; d = cand_rows_fn(runs[(args.candidate, s)])
                rows.append({'seed': s, **paired(c, errors(c), errors(d))})
        return {'per_seed': rows, 'gate': gate(rows)} if rows else None
    report['pairs']['v2main_C_vs_C2mix_posterior_mean'] = pair_set('v2main_C', 'mix', lambda r: r['rows'])
    report['pairs']['v2main_C_vs_C2mix_MAP'] = pair_set('v2main_C', 'map', lambda r: r['map_rows'])
    report['pairs']['v2main_B_vs_C2mix_posterior_mean'] = pair_set('v2main_B', 'mix', lambda r: r['rows'])
    # Ensembles (equal-weight mean of z predictions, majority-vote law).
    groups = {'C_seed_ensemble': [('v2main_C', s) for s in SEEDS], 'B_seed_ensemble': [('v2main_B', s) for s in SEEDS],
              'D_seed_ensemble': [('v2main_D', s) for s in SEEDS],
              'BCD_9run_ensemble': [(g, s) for g in ('v2main_B', 'v2main_C', 'v2main_D') for s in SEEDS],
              'C2mix_seed_ensemble': [(args.candidate, s) for s in SEEDS]}
    for name, keys in groups.items():
        if not all(k in runs for k in keys):
            report['ensembles'][name] = {'missing': [f'{g}_seed{s}' for g, s in keys if (g, s) not in runs]}; continue
        rows = ensemble([runs[k] for k in keys]); err = errors(rows)
        singles = [mae(errors(runs[k]['rows'])) for k in keys]
        entry = {'members': [f'{g}_seed{s}' for g, s in keys], **summary(rows, err), 'member_mae_mean': float(np.mean(singles)),
                 'member_mae': singles,
                 'vs_each_member': [{'member': f'{g}_seed{s}', **paired(runs[(g, s)]['rows'], errors(runs[(g, s)]['rows']), err)}
                                    for g, s in keys]}
        report['ensembles'][name] = entry
        report['ensembles'][name]['_rows'] = rows
    if 'C_seed_ensemble' in report['ensembles'] and 'C2mix_seed_ensemble' in report['ensembles'] and '_rows' in report['ensembles']['C2mix_seed_ensemble']:
        c = report['ensembles']['C_seed_ensemble']['_rows']; d = report['ensembles']['C2mix_seed_ensemble']['_rows']
        report['pairs']['C_ensemble_vs_C2mix_ensemble'] = paired(c, errors(c), errors(d))
    for e in report['ensembles'].values():
        e.pop('_rows', None)
    text = json.dumps(report, indent=1)
    if args.output:
        Path(ROOT / args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
