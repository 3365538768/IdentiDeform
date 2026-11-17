"""V3i ablation report: data scaling (B/C x train-object fraction) and view count (C x 1/2/3 cameras).

Best validation epoch per run (same rule as r1_report.py); 100% / 3-view rows are the existing v2main runs.
Within-law r is the mean over laws of Pearson r inside each GT law on val; with target_space=specific,
'E' is E/rho and 'sigma_y' is sigma_y/rho. Object bootstrap (10 val objects) for the key pairs. Test untouched.

  python scripts/v3i/report.py --output docs/v3i/ABLATIONS.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_report import best_dir
from audit_within_law import audit
from bootstrap_object_ci import compare

SEEDS = (17, 29, 43)

def run_dir(group, seed):
    if group in ('B_f100', 'C_f100', 'C_v3'):
        return Path(f"runs/v2main_{group[0]}_seed{seed}")
    return Path(f"runs/v3i_{group}_seed{seed}")

def summarize(group):
    rows = []
    for s in SEEDS:
        path = run_dir(group, s)
        status = path / 'status.json'
        if not status.exists():
            continue
        st = json.loads(status.read_text())
        if st.get('status') not in {'training_finished', 'early_stopped'}:
            continue
        d, best = best_dir(path)
        a = audit(d)
        wl = {n: float(np.mean([w['pearson'] for w in a['parameters'][n]['within_law'].values() if w['pearson'] is not None]))
              for n in ('E', 'sigma_y')}
        rows.append({'seed': s, 'run': str(path), 'best_epoch': best['epoch'], 'pred_dir': str(d), 'val_mae': best['val_mae'],
                     'law_accuracy': a['law_accuracy'], 'within_law_r_E_over_rho': wl['E'], 'within_law_r_sigma_y_over_rho': wl['sigma_y'],
                     'train_samples': None})
    out = {'seeds': [r['seed'] for r in rows], 'runs': rows}
    for k in ('val_mae', 'law_accuracy', 'within_law_r_E_over_rho', 'within_law_r_sigma_y_over_rho'):
        v = [r[k] for r in rows]
        out[k] = {'mean': float(np.mean(v)) if v else None, 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'per_seed': v}
    return out

def pair(groups, control, candidate):
    rows = []
    c_runs = {r['seed']: r for r in groups[control]['runs']}
    for r in groups[candidate]['runs']:
        if r['seed'] in c_runs:
            ci = compare(Path(c_runs[r['seed']]['pred_dir']) / 'predictions.jsonl', Path(r['pred_dir']) / 'predictions.jsonl')
            rows.append({'seed': r['seed'], 'control': c_runs[r['seed']]['val_mae'], 'candidate': r['val_mae'],
                         'relative_change': (r['val_mae'] - c_runs[r['seed']]['val_mae']) / c_runs[r['seed']]['val_mae'],
                         'object_bootstrap': ci})
    return {'control': control, 'candidate': candidate, 'per_seed': rows,
            'mean_relative_change_candidate_vs_control': float(np.mean([x['relative_change'] for x in rows])) if rows else None,
            'seeds_candidate_worse': sum(x['relative_change'] > 0 for x in rows),
            'seeds_ci_excludes_zero_candidate_worse': sum(x['object_bootstrap']['paired_bootstrap_95_ci'][0] > 0 for x in rows)}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output'); a = p.parse_args()
    manifest = [json.loads(l) for l in open('data/manifests/v2main.jsonl')]
    train_objects = len({r['object_id'] for r in manifest if r['split'] == 'train'})
    scaling = {f'{m}_f{f}': summarize(f'{m}_f{f}') for m in 'BC' for f in ('10', '25', '50', '100')}
    views = {f'C_v{v}': summarize(f'C_v{v}') for v in (1, 2, 3)}
    groups = {**scaling, **views}
    pairs = [pair(groups, f'C_f{f}', f'B_f{f}') for f in ('10', '25', '50', '100')]
    pairs += [pair(groups, 'C_f100', f'C_f{f}') for f in ('10', '25', '50')]
    pairs += [pair(groups, 'C_v3', 'C_v1'), pair(groups, 'C_v3', 'C_v2'), pair(groups, 'C_v2', 'C_v1')]
    report = {
        'protocol': ('V3i: best validation epoch per run (val MAE over standardized specific targets), val split = 283 samples / '
                     '10 objects, test split untouched. 100%/3-view rows reuse runs/v2main_{B,C}_seed*. '
                     'Data scaling: data.train_object_fraction with subset_seed = seed (object-level subsets, F1 mechanism); '
                     f'{train_objects} train objects -> 9/22/44/87 objects. View count: model C trained AND validated on a fixed camera subset '
                     '([0] or [0,1]) by masking the other cameras (valid=False, field_mask=0) on the per-view cached V-JEPA2 tokens; '
                     'view_dropout=0. Within-law r: mean over laws of within-law Pearson r; E/rho and sigma_y/rho are the specific-space targets.'),
        'scaling': scaling, 'views': views, 'pairs': [x for x in pairs if x['per_seed']]}
    text = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).write_text(text + '\n')
    def fmt(x):
        return '  n/a ' if x['mean'] is None else f"{x['mean']:.3f}±{(x['sd'] or 0):.3f}"
    for name, g in groups.items():
        print(f"{name:7s} n={len(g['seeds'])} mae {fmt(g['val_mae'])} law {fmt(g['law_accuracy'])} rE {fmt(g['within_law_r_E_over_rho'])} rS {fmt(g['within_law_r_sigma_y_over_rho'])}")
    for x in report['pairs']:
        print(x['control'], '->', x['candidate'], round(x['mean_relative_change_candidate_vs_control'], 4), 'worse seeds', x['seeds_candidate_worse'], 'CI>0', x['seeds_ci_excludes_zero_candidate_worse'])

if __name__ == '__main__':
    main()
