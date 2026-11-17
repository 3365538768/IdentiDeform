"""Summarize R-line runs: best epoch per run, seed-level and object-level paired comparisons.

  python scripts/r1_report.py --prefix r1 --pairs C:D B:C B:D C:Ch D:Dh C:Cl --output docs/r1/R1_RESULTS.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_object_ci import compare
from audit_within_law import audit


def best_dir(run):
    timing = [json.loads(l) for l in (run / 'timing.jsonl').read_text().splitlines() if l.strip()]
    joint = [t for t in timing if t.get('phase') == 'joint_training' and t.get('val_mae') is not None]
    best = min(joint, key=lambda t: t['val_mae'])
    return run / f"val_epoch_{best['epoch'] - 1:04d}", best


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prefix', default='r1'); p.add_argument('--seeds', default='17,29,43')
    p.add_argument('--pairs', nargs='+', default=['C:D']); p.add_argument('--output'); a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    runs = {}
    for path in sorted(Path('runs').glob(f'{a.prefix}_*_seed*')):
        group, seed = path.name[len(a.prefix) + 1:].rsplit('_seed', 1)
        if (path / 'status.json').exists():
            d, best = best_dir(path)
            runs[(group, int(seed))] = {'dir': str(d), 'best_epoch': best['epoch'], 'val_mae': best['val_mae'],
                                        'status': json.loads((path / 'status.json').read_text()), **audit(d)}
    report = {'protocol': f'{a.prefix}: best validation epoch per run; test split untouched',
              'groups': {}, 'pairs': []}
    for group in sorted({g for g, _ in runs}):
        values = [runs[(group, s)] for s in seeds if (group, s) in runs]
        maes = [v['val_mae'] for v in values]
        report['groups'][group] = {
            'seeds': [s for s in seeds if (group, s) in runs], 'val_mae': maes,
            'mean': float(np.mean(maes)), 'sd': float(np.std(maes, ddof=1)) if len(maes) > 1 else None,
            'law_accuracy': [v['law_accuracy'] for v in values],
            'within_law_r': {name: [float(np.mean([w['pearson'] for w in v['parameters'][name]['within_law'].values()
                                                   if w['pearson'] is not None])) for v in values]
                             for name in values[0]['parameters']}}
    for pair in a.pairs:
        control, candidate = pair.split(':')
        rows = []
        for s in seeds:
            if (control, s) in runs and (candidate, s) in runs:
                ci = compare(Path(runs[(control, s)]['dir']) / 'predictions.jsonl', Path(runs[(candidate, s)]['dir']) / 'predictions.jsonl')
                c, d = runs[(control, s)]['val_mae'], runs[(candidate, s)]['val_mae']
                rows.append({'seed': s, 'control': c, 'candidate': d, 'relative_improvement': (c - d) / c, 'object_bootstrap': ci})
        if rows:
            imp = [r['relative_improvement'] for r in rows]
            report['pairs'].append({'control': control, 'candidate': candidate, 'per_seed': rows,
                                    'mean_relative_improvement': float(np.mean(imp)),
                                    'seeds_candidate_better': sum(i > 0 for i in imp)})
    text = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
