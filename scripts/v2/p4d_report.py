"""P4d: does the low-damping V2 design make identifiable quantities learnable? (validation only)

For each arm and seed the best validation epoch is used (scripts/r1_report.best_dir). Targets are in the
identifiable space: E/rho, nu, rho (expected unidentifiable control), sigma_y/rho. Reports MAE, law accuracy,
within-law Pearson r per law and per action, and object-level paired bootstrap of low vs v1 (same samples).
  .venv/bin/python scripts/v2/p4d_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402
from bootstrap_object_ci import compare  # noqa: E402

PARAMS = ['E_over_rho', 'nu', 'rho', 'sigma_over_rho']
LAWS = {0: 'jelly', 1: 'metal', 2: 'plasticine'}


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 4 and a.std() > 0 and b.std() > 0 else None


def summarize(pred_dir):
    rows = [json.loads(l) for l in (pred_dir / 'predictions.jsonl').read_text().splitlines() if l.strip()]
    pz, tz = np.array([r['prediction_z'] for r in rows]), np.array([r['target_z'] for r in rows])
    m = np.array([r['target_mask'] for r in rows]); law = np.array([r['law_id'] for r in rows])
    act = np.array([r['action'] for r in rows])
    out = {'n': len(rows), 'mae': float(np.abs(pz - tz)[m].mean()),
           'law_accuracy': float(np.mean([r['predicted_law_id'] == r['law_id'] for r in rows])), 'within_law_r': {}}
    for j, name in enumerate(PARAMS):
        for lid, lname in LAWS.items():
            for a in ('all', 'press', 'tap'):
                k = m[:, j] & (law == lid) & ((act == a) if a != 'all' else True)
                r = pearson(pz[k, j], tz[k, j])
                if r is not None:
                    out['within_law_r'][f'{name}/{lname}/{a}'] = r
    return out


def main():
    report = {'protocol': 'P4d: best validation epoch per run; paired V2 pilot arms differ only in damping', 'arms': {}, 'pairs': []}
    for arm in ('low', 'v1'):
        runs = []
        for seed in (17, 29, 43):
            run = ROOT / f'runs/p4d_C{arm}_seed{seed}'
            if not (run / 'status.json').exists():
                continue
            d, best = best_dir(run)
            runs.append({'seed': seed, 'dir': str(d), 'epoch': best['epoch'], 'val_mae': best['val_mae'], **summarize(d)})
        if not runs:
            continue
        keys = sorted({k for r in runs for k in r['within_law_r']})
        report['arms'][arm] = {'runs': runs, 'mae': [float(np.mean([r['mae'] for r in runs])), float(np.std([r['mae'] for r in runs]))],
                               'law_accuracy': float(np.mean([r['law_accuracy'] for r in runs])),
                               'within_law_r': {k: float(np.mean([r['within_law_r'][k] for r in runs if k in r['within_law_r']])) for k in keys}}
    if {'low', 'v1'} <= set(report['arms']):
        for a, b in zip(report['arms']['v1']['runs'], report['arms']['low']['runs']):
            ci = compare(Path(a['dir']) / 'predictions.jsonl', Path(b['dir']) / 'predictions.jsonl')
            report['pairs'].append({'seed': a['seed'], 'relative_improvement_low_vs_v1': (a['mae'] - b['mae']) / a['mae'],
                                    'object_bootstrap_95_ci': ci['paired_bootstrap_95_ci']})
    (ROOT / 'docs/p4d/P4D_RESULTS.json').write_text(json.dumps(report, indent=2))
    for arm, v in report['arms'].items():
        print(f"{arm}: n_seeds={len(v['runs'])} MAE {v['mae'][0]:.3f}±{v['mae'][1]:.3f} law_acc {v['law_accuracy']:.3f}")
        for k, r in v['within_law_r'].items():
            if k.endswith('/all') or k.startswith(('E_over_rho', 'sigma_over_rho')):
                print(f'   {k:36s} {r:+.2f}')
    for p in report['pairs']:
        print('pair', p)


if __name__ == '__main__':
    main()
