"""V3c: V2 main validation baselines vs V-JEPA 2 heads (docs/v3c/BASELINES_VAL.json).

Every method is scored from an evaluation-format predictions.jsonl on the same 283 val samples
(10 val objects) in the same standardized specific space (train statistics). Trained runs use the
best validation epoch (same rule as scripts/r1_report.py for V2 main; optimistic for every trained
method alike). Object bootstrap: scripts/bootstrap_object_ci.compare(method, V-JEPA C), seed-matched
for trained methods, against each C seed for deterministic ones; negative delta favors V-JEPA C.
  .venv/bin/python scripts/v3c/baselines_report.py --output docs/v3c/BASELINES_VAL.json
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from bootstrap_object_ci import compare
from audit_within_law import audit
from r1_report import best_dir

PARAMS = ['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho']
SEEDS = [17, 29, 43]
TRAINED = {'dinov3_static_first_frame_B': 'runs/v3c_dino_S_seed{}', 'dinov3_16f_B': 'runs/v3c_dino_B_seed{}',
           'dinov3_16f_C': 'runs/v3c_dino_C_seed{}', 'vjepa2_32f_B (reference)': 'runs/v2main_B_seed{}',
           'vjepa2_32f_C (reference)': 'runs/v2main_C_seed{}', 'vjepa2_32f_D (reference)': 'runs/v2main_D_seed{}'}
SINGLE = {'constant_train_mean': 'runs/v3c_const', 'law_mean_gt_privileged': 'runs/v3c_lawmean_gt',
          'qwen3vl8b_zeroshot': 'runs/v3c_vlm_qwen3vl8b_val'}
REFERENCE = 'vjepa2_32f_C (reference)'


def merge_vlm(directory):
    d = Path(directory); shards = sorted(d.glob('predictions.shard*of*.jsonl'))
    rows = {}
    for s in shards:
        for line in s.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows[r['sample_id']] = r
    rows = [rows[k] for k in sorted(rows)]
    (d / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    fallback = np.array([r.get('z_fallback', [False] * 4) for r in rows]); mask = np.array([r['target_mask'] for r in rows])
    return {'samples': len(rows), 'unparseable_answers': int(sum(not r['parsed'] for r in rows)),
            'field_fallback_rate_on_scored_entries': {p: float(fallback[mask[:, j], j].mean()) for j, p in enumerate(PARAMS)}}


def score(pred_dir):
    rows = [json.loads(l) for l in open(Path(pred_dir) / 'predictions.jsonl') if l.strip()]
    z = np.array([r['prediction_z'] for r in rows]); t = np.array([r['target_z'] for r in rows]); m = np.array([r['target_mask'] for r in rows], bool)
    err = np.abs(z - t)
    obj = {}
    for r, e, mm in zip(rows, err, m):
        obj.setdefault(r['object_id'], []).append(e[mm].mean())
    a = audit(pred_dir)
    within = {}
    for j, p in enumerate(PARAMS):
        name = ['E', 'nu', 'rho', 'sigma_y'][j]
        if name in a['parameters']:
            per = a['parameters'][name]['within_law']
            vals = [w['pearson'] for w in per.values() if w['pearson'] is not None]
            within[p] = {'mean_over_laws': float(np.mean(vals)) if vals else None,
                         'per_law': {k: w['pearson'] for k, w in per.items()},
                         'global_pearson': a['parameters'][name]['pearson']}
    return {'samples': len(rows), 'standardized_mae': float(err[m].mean()),
            'object_macro_mae': float(np.mean([np.mean(v) for v in obj.values()])),
            'per_parameter_mae_z': {p: float(err[m[:, j], j].mean()) for j, p in enumerate(PARAMS)},
            'law_accuracy': a['law_accuracy'], 'within_law_r': within}


def agg(values):
    v = [x for x in values if x is not None]
    return {'mean': float(np.mean(v)) if v else None, 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'values': values}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    report = {'protocol': {'split': 'V2 main val (283 samples, 10 objects); test untouched',
                           'metric': 'standardized MAE over all scored (sample, parameter) entries in specific space (E/rho, nu, rho, sigma_y/rho; ln1p + train z-score); rho is an expected-unidentifiable control',
                           'identifiable': ['E_over_rho', 'sigma_y_over_rho'],
                           'within_law_r': 'Pearson r inside each GT law, averaged over laws (scripts/audit_within_law.py)',
                           'selection': 'trained runs: epoch with best val standardized MAE (same rule as V2 main report); deterministic baselines: none',
                           'bootstrap': 'paired object bootstrap (2000 draws, seed 17) of object-macro MAE difference method - V-JEPA C; negative favors V-JEPA C',
                           'caveat': 'only 10 val objects: object bootstrap CIs are wide; val is also the selection split for trained runs'},
              'methods': {}}
    c_dirs = {}
    for s in SEEDS:
        run = Path(TRAINED[REFERENCE].format(s))
        if (run / 'timing.jsonl').exists():
            c_dirs[s] = best_dir(run)[0]
    for name, directory in SINGLE.items():
        if not Path(directory).exists():
            report['methods'][name] = {'status': 'missing'}; continue
        extra = merge_vlm(directory) if name.startswith('qwen') else {}
        if name.startswith('qwen') and extra['samples'] != 283:
            report['methods'][name] = {'status': f"incomplete ({extra['samples']}/283)", **extra}; continue
        m = score(directory); m.update(extra)
        m['vs_vjepa_C'] = {str(s): compare(Path(directory) / 'predictions.jsonl', d / 'predictions.jsonl') for s, d in c_dirs.items()}
        if name in ('law_mean_gt_privileged', 'constant_train_mean'):
            m['within_law_r'] = {p_: {'mean_over_laws': None, 'note': 'undefined: prediction constant within each law'} for p_ in PARAMS}
        if name == 'law_mean_gt_privileged':
            m['law_accuracy'] = None; m['note'] = 'uses GT law (privileged); law accuracy not applicable'
        if name == 'constant_train_mean':
            m['note'] = 'law prediction = majority train law'
        report['methods'][name] = m
    for name, pattern in TRAINED.items():
        per = {}
        for s in SEEDS:
            run = Path(pattern.format(s)); status = run / 'status.json'
            if not status.exists() or json.loads(status.read_text()).get('status') not in {'training_finished', 'early_stopped'}:
                continue
            d, best = best_dir(run)
            m = score(d); m['best_epoch'] = best['epoch']; m['dir'] = str(d)
            if name != REFERENCE and s in c_dirs:
                m['vs_vjepa_C_same_seed'] = compare(d / 'predictions.jsonl', c_dirs[s] / 'predictions.jsonl')
            per[str(s)] = m
        if not per:
            report['methods'][name] = {'status': 'missing'}; continue
        vals = list(per.values())
        report['methods'][name] = {
            'seeds': per, 'standardized_mae': agg([v['standardized_mae'] for v in vals]),
            'law_accuracy': agg([v['law_accuracy'] for v in vals]),
            'per_parameter_mae_z': {p_: agg([v['per_parameter_mae_z'][p_] for v in vals]) for p_ in PARAMS},
            'within_law_r': {p_: agg([v['within_law_r'].get(p_, {}).get('mean_over_laws') for v in vals]) for p_ in PARAMS}}
    rows = []
    for name, m in report['methods'].items():
        if 'standardized_mae' not in m:
            continue
        g = lambda x: x['mean'] if isinstance(x, dict) else x
        rows.append({'method': name, 'standardized_mae': g(m['standardized_mae']), 'law_accuracy': g(m['law_accuracy']),
                     'E_over_rho_within_law_r': g(m['within_law_r']['E_over_rho']) if 'seeds' in m else m['within_law_r']['E_over_rho']['mean_over_laws'],
                     'sigma_y_over_rho_within_law_r': g(m['within_law_r']['sigma_y_over_rho']) if 'seeds' in m else m['within_law_r']['sigma_y_over_rho']['mean_over_laws'],
                     'seeds': len(m.get('seeds', {})) or None})
    report['table'] = sorted(rows, key=lambda r: r['standardized_mae'])
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['table'], indent=1))


if __name__ == '__main__':
    main()
