"""P2: one-shot test-set table following docs/p2/P2_PROTOCOL.json.

Standardized MAE = mean |z_pred - z_gt| over all valid (sample, parameter) entries, identical to
fieldbridge.evaluation. Constant and law-mean references are computed on the same test records.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_within_law import audit  # noqa: E402
from bootstrap_object_ci import compare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ['E', 'nu', 'rho', 'sigma_y']


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def mae(records, pred=None):
    errors = []
    for r in records:
        p = pred(r) if pred else r['prediction_z']
        errors += [abs(a - b) for a, b, m in zip(p, r['target_z'], r['target_mask']) if m]
    return float(np.mean(errors))


def summarize(path):
    a = audit(Path(path).parent)
    return {'mae': mae(rows(path)), 'law_accuracy': a['law_accuracy'],
            'within_law_r': {k: float(np.nanmean([w['pearson'] if w['pearson'] is not None else np.nan
                                                  for w in v['within_law'].values()])) for k, v in a['parameters'].items()},
            'per_parameter_mae': {k: v['model_mae'] for k, v in a['parameters'].items()}}


def law_means():
    """Train-set mean z per law (native statistics), for the privileged law-mean reference."""
    from fieldbridge.data import ParameterCodec
    codec = ParameterCodec(json.loads((ROOT / 'data/manifests/deformfield_v1.stats.json').read_text()))
    sums = {}
    for r in rows(ROOT / 'data/manifests/deformfield_v1.jsonl'):
        if r['split'] != 'train':
            continue
        z, m = codec.encode(r['parameters'])
        s = sums.setdefault(r['law_id'], [np.zeros(4), np.zeros(4)])
        s[0] += np.where(m.numpy(), z.numpy(), 0); s[1] += m.numpy()
    return {k: (v[0] / np.maximum(v[1], 1)).tolist() for k, v in sums.items()}


def main():
    protocol = json.loads((ROOT / 'docs/p2/P2_PROTOCOL.json').read_text())
    report = {'protocol': 'docs/p2/P2_PROTOCOL.json', 'split': 'test', 'native': {}, 'identifiable': {}, 'pairs': []}
    for space, table in [('native', protocol['runs']['native_space_main_table']),
                         ('identifiable', protocol['runs']['identifiable_space_supplement'])]:
        for group, runs in table.items():
            if not isinstance(runs, list):
                continue
            per_seed = [summarize(ROOT / f'runs/{r}/eval_test/predictions.jsonl') for r in runs]
            report[space][group] = {'runs': runs, 'per_seed': per_seed,
                                    'mae_mean': float(np.mean([s['mae'] for s in per_seed])),
                                    'mae_sd': float(np.std([s['mae'] for s in per_seed], ddof=1)),
                                    'law_accuracy_mean': float(np.mean([s['law_accuracy'] for s in per_seed]))}
    reference = rows(ROOT / 'runs/r1_C_seed17/eval_test/predictions.jsonl')
    means = law_means()
    report['native']['constant'] = {'mae_mean': mae(reference, lambda r: [0., 0., 0., 0.])}
    report['native']['law_mean_reference'] = {'mae_mean': mae(reference, lambda r: means[r['law_id']]),
                                              'note': 'uses GT law; privileged, not deployable'}
    vlm = ROOT / 'runs/p2_vlm_qwen3vl8b_test/predictions.jsonl'
    if vlm.exists():
        report['native']['VLM_Qwen3VL8B_zero_shot'] = summarize(vlm)
    for space, control, candidate in [('native', 'DINOv3_C', 'VJEPA2L_C'), ('native', 'VJEPA2L_C', 'VJEPA2L_D'),
                                      ('native', 'DINOv3_C', 'DINOv3_D'), ('native', 'VJEPA2L_B', 'VJEPA2L_C'),
                                      ('native', 'DINOv3_B', 'DINOv3_C'), ('native', 'Static_first_frame', 'VJEPA2L_B'),
                                      ('identifiable', 'DINOv3_C', 'VJEPA2L_C')]:
        a, b = report[space][control], report[space][candidate]
        seeds = []
        for ra, rb, sa, sb in zip(a['runs'], b['runs'], a['per_seed'], b['per_seed']):
            ci = compare(ROOT / f'runs/{ra}/eval_test/predictions.jsonl', ROOT / f'runs/{rb}/eval_test/predictions.jsonl')
            seeds.append({'control_run': ra, 'candidate_run': rb, 'relative_improvement': (sa['mae'] - sb['mae']) / sa['mae'],
                          'object_bootstrap_95_ci': ci['paired_bootstrap_95_ci']})
        report['pairs'].append({'space': space, 'control': control, 'candidate': candidate, 'per_seed': seeds,
                                'mean_relative_improvement': float(np.mean([s['relative_improvement'] for s in seeds]))})
    out = ROOT / 'docs/p2/P2_TEST_RESULTS.json'
    out.write_text(json.dumps(report, indent=2) + '\n'); print(out)


if __name__ == '__main__':
    main()
