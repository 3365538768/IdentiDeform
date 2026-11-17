"""V3c: V2 main TEST report (run once, after all test_once outputs exist) -> docs/v3c/BASELINES_TEST.json.

Same scoring as scripts/v3c/baselines_report.py (imported). Trained runs: <run>/test_once
(checkpoint = best_material.pt, i.e. the val-selected epoch). No re-selection on test.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_object_ci import compare
from baselines_report import score, merge_vlm, agg, PARAMS, SEEDS

TRAINED = {'dinov3_static_first_frame_B': 'runs/v3c_dino_S_seed{}', 'dinov3_16f_B': 'runs/v3c_dino_B_seed{}',
           'dinov3_16f_C': 'runs/v3c_dino_C_seed{}', 'vjepa2_32f_B': 'runs/v2main_B_seed{}',
           'vjepa2_32f_C': 'runs/v2main_C_seed{}', 'vjepa2_32f_D': 'runs/v2main_D_seed{}'}
SINGLE = {'constant_train_mean': 'runs/v3c_const_test', 'law_mean_gt_privileged': 'runs/v3c_lawmean_gt_test',
          'qwen3vl8b_zeroshot': 'runs/v3c_vlm_qwen3vl8b_test'}
REF = 'vjepa2_32f_C'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default='docs/v3c/BASELINES_TEST.json'); a = p.parse_args()
    c = {s: Path(TRAINED[REF].format(s)) / 'test_once' for s in SEEDS}
    missing = [str(d) for d in c.values() if not (d / 'predictions.jsonl').exists()]
    if missing:
        raise SystemExit('missing reference outputs: ' + str(missing))
    rep = {'protocol': 'docs/v3c/TEST_PROTOCOL.json + docs/v3c/TEST_READY.json; test evaluated once; checkpoints selected on val', 'methods': {}}
    for name, d in SINGLE.items():
        if not Path(d).exists():
            rep['methods'][name] = {'status': 'missing'}; continue
        extra = merge_vlm(d) if name.startswith('qwen') else {}
        m = score(d); m.update(extra)
        if name != 'qwen3vl8b_zeroshot':
            m['within_law_r'] = {k: {'mean_over_laws': None} for k in PARAMS}
        if name == 'law_mean_gt_privileged':
            m['law_accuracy'] = None
        m['vs_vjepa_C'] = {str(s): compare(Path(d) / 'predictions.jsonl', c[s] / 'predictions.jsonl') for s in SEEDS}
        rep['methods'][name] = m
    for name, pat in TRAINED.items():
        per = {}
        for s in SEEDS:
            d = Path(pat.format(s)) / 'test_once'
            if not (d / 'predictions.jsonl').exists():
                continue
            m = score(d)
            if name != REF:
                m['vs_vjepa_C_same_seed'] = compare(d / 'predictions.jsonl', c[s] / 'predictions.jsonl')
            per[str(s)] = m
        if not per:
            rep['methods'][name] = {'status': 'missing'}; continue
        v = list(per.values())
        rep['methods'][name] = {'seeds': per, 'standardized_mae': agg([x['standardized_mae'] for x in v]),
                                'law_accuracy': agg([x['law_accuracy'] for x in v]),
                                'per_parameter_mae_z': {k: agg([x['per_parameter_mae_z'][k] for x in v]) for k in PARAMS},
                                'within_law_r': {k: agg([x['within_law_r'].get(k, {}).get('mean_over_laws') for x in v]) for k in PARAMS}}
    g = lambda x: x['mean'] if isinstance(x, dict) else x
    rows = []
    for n, m in rep['methods'].items():
        if 'standardized_mae' in m:
            rows.append({'method': n, 'standardized_mae': g(m['standardized_mae']), 'law_accuracy': g(m['law_accuracy']),
                         'E_over_rho_within_law_r': g(m['within_law_r']['E_over_rho']) if 'seeds' in m else m['within_law_r']['E_over_rho']['mean_over_laws'],
                         'sigma_y_over_rho_within_law_r': g(m['within_law_r']['sigma_y_over_rho']) if 'seeds' in m else m['within_law_r']['sigma_y_over_rho']['mean_over_laws'],
                         'seeds': len(m.get('seeds', {})) or None})
    rep['table'] = sorted(rows, key=lambda r: r['standardized_mae'])
    Path(a.output).write_text(json.dumps(rep, indent=2) + '\n'); print(json.dumps(rep['table'], indent=1))


if __name__ == '__main__':
    main()
