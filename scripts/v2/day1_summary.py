"""Collect the day-1 automatic results into docs/day1/DAY1_SUMMARY.json (no selection, no test split)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(p):
    try:
        return json.loads((ROOT / p).read_text())
    except (OSError, ValueError):
        return None


def groups(rep):
    return None if rep is None else {g: {'mean_val_mae': v['mean'], 'sd': v['sd'], 'seeds': v['seeds']} for g, v in rep['groups'].items()}


def pairs(rep):
    return None if rep is None else [{'control': p['control'], 'candidate': p['candidate'], 'mean_relative_improvement': p['mean_relative_improvement'],
                                      'seeds_candidate_better': p['seeds_candidate_better'],
                                      'ci': [r['object_bootstrap']['paired_bootstrap_95_ci'] for r in p['per_seed']]} for p in rep['pairs']]


f1, p4d, v2 = read('docs/f1/F1_RESULTS.json'), read('docs/f1/P4D_BC_RESULTS.json'), read('docs/v2main/V2MAIN_RESULTS.json')
replay = read('docs/replay/REPLAY_p4d_Clow_seed17.json')
pac = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'data/external_datasets/pacnerf').rglob('*') if p.is_dir())[:40] if (ROOT / 'data/external_datasets/pacnerf').exists() else None
summary = {'field_learning_curve_v1': {'groups': groups(f1), 'pairs': pairs(f1)},
           'field_v2_pilot_B_vs_C': {'groups': groups(p4d), 'pairs': pairs(p4d)},
           'replay_unseen_action': None if replay is None else {k: v for k, v in replay.items() if k != 'rows'},
           'v2_main': {'groups': groups(v2), 'pairs': pairs(v2)},
           'pacnerf_download_dirs': pac}
(ROOT / 'docs/day1/DAY1_SUMMARY.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1)[:4000])
