"""V3c: add the equal-weight V-JEPA ensembles (BCD9, C3) to docs/v3c/BASELINES_TEST.json, with their val scores.

Run after scripts/v3c/test_report.py and `ensembles.py --split val/test`. Same scoring/bootstrap as the
other methods (vs each V-JEPA C seed's test_once / best-val predictions). Rule fixed a priori (v3b).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_object_ci import compare
from baselines_report import score
from r1_report import best_dir

rep = json.loads(Path('docs/v3c/BASELINES_TEST.json').read_text())
ref = {'test': {s: Path(f'runs/v2main_C_seed{s}/test_once') for s in (17, 29, 43)},
       'val': {s: best_dir(Path(f'runs/v2main_C_seed{s}'))[0] for s in (17, 29, 43)}}
for name in ('bcd9', 'c3'):
    key = f'vjepa2_ensemble_{name.upper()}'
    m = {}
    for split in ('test', 'val'):
        d = Path(f'runs/v3c_ens_{name}_{split}')
        x = score(d); x['vs_vjepa_C'] = {str(s): compare(d / 'predictions.jsonl', r / 'predictions.jsonl') for s, r in ref[split].items()}
        m[split] = x
    rep['methods'][key] = dict(m['test'], val=m['val'], members=json.loads((Path(f'runs/v3c_ens_{name}_test') / 'members.json').read_text()))
    rep['table'] = [r for r in rep['table'] if r['method'] != key] + [{'method': key, 'standardized_mae': m['test']['standardized_mae'],
        'law_accuracy': m['test']['law_accuracy'], 'E_over_rho_within_law_r': m['test']['within_law_r']['E_over_rho']['mean_over_laws'],
        'sigma_y_over_rho_within_law_r': m['test']['within_law_r']['sigma_y_over_rho']['mean_over_laws'], 'seeds': None}]
rep['table'].sort(key=lambda r: r['standardized_mae'])
Path('docs/v3c/BASELINES_TEST.json').write_text(json.dumps(rep, indent=2) + '\n')
print(json.dumps(rep['table'], indent=1))
