"""V3h context: budget-0 (no simulation) identifiable z-MAE of each initialization on the 60 C1 val samples, the
30-sample V3h subset and the full val split (prediction files only). -> docs/v3h/INIT_QUALITY_VAL.json"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402
def load(run):
    d, _ = best_dir(ROOT / 'runs' / run)
    return {r['sample_id']: r for r in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines())}
P = {r: load(r) for r in [f'v3c_dino_C_seed{s}' for s in (17, 29, 43)] + ['v2main_C_seed29']}
ens = json.loads((ROOT / 'docs/v3b/ENSEMBLE_BCD9_VAL.json').read_text())['samples']
ref = P['v2main_C_seed29']
def err(z, sid):
    t, m = ref[sid]['target_z'], ref[sid]['target_mask']
    return [abs(z[j] - t[j]) for j in (0, 3) if m[j]]
def law_ok(sid, laws):
    return float(np.bincount(laws, minlength=3).argmax() == ref[sid]['law_id'])
init = {
    'vjepa_C_seed29': lambda s: (P['v2main_C_seed29'][s]['prediction_z'], ref[s]['predicted_law_id'] == ref[s]['law_id']),
    'vjepa_BCD9_ens': lambda s: (ens[s]['prediction_z'], int(np.argmax(ens[s]['law_posterior_votes'])) == ref[s]['law_id']),
    'dinov3_C_3ens': lambda s: (np.mean([P[f'v3c_dino_C_seed{k}'][s]['prediction_z'] for k in (17, 29, 43)], 0).tolist(),
                                bool(law_ok(s, [P[f'v3c_dino_C_seed{k}'][s]['predicted_law_id'] for k in (17, 29, 43)]))),
    'dinov3_C_seed29': lambda s: (P['v3c_dino_C_seed29'][s]['prediction_z'], P['v3c_dino_C_seed29'][s]['predicted_law_id'] == ref[s]['law_id'])}
sets = {'full_val': sorted(ref),
        'c1_60': [f"v2main/val/{p['index']:06d}" for p in json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']],
        'v3h_30': [f'v2main/val/{i:06d}' for i in json.loads((ROOT / 'docs/v3h/PROTOCOL.json').read_text())['subset']['indices']]}
out = {}
for name, ids in sets.items():
    out[name] = {'n': len(ids)}
    for k, f in init.items():
        e = []; acc = []
        for s in ids:
            z, ok = f(s); e += err(z, s); acc.append(float(ok))
        out[name][k] = {'identifiable_mae': float(np.mean(e)), 'law_accuracy': float(np.mean(acc))}
(ROOT / 'docs/v3h/INIT_QUALITY_VAL.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
