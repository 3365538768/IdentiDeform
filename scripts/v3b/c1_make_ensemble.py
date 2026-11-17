"""9-run B/C/D ensemble on v2main val (best val epoch each): mean z prediction + law vote fractions.
Predictions.jsonl stores only the argmax law, so the ensemble law posterior is the vote fraction over the 9 runs.
Writes docs/v3b/ENSEMBLE_BCD9_VAL.json (validation only)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402
runs = [f'v2main_{g}_seed{s}' for g in 'BCD' for s in (17, 29, 43)]
rows = {}
for r in runs:
    d, _ = best_dir(ROOT / 'runs' / r)
    for x in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines()):
        e = rows.setdefault(x['sample_id'], {'z': [], 'votes': [0, 0, 0], 'logvar': []})
        e['z'].append(x['prediction_z']); e['logvar'].append(x['logvar']); e['votes'][x['predicted_law_id']] += 1
out = {sid: {'prediction_z': np.mean(e['z'], 0).tolist(), 'z_std_across_runs': np.std(e['z'], 0).tolist(),
             'law_posterior_votes': (np.array(e['votes']) / len(runs)).tolist()} for sid, e in rows.items()}
(ROOT / 'docs/v3b/ENSEMBLE_BCD9_VAL.json').write_text(json.dumps({'runs': runs, 'samples': out}))
print(len(out))
