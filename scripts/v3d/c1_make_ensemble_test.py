"""V3d: 9-run B/C/D ensemble TEST predictions in the format of docs/v3b/ENSEMBLE_BCD9_VAL.json
(scripts/v3b/c1_make_ensemble.py rule: mean prediction_z, law_posterior_votes = vote fractions), from the saved
single-run test predictions runs/v2main_{B,C,D}_seed{17,29,43}/test_once (evaluated once in v3c), and checked
against the saved ensemble runs/v3c_ens_bcd9_test/predictions.jsonl. Writes docs/v3d/ENSEMBLE_BCD9_TEST.json."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
runs = [f'v2main_{g}_seed{s}' for g in 'BCD' for s in (17, 29, 43)]
rows = {}
for r in runs:
    for x in map(json.loads, (ROOT / 'runs' / r / 'test_once/predictions.jsonl').read_text().splitlines()):
        e = rows.setdefault(x['sample_id'], {'z': [], 'votes': [0, 0, 0]})
        e['z'].append(x['prediction_z']); e['votes'][x['predicted_law_id']] += 1
assert len(rows) == 300 and all(len(e['z']) == 9 for e in rows.values())
out = {sid: {'prediction_z': np.mean(e['z'], 0).tolist(), 'z_std_across_runs': np.std(e['z'], 0).tolist(),
             'law_posterior_votes': (np.array(e['votes']) / len(runs)).tolist()} for sid, e in rows.items()}
saved = {x['sample_id']: x for x in map(json.loads, (ROOT / 'runs/v3c_ens_bcd9_test/predictions.jsonl').read_text().splitlines())}
assert set(saved) == set(out)
for sid, x in saved.items():
    assert np.allclose(x['prediction_z'], out[sid]['prediction_z'], atol=1e-9), sid
    assert x['law_votes'] == rows[sid]['votes'], sid
(ROOT / 'docs/v3d/ENSEMBLE_BCD9_TEST.json').write_text(json.dumps({'runs': runs, 'split': 'test', 'source': 'runs/<run>/test_once; '
    'identical to runs/v3c_ens_bcd9_test (checked)', 'samples': out}))
print(len(out), 'matches runs/v3c_ens_bcd9_test')
