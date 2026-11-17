"""V3c: equal-weight V-JEPA ensembles from saved predictions (no fitting, no selection on test).

Rule = scripts/v3b/c1_make_ensemble.py: mean prediction_z over runs; law = majority vote of the runs'
argmax laws (ties -> lowest law id, np.argmax). Members: BCD9 = v2main_{B,C,D}_seed{17,29,43};
C3 = v2main_C_seed{17,29,43}. val uses each run's best val epoch dir (= best_material.pt);
test uses <run>/test_once. Writes runs/v3c_ens_{bcd9,c3}_{val,test}/predictions.jsonl.
  .venv/bin/python scripts/v3c/ensembles.py --split val|test
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_report import best_dir

SETS = {'bcd9': [f'v2main_{g}_seed{s}' for g in 'BCD' for s in (17, 29, 43)], 'c3': [f'v2main_C_seed{s}' for s in (17, 29, 43)]}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--split', choices=['val', 'test'], required=True); a = p.parse_args()
    for name, runs in SETS.items():
        rows = {}
        for r in runs:
            d = best_dir(Path('runs') / r)[0] if a.split == 'val' else Path('runs') / r / 'test_once'
            for x in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines()):
                e = rows.setdefault(x['sample_id'], {'base': x, 'z': [], 'votes': [0, 0, 0]})
                if e['base']['target_z'] != x['target_z'] or e['base']['target_mask'] != x['target_mask']:
                    raise ValueError('target mismatch ' + x['sample_id'])
                e['z'].append(x['prediction_z']); e['votes'][x['predicted_law_id']] += 1
        if any(len(e['z']) != len(runs) for e in rows.values()):
            raise ValueError('incomplete member coverage')
        out = Path(f'runs/v3c_ens_{name}_{a.split}'); out.mkdir(parents=True, exist_ok=True)
        with (out / 'predictions.jsonl').open('w') as f:
            for sid in sorted(rows):
                e = rows[sid]; b = e['base']
                f.write(json.dumps({k: b[k] for k in ('sample_id', 'object_id', 'source', 'action', 'law_id', 'target_z', 'target_mask')}
                                   | {'prediction_z': np.mean(e['z'], 0).tolist(), 'logvar': [0.] * 4,
                                      'predicted_law_id': int(np.argmax(e['votes'])), 'law_votes': e['votes']}) + '\n')
        (out / 'members.json').write_text(json.dumps({'runs': runs, 'rule': 'mean z; majority-vote law (ties -> lowest id)'}))
        print(name, a.split, len(rows))


if __name__ == '__main__':
    main()
