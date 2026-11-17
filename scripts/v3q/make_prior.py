"""V3q: V3 main prior ensemble per split -> docs/v3q/PRIOR_ENSEMBLE_{val,test}.json, from docs/v3p/HANDOFF.json.

Rule = scripts/v3b/c1_make_ensemble.py / scripts/v3d/c1_make_ensemble_test.py (V2 ENSEMBLE_BCD9): over the hand-off's
prediction runs, mean prediction_z and law_posterior_votes = fraction of runs predicting each law. The mean z is decoded
with the prior's specific stats (expm1(z*std+mean)) to log10 E/rho and log10 sigma_y/rho (the search space), and nu.
target_z / target_mask are copied (identical across runs; checked).

  .venv/bin/python scripts/v3q/make_prior.py [--handoff docs/v3p/HANDOFF.json]
"""
import argparse, json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]


def find_pred_files(entry):
    """entry: list of dirs / predictions.jsonl paths, or dict with such a list."""
    if isinstance(entry, dict):
        for k in ('prediction_dirs', 'dirs', 'predictions', 'runs', 'paths', 'files'):
            if k in entry:
                return find_pred_files(entry[k])
        raise SystemExit(f'cannot find prediction list in {list(entry)}')
    out = []
    for p in entry:
        p = ROOT / p if not str(p).startswith('/') else Path(p)
        out.append(p if p.name.endswith('.jsonl') else p / 'predictions.jsonl')
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--handoff', default='docs/v3p/HANDOFF.json')
    a.add_argument('--val-key', default=None); a.add_argument('--test-key', default=None); a.add_argument('--stats', default=None)
    args = a.parse_args()
    H = json.loads((ROOT / args.handoff).read_text())
    stats_path = args.stats or H.get('stats') or H.get('specific_stats') or H.get('stats_specific')
    assert stats_path, 'stats path not in hand-off; pass --stats'
    ST = json.loads((ROOT / stats_path).read_text())
    RUNS = [f'v3main_{g}_seed{sd}' for g in 'BCD' for sd in (17, 29, 43)]  # 9-run B/C/D ensemble (V2 rule)
    for split in ('val', 'test'):
        if 'runs' in H:
            if not all(split in H['runs'].get(r, {}) for r in RUNS):
                print(split, 'incomplete:', [r for r in RUNS if split not in H['runs'].get(r, {})]); continue
            files = [ROOT / H['runs'][r][split]['predictions'] for r in RUNS]
        else:
            files = find_pred_files(H[getattr(args, f'{split}_key') or split])
        rows = {}
        for f in files:
            for x in map(json.loads, f.read_text().splitlines()):
                if not x['sample_id'].startswith(f'v3main/{split}/'):
                    continue
                e = rows.setdefault(x['sample_id'], {'z': [], 'votes': [0, 0, 0], 'tz': x['target_z'], 'tm': x['target_mask']})
                assert np.allclose(e['tz'], x['target_z']) and e['tm'] == x['target_mask'], x['sample_id']
                e['z'].append(x['prediction_z']); e['votes'][x['predicted_law_id']] += 1
        n = len(files)
        assert all(len(e['z']) == n for e in rows.values()), 'runs disagree on samples'
        out = {}
        for sid, e in rows.items():
            z = np.mean(e['z'], 0)
            dec = [float(math.expm1(min(max(v * ST['std'][j] + ST['mean'][j], 0.), 40.))) if j != 1 else float(v * ST['std'][j] + ST['mean'][j]) for j, v in enumerate(z)]
            out[str(int(sid.rsplit('/', 1)[1]))] = {
                'sample_id': sid, 'prediction_z': z.tolist(), 'z_std_across_runs': np.std(e['z'], 0).tolist(),
                'law_posterior_votes': (np.array(e['votes']) / n).tolist(), 'nu': dec[1],
                'x_log10': [math.log10(max(dec[0], 1e-3)), math.log10(max(dec[3], 1e-3))], 'target_z': e['tz'], 'target_mask': e['tm']}
        res = {'source': f'{args.handoff} ({n} runs)', 'files': [str(f.relative_to(ROOT)) for f in files], 'split': split,
               'stats': ST, 'stats_source': stats_path, 'rule': 'mean prediction_z, vote fractions (V2 ENSEMBLE_BCD9 rule)', 'samples': out}
        ej = H.get('ensemble', {}).get(split, {}).get('json')
        if ej and (ROOT / ej).exists():  # cross-check against the v3p ensemble file
            E = json.loads((ROOT / ej).read_text())['samples']
            for sid, e in rows.items():
                assert np.allclose(E[sid]['prediction_z'], np.mean(e['z'], 0), atol=1e-9) and np.allclose(E[sid]['law_posterior_votes'], np.array(e['votes']) / n), sid
            res['checked_against'] = ej
        (ROOT / f'docs/v3q/PRIOR_ENSEMBLE_{split}.json').write_text(json.dumps(res) + '\n')
        print(split, len(out), 'samples from', n, 'runs')


if __name__ == '__main__':
    main()
