"""V3p: non-visual reference baselines (rule of scripts/v3c/simple_baselines.py, re-pointed to a V3 manifest).
constant: z=0 (train mean) + majority train law. class_mean_gt (PRIVILEGED): GT law -> that law's train mean z.
  .venv/bin/python scripts/v3p/baselines.py --manifest v3main_t2000 --prefix v3main --split val
"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
from fieldbridge.data import ParameterCodec, specific_parameters
ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--prefix', required=True)
    ap.add_argument('--split', required=True); a = ap.parse_args()
    codec = ParameterCodec(json.loads((ROOT / f'data/manifests/{a.manifest}.specific.stats.json').read_text()))
    rows = [json.loads(l) for l in open(ROOT / f'data/manifests/{a.manifest}.jsonl')]
    enc = lambda r: [x.tolist() for x in codec.encode(specific_parameters(r['parameters']))]
    train = [(r, *enc(r)) for r in rows if r['split'] == 'train']; ev = [(r, *enc(r)) for r in rows if r['split'] == a.split]
    majority = Counter(r['law_id'] for r, _, _ in train).most_common(1)[0][0]
    law_mean = {}
    for law in sorted({r['law_id'] for r, _, _ in train}):
        z = np.array([t for r, t, m in train if r['law_id'] == law]); m = np.array([m for r, t, m in train if r['law_id'] == law], bool)
        law_mean[law] = [float(z[m[:, j], j].mean()) if m[:, j].any() else 0. for j in range(4)]
    for name, predict in [('const', lambda r: ([0.] * 4, majority)), ('lawmean_gt', lambda r: (law_mean[r['law_id']], r['law_id']))]:
        out = ROOT / f'runs/{a.prefix}_{name}_{a.split}'; out.mkdir(parents=True, exist_ok=True)
        with (out / 'predictions.jsonl').open('w') as f:
            for r, t, m in ev:
                z, law = predict(r)
                f.write(json.dumps({'sample_id': r['sample_id'], 'object_id': r['object_id'], 'source': r.get('source', ''),
                                    'action': r['action'], 'law_id': r['law_id'], 'predicted_law_id': int(law),
                                    'prediction_z': z, 'logvar': [0.] * 4, 'target_z': t, 'target_mask': m}) + '\n')
        (out / 'law_means.json').write_text(json.dumps({'manifest': a.manifest, 'majority_train_law': majority, 'train_law_mean_z': law_mean}, indent=1))
    print('ok', len(train), len(ev), majority)


if __name__ == '__main__':
    main()
