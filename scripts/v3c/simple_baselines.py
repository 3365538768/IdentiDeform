"""V3c: non-visual reference baselines on V2 main val, written in evaluation predictions.jsonl format.

constant      : every val sample gets the train mean in standardized specific space (z=0 for every
                parameter by construction) and the majority train law.
law_mean_gt   : PRIVILEGED. Uses the ground-truth law of the val sample and predicts that law's
                train-split mean (z). Law accuracy is 1 by construction (not reported as a result).
  .venv/bin/python scripts/v3c/simple_baselines.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fieldbridge.data import ParameterCodec, specific_parameters

MANIFEST = 'data/manifests/v2main.jsonl'
STATS = 'data/manifests/v2main.specific.stats.json'


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='val'); a = ap.parse_args()
    suffix = '' if a.split == 'val' else '_' + a.split
    codec = ParameterCodec(json.loads(Path(STATS).read_text()))
    rows = [json.loads(l) for l in open(MANIFEST)]
    enc = lambda r: [x.tolist() for x in codec.encode(specific_parameters(r['parameters']))]
    train = [(r, *enc(r)) for r in rows if r['split'] == 'train']
    val = [(r, *enc(r)) for r in rows if r['split'] == a.split]
    majority = Counter(r['law_id'] for r, _, _ in train).most_common(1)[0][0]
    law_mean = {}
    for law in sorted({r['law_id'] for r, _, _ in train}):
        z = np.array([t for r, t, m in train if r['law_id'] == law]); m = np.array([m for r, t, m in train if r['law_id'] == law], bool)
        law_mean[law] = [float(z[m[:, j], j].mean()) if m[:, j].any() else 0. for j in range(4)]
    for name, predict in [('const', lambda r: ([0.] * 4, majority)), ('lawmean_gt', lambda r: (law_mean[r['law_id']], r['law_id']))]:
        out = Path('runs/v3c_' + name + suffix); out.mkdir(parents=True, exist_ok=True)
        with (out / 'predictions.jsonl').open('w') as f:
            for r, t, m in val:
                z, law = predict(r)
                f.write(json.dumps({'sample_id': r['sample_id'], 'object_id': r['object_id'], 'source': r.get('source', ''),
                                    'action': r['action'], 'law_id': r['law_id'], 'predicted_law_id': int(law),
                                    'prediction_z': z, 'logvar': [0.] * 4, 'target_z': t, 'target_mask': m}) + '\n')
    Path('runs/v3c_lawmean_gt' + suffix + '/law_means.json').write_text(json.dumps({'majority_train_law': majority, 'train_law_mean_z': law_mean}, indent=2))
    print('ok', len(train), len(val), majority, law_mean)


if __name__ == '__main__':
    main()
