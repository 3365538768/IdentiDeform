"""P4d: manifest + train statistics for one P4c arm, in the DeformFieldBench v1 manifest format.

  .venv/bin/python scripts/v2/build_v2_manifest.py --arm low
Writes data/manifests/v2pilot_{arm}.jsonl, .stats.json (native) and .specific.stats.json. Only samples whose
sample_pack.npz exists in both arms are kept, so the two arms stay paired. Censored plasticine sigma_y
is written as null (masked), with sigma_censored=true.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def stats(rows, specific, manifest_sha):
    cols = [[], [], [], []]
    for r in rows:
        p = dict(r['parameters']); scale = p['rho'] / 1000.
        if specific:
            p['E'] /= scale
            if p['sigma_y'] is not None:
                p['sigma_y'] /= scale
        cols[0].append(math.log1p(p['E'])); cols[1].append(p['nu']); cols[2].append(math.log1p(p['rho']))
        if p['sigma_y'] is not None:
            cols[3].append(math.log1p(p['sigma_y']))
    out = {'mean': [float(np.mean(c)) for c in cols], 'std': [float(np.std(c)) for c in cols], 'count': [len(c) for c in cols],
           'transform': 'ln1p_E_rho_sigma_y;linear_nu', 'fit_split': 'train', 'manifest_sha256': manifest_sha}
    if specific:
        out.update(target_space='specific', note='E and sigma_y divided by rho/1000 before ln1p')
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--arm', choices=['low', 'v1'])
    a.add_argument('--root', help='unpaired dataset root, e.g. data/v2_main'); a.add_argument('--name'); args = a.parse_args()
    paired = args.root is None
    root = ROOT / (f'data/v2_pilot/p4c_{args.arm}' if paired else args.root)
    name = f'v2pilot_{args.arm}' if paired else args.name
    other = 'v1' if args.arm == 'low' else 'low'
    rows = []
    for split in ('train', 'val', 'test'):
        for d in sorted((root / split).glob('[0-9]*')):
            packs = list(d.glob('*/sample_pack.npz'))
            result = d / 'result.json'
            if not packs or not result.exists() or json.loads(result.read_text()).get('exit') != 0:
                continue
            if paired and not list((ROOT / f'data/v2_pilot/p4c_{other}/{split}/{d.name}').glob('*/sample_pack.npz')):
                continue
            gt = json.loads((d / 'gt.json').read_text())
            rp = json.loads((packs[0].parent / 'meta/run_parameters.json').read_text())
            idx = rp['render_frame_indices']; dt = float(rp['time_params']['frame_dt'])
            sigma = gt['regression']['yield_stress'] if gt['law'] != 'jelly' and not gt['labels']['censored_sigma'] else None
            rows.append({'sample_id': f"{'v2pilot' if paired else name}/{split}/{d.name}", 'object_id': gt['object'], 'asset_group': 'deformfield:' + gt['object'],
                         'source': (f'DeformFieldBench-V2pilot-{args.arm}' if paired else f'DeformFieldBench-{name}'), 'pack': str(packs[0]),
                         'parameters': {'E': gt['regression']['E'], 'nu': gt['regression']['nu'], 'rho': gt['regression']['density'],
                                        'sigma_y': sigma},
                         'law_id': gt['law_id'], 'action': gt['action'], 'field_mode': 'legacy_visual',
                         'label_provenance': 'simulator_config', 'field_warning': 'visualized flow/stress; contact mask is not force',
                         'timestamps_seconds': [(i - idx[0]) * dt for i in idx], 'time_provenance': 'render_frame_indices * frame_dt',
                         'published_split': split, 'split': split, 'sigma_censored': gt['labels']['censored_sigma'],
                         'damping': gt['damping']})
    manifest = ROOT / f'data/manifests/{name}.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    train = [r for r in rows if r['split'] == 'train']
    (ROOT / f'data/manifests/{name}.stats.json').write_text(json.dumps(stats(train, False, sha), indent=1))
    (ROOT / f'data/manifests/{name}.specific.stats.json').write_text(json.dumps(stats(train, True, sha), indent=1))
    print(manifest, {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')})


if __name__ == '__main__':
    main()
