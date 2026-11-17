"""Score replays: per-frame object-mask IoU and RGB MAE (inside the union mask) of pred/lawmean vs gt."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(d):
    # result.json is written after the pack, so only finished simulations are read.
    p = list(d.glob('*/sample_pack.npz'))
    if not p or not (d / 'result.json').exists() or json.loads((d / 'result.json').read_text()).get('exit') != 0:
        return None
    n = np.load(p[0]); return n['object_mask'][:, :, 0] > 127, n['rgb'].astype(np.float32)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--run', required=True); args = a.parse_args()
    rows = []
    for d in sorted((ROOT / f'data/v2_replay/{args.run}').glob('[0-9]*')):
        gt = load(d / 'gt')
        if gt is None:
            continue
        row = {'index': d.name, 'replay': json.loads((d / 'gt/replay.json').read_text())['replay_action']}
        for name in ('pred', 'lawmean'):
            c = load(d / name)
            if c is None:
                continue
            inter = (gt[0] & c[0]).sum((-1, -2)); union = (gt[0] | c[0]).sum((-1, -2))
            iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
            rgb = np.abs(gt[1] - c[1]).mean(2)
            u = gt[0] | c[0]
            row[name] = {'iou': float(iou.mean()), 'iou_last': float(iou[:, -1].mean()),
                         'rgb_mae_union': float((rgb * u).sum() / max(u.sum(), 1))}
        if 'pred' in row and 'lawmean' in row:
            rows.append(row)
    def agg(name, key):
        v = np.array([r[name][key] for r in rows]); return float(v.mean())
    out = {'run': args.run, 'n': len(rows), 'note': 'unseen action replay; lawmean uses the true law (privileged)',
           'pred': {k: agg('pred', k) for k in ('iou', 'iou_last', 'rgb_mae_union')},
           'lawmean': {k: agg('lawmean', k) for k in ('iou', 'iou_last', 'rgb_mae_union')},
           'pred_better_iou_fraction': float(np.mean([r['pred']['iou'] > r['lawmean']['iou'] for r in rows])),
           'rows': rows}
    rng = np.random.default_rng(0); diffs = np.array([r['pred']['iou'] - r['lawmean']['iou'] for r in rows])
    boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
    out['iou_gain_95ci'] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    (ROOT / f'docs/replay/REPLAY_{args.run}.json').write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != 'rows'})


if __name__ == '__main__':
    main()
