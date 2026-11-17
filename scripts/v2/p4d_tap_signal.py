"""P4d diagnostic: is E/rho visible in tap videos without any learning?

From the 128-frame object masks (32 fps) of each tap sample, take the post-release window (0.6-4 s), track
per-view mask height and area, and compute (a) dominant frequency, (b) RMS of the detrended signal.
Within each law, Spearman correlation of these hand features with log E/rho, per arm and split.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 4 else None


def features(pack):
    m = np.load(pack)['object_mask'][:, :, 0] > 127                     # [V, T, H, W]
    rows = m.any(-1)                                                     # [V, T, H]
    top = np.where(rows.any(-1), rows.argmax(-1), np.nan)                # topmost row per frame
    area = m.sum((-1, -2)).astype(float)
    out = []
    for sig in (top, area):
        x = sig[:, 19:]                                                  # after release (0.6 s)
        t = np.arange(x.shape[1])
        res = np.stack([v - np.polyval(np.polyfit(t, v, 2), t) for v in np.nan_to_num(x)])   # remove drift
        spec = np.abs(np.fft.rfft(res * np.hanning(len(t)), axis=1)) ** 2
        f = np.fft.rfftfreq(len(t), 1 / 32.)
        out += [float(np.median(f[1:][spec[:, 1:].argmax(1)])), float(np.log(res.std(1).mean() + 1e-6))]
    return out


def main():
    report = {}
    for arm in ('low', 'v1'):
        for split in ('train', 'val'):
            rows = []
            for gt_path in sorted((ROOT / f'data/v2_pilot/p4c_{arm}/{split}').glob('*/gt.json')):
                gt = json.loads(gt_path.read_text()); packs = list(gt_path.parent.glob('*/sample_pack.npz'))
                if gt['action'] != 'tap' or not packs:
                    continue
                rows.append((gt['law'], np.log10(gt['labels']['E_over_rho_specific']), features(packs[0])))
            for law in ('jelly', 'metal', 'plasticine'):
                sub = [r for r in rows if r[0] == law]
                y = np.array([r[1] for r in sub]); X = np.array([r[2] for r in sub])
                report[f'{arm}/{split}/{law}'] = {'n': len(sub), **{name: spearman(X[:, i], y) for i, name in
                                                  enumerate(['top_freq', 'top_rms', 'area_freq', 'area_rms'])}}
    (ROOT / 'docs/p4d/P4D_TAP_SIGNAL.json').write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(f"{k:24s} n={v['n']:3d} " + ' '.join(f"{n}={v[n]:+.2f}" for n in ('top_freq', 'top_rms', 'area_freq', 'area_rms')))


if __name__ == '__main__':
    main()
