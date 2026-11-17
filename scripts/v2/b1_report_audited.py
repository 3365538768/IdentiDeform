"""B1 report: per excitation, is jelly E/rho visible on real (3DGS) objects without any learning?

Each B1 simulation pair differs only in E/rho (x sqrt(10)); 4 stiffness levels per object (10^2.75..10^4.25).
(a) pair pixel difference: mean over frames/views of |mask_a XOR mask_b| / |mask_a| (0 = indistinguishable).
(b) hand features from the per-view object mask (top row, area, centroid row; normalized by frame 0):
    min/final top drop, min/final area ratio, centroid sag, post-0.6 s residual RMS and dominant frequency.
    Spearman with log E/rho after removing the per-object mean (so object size/shape cannot drive it).
Decision gate (docs/P4D_RESULTS_AND_NEXT_PLAN_2026-11-11_ZH.md): an action enters B2 if max |Spearman| >= 0.5.
  .venv/bin/python scripts/v2/b1_report.py   ->  docs/b1/B1_RESULTS.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
B1 = ROOT / 'data/v2_pilot/b1'
FEATURES = ['top_drop_max', 'top_drop_final', 'area_min', 'area_final', 'centroid_sag', 'resid_rms', 'resid_freq']


def spearman(a, b):
    ra, rb = rankdata(a, method="average"), rankdata(b, method="average")
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 4 and np.std(ra) > 0 and np.std(rb) > 0 else None


def masks(sample_dir):
    packs = list(sample_dir.glob('*/sample_pack.npz'))
    return np.load(packs[0])['object_mask'][:, :, 0] > 127 if packs else None   # [V, T, H, W]


def features(m):
    rows = m.any(-1)                                                          # [V, T, H]
    h = m.shape[-2]
    top = np.where(rows.any(-1), rows.argmax(-1), h).astype(float)            # topmost row per frame
    area = m.sum((-1, -2)).astype(float)
    ys = np.arange(h)[None, None, :]
    cy = (rows * ys).sum(-1) / np.maximum(rows.sum(-1), 1)
    size = np.maximum(h - top[:, :1], 1.0)                                   # rough object height in pixels
    drop = (top - top[:, :1]) / size
    ratio = area / np.maximum(area[:, :1], 1.0)
    x = drop[:, 19:]                                                          # after 0.6 s
    t = np.arange(x.shape[1])
    res = np.stack([v - np.polyval(np.polyfit(t, v, 2), t) for v in x])
    spec = np.abs(np.fft.rfft(res * np.hanning(len(t)), axis=1)) ** 2
    freq = np.fft.rfftfreq(len(t), 1 / 32.)
    vals = [drop.max(1), drop[:, -8:].mean(1), ratio.min(1), ratio[:, -8:].mean(1),
            ((cy - cy[:, :1]) / size).max(1), np.log(res.std(1) + 1e-6), freq[1:][spec[:, 1:].argmax(1)]]
    return [float(np.mean(v)) for v in vals]


def main():
    report = {'actions': {}, 'gate': 0.5}
    for action_dir in sorted(p for p in B1.iterdir() if p.is_dir()):
        samples, pair_diff, missing = {}, [], 0
        union_diff, empty_rates, border_rates = [], [], []
        for d in sorted(action_dir.iterdir()):
            r = d / 'result.json'
            if not r.exists() or json.loads(r.read_text()).get('exit') != 0:
                missing += 1; continue
            m = masks(d)
            if m is None:
                missing += 1; continue
            empty_rates.append(float((m.sum((-1,-2)) == 0).mean()))
            border_rates.append(float((m[...,0,:].any(-1) | m[...,-1,:].any(-1) | m[...,:,0].any(-1) | m[...,:,-1].any(-1)).mean()))
            gt = json.loads((d / 'gt.json').read_text())
            samples[d.name] = dict(obj=gt['object'], base=gt['base'], member=gt['member'],
                                   log_e=float(np.log10(gt['E_over_rho_specific'])), m=m)
        for name, s in samples.items():
            if s['member'] != 'a':
                continue
            other = samples.get(name[:-1] + 'b')
            if other is None:
                continue
            a, b = s['m'], other['m']
            union_diff.append(float(((a ^ b).sum((-1,-2)) / np.maximum((a | b).sum((-1,-2)),1)).mean()))
            pair_diff.append(float(((a ^ b).sum((-1, -2)) / np.maximum(a.sum((-1, -2)), 1)).mean()))
        feats = {k: features(s['m']) for k, s in samples.items()}
        out = {'rank_method': 'average_ties', 'exploratory_only': True, 'xor_over_union_mean': float(np.mean(union_diff)) if union_diff else None, 'empty_frame_fraction': float(np.mean(empty_rates)) if empty_rates else None, 'border_touch_fraction': float(np.mean(border_rates)) if border_rates else None, 'n_samples': len(samples), 'n_missing': missing, 'n_pairs': len(pair_diff),
               'pair_xor_mean': float(np.mean(pair_diff)) if pair_diff else None,
               'pair_xor_median': float(np.median(pair_diff)) if pair_diff else None, 'spearman': {}}
        if len(samples) > 4:
            names = list(samples)
            objs = np.array([samples[n]['obj'] for n in names])
            log_e = np.array([samples[n]['log_e'] for n in names])
            f = np.array([feats[n] for n in names])
            for o in set(objs):                                              # remove per-object offsets
                f[objs == o] -= f[objs == o].mean(0)
            out['spearman'] = {k: spearman(f[:, i], log_e) for i, k in enumerate(FEATURES)}
            best = max(((k, v) for k, v in out['spearman'].items() if v is not None), key=lambda kv: abs(kv[1]), default=(None, None))
            out['best_feature'], out['best_abs_spearman'] = best[0], abs(best[1]) if best[1] is not None else None
            out['passes_gate'] = bool(out['best_abs_spearman'] is not None and out['best_abs_spearman'] >= report['gate'])
        report['actions'][action_dir.name] = out
        print(action_dir.name, {k: v for k, v in out.items() if k != 'spearman'}, flush=True)
    path = ROOT / 'docs/b1/B1_AUDITED_RESULTS.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
