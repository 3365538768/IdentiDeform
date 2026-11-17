"""P4a: how much temporal signal lies above what the V1 sampling can represent.

For each 32 fps replay (128 frames over 4 s) the per-pixel RGB change inside the object mask is
detrended per pixel; the power spectrum (mean over pixels and views) gives the energy fraction above
8 Hz (Nyquist of the 16 fps V1 videos), above 4 Hz (V-JEPA tokens: 32 steps / 4 s) and above 2 Hz
(DINO runs: 16 frames / 4 s). Computed on the whole clip and on the free phase after tool release (3-4 s).
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAW = {'jelly': 'jelly', 'plasticine': 'plasticine', 'metal': 'metal'}


def fractions(x, fps):
    x = x - x.mean(0, keepdims=True)
    w = np.hanning(len(x))[:, None]
    p = (np.abs(np.fft.rfft(x * w, axis=0)) ** 2).mean(1)[1:]
    f = np.fft.rfftfreq(len(x), 1 / fps)[1:]
    total = p.sum() + 1e-12
    return {f'above_{c}hz': float(p[f > c].sum() / total) for c in (2, 4, 8)}, float(f[p.argmax()])


def main():
    rows = []
    for pack in sorted(glob.glob(str(ROOT / 'data/v2_pilot/replay32/*/*/sample_pack.npz'))):
        d = Path(pack).parents[1]; gt = json.loads((d / 'gt.json').read_text()); n = np.load(pack)
        rgb = n['rgb'].astype(np.float32).mean(2); mask = n['object_mask'][:, :, 0] > 127   # [V, T, H, W]
        sel = mask.any(1)                                                                  # pixels ever covered
        sig = np.concatenate([rgb[v][:, sel[v]] for v in range(rgb.shape[0])], 1)          # [T, P]
        whole, peak = fractions(sig, 32.); free, peak_free = fractions(sig[96:], 32.)
        rows.append(dict(sample=d.name, law=gt['params']['material'], action=gt['action'], E=gt['regression']['E'],
                         rho=gt['regression']['density'], whole=whole, free=free, peak_hz=peak, peak_free_hz=peak_free,
                         motion=float(np.abs(np.diff(sig, axis=0)).mean())))
    table = {}
    for key in sorted({(r['law'], r['action']) for r in rows}):
        sub = [r for r in rows if (r['law'], r['action']) == key]
        table['/'.join(key)] = {ph: {k: float(np.mean([r[ph][k] for r in sub])) for k in sub[0][ph]} for ph in ('whole', 'free')}
    for law in sorted({r['law'] for r in rows}):
        sub = [r for r in rows if r['law'] == law]
        table[law] = {ph: {k: float(np.mean([r[ph][k] for r in sub])) for k in sub[0][ph]} for ph in ('whole', 'free')}
    out = ROOT / 'docs/p4/P4A_SPECTRUM.json'; out.write_text(json.dumps({'samples': rows, 'summary': table}, indent=1))
    for k, v in table.items():
        print(f"{k:22s} whole >2/4/8Hz {v['whole']['above_2hz']:.3f} {v['whole']['above_4hz']:.3f} {v['whole']['above_8hz']:.3f} | "
              f"free {v['free']['above_2hz']:.3f} {v['free']['above_4hz']:.3f} {v['free']['above_8hz']:.3f}")


if __name__ == '__main__':
    main()
