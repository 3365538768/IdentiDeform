"""R2b: how distinguishable are neighbouring E/rho or sigma_y/rho values in the original MPM?

Sweeps one scale-invariant group over the dataset range (half-decade steps) on the R2 cube and
records, per value, the object's centroid-removed shape signals. For adjacent values it reports
the deformation RMS difference in pixels (assumed 240 px per grid unit, i.e. the 0.5-unit cube
spans ~120 px of a 224 image) using every simulated frame, the dataset's 64 frames and the 16
frames used in training, plus the dominant oscillation frequency of the object's height.

  PYTHONPATH=.venv/lib/python3.12/site-packages .engine-envs/warp/bin/python scripts/r2b_mpm_sweep.py --material jelly --sweep E
"""
import argparse
import json
from pathlib import Path

import numpy as np

from r2_mpm_identifiability import REFERENCE, ROOT, simulate, wp

PX_PER_UNIT = 240.


def shape_signals(traj):
    centered = traj - traj.mean(1, keepdims=True)
    height = np.quantile(centered[..., 2], .98, axis=1) - np.quantile(centered[..., 2], .02, axis=1)
    return centered, height


def dominant_hz(signal, dt):
    x = signal - signal[len(signal) // 4:].mean()
    spectrum = np.abs(np.fft.rfft(x[len(x) // 4:] * np.hanning(len(x) - len(x) // 4)))
    freqs = np.fft.rfftfreq(len(x) - len(x) // 4, dt)
    return float(freqs[1 + spectrum[1:].argmax()]) if len(spectrum) > 1 else None


def main():
    p = argparse.ArgumentParser(); p.add_argument('--material', required=True)
    p.add_argument('--sweep', choices=['E', 'yield_stress'], required=True)
    p.add_argument('--actions', default='press,drop'); p.add_argument('--frame-dt', type=float, default=1 / 60)
    p.add_argument('--decades', type=float, nargs=2, default=None); a = p.parse_args()
    wp.init(); out = ROOT / Path(f'docs/r2/MPM_SWEEP_{a.material}_{a.sweep}.json')
    ref = REFERENCE[a.material]; rho = ref['density']
    # Dataset ranges of the swept group (value / (rho/1000)), from material_space_example.json.
    ranges = {('jelly', 'E'): (2, 6), ('plasticine', 'E'): (2, 6), ('metal', 'E'): (3.5, 7),
              ('plasticine', 'yield_stress'): (0, 6), ('metal', 'yield_stress'): (0, 5)}
    lo, hi = a.decades or ranges[(a.material, a.sweep)]
    groups = 10. ** np.arange(lo, hi + 1e-9, .5)   # specific value in units of (rho/1000)
    report = {'material': a.material, 'sweep': a.sweep, 'specific_values': groups.tolist(), 'density': rho,
              'px_per_unit_assumed': PX_PER_UNIT, 'frame_dt': a.frame_dt, 'actions': {}}
    for action in a.actions.split(','):
        seconds = 4.0 if action == 'press' else 2.0
        runs = []
        for g in groups:
            traj = simulate(a.material, {a.sweep: float(g * rho / 1000.)}, action, seconds, a.frame_dt, 1e-4)
            centered, height = shape_signals(traj)
            runs.append((centered, height))
            print(json.dumps({'action': action, 'value': float(g), 'height_range': float(np.ptp(height))}), flush=True)
        n = len(runs[0][1]); idx64 = np.linspace(0, n - 1, 64).round().astype(int); idx16 = np.linspace(0, n - 1, 16).round().astype(int)
        rows = []
        for i, (g, (centered, height)) in enumerate(zip(groups, runs)):
            row = {'value': float(g), 'dominant_height_hz': dominant_hz(height, a.frame_dt),
                   'height_change_px': float(np.ptp(height) * PX_PER_UNIT)}
            if i + 1 < len(runs):
                diff = np.sqrt(((centered - runs[i + 1][0]) ** 2).sum(-1).mean(1)) * PX_PER_UNIT  # per-frame RMS px
                row.update(next_value=float(groups[i + 1]), diff_px_all=float(diff.mean()),
                           diff_px_64=float(diff[idx64].mean()), diff_px_16=float(diff[idx16].mean()),
                           diff_px_max=float(diff.max()))
            rows.append(row)
        report['actions'][action] = rows
        out.write_text(json.dumps(report, indent=2))
    report['status'] = 'complete'; out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
