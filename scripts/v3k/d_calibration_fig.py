"""V3k D figure: reliability curves (nominal central-interval level vs empirical coverage) from docs/v3k/D_CALIBRATION.json.
  .engine-envs/plot/bin/python scripts/v3k/d_calibration_fig.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / 'docs/v3k/D_CALIBRATION.json').read_text())
T = ['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho']; LBL = {'E_over_rho': 'E/ρ', 'nu': 'ν', 'rho': 'ρ', 'sigma_y_over_rho': 'σ_y/ρ'}
KIND = {'ens': 'σ = across-run std (9 runs)', 'total': 'σ = sqrt(aleatoric + across-run)'}
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8984'
fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), sharex=True, sharey=True)
for i, kind in enumerate(('ens', 'total')):
    for j, t in enumerate(T):
        ax = axes[i, j]; r = d['results'][kind][t]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, zorder=1)
        for key, c, ls, lab in [('val_raw', BLUE, '-', 'val, raw σ'), ('val_scaled_in_sample', BLUE, '--', 'val, T·σ (in-sample)'),
                                ('test_raw', ORANGE, '-', 'test, raw σ'), ('test_val_scaled', ORANGE, '--', 'test, val-fitted T·σ')]:
            rel = r[key]['reliability']; xs = [float(k) for k in rel]; ys = list(rel.values())
            ax.plot(xs, ys, color=c, ls=ls, lw=2, marker='o', ms=4, label=lab, zorder=3)
        ax.set_title(f"{LBL[t]}  (T={r['T_val']:.2f}, n={r['val_raw']['n']}/{r['test_raw']['n']})", fontsize=10, color=INK)
        ax.grid(color='#e6e5e0', lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        if j == 0: ax.set_ylabel(f"{KIND[kind]}\nempirical coverage", fontsize=9)
        if i == 1: ax.set_xlabel('nominal central-interval level', fontsize=9)
axes[0, 0].legend(fontsize=8, frameon=False, loc='upper left')
fig.suptitle('9-run ensemble calibration (z space; val n/test n; T fitted on val only)', fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / 'docs/v3k/calibration.png', dpi=150)
print('ok')
