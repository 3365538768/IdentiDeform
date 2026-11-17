"""R0 figure for reviewer a1nG: predicted vs ground truth, globally and within each constitutive law.

Rows are runs (e.g. native E from R1, E/rho from R1b); columns are all laws together, then each law.
Values are converted back to log10 physical units with the run's train statistics.

  .engine-envs/plot/bin/python scripts/plot_within_law.py runs/r1_C_seed17:E:native runs/r1b_C_seed17:E:specific \
      --output docs/r1/within_law_scatter.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAWS = {0: 'jelly', 1: 'metal', 2: 'plasticine'}
COLORS = {0: '#2a78d6', 1: '#eb6834', 2: '#1baf7a'}  # validated categorical slots 1-3
PARAM_INDEX = {'E': 0, 'nu': 1, 'rho': 2, 'sigma_y': 3}
LABELS = {('E', 'native'): 'E', ('E', 'specific'): 'E/ρ', ('sigma_y', 'native'): 'σ_y',
          ('sigma_y', 'specific'): 'σ_y/ρ', ('rho', 'native'): 'ρ', ('rho', 'specific'): 'ρ', ('nu', 'native'): 'ν'}


def best_predictions(run):
    timing = [json.loads(l) for l in (run / 'timing.jsonl').read_text().splitlines() if l.strip()]
    best = min((t for t in timing if t.get('phase') == 'joint_training'), key=lambda t: t['val_mae'])
    return [json.loads(l) for l in (run / f"val_epoch_{best['epoch'] - 1:04d}/predictions.jsonl").read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(); p.add_argument('specs', nargs='+', help='RUN:PARAM:SPACE')
    p.add_argument('--output', required=True); a = p.parse_args()
    fig, axes = plt.subplots(len(a.specs), 4, figsize=(13, 3.2 * len(a.specs)), squeeze=False)
    for row, spec in zip(axes, a.specs):
        run, name, space = spec.split(':'); run = Path(run); j = PARAM_INDEX[name]
        cfg = json.loads((run / 'run.json').read_text()) if (run / 'run.json').exists() else {}
        stats_path = cfg.get('config', cfg).get('data', {}).get('stats') or json.loads(
            next(run.parent.parent.glob(f'configs/*/{run.name}.json')).read_text())['data']['stats']
        stats = json.loads(Path(stats_path).read_text())
        records = [r for r in best_predictions(run) if r['target_mask'][j]]
        to_log10 = lambda z: (np.asarray(z) * stats['std'][j] + stats['mean'][j]) / np.log(10) if name != 'nu' else np.asarray(z) * stats['std'][j] + stats['mean'][j]
        gt = to_log10([r['target_z'][j] for r in records]); pr = to_log10([r['prediction_z'][j] for r in records])
        law = np.array([r['law_id'] for r in records]); label = LABELS[(name, space)]
        unit = '' if name == 'nu' else 'log10 '
        lo, hi = min(gt.min(), pr.min()), max(gt.max(), pr.max())
        for k, ax in enumerate(row):
            sel = np.ones(len(gt), bool) if k == 0 else law == k - 1
            if sel.sum() < 3:
                ax.axis('off'); continue
            for L in sorted(set(law[sel])):
                m = sel & (law == L)
                ax.scatter(gt[m], pr[m], s=12, color=COLORS[L], alpha=.7, linewidths=0, label=LAWS[L])
            r = np.corrcoef(gt[sel], pr[sel])[0, 1]
            a_lo, a_hi = (lo, hi) if k == 0 else (min(gt[sel].min(), pr[sel].min()), max(gt[sel].max(), pr[sel].max()))
            ax.plot([a_lo, a_hi], [a_lo, a_hi], color='#8a8984', lw=1, ls='--')
            title = 'all laws' if k == 0 else LAWS[k - 1]
            ax.set_title(f'{title}: r = {r:.2f}  (n={sel.sum()})', fontsize=10)
            ax.set_xlabel(f'GT {unit}{label}', fontsize=9); ax.set_ylabel(f'pred {unit}{label}', fontsize=9)
            ax.tick_params(labelsize=8); ax.grid(color='#e4e3df', lw=.5)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            if k == 0:
                ax.legend(fontsize=8, frameon=False, loc='upper left')
        row[0].annotate(f'{run.name}', xy=(0, 1.18), xycoords='axes fraction', fontsize=10, fontweight='bold')
    fig.tight_layout(); Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160); print(a.output)


if __name__ == '__main__':
    main()
