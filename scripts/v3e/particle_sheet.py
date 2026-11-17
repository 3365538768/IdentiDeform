"""V3e pilot check: particle contact sheet (side view, squeeze axis horizontal, z up) for kept squeeze runs.

  .venv/bin/python scripts/v3e/particle_sheet.py OUT.png DIR [DIR ...]   (DIR = a cache/<key>/<d> run dir)
Columns: t=0, end of closing (1.0 s), end of hold (1.5 s), final 0.5 s mean; plates drawn at their positions.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out, dirs = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    fig, axes = plt.subplots(len(dirs), 4, figsize=(12, 2.8 * len(dirs)), squeeze=False)
    for row, d in zip(axes, dirs):
        t = np.load(d / 'traj.npz'); bc = json.loads((d / 'squeeze_bc.json').read_text()); a = bc['axis']
        sub = np.random.default_rng(1).choice(len(t['x0']), min(20000, len(t['x0'])), replace=False)
        lo, hi = np.array(bc['lo']), np.array(bc['hi']); W = bc['width']; pad = 0.3 * max(bc['extents'])
        for ax, (name, key) in zip(row, [('t=0', 'x0'), ('closed 1.0s', 'x_closed'), ('hold end 1.5s', 'x_open'), ('final (3.5-4s mean)', 'x_tail')]):
            x = t[key][sub]
            ax.scatter(x[:, a], x[:, 2], s=0.2, c=x[:, 1 - a], cmap='viridis', rasterized=True)
            ref = t['x0'][sub]
            ax.axvline(ref[:, a].min(), color='0.7', lw=0.5); ax.axvline(ref[:, a].max(), color='0.7', lw=0.5)
            if bc['d'] > 0 and key in ('x_closed', 'x_open'):
                off = bc['d'] * W / 2
                for xx in (lo[a] + off, hi[a] - off):
                    ax.axvline(xx, color='r', lw=1.0)
            ax.set_xlim(lo[a] - pad, hi[a] + pad); ax.set_ylim(lo[2] - 0.1 * pad, hi[2] + pad); ax.set_aspect('equal')
            ax.set_title(f'{d.parent.parent.parent.name} d={bc["d"]:.2f} {name}', fontsize=7); ax.tick_params(labelsize=5)
    fig.tight_layout(); fig.savefig(out, dpi=130)


if __name__ == '__main__':
    main()
