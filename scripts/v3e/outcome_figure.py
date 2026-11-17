"""V3e paper figure: GT-parameter execution outcomes of the depths chosen by each planner (particle view).

  python3 scripts/v3e/outcome_figure.py --json docs/v3e/GRIPPER_TASK.json --indices 293,29,53 --out docs/v3e/figs/outcomes.png
For each sample (row): initial shape (grey) vs shape after release (colour) of the GT object squeezed at the depth chosen
by each method (abstain = not grasped), top view (squeeze axis horizontal). Uses the 4096-particle subsets kept by
squeeze_run.py (cache/<key>/<d>/subset.npz) of the GT runs.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHOW = [('gt', 'oracle (GT twin)'), ('refined_top2', 'ours: prior+12 sims'), ('ensemble_prior', 'prior, 0 sims'),
        ('refined_best3', 'uninformed+12 sims'), ('lawmean_pred', 'law mean'), ('fixed_loo', 'fixed depth')]


def main():
    a = argparse.ArgumentParser(); a.add_argument('--json', default='docs/v3e/GRIPPER_TASK.json'); a.add_argument('--indices', required=True)
    a.add_argument('--out', required=True); args = a.parse_args()
    res = {r['index']: r for r in json.loads((ROOT / args.json).read_text())['samples']}
    idx = [int(i) for i in args.indices.split(',')]
    tau = json.loads((ROOT / args.json).read_text())['protocol']['tau']
    fig, axes = plt.subplots(len(idx), len(SHOW), figsize=(2.3 * len(SHOW), 2.5 * len(idx)), squeeze=False)
    for row, i in zip(axes, idx):
        r = res[i]; key = json.loads((ROOT / f'data/v3e/runs/{i:06d}/gt/twin.json').read_text())['cache']
        cache = ROOT / f'data/v3e/runs/{i:06d}/cache/{key}'
        for ax, (m, lab) in zip(row, SHOW):
            ax.set_xticks([]); ax.set_yticks([])
            if m not in r['methods']:
                ax.axis('off'); continue
            d = r['methods'][m]['chosen_d']
            if d is None:
                sub = np.load(cache / '0.000' / 'subset.npz') if (cache / '0.000' / 'subset.npz').exists() else None
                txt = 'abstain'; ok = r['methods'][m]['success']
            else:
                sub = np.load(cache / f'{d:.3f}' / 'subset.npz'); bc = json.loads((cache / f'{d:.3f}' / 'squeeze_bc.json').read_text())
                c = r['gt_dent'][f'{d:.2f}']; txt = f'd={d:.2f}  crush={100 * c:.1f}%'; ok = r['methods'][m]['success']
            if sub is not None:
                bc = json.loads((cache / '0.000' / 'squeeze_bc.json').read_text()) if d is None else bc
                ax_a = bc['axis']; ax_b = 1 - ax_a
                x0, xt = sub['x0'], sub['x_tail'] if d is not None else sub['x0']
                ms, md = xt.mean(0), x0.mean(0)  # rigid (Procrustes) alignment to the initial shape, as in the metric
                u, _, vt = np.linalg.svd((xt - ms).T @ (x0 - md)); sg = np.eye(3); sg[2, 2] = np.sign(np.linalg.det(u @ vt))
                xt = (xt - ms) @ (u @ sg @ vt) + md
                ax.scatter(x0[:, ax_a], x0[:, ax_b], s=0.3, c='0.8', rasterized=True)
                ax.scatter(xt[:, ax_a], xt[:, ax_b], s=0.3, c=('tab:green' if ok else 'tab:red'), rasterized=True)
                ax.set_aspect('equal')
            ax.set_title(f'{lab}\n{txt}  {"OK" if ok else "FAIL"}', fontsize=7, color=('darkgreen' if ok else 'darkred'))
        row[0].set_ylabel(f'#{i} {r["law"]}\noracle d={r["oracle_d"]}', fontsize=8)
    fig.suptitle(f'GT execution of planned squeeze depth (top view, grey = initial, colour = after release (rigidly aligned); tau={100 * tau:.0f}% width loss)', fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / args.out, dpi=170)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
