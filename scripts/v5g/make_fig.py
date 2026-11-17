"""F6b: twin-planned grasp-and-lift (docs/v5g/RESULTS.json) -> docs/paper_figs/F6b_grasp.{pdf,png}.
(a) executed outcome per arm: success / drop / damage / missed (abstained although a GT-feasible grasp existed), stacked to 100 %,
    success with its object-bootstrap 95 % CI; (c) per-candidate outcome-prediction macro-F1 (primary, PREREG_ADDENDUM) for twins and non-twin baselines; (b) relative grip-force regret (F_GT(chosen) - F_GT(oracle)) / F_GT(oracle) on successful grasps (n shown), CI.
Style (fonts, sizes, Okabe-Ito colours, save()) imported from scripts/paper_figs/make_figs.py.
  .engine-envs/plot/bin/python scripts/v5g/make_fig.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/paper_figs'))
import make_figs as MF  # noqa: E402  (sets rcParams; provides OI, COL, save, plabel)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

ORDER = ['gt', 'prior_vt', 'prior_vision', 'lawmean_pred', 'fixed_loo']
LAB = {'gt': 'GT oracle', 'prior_vt': 'Vision+touch twin', 'prior_vision': 'Vision twin', 'lawmean_pred': 'Class-mean twin',
       'fixed_loo': 'Fixed grasp (LOO)'}
SEG = [('success', 'Success (incl. correct abstain)', MF.OI['green']), ('drop', 'Drop', MF.OI['orange']), ('damage', 'Damage', MF.OI['verm']),
       ('missed', 'Missed (abstain)', MF.OI['lgrey'])]


def main():
    R = json.loads((ROOT / 'docs/v5g/RESULTS.json').read_text()); S = R['summary']
    arms = [a for a in ORDER if a in S]; y = np.arange(len(arms))[::-1]
    fig = plt.figure(figsize=(MF.FULL, 1.85)); gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1.5], wspace=0.55)
    ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1], sharey=ax); cx = fig.add_subplot(gs[2])
    for yi, a in zip(y, arms):
        left = 0.0
        for k, _, c in SEG:
            v = S[a][k] or 0.0
            ax.barh(yi, v, left=left, height=0.62, color=c, edgecolor='white', lw=0.8)
            left += v
        lo, hi = S[a]['success_ci95']
        if lo is not None:
            ax.errorbar(S[a]['success'], yi, xerr=[[S[a]['success'] - lo], [hi - S[a]['success']]], fmt='none', ecolor='#333333', elinewidth=0.7)
        r = S[a]['force_regret_rel']; rl, rh = S[a]['force_regret_rel_ci95']
        if r is None:
            continue
        bx.barh(yi, r, height=0.62, color=MF.OI['blue'], edgecolor='white', lw=0.5)
        bx.text(0.03, yi + 0.42, f"n={S[a]['n_force_regret']}", va='center', ha='left', fontsize=6, color='#555555',
                transform=matplotlib.transforms.blended_transform_factory(bx.transAxes, bx.transData))
        if rl is not None:
            bx.errorbar(r, yi, xerr=[[r - rl], [rh - r]], fmt='none', ecolor='#333333', elinewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels([LAB[a] for a in arms], fontsize=7); ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1'])
    ax.set_xlabel(f"Executed outcome (n={R['n_samples']})"); ax.grid(axis='y', visible=False)
    bx.set_xlabel('Grip-force regret (rel. to oracle)'); bx.grid(axis='y', visible=False); plt.setp(bx.get_yticklabels(), visible=False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEG]
    ax.legend(hs, [l for _, l, _ in SEG], loc='lower center', ncol=2, fontsize=6, bbox_to_anchor=(0.6, 1.07), handlelength=1.0, columnspacing=0.8)
    PR = R.get('primary_prediction')
    if PR:
        names = [('prior_vt', 'Vision+touch twin'), ('prior_vision', 'Vision twin'), ('lawmean_pred', 'Class-mean twin'),
                 ('base_cand_law', 'Per-grasp x law majority'), ('base_cand', 'Per-grasp majority'), ('base_majority', 'Majority class')]
        names = [(k, l) for k, l in names if k in PR['predictors']]; yy = np.arange(len(names))[::-1]
        for yi, (k, _) in zip(yy, names):
            e = PR['predictors'][k]; v = e['macro_f1']; lo, hi = e['ci95']['macro_f1']
            cx.barh(yi, v, height=0.62, color=MF.OI['blue'] if not k.startswith('base') else MF.OI['grey'], edgecolor='white', lw=0.5)
            if lo is not None:
                cx.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt='none', ecolor='#333333', elinewidth=0.7)
        cx.set_yticks(yy); cx.set_yticklabels([l for _, l in names], fontsize=7); cx.tick_params(axis='y', length=0)
        cx.set_xlim(0, 1); cx.set_xlabel(f"Outcome prediction macro-F1 (n={PR['n_samples']}x{PR['n_candidates']})"); cx.grid(axis='y', visible=False)
        MF.plabel(cx, 'c')
    MF.plabel(ax, 'a'); MF.plabel(bx, 'b')
    MF.save(fig, 'F6b_grasp')


if __name__ == '__main__':
    main()
