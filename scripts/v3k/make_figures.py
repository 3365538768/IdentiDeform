"""V3k figures from docs/v3k/RESULTS.json (run with .engine-envs/plot/bin/python).
  springgaus_refine.png : A — GT vs estimated log10 E (scatter) + |dlog10 E| / Spearman vs simulations
  bc_refine.png         : B/C — identifiable z-MAE vs simulations (variant vs matched GT-asset reference)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / 'docs/v3k/RESULTS.json').read_text())
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a8985', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6})

A = R['A_springgaus_in_target_refinement']
if A.get('n_scenes'):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ps = A['per_scene']; sc = A['scenes']
    gt = [ps[s]['gt_log10E'] for s in sc]
    ax[0].plot([3.5, 6.6], [3.5, 6.6], color=GRAY, lw=1, ls='--', zorder=0)
    ax[0].scatter(gt, [ps[s]['prior_log10E'] for s in sc], s=36, color=GRAY, label='zero-shot prior (k=0)', edgecolor='white', lw=1)
    ax[0].scatter(gt, [ps[s]['prior_k12'] for s in sc], s=36, color=BLUE, label='prior + 12 sims', edgecolor='white', lw=1)
    ax[0].scatter(gt, [ps[s]['jellymean_k12'] for s in sc], s=36, color=ORANGE, marker='s', label='jelly train mean + 12 sims', edgecolor='white', lw=1)
    ax[0].set_xlabel('GT log10 E (Pa, rho=1000)'); ax[0].set_ylabel('estimate log10 E'); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('Spring-Gaus synthetic, 14 scenes', fontsize=9, color=INK)
    ks = list(range(13))
    for arm, c, lab in (('prior', BLUE, 'prior (9-run ens.)'), ('jellymean', ORANGE, 'jelly train mean')):
        pb = A['per_budget'][arm]
        ax[1].plot(ks, [pb[str(k)]['mean_abs_dlog10E'] for k in ks], color=c, lw=2, marker='o', ms=4, label=lab)
        ax[2].plot(ks, [pb[str(k)]['spearman'] for k in ks], color=c, lw=2, marker='o', ms=4, label=lab)
    sm = A['controls']['support_mean_no_video_v3f_LOSO']
    ax[1].axhline(sm['mean_abs_dlog10E'], color=GRAY, ls=':', lw=1.5, label='no-video support mean (v3f)')
    ax[2].axhline(sm['spearman'], color=GRAY, ls=':', lw=1.5, label='no-video support mean (v3f)')
    ax[2].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_xlabel('simulations'); ax[1].set_ylabel('mean |dlog10 E| vs GT'); ax[1].legend(frameon=False, fontsize=8)
    ax[2].set_xlabel('simulations'); ax[2].set_ylabel('Spearman(log10 E, GT)'); ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / 'docs/v3k/springgaus_refine.png', dpi=160)

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
for a_, key, lab in ((ax[0], 'B_reconstructed_geometry', 'visual-hull geometry'), (ax[1], 'C_simulator_mismatch', 'perturbed simulator (n_grid x0.75, damping x10)')):
    r = R[key]
    if not r.get('ident_zmae'):
        a_.set_title(f'{lab}: no data', fontsize=9); continue
    ks = [0, 4, 12]
    for which, c, l in (('reference_matched_gt_asset', GRAY, 'GT asset, matched simulator'), ('variant', BLUE if key[0] == 'B' else ORANGE, lab)):
        m = [r['ident_zmae'][str(k)][which] for k in ks]
        a_.errorbar(ks, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], color=c, lw=2, marker='o', ms=5, capsize=3, label=l)
    a_.set_xlabel('simulations'); a_.set_title(f"{key[0]}: n={r['n_complete']} samples", fontsize=9, color=INK); a_.legend(frameon=False, fontsize=8)
ax[0].set_ylabel('identifiable z-MAE (E/rho, sigma_y/rho)')
fig.tight_layout(); fig.savefig(ROOT / 'docs/v3k/bc_refine.png', dpi=160)
print('ok')
