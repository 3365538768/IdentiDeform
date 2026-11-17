"""V3j figure: docs/v3j/RESULTS.json -> docs/v3j/curves.png (identifiable z-MAE vs #simulations, one line per arm).

  .engine-envs/plot/bin/python scripts/v3j/make_figure.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = {'prior_top1': 'V-JEPA prior, compass 2-D', 'lawmean_gt': 'GT-law mean (priv.), compass 2-D',
         'lawmean_best3': 'best-of-3 law means, compass 2-D', 'cma_prior': 'V-JEPA prior, CMA-ES (s0=0.5)',
         'cma_gtmean': 'GT-law mean (priv.), CMA-ES (s0=1)', 'cma_best3': 'best-of-3 law means, CMA-ES (s0=1)',
         'u4_prior': 'V-JEPA prior, compass 4-D (unfrozen h, nu)', 'u4_gtmean': 'GT-law mean (priv.), compass 4-D'}


def figure(res):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = {'prior_top1': '#2a78d6', 'cma_prior': '#2a78d6', 'u4_prior': '#2a78d6', 'lawmean_gt': '#eb6834', 'cma_gtmean': '#eb6834',
           'u4_gtmean': '#eb6834', 'lawmean_best3': '#1baf7a', 'cma_best3': '#1baf7a'}
    ls = lambda a: '--' if a.startswith('cma') or a.startswith('u4') else '-'
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title in ((axs[0], 'exp1_curves', 'Exp 1: compass (solid) vs CMA-ES (dashed)'),
                           (axs[1], 'exp2_curves', 'Exp 2: frozen 2-D (solid) vs unfrozen 4-D (dashed)')):
        c = res[key] if res[key]['n_samples'] else res.get(key + '_24', res[key])
        for a, v in c['arms'].items():
            ks = sorted(int(k) for k in v); ax.plot(ks, [v[str(k)] for k in ks], ls(a), color=col[a], lw=2, label=LABEL[a])
        if 'target' in res.get('sims_to_target_exp1', {}) and key == 'exp1_curves':
            ax.axhline(res['sims_to_target_exp1']['target'], color='#888888', lw=1, ls=':', label='target: GT-law mean compass @24')
        ax.set_title(f"{title}  (n = {c['n_samples']})", fontsize=10); ax.set_xlabel('# simulations k (best of first k)')
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24]); ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=7.5, frameon=False)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel('identifiable z-MAE (E/rho, sigma_y/rho)')
    fig.tight_layout(); fig.savefig(ROOT / 'docs/v3j/curves.png', dpi=160)



if __name__ == '__main__':
    figure(json.loads((ROOT / 'docs/v3j/RESULTS.json').read_text()))
