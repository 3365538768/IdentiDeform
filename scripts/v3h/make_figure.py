"""V3h figure from docs/v3h/PRIOR_QUALITY.json -> docs/v3h/prior_quality.png (needs matplotlib: .engine-envs/plot).

  .engine-envs/plot/bin/python scripts/v3h/make_figure.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#b5b4ad'
BL = {0: '#2a78d6', 4: '#eb6834', 12: '#1baf7a'}
SHORT = {'prior_top1': 'V-JEPA 9-ens top-1 law', 'prior': 'V-JEPA single', 'prior_top2': 'V-JEPA 9-ens +top-2 probes', 'ens_map': 'V-JEPA 9-ens', 'dino_c': 'DINOv3 C ens',
         'lawmean_gt': 'GT-law mean*', 'lawmean_best3': 'best-of-3 law means', 'gtnoise_0.25': 'GT + 0.25 dec*',
         'gtnoise_0.5': 'GT + 0.5 dec*', 'gtnoise_1': 'GT + 1 dec*'}


def figure(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    A = res['arms']
    for r in A.values():
        r['curve'] = {int(k): v for k, v in r['curve'].items()}
    noise = [a for a in ('gtnoise_0.25', 'gtnoise_0.5', 'gtnoise_1') if a in A]
    learned = [a for a in ('prior', 'prior_top1', 'ens_map', 'prior_top2', 'dino_c') if a in A]
    uninf = [a for a in ('lawmean_gt', 'lawmean_best3') if a in A]
    x0 = lambda a: A[a]['curve'][A[a]['init_budget']]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={'width_ratios': [1.2, 1]})
    lim = max(x0(a) for a in A) * 1.08
    ymax = max(max(A[a]['curve'].get(k, 0) for k in (4, 12)) for a in A) * 1.15
    ax.plot([0, lim], [0, lim], color=GRID, lw=1, ls=':', zorder=0)
    ax.text(ymax * .93, ymax * .93, 'y = x', ha='right', va='bottom', fontsize=8, color=INK2)
    t24 = res['targets']['T24_lawmean_gt']
    ax.axhline(t24, color=INK2, lw=1, ls='--', zorder=0)
    ax.text(lim * .99, t24 + .006, f'target: GT-law mean @24 sims = {t24:.3f}', ha='right', va='bottom', fontsize=8, color=INK2)
    for k in (4, 12):
        if noise:
            ax.plot([x0(a) for a in noise], [A[a]['curve'][k] for a in noise], color=BL[k], lw=2, zorder=3)
        for a in noise + learned + uninf:
            if k in A[a]['curve']:
                mk = 'o' if a in noise else 's' if a in learned else '^'
                ax.plot([x0(a)], [A[a]['curve'][k]], ls='none', marker=mk, ms=8, color=BL[k], mec='white', mew=1.5, zorder=4)
    for k in (4, 12):
        ax.plot([], [], color=BL[k], lw=2, label=f'after {k} sims')
    ax.plot([], [], ls='none', marker='o', color=INK2, label='GT + noise* (joined)')
    ax.plot([], [], ls='none', marker='s', color=INK2, label='learned prior')
    ax.plot([], [], ls='none', marker='^', color=INK2, label='uninformed init')
    ax.set_xlim(0, lim); ax.set_ylim(0, ymax)
    ax.set_xlabel('initial error = z-MAE at 0 sims (best-of-3: after its 3 probes)')
    ax.set_ylabel('identifiable z-MAE after refinement')
    ax.set_title(f'Final vs initial error (val subset, n={res["n_samples"]})', fontsize=10, loc='left')
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    order = sorted(A, key=x0)
    ys = list(range(len(order)))[::-1]
    for y, a in zip(ys, order):
        pts = [(0, x0(a))] + [(k, A[a]['curve'][k]) for k in (4, 12) if k in A[a]['curve']]
        vals = [v for _, v in pts]
        bx.plot([min(vals), max(vals)], [y, y], color=GRID, lw=1, zorder=1)
        for k, v in pts:
            bx.plot([v], [y], marker='o', ms=7, color=BL[k], mec='white', mew=1.2, ls='none', zorder=3)
        st = A[a]['sims_to_target']['T24_lawmean_gt']
        s = f"{st['estimate']:.1f}" if st['estimate'] is not None else f">{st['censored_above']}"
        bx.text(1.02, y, s, transform=bx.get_yaxis_transform(), va='center', fontsize=8, color=INK)
    bx.text(1.02, ys[0] + .6, 'sims to\ntarget', transform=bx.get_yaxis_transform(), va='bottom', fontsize=7.5, color=INK2)
    bx.axvline(t24, color=INK2, lw=1, ls='--', zorder=0)
    bx.set_yticks(ys); bx.set_yticklabels([SHORT[a] for a in order], fontsize=8)
    bx.set_ylim(-.7, len(order) + .2)
    for k in (0, 4, 12):
        bx.plot([], [], marker='o', ls='none', color=BL[k], label=f'{k} sims')
    bx.legend(fontsize=7.5, frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.0))
    bx.set_xlabel('identifiable z-MAE'); bx.set_title('Per initialization, sorted by initial error', fontsize=10, loc='left')
    for z in (ax, bx):
        z.spines[['top', 'right']].set_visible(False); z.grid(alpha=.25, lw=.6)
    fig.text(0.01, 0.01, '* privileged (GT law). "sims to target": first (interpolated) budget whose pooled error <= target; '
             '">k": not reached within the arm\'s budget (new arms 12, reused 24). CIs in PRIOR_QUALITY.json.', fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, .04, .96, 1)); fig.savefig(path, dpi=180); plt.close(fig)


if __name__ == '__main__':
    figure(json.loads((ROOT / 'docs/v3h/PRIOR_QUALITY.json').read_text())['main'], ROOT / 'docs/v3h/prior_quality.png')
