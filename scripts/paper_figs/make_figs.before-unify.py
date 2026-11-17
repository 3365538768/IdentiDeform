"""CVPR paper figures (vector PDF + 300-dpi PNG) -> docs/paper_figs/.

Every plotted number is read from the result files listed in docs/paper_figs/SOURCES.md; F1 bands come from
docs/paper_figs/data/f1_bands.json (scripts/paper_figs/compute_f1_bands.py, object bootstrap of stored per-sample
errors, verified against the stored curves/CIs).

  .engine-envs/plot/bin/python scripts/paper_figs/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/paper_figs'
J = lambda p: json.loads((ROOT / p).read_text())

COL, FULL = 3.25, 6.875
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'lines.linewidth': 1.3, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'errorbar.capsize': 1.5,
})
# Okabe-Ito (colour-blind safe); each method keeps its colour in every figure.
OI = dict(blue='#0072B2', sky='#56B4E9', orange='#E69F00', green='#009E73', verm='#D55E00', purple='#CC79A7',
          yellow='#F0E442', black='#000000', grey='#7f7f7f', lgrey='#bdbdbd')
STY = {
    'prior_top1': dict(c=OI['blue'], ls='-', m='o', label='Video prior, top-1 law (ours)'),
    'prior_top2': dict(c=OI['sky'], ls='--', m='o', label='Video prior + top-2 law probes'),
    'lawmean_gt': dict(c=OI['orange'], ls='-.', m='^', label='GT-law mean (privileged)'),
    'lawmean_best3': dict(c=OI['purple'], ls=':', m='^', label='Best-of-3 law means'),
    'prior': dict(c=OI['verm'], ls='-', m='o', label='V-JEPA 2 single'),
    'dino_c': dict(c=OI['green'], ls='-', m='o', label='DINOv3 ens.'),
    'gtnoise_0.25': dict(c='#4d4d4d', ls='-', m='s', label='GT + 0.25 dec (priv.)'),
    'gtnoise_0.5': dict(c='#808080', ls='-', m='s', label='GT + 0.5 dec (priv.)'),
    'gtnoise_1': dict(c='#b3b3b3', ls='-', m='s', label='GT + 1 dec (priv.)'),
}


PADX, PADY = 0.03, 0.02


def save(fig, name):
    """Exact figure width (figsize width) with symmetric left/right whitespace.

    The tight bounding box of all artists is measured and the axes are rescaled horizontally about their common
    centre (decorations keep their size) until the content spans W - 2*PADX. The saved page is then
    exactly W wide, centred on the content, and tight in the vertical direction."""
    from matplotlib.transforms import Bbox
    W = fig.get_figwidth()
    prev = None
    for _ in range(10):
        fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer())
        if abs(bb.width - (W - 2 * PADX)) < 0.01 or (prev is not None and abs(prev - bb.width) < 1e-3 and bb.width <= W - 2 * PADX):
            break
        prev = bb.width
        pos = [a.get_position() for a in fig.axes]
        a0 = min(p_.x0 for p_ in pos) * W; a1 = max(p_.x1 for p_ in pos) * W
        decor = bb.width - (a1 - a0)
        sc = (W - 2 * PADX - decor) / (a1 - a0)
        c = (a0 + a1) / 2
        for a, p_ in zip(fig.axes, pos):
            x0 = (c + (p_.x0 * W - c) * sc) / W; x1 = (c + (p_.x1 * W - c) * sc) / W
            a.set_position([x0, p_.y0, x1 - x0, p_.height])
    fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer())
    cx = (bb.x0 + bb.x1) / 2
    box = Bbox.from_extents(cx - W / 2, bb.y0 - PADY, cx + W / 2, bb.y1 + PADY)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches=box)
    plt.close(fig)
    print('wrote', name, f'width={box.width:.4f}in margins L={bb.x0 - box.x0:.3f} R={box.x1 - bb.x1:.3f}')


def plabel(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, f'({s})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, fontweight='bold')


# --------------------------------------------------------------------------------------------- F1
def curve_panel(ax, arms_rows, order, annot, above=True):
    for a in order:
        rows = arms_rows[a]
        ks = sorted(int(k) for k in rows if not (a == 'lawmean_best3' and int(k) < 3))
        m = np.array([rows[str(k)]['mean'] for k in ks])
        lo = np.array([rows[str(k)]['ci95'][0] for k in ks]); hi = np.array([rows[str(k)]['ci95'][1] for k in ks])
        s = STY[a]
        if a != 'prior_top2':
            ax.fill_between(ks, lo, hi, color=s['c'], alpha=0.12, lw=0)
        mk = [i for i, k in enumerate(ks) if k in (0, 3, 4, 8, 12, 16, 24)]
        ax.plot(ks, m, color=s['c'], ls=s['ls'], marker='o' if a == 'prior_top1' else None, ms=3.2, markevery=mk,
                label=s['label'],
                mec='white', mew=0.4, zorder=3)
    y0, k_match, txt = annot
    ax.annotate('', xy=(k_match, y0), xytext=(0, y0),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=0.7, shrinkA=0, shrinkB=0), zorder=4)
    ax.plot([k_match, k_match], [y0, y0 + 0.0], color='#333333', lw=0.6)
    ax.text(0.3, y0 + (0.015 if above else -0.015), txt, ha='left', va='bottom' if above else 'top', fontsize=7,
            color='#222222')
    ax.set_xlim(-0.5, 24.5); ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Number of simulations $k$')


def fig1():
    B = J('docs/paper_figs/data/f1_bands.json')
    c1 = J('docs/v3b/C1_REFINE_RESULTS.json'); t = J('docs/v3d/C1_TEST_RESULTS.json')
    gr = J('docs/v3e/GRIPPER_TASK.json')
    order = ['lawmean_best3', 'lawmean_gt', 'prior_top2', 'prior_top1']
    k_val = c1['prior_top2']['speedup']['lawmean_gt']['by_prior_budget']['0']['uninformed_sims_to_match']
    k_test = t['prior_top1_addendum']['speedup']['lawmean_gt']['by_prior_budget']['0']['uninformed_sims_to_match']
    y_val = B['val60']['arms']['prior_top1']['0']['mean']; y_test = B['test30']['arms']['prior_top1']['0']['mean']
    fig = plt.figure(figsize=(FULL, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.55], wspace=0.42)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax1); ax3 = fig.add_subplot(gs[2])
    curve_panel(ax1, B['val60']['arms'], order, (y_val, k_val, f'{k_val:.1f} sims saved'))
    curve_panel(ax2, B['test30']['arms'], order, (y_test, k_test, f'{k_test:.1f} sims saved'), above=False)
    ax1.set_ylabel('Identifiable error (z-MAE)')
    ax1.set_ylim(0, 1.0)
    ax1.text(0.98, 0.97, f"Val, {B['val60']['n_samples']} samples / {B['val60']['n_objects']} objects",
             transform=ax1.transAxes, ha='right', va='top', fontsize=7)
    ax2.text(0.98, 0.97, f"Test, {B['test30']['n_samples']} samples / {B['test30']['n_objects']} objects",
             transform=ax2.transAxes, ha='right', va='top', fontsize=7)
    plt.setp(ax2.get_yticklabels(), visible=False)
    h, l = ax1.get_legend_handles_labels()
    plabel(ax1, 'a'); plabel(ax2, 'b')
    # legend left edge flush with the left edge of panel (a) incl. its y label / panel letter; the offset is in
    # inches, so it survives the horizontal axes rescaling in save()
    from matplotlib.transforms import ScaledTranslation
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    dx = (ax1.get_window_extent(r).x0 - ax1.get_tightbbox(r).x0) / fig.dpi
    fig.legend(h[::-1], l[::-1], loc='lower left', ncol=4, bbox_to_anchor=(0, 1.10), handlelength=2.6,
               columnspacing=1.0, borderaxespad=0, borderpad=0,
               bbox_transform=ax1.transAxes + ScaledTranslation(-dx, 0, fig.dpi_scale_trans))
    # (c) gripper success
    gripper_bars(ax3, gr, 'success', compact=True)
    plabel(ax3, 'c', x=-0.48)
    save(fig, 'F1_teaser')


GRIP_ORDER = ['gt', 'refined_top2', 'ensemble_prior', 'refined_best3', 'fixed_loo', 'lawmean_pred']
GRIP_LAB = {'gt': 'GT parameters (oracle)', 'refined_top2': 'Prior + 12 sims (ours)',
            'ensemble_prior': 'Prior, 0 sims (ours)', 'refined_best3': 'Best-of-3 law means + 12 sims',
            'fixed_loo': 'Fixed depth (LOO-tuned)', 'lawmean_pred': 'Law mean, predicted law'}
GRIP_SHORT = {'gt': 'GT oracle', 'refined_top2': 'Prior+12', 'ensemble_prior': 'Prior',
              'refined_best3': 'Best3+12', 'fixed_loo': 'Fixed', 'lawmean_pred': 'Law mean'}
GRIP_COL = {'gt': OI['black'], 'refined_top2': OI['blue'], 'ensemble_prior': OI['sky'], 'refined_best3': OI['purple'],
            'fixed_loo': OI['lgrey'], 'lawmean_pred': OI['orange']}


def gripper_bars(ax, gr, key, compact=False):
    S = gr['summary']; y = np.arange(len(GRIP_ORDER))[::-1]
    for yi, a in zip(y, GRIP_ORDER):
        r = S[a]['all']
        v = r['success'] if key == 'success' else r['regret_mean']
        lo, hi = r[f'{key}_ci95_object_bootstrap'] if key == 'success' else r['regret_ci95_object_bootstrap']
        ax.barh(yi, v, height=0.62, color=GRIP_COL[a], edgecolor='white', lw=0.5)
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt='none', ecolor='#333333', elinewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels([(GRIP_SHORT if compact else GRIP_LAB)[a] for a in GRIP_ORDER], fontsize=7)
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)
    if key == 'success':
        ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel(f"Gripper success (n={gr['n_samples']})")
    else:
        ax.set_xlabel('Depth regret $|d-d^*|$')


# --------------------------------------------------------------------------------------------- F2
def fig2():
    P = J('docs/v3h/PRIOR_QUALITY.json'); A = P['main']['arms']; fit = P['scaling_fit_no_probe_arms']
    arms = ['prior_top1', 'prior', 'dino_c', 'gtnoise_0.25', 'gtnoise_0.5', 'gtnoise_1', 'lawmean_gt', 'lawmean_best3']
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.85), sharey=True, gridspec_kw=dict(wspace=0.12))
    k_floor = '12'
    floor_arms = [a for a in fit['12']['arms'] if A[a]['curve']['0'] <= 0.35]
    fl = [A[a]['curve'][k_floor] for a in floor_arms]
    for ax, k in zip(axs, ('4', '12')):
        ax.axhspan(min(fl), max(fl), color=OI['yellow'], alpha=0.35, lw=0, zorder=0)
        xs = np.linspace(0.12, 0.72, 10)
        ax.plot(xs, xs, color='#999999', lw=0.6, ls=(0, (2, 2)), zorder=1)
        f = fit[k]
        ax.plot(xs, f['slope'] * xs + f['intercept'], color='#333333', lw=0.8, zorder=2)
        ax.text(0.97, 0.03, f"slope {f['slope']:.2f}, $r$={f['r']:.2f}", transform=ax.transAxes, ha='right',
                va='bottom', fontsize=7)
        for a in arms:
            s = STY[a]; k0 = str(A[a]['init_budget'])
            x, (xl, xh) = A[a]['mae'][k0]['mean'], A[a]['mae'][k0]['ci95']
            yv, (yl, yh) = A[a]['mae'][k]['mean'], A[a]['mae'][k]['ci95']
            hollow = a not in fit[k]['arms']
            ax.errorbar(x, yv, xerr=[[x - xl], [xh - x]], yerr=[[yv - yl], [yh - yv]], fmt='none',
                        ecolor=s['c'], elinewidth=0.5, alpha=0.6, zorder=2)
            ax.plot(x, yv, marker=s['m'], ms=4.2, color=s['c'], mfc='white' if hollow else s['c'],
                    mec=s['c'] if hollow else 'white', mew=0.8 if hollow else 0.4, ls='none', zorder=3)
        ax.set_xlim(0.12, 0.92); ax.set_ylim(0.1, 0.92)
        ax.set_xlabel('Initial error (0 sims)')
        ax.text(0.03, 0.97, f'after {k} sims', transform=ax.transAxes, ha='left', va='top', fontsize=7)
    axs[0].set_ylabel('Error after $k$ sims')
    axs[1].text(0.905, (min(fl) + max(fl)) / 2, 'floor', ha='right', va='center', fontsize=7, color='#6b5d00')
    hs = [Line2D([], [], marker=STY[a]['m'], color=STY[a]['c'], ls='none', ms=4,
                 mfc='white' if a == 'lawmean_best3' else STY[a]['c'],
                 label=STY[a]['label'].replace(' (ours)', '') + (' (3 probes)' if a == 'lawmean_best3' else ''))
          for a in arms]
    hs += [Line2D([], [], color='#333333', lw=0.8, label='Linear fit (filled arms)'),
           Patch(color=OI['yellow'], alpha=0.35, label='Refinement floor')]
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=7,
               handletextpad=0.3, columnspacing=0.8, labelspacing=0.25)
    plabel(axs[0], 'a'); plabel(axs[1], 'b')
    save(fig, 'F2_init_vs_refined')
    return floor_arms, min(fl), max(fl)


# --------------------------------------------------------------------------------------------- F3
def mean_over_laws(run, prefix, laws):
    return float(np.mean([run['within_law_r'][f'{prefix}/{l}/all'] for l in laws]))


def dotbar(ax, x, vals, color, w, hatch=None, label=None):
    m = float(np.mean(vals))
    ax.bar(x, m, w, color=color, edgecolor='white' if not hatch else color, lw=0.5, hatch=hatch,
           fill=hatch is None, label=label)
    ax.plot(np.full(len(vals), x) + np.linspace(-w / 4, w / 4, len(vals)), vals, 'o', ms=2, color='#222222',
            zorder=4)
    return m


V1C, V2C = '#b0b0b0', OI['blue']  # "V1 / baseline condition" vs "V2 / improved condition", used in F3 and F3s


def fig3_bars(ax, groups, metrics=(r'$r(E/\rho)$', r'$r(\sigma_y/\rho)$'), w=0.34):
    """groups: list of (label, colour, [per-seed values metric0], [per-seed values metric1])."""
    n = len(groups)
    for gi, (lab, c, *vals) in enumerate(groups):
        off = (gi - (n - 1) / 2) * (w + 0.02)
        for mi, v in enumerate(vals):
            ax.bar(mi + off, float(np.mean(v)), w, color=c, lw=0, label=lab if mi == 0 else None)
            ax.plot(np.full(len(v), mi + off) + np.linspace(-w / 5, w / 5, len(v)), v, 'o', ms=2.2,
                    color='#222222', zorder=4)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.grid(axis='x', visible=False)


def fig3():
    p1c = J('docs/p1c/P1C_RESULTS.json')['groups']['Cvg']['within_law_r']
    v2 = J('docs/v2main/V2MAIN_RESULTS.json')['groups']['C']['within_law_r']
    p4 = J('docs/p4d/P4D_RESULTS.json')['arms']
    fig = plt.figure(figsize=(FULL, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    fig3_bars(ax, [('V1 data', V1C, p1c['E'], p1c['sigma_y']), ('V2 data', V2C, v2['E'], v2['sigma_y'])])
    ax.set_ylabel('Within-law Pearson $r$')
    ax.legend(loc='upper left', handlelength=1.0, handletextpad=0.4)
    plabel(ax, 'a')
    ax = fig.add_subplot(gs[1])
    pe = lambda a: [mean_over_laws(r, 'E_over_rho', ('jelly', 'metal', 'plasticine')) for r in p4[a]['runs']]
    ps = lambda a: [mean_over_laws(r, 'sigma_over_rho', ('metal', 'plasticine')) for r in p4[a]['runs']]
    fig3_bars(ax, [('V1 damping', V1C, pe('v1'), ps('v1')), ('Low damping', V2C, pe('low'), ps('low'))])
    ax.legend(loc='upper left', handlelength=1.0, handletextpad=0.4)
    plabel(ax, 'b')
    ax = fig.add_subplot(gs[2])
    R = {m: {r['case']: r['relative_rms'] for r in J(f'docs/r2/MPM_IDENTIFIABILITY_{m}.json')['results']
             if r['action'] == 'press'} for m in ('jelly', 'plasticine', 'metal')}
    joint = [R[m]['E_rho_sigma_x4'] for m in ('jelly', 'plasticine', 'metal')]
    bars = [(r'$E\times4$' + '\njelly', R['jelly']['E_x4'], OI['blue']),
            (r'$E\times4$' + '\nmetal', R['metal']['E_x4'], OI['blue']),
            (r'$(E,\rho,\sigma_y)\times4$' + '\nall materials', max(joint), '#333333')]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, 0.55, color=c, lw=0)
    ax.plot(np.full(3, 2) + np.array([-0.12, 0, 0.12]), joint, 'o', ms=2.2, color='white', mec='#222222', mew=0.5,
            zorder=4)
    ax.axhline(max(joint), color=OI['verm'], lw=0.8, ls='--', zorder=3)
    ax.text(-0.35, max(joint) * 2.2, 'numerical floor', color=OI['verm'], fontsize=7, ha='left', va='bottom')
    ax.set_yscale('log'); ax.set_ylim(3e-7, 2)
    ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel('Relative trajectory change'); ax.grid(axis='x', visible=False); ax.minorticks_off()
    plabel(ax, 'c')
    save(fig, 'F3_identifiability')


def fig3s():
    """Supplementary: V1 vs V2 within-law r for both backbone families."""
    r1b = J('docs/r1b/R1B_RESULTS.json')['groups']['C']['within_law_r']
    p1c = J('docs/p1c/P1C_RESULTS.json')['groups']['Cvg']['within_law_r']
    v2 = J('docs/v2main/V2MAIN_RESULTS.json')['groups']['C']['within_law_r']
    bv = J('docs/v3c/BASELINES_VAL.json')['methods']['dinov3_16f_C']['within_law_r']
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    fig3_bars(axs[0], [('V1 data', V1C, r1b['E'], r1b['sigma_y']),
                       ('V2 data', V2C, bv['E_over_rho']['values'], bv['sigma_y_over_rho']['values'])])
    fig3_bars(axs[1], [('V1 data', V1C, p1c['E'], p1c['sigma_y']), ('V2 data', V2C, v2['E'], v2['sigma_y'])])
    axs[0].set_ylabel('Within-law Pearson $r$')
    axs[0].set_xlabel('DINOv3 (C)'); axs[1].set_xlabel('V-JEPA 2 (C; V1 ViT-g, V2 ViT-L)')
    axs[0].legend(loc='upper left', handlelength=1.0, handletextpad=0.4)
    plabel(axs[0], 'a'); plabel(axs[1], 'b')
    save(fig, 'F3s_identifiability_dinov3')


# --------------------------------------------------------------------------------------------- F4
def fig4():
    A = J('docs/v3i/ABLATIONS.json')
    objs = {'f10': 9, 'f25': 22, 'f50': 44, 'f100': 87}  # stated in ABLATIONS.json protocol string
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.55), gridspec_kw=dict(wspace=0.42))
    for g, c, m, lab in (('B', OI['sky'], 'o', 'B (no field sup.)'), ('C', OI['blue'], 'o', 'C (field sup.)')):
        xs = [objs[f] for f in objs]
        rows = [A['scaling'][f'{g}_{f}'] for f in objs]
        mu = [r['val_mae']['mean'] for r in rows]; sd = [r['val_mae']['sd'] for r in rows]
        axs[0].errorbar(xs, mu, yerr=sd, color=c, marker=m, ms=3.5, label=lab, ls='-' if g == 'C' else '--',
                        mec='white', mew=0.4)
    for key, c, m, lab in (('within_law_r_E_over_rho', OI['blue'], 'o', r'$r(E/\rho)$'),
                           ('within_law_r_sigma_y_over_rho', OI['verm'], 'o', r'$r(\sigma_y/\rho)$')):
        xs = [objs[f] for f in objs]; rows = [A['scaling'][f'C_{f}'] for f in objs]
        axs[1].errorbar(xs, [r[key]['mean'] for r in rows], yerr=[r[key]['sd'] for r in rows], color=c, marker=m,
                        ms=3.5, label=lab, mec='white', mew=0.4)
        vrows = [A['views'][f'C_v{v}'] for v in (1, 2, 3)]
        axs[3].errorbar([1, 2, 3], [r[key]['mean'] for r in vrows], yerr=[r[key]['sd'] for r in vrows], color=c,
                        marker=m, ms=3.5, label=lab, mec='white', mew=0.4)
    vrows = [A['views'][f'C_v{v}'] for v in (1, 2, 3)]
    axs[2].errorbar([1, 2, 3], [r['val_mae']['mean'] for r in vrows], yerr=[r['val_mae']['sd'] for r in vrows],
                    color=OI['blue'], marker='o', ms=3.5, mec='white', mew=0.4, label='C (field sup.)')
    for ax in axs[:2]:
        ax.set_xscale('log'); ax.set_xticks([9, 22, 44, 87]); ax.set_xticklabels(['9', '22', '44', '87'])
        ax.minorticks_off(); ax.set_xlabel('Training objects')
    for ax in axs[2:]:
        ax.set_xticks([1, 2, 3]); ax.set_xlim(0.7, 3.3); ax.set_xlabel('Camera views')
    axs[0].set_ylabel('Val z-MAE (all 4)'); axs[2].set_ylabel('Val z-MAE (all 4)')
    axs[1].set_ylabel('Within-law $r$'); axs[3].set_ylabel('Within-law $r$')
    axs[1].set_ylim(0.3, 0.95); axs[3].set_ylim(0.3, 0.95)
    axs[0].legend(loc='upper right', fontsize=7, handlelength=1.6)
    axs[1].legend(loc='lower right', fontsize=7, handlelength=1.2)
    for ax, s in zip(axs, 'abcd'):
        plabel(ax, s)
    save(fig, 'F4_scaling_views')


# --------------------------------------------------------------------------------------------- F5
def fig5():
    V = J('docs/v3c/BASELINES_VAL.json')['methods']; T = J('docs/v3c/BASELINES_TEST.json')['methods']
    PX = J('docs/v3f/PIXIE_BASELINE.json')

    def get(src, key):
        v = src[key]['standardized_mae']
        if isinstance(v, dict):
            return v['mean'], v['sd'], v['values']
        return v, None, None
    rows = [  # (label, val getter, test getter, colour)
        ('Qwen3-VL-8B zero-shot', get(V, 'qwen3vl8b_zeroshot'), get(T, 'qwen3vl8b_zeroshot'), OI['lgrey']),
        ('PIXIE (released)', (PX['val']['filled']['standardized_mae'], None, None),
         (PX['test']['filled']['standardized_mae'], None, None), OI['verm']),
        ('PIXIE (train-calibrated)', (PX['val']['filled_train_calibrated']['standardized_mae'], None, None),
         (PX['test']['filled_train_calibrated']['standardized_mae'], None, None), OI['verm']),
        ('Constant train mean', get(V, 'constant_train_mean'), get(T, 'constant_train_mean'), OI['lgrey']),
        ('DINOv3 first frame', get(V, 'dinov3_static_first_frame_B'), get(T, 'dinov3_static_first_frame_B'), OI['green']),
        ('GT-law mean (privileged)', get(V, 'law_mean_gt_privileged'), get(T, 'law_mean_gt_privileged'), OI['orange']),
        ('DINOv3 16f, B', get(V, 'dinov3_16f_B'), get(T, 'dinov3_16f_B'), OI['green']),
        ('DINOv3 16f, C', get(V, 'dinov3_16f_C'), get(T, 'dinov3_16f_C'), OI['green']),
        ('V-JEPA 2, B', get(V, 'vjepa2_32f_B (reference)'), get(T, 'vjepa2_32f_B'), OI['blue']),
        ('V-JEPA 2, C', get(V, 'vjepa2_32f_C (reference)'), get(T, 'vjepa2_32f_C'), OI['blue']),
        ('V-JEPA 2, D', get(V, 'vjepa2_32f_D (reference)'), get(T, 'vjepa2_32f_D'), OI['blue']),
        ('V-JEPA 2 ensemble (9)', (T['vjepa2_ensemble_BCD9']['val']['standardized_mae'], None, None),
         get(T, 'vjepa2_ensemble_BCD9'), OI['blue']),
    ]
    rows.sort(key=lambda r: -r[2][0])  # ordered by test MAE, best at bottom
    fig = plt.figure(figsize=(COL, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1], wspace=0.06)
    axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1], sharey=axL)
    y = np.arange(len(rows)); h = 0.38
    for ax in (axL, axR):
        for yi, (lab, v, t, c) in enumerate(rows):
            for off, (m, sd, vals), alpha, hatch in ((h / 2, v, 0.45, None), (-h / 2, t, 1.0, None)):
                ax.barh(yi + off, m, h * 0.95, color=c, alpha=alpha, lw=0)
                if sd is not None:
                    ax.errorbar(m, yi + off, xerr=sd, fmt='none', ecolor='#222222', elinewidth=0.6, capsize=1.2)
                    ax.plot(vals, np.full(3, yi + off), '|', ms=2.5, color='#222222', mew=0.5)
        ax.grid(axis='y', visible=False)
    axL.set_xlim(0, 0.9); axR.set_xlim(1.3, 1.65)
    axL.set_yticks(y); axL.set_yticklabels([r[0] for r in rows], fontsize=7)
    axL.tick_params(axis='y', length=0)
    plt.setp(axR.get_yticklabels(), visible=False); axR.tick_params(axis='y', length=0)
    axR.spines['left'].set_visible(False); axR.set_xticks([1.4, 1.6])
    d = 0.015
    for ax, xs in ((axL, (1,)), (axR, (0,))):
        for x in xs:
            ax.plot([x - d * (4.2 if x == 0 else 1), x + d * (4.2 if x == 0 else 1)], [-d * 2, d * 2],
                    transform=ax.transAxes, color='k', lw=0.6, clip_on=False)
    axL.set_xlabel('Standardized MAE, all 4 targets (lower is better)', x=0.6)
    hs = [Patch(color='#555555', alpha=0.45, label='Val (283 samples)'), Patch(color='#555555', label='Test (300 samples)')]
    axL.legend(handles=hs, loc='lower center', ncol=2, fontsize=7, bbox_to_anchor=(0.6, 0.995))
    save(fig, 'F5_baselines')


# --------------------------------------------------------------------------------------------- F6
def fig6():
    gr = J('docs/v3e/GRIPPER_TASK.json')
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.6), sharey=True, gridspec_kw=dict(wspace=0.16))
    gripper_bars(axs[0], gr, 'success'); gripper_bars(axs[1], gr, 'regret')
    axs[0].set_xticks([0, 0.5, 1]); axs[0].set_xticklabels(['0', '0.5', '1'])
    axs[1].set_xticks([0, 0.05, 0.1, 0.15]); axs[1].set_xticklabels(['0', '.05', '.10', '.15'])
    axs[0].set_xlabel('Success rate'); axs[1].set_xlabel('Regret $|d-d^*|$')
    plt.setp(axs[1].get_yticklabels(), visible=False)
    plabel(axs[0], 'a'); plabel(axs[1], 'b')
    save(fig, 'F6_gripper')


# --------------------------------------------------------------------------------------------- F7
def fig7():
    b = J('docs/v3h/PRIOR_QUALITY.json')['breakdown_60']
    groups = [(a, b['action'][a]) for a in ('bend', 'drop', 'press', 'shear', 'stretch', 'tap30hold')] + \
             [(l, b['gt_law'][l]) for l in ('jelly', 'metal', 'plasticine')]
    names = {'tap30hold': 'tap-hold'}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FULL, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.22))
    x = np.arange(len(groups)) + np.array([0] * 6 + [0.6] * 3)
    w = 0.19
    spec = [('prior_top1', '0', 0.45), ('lawmean_gt', '0', 0.45), ('prior_top1', '24', 1.0), ('lawmean_gt', '24', 1.0)]
    for si, (a, k, al) in enumerate(spec):
        vals = [g[a][k] for _, g in groups]
        ax.bar(x + (si - 1.5) * w, vals, w * 0.92, color=STY[a]['c'], alpha=al, lw=0)
    ax.set_ylabel('Identifiable z-MAE'); ax.grid(axis='x', visible=False)
    hs = [Patch(color=STY['prior_top1']['c'], alpha=0.45, label='Prior, 0 sims'),
          Patch(color=STY['lawmean_gt']['c'], alpha=0.45, label='GT-law mean, 0 sims'),
          Patch(color=STY['prior_top1']['c'], label='Prior, 24 sims'),
          Patch(color=STY['lawmean_gt']['c'], label='GT-law mean, 24 sims')]
    ax.legend(handles=hs, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    for k, off, c in (('0', -0.12, OI['lgrey']), ('24', 0.12, '#333333')):
        d = [g['diff'][f'prior_top1_minus_lawmean_gt_k{k}'] for _, g in groups]
        m = np.array([v['mean'] for v in d]); lo = np.array([v['ci95'][0] for v in d]); hi = np.array([v['ci95'][1] for v in d])
        ax2.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt='o', ms=3, color=c, ecolor=c, elinewidth=0.8,
                     label=f'{k} sims', mec='white', mew=0.3)
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_ylabel('Difference\n(95% CI)')
    ax2.set_ylim(-1.0, 0.35)
    ax2.legend(loc='lower right', ncol=2, title='Prior $-$ GT-law mean at', title_fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{names.get(n, n)}\n(n={g['n']})" for n, g in groups])
    ax2.grid(axis='x', visible=False)
    ax2.text((x[0] + x[5]) / 2, -0.58, 'observed action', ha='center', va='top', transform=ax2.get_xaxis_transform(), fontsize=7.5)
    ax2.text((x[6] + x[8]) / 2, -0.58, 'GT material law', ha='center', va='top', transform=ax2.get_xaxis_transform(), fontsize=7)
    ax.set_ylim(0, 0.92)
    plabel(ax, 'a'); plabel(ax2, 'b')
    save(fig, 'F7_breakdown')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fl = fig2(); print('F2 floor arms / band:', fl)
    fig3(); fig3s(); fig4(); fig5(); fig6(); fig7()
