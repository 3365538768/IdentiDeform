"""F9: real-video replay on Spring-Gaus real captures (V3v). Left: held-out-camera strip (observed vs. replayed twins,
2 objects x 3 times; white contour = observed mask). Right: IoU and PSNR per arm (held-out camera / future frames,
mean over n=5 objects, bootstrap 95% CI). Reads docs/v3v/RESULTS.json and data/v3v/fig/<obj>_<cam>.npz.
  .engine-envs/plot/bin/python scripts/paper_figs/make_fig_real.py [--objs bun potato] [--cam C0787]
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/paper_figs'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'errorbar.capsize': 1.2})
OI = dict(blue='#0072B2', verm='#D55E00', green='#009E73', orange='#E69F00', grey='#999999', black='#000000')
ARMS = [('prior', 'Prior'), ('prior_refine', 'Prior+\nref. $E$'), ('prior_refine_damp', 'Prior+ref.\n$E$,damp$^*$'),
        ('class_mean', 'Class\nmean'), ('global_constant_loo', 'Global\nconst.'), ('oracle', 'Oracle')]
SPLITS = [('heldout_camera', 'Held-out camera', OI['blue']), ('future_frames', 'Future frames', OI['verm'])]

ap = argparse.ArgumentParser(); ap.add_argument('--objs', nargs=2, default=['bun', 'potato']); ap.add_argument('--cam', default='C0787')
a = ap.parse_args()
R = json.loads((ROOT / 'docs/v3v/RESULTS.json').read_text()); S = R['summary']
arms = [(k, l) for k, l in ARMS if k in S and 'heldout_camera' in S[k]]

fig = plt.figure(figsize=(6.9, 2.75))
gl = fig.add_gridspec(3, 7, left=0.045, right=0.6, top=0.88, bottom=0.03, wspace=0.04, hspace=0.06, width_ratios=[1, 1, 1, 0.12, 1, 1, 1])
ROWS = [('obs', 'Observed'), ('prior', 'Prior'), ('prior_refine', 'Prior+refine')]
for j, obj in enumerate(a.objs):
    z = np.load(ROOT / 'data/v3v/fig' / f'{obj}_{a.cam}.npz'); frames = [int(f) for f in z['frames']]; lg = z['lg']
    for r, (arm, lab) in enumerate(ROWS):
        for t, k in enumerate(frames):
            ax = fig.add_subplot(gl[r, 4 * j + t]); ax.imshow(z[f'{arm}_{k}_img']); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if arm != 'obs':
                ax.contour(z[f'obs_{k}_msk'].astype(float), levels=[0.5], colors='white', linewidths=0.5)
            if r == 0:
                ax.set_title(f'{obj}, t = {k / 119.88 * 1000:.0f} ms' if t == 0 else f't = {k / 119.88 * 1000:.0f} ms', fontsize=6.5, pad=2)
            if t == 0 and j == 0:
                ax.set_ylabel(lab, fontsize=7)
            if arm != 'obs' and t == 2:
                ax.text(0.97, 0.04, f'log E/ρ={lg[r - 1]:.2f}', transform=ax.transAxes, ha='right', va='bottom', color='white', fontsize=5)
fig.text(0.32, 0.965, f'Held-out camera ({a.cam}); white contour = observed mask', ha='center', fontsize=7)

x = np.arange(len(arms)); w = 0.38
for i, (met, ylab, lim) in enumerate([('iou', 'Mask IoU', (0, 0.75)), ('psnr', 'PSNR (dB)', (5, 13))]):
    ax = fig.add_axes([0.675, 0.58 - i * 0.40, 0.315, 0.30])
    for s, (blk, lab, c) in enumerate(SPLITS):
        m = np.array([S[k][blk][met]['mean'] for k, _ in arms])
        ci = np.array([S[k][blk][met]['ci95_bootstrap_objects'] for k, _ in arms])
        ax.bar(x + (s - 0.5) * w, m - lim[0], w * 0.92, bottom=lim[0], color=c, label=lab,
               yerr=[m - ci[:, 0], ci[:, 1] - m], error_kw=dict(lw=0.6, ecolor='#333333'))
    ax.set_xlim(-0.6, len(arms) - 0.4); ax.set_ylim(*lim); ax.set_ylabel(ylab)
    ax.grid(axis='y', color='#e3e3e3', lw=0.5); ax.set_axisbelow(True)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms] if i == 1 else [], fontsize=6)
    if i == 0:
        ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.98), handlelength=1.0, columnspacing=0.8)
fig.text(0.83, 0.012, '$n{=}5$ objects; bars: bootstrap 95% CI.  $^*$post hoc', ha='center', fontsize=5.5, color='#555555')
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'F9_real.pdf'); fig.savefig(OUT / 'F9_real.png', dpi=300)
print('wrote', OUT / 'F9_real.pdf')
